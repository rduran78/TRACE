 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The suspicion is correct: **the computational bottleneck is the spatial neighbor feature construction, not Random Forest inference.**

Here is a precise breakdown of why:

### `build_neighbor_lookup` — O(N) `lapply` with expensive string operations

For each of the ~6.46 million rows, the function:

1. Converts an integer ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Calls `paste()` to construct `"id_year"` key strings for every neighbor of that row.
4. Looks up each key string in `idx_lookup` (a named character vector — i.e., linear-hashed lookup per key).
5. Filters `NA`s and coerces to integer.

With ~6.46M rows and an average of ~4 rook neighbors per cell, this produces roughly **25.8 million `paste` + named-vector lookups**, all inside an interpreted R `lapply` loop. Named vector lookup in R is hash-based but carries significant per-call overhead when done millions of times inside `lapply`. The `paste` calls alone generate tens of millions of temporary string allocations.

**Estimated cost:** The `build_neighbor_lookup` step alone likely accounts for **60–70%** of the 86+ hour runtime.

### `compute_neighbor_stats` — O(N) `lapply` with per-row subsetting

For each of 5 variables × 6.46M rows, the function:

1. Subsets `vals[idx]` (a numeric vector index — fast in isolation).
2. Removes `NA`s.
3. Computes `max`, `min`, `mean`.

This is called 5 times, so ~32.3 million iterations total. The `lapply` + anonymous function overhead and the `do.call(rbind, ...)` on a 6.46M-element list (each a length-3 vector) are both costly. `do.call(rbind, ...)` on a long list is notoriously slow because it incrementally allocates.

**Estimated cost:** ~25–35% of total runtime.

### Summary of root causes

| Cause | Location | Severity |
|---|---|---|
| Millions of `paste()` string constructions | `build_neighbor_lookup` | **Critical** |
| Named-vector lookups inside `lapply` | `build_neighbor_lookup` | **Critical** |
| Per-row `lapply` over 6.46M rows (×5 vars) | `compute_neighbor_stats` | **High** |
| `do.call(rbind, list_of_6.46M_vectors)` | `compute_neighbor_stats` | **High** |
| Redundant per-variable passes over same neighbor structure | Outer loop | **Moderate** |

---

## Optimization Strategy

The key insight is: **replace row-level R loops and string-key lookups with vectorized joins and grouped aggregations using `data.table`.**

### Step 1: Replace `build_neighbor_lookup` entirely

Instead of building a list of 6.46M integer vectors (one per row), construct a **long-format edge table** that maps every `(cell-year row) → (neighbor cell-year row)` using integer joins. This eliminates all `paste` and named-vector lookups.

- Create a `data.table` of the panel with a row index column.
- Expand the `nb` object into a two-column edge data.table: `(id, neighbor_id)`.
- Join on `(neighbor_id, year)` to get the row index of each neighbor in each year.

This is a single equi-join — `data.table` does this in seconds on 25M rows.

### Step 2: Replace `compute_neighbor_stats` with grouped `data.table` aggregation

Once we have the long-format edge table `(focal_row, neighbor_row)`, we attach the neighbor's variable values and compute `max`, `min`, `mean` as a single grouped aggregation per variable — fully vectorized, no `lapply`.

### Step 3: Process all 5 variables in one pass (or 5 fast passes)

Since the edge table is the same for all variables, we can either compute all 5 variables' stats in a single grouped aggregation, or loop over 5 variables with the same edge table. Both are fast.

### Expected speedup

| Component | Before | After | Speedup |
|---|---|---|---|
| Neighbor lookup construction | ~50–60 hours | ~10–30 seconds | **~6000×** |
| Neighbor stats (5 vars) | ~25–30 hours | ~30–90 seconds | **~1500×** |
| **Total neighbor features** | **~80+ hours** | **~1–3 minutes** | **~2000×** |

### What is preserved

- The trained Random Forest model is untouched.
- The numerical output (max, min, mean of non-NA neighbor values per cell-year per variable) is identical.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED SPATIAL NEIGHBOR FEATURE CONSTRUCTION
# Replaces: build_neighbor_lookup + compute_neighbor_stats + outer loop
# Preserves: exact numerical output (max, min, mean of non-NA neighbor values)
# Requires: data.table
# =============================================================================

library(data.table)

#' Convert an spdep nb object into a two-column data.table of directed edges.
#'
#' @param neighbors  An nb object (list of integer vectors of neighbor indices).
#' @param id_order   The vector of cell IDs corresponding to each nb index.
#' @return A data.table with columns: id (focal cell), neighbor_id (neighbor cell).
nb_to_edge_dt <- function(neighbors, id_order) {
  # Determine the number of neighbors per cell (handles 0-neighbor cells)
  n_neighbors <- vapply(neighbors, function(x) {
    # spdep nb objects use 0L to indicate no neighbors
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1))

  focal_idx <- rep(seq_along(neighbors), times = n_neighbors)

  neighbor_idx <- unlist(lapply(neighbors, function(x) {
    if (length(x) == 1L && x[1] == 0L) integer(0) else x
  }), use.names = FALSE)

  data.table(
    id          = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

#' Build all neighbor features for the specified variables and attach them
#' to the panel data.table.
#'
#' @param cell_data              A data.frame or data.table with columns: id, year,
#'                               and all columns named in neighbor_source_vars.
#' @param id_order               Integer vector of cell IDs matching the nb object.
#' @param rook_neighbors_unique  An spdep nb object (precomputed).
#' @param neighbor_source_vars   Character vector of variable names to aggregate.
#' @return cell_data with new columns: <var>_max, <var>_min, <var>_mean for each var.
build_all_neighbor_features <- function(cell_data,
                                        id_order,
                                        rook_neighbors_unique,
                                        neighbor_source_vars) {

  # --- Convert to data.table if needed (by reference if already one) ----------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # --- Step 1: Add a row-index column ----------------------------------------
  cell_data[, .row_idx := .I]

  # --- Step 2: Build the edge table from the nb object -----------------------
  edges <- nb_to_edge_dt(rook_neighbors_unique, id_order)
  # edges now has columns: id (focal), neighbor_id

  # --- Step 3: Create a keyed lookup from (id, year) -> row index ------------
  #     This replaces the entire build_neighbor_lookup function.
  row_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # --- Step 4: Expand edges across all years ---------------------------------
  #     For every (focal_id, neighbor_id) pair, we need every year in the panel.
  years <- sort(unique(cell_data$year))

  # Cross join edges × years  (~1.37M edges × 28 years ≈ 38.5M rows)
  # This is the full set of (focal_id, neighbor_id, year) triples.
  edge_year <- CJ_dt(edges, years)

  # --- Step 5: Attach focal row index ----------------------------------------
  setkey(edge_year, id, year)
  edge_year[row_lookup, focal_row := i..row_idx, on = .(id, year)]

  # --- Step 6: Attach neighbor row index -------------------------------------
  setnames(row_lookup, c("id", "year", ".row_idx"),
           c("neighbor_id", "year", "neighbor_row"))
  setkey(row_lookup, neighbor_id, year)
  edge_year[row_lookup, neighbor_row := i.neighbor_row,
            on = .(neighbor_id, year)]

  # Restore row_lookup names for safety
  setnames(row_lookup, c("neighbor_id", "year", "neighbor_row"),
           c("id", "year", ".row_idx"))

  # --- Step 7: Drop edges where either focal or neighbor is missing ----------
  edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row)]

  # --- Step 8: Compute grouped stats for each variable -----------------------
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)

    # Attach the neighbor's value for this variable
    edge_year[, nval := cell_data[[var_name]][neighbor_row]]

    # Grouped aggregation — only over non-NA neighbor values
    stats <- edge_year[!is.na(nval),
                       .(var_max  = max(nval),
                         var_min  = min(nval),
                         var_mean = mean(nval)),
                       by = focal_row]

    # Initialize new columns to NA
    max_col  <- paste0(var_name, "_max")
    min_col  <- paste0(var_name, "_min")
    mean_col <- paste0(var_name, "_mean")

    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Write results back by row index
    cell_data[stats$focal_row, (max_col)  := stats$var_max]
    cell_data[stats$focal_row, (min_col)  := stats$var_min]
    cell_data[stats$focal_row, (mean_col) := stats$var_mean]

    # Clean up the temporary column
    edge_year[, nval := NULL]
  }

  # --- Clean up ---------------------------------------------------------------
  cell_data[, .row_idx := NULL]

  return(cell_data)
}

#' Helper: Cross join an edge data.table with a vector of years.
#' More memory-efficient than a full CJ on three columns.
CJ_dt <- function(edges, years) {
  # edges: data.table with columns id, neighbor_id
  # years: integer/numeric vector
  n_edges <- nrow(edges)
  n_years <- length(years)

  data.table(
    id          = rep(edges$id,          times = n_years),
    neighbor_id = rep(edges$neighbor_id, times = n_years),
    year        = rep(years, each = n_edges)
  )
}

# =============================================================================
# USAGE — drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_all_neighbor_features(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# cell_data now contains the same 15 new columns (5 vars × 3 stats) as before.
# The trained Random Forest model can be applied directly — no retraining needed.
```

---

## Memory Considerations (16 GB laptop)

The largest intermediate object is `edge_year` after the cross join:

- ~1.37M edges × 28 years ≈ **38.5M rows** × 5 columns (3 integer/numeric IDs + 2 row indices) ≈ **1.5 GB**.
- `cell_data` itself at 6.46M rows × ~125 columns ≈ **6–7 GB**.
- Total peak ≈ **~10–11 GB**, which fits within 16 GB.

If memory is tight, the cross join can be chunked by year (process 4–7 years at a time) with negligible speed penalty. Here is a chunked variant of Step 4 that can be substituted:

```r
# Memory-safe chunked variant (replaces Steps 4–8 above)
build_all_neighbor_features_chunked <- function(cell_data,
                                                 id_order,
                                                 rook_neighbors_unique,
                                                 neighbor_source_vars,
                                                 chunk_size = 7L) {
  if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)
  cell_data[, .row_idx := .I]

  edges <- nb_to_edge_dt(rook_neighbors_unique, id_order)

  row_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  years <- sort(unique(cell_data$year))

  # Pre-allocate result columns
  for (var_name in neighbor_source_vars) {
    cell_data[, paste0(var_name, "_max")  := NA_real_]
    cell_data[, paste0(var_name, "_min")  := NA_real_]
    cell_data[, paste0(var_name, "_mean") := NA_real_]
  }

  # Process in year chunks
  year_chunks <- split(years, ceiling(seq_along(years) / chunk_size))

  for (yc in year_chunks) {
    message("Processing years: ", paste(range(yc), collapse = "–"))

    edge_year <- CJ_dt(edges, yc)

    # Focal row
    setkey(edge_year, id, year)
    edge_year[row_lookup, focal_row := i..row_idx, on = .(id, year)]

    # Neighbor row
    rl2 <- copy(row_lookup)
    setnames(rl2, c("id", "year", ".row_idx"),
             c("neighbor_id", "year", "neighbor_row"))
    setkey(rl2, neighbor_id, year)
    edge_year[rl2, neighbor_row := i.neighbor_row,
              on = .(neighbor_id, year)]

    edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row)]

    for (var_name in neighbor_source_vars) {
      edge_year[, nval := cell_data[[var_name]][neighbor_row]]

      stats <- edge_year[!is.na(nval),
                         .(var_max  = max(nval),
                           var_min  = min(nval),
                           var_mean = mean(nval)),
                         by = focal_row]

      max_col  <- paste0(var_name, "_max")
      min_col  <- paste0(var_name, "_min")
      mean_col <- paste0(var_name, "_mean")

      set(cell_data, i = stats$focal_row, j = max_col,  value = stats$var_max)
      set(cell_data, i = stats$focal_row, j = min_col,  value = stats$var_min)
      set(cell_data, i = stats$focal_row, j = mean_col, value = stats$var_mean)

      edge_year[, nval := NULL]
    }

    rm(edge_year, rl2)
    gc()
  }

  cell_data[, .row_idx := NULL]
  return(cell_data)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Paradigm** | Row-level `lapply` + string keys | Vectorized `data.table` joins + grouped aggregation |
| **`build_neighbor_lookup`** | ~6.46M `paste` + named-vector lookups | Eliminated; replaced by a single keyed join |
| **`compute_neighbor_stats`** | `lapply` over 6.46M rows × 5 vars + `do.call(rbind, ...)` | `data.table` grouped `max`/`min`/`mean` by `focal_row` |
| **Estimated runtime** | **86+ hours** | **1–3 minutes** |
| **Peak memory** | Lower but irrelevant (time-dominated) | ~10–11 GB (fits 16 GB; chunked variant ~7–8 GB) |
| **Numerical output** | max, min, mean of non-NA neighbor values | **Identical** |
| **Trained RF model** | — | **Untouched** |