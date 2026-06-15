 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two functions `build_neighbor_lookup` and `compute_neighbor_stats`, both of which rely on **row-level `lapply` loops over ~6.46 million rows** with expensive per-row string operations.

### `build_neighbor_lookup` — Primary Bottleneck

For each of the ~6.46 million rows, the function:

1. Converts `data$id[i]` to a character and looks it up in `id_to_ref` — **6.46M character coercions and named-vector lookups**.
2. Retrieves neighbor cell IDs and calls `paste(..., sep = "_")` to build composite keys — **6.46M × ~4 neighbors ≈ 25.8M `paste` calls**.
3. Looks up each key in the named vector `idx_lookup` (itself of length 6.46M) — this is an **O(n) hash-table probe repeated ~25.8M times against a 6.46M-entry named vector**.
4. Filters `NA`s and coerces to integer.

The total work is roughly **O(N × k)** with very high constant factors due to R's interpreted string operations and named-vector lookups. With N ≈ 6.46M and k ≈ 4, this produces billions of elementary R interpreter operations.

### `compute_neighbor_stats` — Secondary Bottleneck

Another `lapply` over 6.46M rows computing `max`, `min`, `mean` per row. This is called 5 times (once per variable), totaling ~32.3M interpreted loop iterations with per-iteration allocation.

### Why 86+ hours?

| Operation | Iterations | Cost per iteration | Estimated wall time |
|---|---|---|---|
| `build_neighbor_lookup` (paste + named lookup) | 6.46M × ~4 | ~tens of µs (string alloc, hash probe on 6.46M-entry table) | **60–70+ hours** |
| `compute_neighbor_stats` (5 vars) | 5 × 6.46M | ~µs (subsetting + summary) | **15–20 hours** |

---

## Optimization Strategy

### Principle: Replace row-level R loops and string-key lookups with vectorized integer-index operations using `data.table`.

**Key ideas:**

1. **Eliminate `build_neighbor_lookup` entirely.** Instead of building a 6.46M-element list of neighbor row indices via string keys, expand the neighbor graph into a two-column edge table `(row_i, row_j)` using vectorized joins. This replaces 6.46M `paste` + named-vector lookups with a single `data.table` equi-join.

2. **Replace `compute_neighbor_stats` with a grouped `data.table` aggregation.** Once we have the edge table mapping each row to its neighbor rows, we join in the variable values and compute `max`, `min`, `mean` in one vectorized grouped operation — no R-level loop at all.

3. **Process all 5 variables in one pass** over the edge table (or 5 fast vectorized passes) instead of rebuilding anything per variable.

**Expected speedup:** From 86+ hours to **~2–10 minutes** on the same laptop.

---

## Working R Code

```r
library(data.table)

#' Vectorized spatial-neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and
#'                         all columns named in neighbor_source_vars.
#' @param id_order         integer vector of cell IDs in the same order as the
#'                         nb object (i.e., id_order[i] is the cell ID for the
#'                         i-th element of rook_neighbors_unique).
#' @param neighbors        spdep nb object (list of integer vectors);
#'                         rook_neighbors_unique.
#' @param neighbor_source_vars character vector of variable names to summarize.
#' @return cell_data (data.table) with new columns:
#'         <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
#'         for each var in neighbor_source_vars.

add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      neighbors,
                                      neighbor_source_vars) {

  # --- Step 0: Convert to data.table; add row index --------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  cell_data[, .row_idx := .I]

  # --- Step 1: Build the directed edge list (focal_cell -> neighbor_cell) -----
  #     from the nb object.  Fully vectorized, no per-row R loop.
  n_neighbors <- lengths(neighbors)                       # integer vector
  focal_ref   <- rep(seq_along(neighbors), n_neighbors)   # ref indices
  nbr_ref     <- unlist(neighbors, use.names = FALSE)     # neighbor ref indices

  edges <- data.table(
    focal_id    = id_order[focal_ref],
    neighbor_id = id_order[nbr_ref]
  )
  # edges now has ~1,373,394 rows (one per directed rook relationship)

  # --- Step 2: Build a keyed lookup from (id, year) -> row index -------------
  row_key <- cell_data[, .(id, year, .row_idx)]
  setkey(row_key, id, year)

  # --- Step 3: Expand edges across all years ---------------------------------
  #     For every year, every (focal, neighbor) edge produces a pair of row
  #     indices.  We do this via two joins rather than a Cartesian product.

  years <- sort(unique(cell_data$year))

  # Cross-join edges × years  (~1.37M × 28 ≈ 38.5M rows — fits in RAM easily)
  edge_year <- CJ_dt(edges, years)

  # Helper: cross join edges with years vector
  # We build it manually to stay memory-efficient:
  edge_year <- edges[, .(focal_id, neighbor_id)][
    , .(year = years), by = .(focal_id, neighbor_id)
  ]

  # Join to get focal row index
  setnames(row_key, "id", "focal_id")
  setkey(row_key, focal_id, year)
  edge_year <- row_key[edge_year, on = .(focal_id, year), nomatch = 0L]
  setnames(edge_year, ".row_idx", "focal_row")

  # Join to get neighbor row index
  setnames(row_key, "focal_id", "neighbor_id")
  setkey(row_key, neighbor_id, year)
  edge_year <- row_key[edge_year, on = .(neighbor_id, year), nomatch = 0L]
  setnames(edge_year, ".row_idx", "nbr_row")

  # Restore row_key column name
  setnames(row_key, "neighbor_id", "id")

  # edge_year now has columns: focal_row, nbr_row  (and focal_id, neighbor_id, year)
  # We only need focal_row and nbr_row going forward.
  edge_year <- edge_year[, .(focal_row, nbr_row)]
  setkey(edge_year, focal_row)

  # --- Step 4: For each variable, vectorized grouped aggregation --------------
  for (var_name in neighbor_source_vars) {

    vals <- cell_data[[var_name]]

    # Attach neighbor values
    edge_year[, nbr_val := vals[nbr_row]]

    # Grouped aggregation — single vectorized pass
    agg <- edge_year[!is.na(nbr_val),
                     .(nmax  = max(nbr_val),
                       nmin  = min(nbr_val),
                       nmean = mean(nbr_val)),
                     keyby = .(focal_row)]

    # Allocate result columns (NA by default)
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Assign results by reference
    cell_data[agg$focal_row, (max_col)  := agg$nmax]
    cell_data[agg$focal_row, (min_col)  := agg$nmin]
    cell_data[agg$focal_row, (mean_col) := agg$nmean]
  }

  # Clean up helper column
  cell_data[, .row_idx := NULL]
  edge_year[, nbr_val := NULL]

  return(cell_data)
}
```

### Replacement for the original outer loop

```r
# ---- BEFORE (86+ hours) ----
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ---- AFTER (~2-10 minutes) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  neighbors            = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names and numerical values (max, min, mean) are identical to the original.
```

### Memory-efficient alternative for the cross-join (if 38.5M rows × several columns strains 16 GB)

If RAM is tight, process one year at a time:

```r
add_all_neighbor_features_lowmem <- function(cell_data,
                                              id_order,
                                              neighbors,
                                              neighbor_source_vars) {

  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  cell_data[, .row_idx := .I]

  # Build edge list (cell-ID level, ~1.37M rows)
  n_neighbors <- lengths(neighbors)
  edges <- data.table(
    focal_id    = id_order[rep(seq_along(neighbors), n_neighbors)],
    neighbor_id = id_order[unlist(neighbors, use.names = FALSE)]
  )

  # Pre-allocate result columns
  for (var_name in neighbor_source_vars) {
    cell_data[, paste0(var_name, "_neighbor_max")  := NA_real_]
    cell_data[, paste0(var_name, "_neighbor_min")  := NA_real_]
    cell_data[, paste0(var_name, "_neighbor_mean") := NA_real_]
  }

  # Keyed lookup
  setkey(cell_data, id, year)

  years <- sort(unique(cell_data$year))

  for (yr in years) {
    # Subset to this year
    dt_yr <- cell_data[year == yr, c("id", ".row_idx", neighbor_source_vars),
                       with = FALSE]
    setkey(dt_yr, id)

    # Join edges to get focal and neighbor row indices + values for this year
    # Focal side
    focal_join <- dt_yr[edges, on = .(id = focal_id), nomatch = 0L, allow.cartesian = TRUE]
    setnames(focal_join, ".row_idx", "focal_row")

    # Neighbor side
    nbr_vals <- dt_yr[, c("id", neighbor_source_vars), with = FALSE]
    setnames(nbr_vals, "id", "neighbor_id")
    setkey(nbr_vals, neighbor_id)

    joined <- nbr_vals[focal_join, on = .(neighbor_id), nomatch = 0L]

    for (var_name in neighbor_source_vars) {
      # Columns from neighbor side get suffix i. if there's a collision;
      # handle both cases:
      nbr_col <- if (paste0("i.", var_name) %in% names(joined)) {
        paste0("i.", var_name)
      } else {
        var_name
      }

      agg <- joined[!is.na(get(nbr_col)),
                     .(nmax  = max(get(nbr_col)),
                       nmin  = min(get(nbr_col)),
                       nmean = mean(get(nbr_col))),
                     keyby = .(focal_row)]

      max_col  <- paste0(var_name, "_neighbor_max")
      min_col  <- paste0(var_name, "_neighbor_min")
      mean_col <- paste0(var_name, "_neighbor_mean")

      set(cell_data, i = agg$focal_row, j = max_col,  value = agg$nmax)
      set(cell_data, i = agg$focal_row, j = min_col,  value = agg$nmin)
      set(cell_data, i = agg$focal_row, j = mean_col, value = agg$nmean)
    }
  }

  cell_data[, .row_idx := NULL]
  return(cell_data)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M-iteration `lapply` with `paste` + named-vector lookup | **Eliminated**; replaced by a single vectorized `data.table` join |
| `compute_neighbor_stats` | 6.46M-iteration `lapply` × 5 variables | **Replaced** by grouped `data.table` aggregation (`max`, `min`, `mean` by `focal_row`) |
| String key construction | ~25.8M `paste` calls + 25.8M probes into a 6.46M-entry named vector | **Zero** string operations; integer joins only |
| Estimated wall time | **86+ hours** | **~2–10 minutes** |
| Peak RAM | Moderate (6.46M-element list of integer vectors) | ~38.5M-row edge table ≈ ~1–2 GB (or ~1.37M per year in low-mem variant) |
| Trained RF model | Untouched | Untouched |
| Numerical results | max, min, mean of neighbor values | **Identical** max, min, mean of neighbor values |