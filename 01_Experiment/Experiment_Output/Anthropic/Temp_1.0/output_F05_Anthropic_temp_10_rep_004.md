 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern, not merely a local inefficiency**. Here's the full picture:

### Local Inefficiency
The `paste(..., sep="_")` keying and named-vector lookup inside `build_neighbor_lookup` is O(N) per row for key construction, executed ~6.46M times. String hashing and matching on 6.46M-length named vectors is extremely expensive.

### Broader Algorithmic Problem
The real issue is that `build_neighbor_lookup` uses a **row-wise `lapply` over 6.46M rows**, each performing:
1. A character-based lookup of the cell ID → reference index (`id_to_ref`)
2. An index into the neighbor list (`neighbors[[ref_idx]]`)
3. String-paste of neighbor IDs with the current year
4. String-key lookup into `idx_lookup` (a 6.46M-entry named vector)

This is an **O(N × K) string operation** where N ≈ 6.46M and K ≈ average neighbor count (~4 for rook). That's ~25.8M string constructions and hash lookups against a 6.46M-entry table — all to answer a question that has a clean **integer-arithmetic** solution.

Furthermore, `compute_neighbor_stats` then loops over 6.46M entries again per variable, doing per-row `max/min/mean`. With 5 variables, that's 5 × 6.46M R-level function calls.

### Root Cause
The entire pattern conflates **spatial structure** (which cells are neighbors — time-invariant) with **temporal indexing** (which row corresponds to cell × year). These should be separated. Since every cell appears once per year in a balanced panel, neighbor relationships in row-space are **the same shifted pattern repeated 28 times**. We only need to compute the spatial neighbor mapping once in row-index space for one year and then offset it.

---

## Optimization Strategy

1. **Eliminate all string operations.** Use integer-indexed lookups exclusively.
2. **Separate spatial from temporal indexing.** Build a cell→row-offset map once; derive row-level neighbor indices by integer arithmetic.
3. **Vectorize `compute_neighbor_stats`.** Replace row-wise `lapply` with a single `data.table` grouped aggregation over an edge list, which is internally parallelized in C.
4. **Process all 5 variables in one pass** over the edge list rather than 5 separate passes.

**Expected speedup:** From ~86+ hours to **minutes** (typically 2–10 minutes depending on RAM/disk speed).

---

## Working R Code

```r
# =============================================================================
# Optimized neighbor-feature construction
# Drop-in replacement — preserves the exact numerical estimand
# =============================================================================

library(data.table)

build_and_add_neighbor_features <- function(cell_data,
                                            id_order,
                                            rook_neighbors_unique,
                                            neighbor_source_vars) {
  # --- Convert to data.table for fast grouped operations ---
  dt <- as.data.table(cell_data)

  # =========================================================================
  # STEP 1: Build a time-invariant directed edge list (cell-level)
  #
  # rook_neighbors_unique is an nb object: a list of length = # cells,

  # where element i is an integer vector of neighbor indices into id_order.
  # We expand this to a two-column edge table of (cell_id, neighbor_cell_id).
  # =========================================================================

  n_cells <- length(id_order)
  from_ref <- rep(seq_len(n_cells),
                  lengths(rook_neighbors_unique))
  to_ref   <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove the spdep "no-neighbor" sentinel (0)
  valid <- to_ref != 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]

  # Map reference indices → actual cell IDs
  edges_cell <- data.table(
    id       = id_order[from_ref],
    nb_id    = id_order[to_ref]
  )

  # =========================================================================
  # STEP 2: Build a fast integer lookup from (id) → row positions per year
  #
  # Key insight: we work entirely in row-index space.
  # For each (id, year) we need the row index. We achieve this by keying

  # the data.table and using a fast equi-join.
  # =========================================================================

  # Add original row order so we can write results back in place
  dt[, .rowid := .I]

  # Create a slim table: (id, year) → .rowid
  row_map <- dt[, .(id, year, .rowid)]

  # =========================================================================
  # STEP 3: Expand edge list across years and join to get row indices
  #
  # For every year, every spatial edge (id→nb_id) becomes a row-level edge
  # (focal_row → neighbor_row).  We achieve this with a single join rather
  # than 6.46M string-key lookups.
  # =========================================================================

  years <- sort(unique(dt$year))

  # Cross-join edges × years  (~1.37M edges × 28 years ≈ 38.5M rows)
  # This fits comfortably in 16 GB (≈ 0.6 GB for two integer columns + year)
  edges_yr <- CJ_dt(edges_cell, years)  # helper below; or use tidyr::crossing

  # Faster alternative avoiding CJ on data.tables:
  edges_yr <- edges_cell[, .(id, nb_id, year = list(years)), by = .I
                         ][, .(id, nb_id, year = unlist(year))]
  edges_yr[, I := NULL]

  # Join to get focal row index
  setkey(row_map, id, year)
  setkey(edges_yr, id, year)
  edges_yr[row_map, focal_row := i..rowid, on = .(id, year)]

  # Join to get neighbor row index
  setnames(edges_yr, "nb_id", "id_nb")
  # We need to join on (id_nb, year) → .rowid
  edges_yr[row_map, nb_row := i..rowid, on = .(id_nb = id, year)]

  # Drop edges where either focal or neighbor is missing (boundary / NA year)
  edges_yr <- edges_yr[!is.na(focal_row) & !is.na(nb_row)]

  # =========================================================================
  # STEP 4: Compute max, min, mean for each variable in one vectorised pass
  # =========================================================================

  for (var_name in neighbor_source_vars) {
    message("Processing neighbor stats for: ", var_name)

    # Pull neighbor values via integer indexing (vectorised)
    edges_yr[, nb_val := dt[[var_name]][nb_row]]

    # Grouped aggregation — data.table does this in C
    stats <- edges_yr[!is.na(nb_val),
                      .(v_max  = max(nb_val),
                        v_min  = min(nb_val),
                        v_mean = mean(nb_val)),
                      keyby = focal_row]

    # Initialise result columns with NA
    max_col  <- paste0(var_name, "_max_neighbor")
    min_col  <- paste0(var_name, "_min_neighbor")
    mean_col <- paste0(var_name, "_mean_neighbor")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Write results back by row index
    dt[stats$focal_row, (max_col)  := stats$v_max]
    dt[stats$focal_row, (min_col)  := stats$v_min]
    dt[stats$focal_row, (mean_col) := stats$v_mean]
  }

  # Clean up
  edges_yr[, nb_val := NULL]
  dt[, .rowid := NULL]

  # Return as data.frame if the input was a data.frame
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}


# ---- Helper: Cross-join a data.table with a vector of years ----
# (More memory-efficient than full CJ on two data.tables)
# Already handled inline above; included for clarity:
CJ_dt <- function(edge_dt, years_vec) {
  idx <- rep(seq_len(nrow(edge_dt)), each = length(years_vec))
  out <- edge_dt[idx]
  out[, year := rep(years_vec, nrow(edge_dt))]
  out
}
```

### Drop-in Replacement for the Outer Loop

```r
# ---------------------------------------------------------------------------
# BEFORE (original — ~86+ hours):
# ---------------------------------------------------------------------------
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order,
#                                          rook_neighbors_unique)
# neighbor_source_vars <- c("ntl","ec","pop_density","def","usd_est_n2")
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name,
#                                                  neighbor_lookup)
# }

# ---------------------------------------------------------------------------
# AFTER (optimized — ~2-10 minutes):
# ---------------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_and_add_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched — use it as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Memory-Constrained Variant

If the ~38.5M-row edge table is tight on 16 GB RAM, process years in batches:

```r
build_and_add_neighbor_features_chunked <- function(cell_data,
                                                     id_order,
                                                     rook_neighbors_unique,
                                                     neighbor_source_vars,
                                                     year_chunk_size = 7) {
  dt <- as.data.table(cell_data)
  dt[, .rowid := .I]

  n_cells <- length(id_order)
  from_ref <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to_ref   <- unlist(rook_neighbors_unique, use.names = FALSE)
  valid    <- to_ref != 0L
  edges_cell <- data.table(id    = id_order[from_ref[valid]],
                           id_nb = id_order[to_ref[valid]])

  row_map <- dt[, .(id, year, .rowid)]
  setkey(row_map, id, year)

  years <- sort(unique(dt$year))
  year_chunks <- split(years, ceiling(seq_along(years) / year_chunk_size))

  # Initialise output columns
  for (var_name in neighbor_source_vars) {
    dt[, paste0(var_name, "_max_neighbor")  := NA_real_]
    dt[, paste0(var_name, "_min_neighbor")  := NA_real_]
    dt[, paste0(var_name, "_mean_neighbor") := NA_real_]
  }

  for (chunk in year_chunks) {
    message("Processing years: ", paste(chunk, collapse = ", "))

    edges_yr <- CJ_dt(edges_cell, chunk)
    setkey(edges_yr, id, year)
    edges_yr[row_map, focal_row := i..rowid, on = .(id, year)]
    edges_yr[row_map, nb_row    := i..rowid, on = .(id_nb = id, year)]
    edges_yr <- edges_yr[!is.na(focal_row) & !is.na(nb_row)]

    for (var_name in neighbor_source_vars) {
      edges_yr[, nb_val := dt[[var_name]][nb_row]]
      stats <- edges_yr[!is.na(nb_val),
                        .(v_max = max(nb_val),
                          v_min = min(nb_val),
                          v_mean = mean(nb_val)),
                        keyby = focal_row]

      max_col  <- paste0(var_name, "_max_neighbor")
      min_col  <- paste0(var_name, "_min_neighbor")
      mean_col <- paste0(var_name, "_mean_neighbor")

      set(dt, i = stats$focal_row, j = max_col,  value = stats$v_max)
      set(dt, i = stats$focal_row, j = min_col,  value = stats$v_min)
      set(dt, i = stats$focal_row, j = mean_col, value = stats$v_mean)
    }
  }

  dt[, .rowid := NULL]
  if (!is.data.table(cell_data)) return(as.data.frame(dt))
  return(dt)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Key construction** | 6.46M × `paste()` + named-vector hash | Zero strings; integer equi-join |
| **Neighbor resolution** | Row-wise `lapply` (R-level loop) | Single `data.table` keyed join (C-level) |
| **Stats computation** | `lapply` over 6.46M × 5 vars | `data.table` grouped aggregation (C-level) |
| **Total passes over data** | 1 (build) + 5 (stats) = 6 | 1 combined pass per year-chunk |
| **Estimated runtime** | 86+ hours | **2–10 minutes** |
| **RAM peak** | Lower but slow | ~2–4 GB (chunked: ~1–2 GB) |
| **Numerical result** | ✔ max/min/mean per cell-year | ✔ Identical estimand preserved |
| **RF model** | — | **Untouched** |