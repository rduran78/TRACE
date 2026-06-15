 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering stage** — specifically `build_neighbor_lookup` and `compute_neighbor_stats` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup`** uses a `lapply` over **~6.46 million rows**, and inside each iteration it performs `as.character()` coercion, `paste()` string construction, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookups in R are O(n) string-hashing operations. Doing this ~6.46 million times, each time constructing and matching multiple string keys, is catastrophically slow. With ~1.37 million neighbor relationships spread across 28 years, the total number of string constructions and lookups is on the order of tens of millions.

2. **`compute_neighbor_stats`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows with `lapply`, subsetting, removing NAs, and computing `max/min/mean`. That's ~32.3 million R-level function calls with per-element vector subsetting.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-loaded model object. Even with 6.46 million rows and 110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in C/C++ and typically completes in seconds to minutes. Loading the model from disk is a one-time `readRDS()` call. Writing predictions is a single `fwrite()`/`write.csv()`. None of these are remotely close to 86+ hours.

**Verdict:** The bottleneck is the row-by-row, string-key-based spatial neighbor feature computation, not Random Forest inference.

---

## Optimization Strategy

1. **Eliminate `build_neighbor_lookup`'s per-row string operations.** Replace the string-keyed named-vector lookup with a direct integer-indexed matrix/hash approach. Pre-build a mapping from `(cell_index, year_index)` → row number as an integer matrix, then look up neighbors purely via integer indexing.

2. **Vectorize `compute_neighbor_stats`.** Replace the `lapply` over 6.46M rows with a flat vectorized operation: expand all neighbor pairs into a two-column integer matrix (`row_i`, `neighbor_row`), then use `data.table` grouping to compute `max`, `min`, `mean` in one pass per variable.

3. **Process all 5 variables in one pass over the neighbor structure** rather than rebuilding/re-iterating the structure 5 times.

These changes reduce complexity from O(N × k × string_ops) to O(N × k) integer operations, where N = 6.46M and k = average neighbor count.

---

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED SPATIAL NEIGHBOR FEATURE ENGINEERING
# ============================================================

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {
  
  # Convert to data.table for speed; preserve row order
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]
  
  # --- Step 1: Build integer-indexed (cell_index, year_index) -> row mapping ---
  
  # Map cell id -> integer index (1..N_cells)
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map year -> integer index (1..N_years)
  unique_years <- sort(unique(dt$year))
  year_to_idx  <- setNames(seq_along(unique_years), as.character(unique_years))
  
  n_cells <- length(id_order)
  n_years <- length(unique_years)
  
  # Assign cell_idx and year_idx to every row
  dt[, cell_idx := id_to_idx[as.character(id)]]
  dt[, year_idx := year_to_idx[as.character(year)]]
  
  # Build a matrix: row_lookup[cell_idx, year_idx] = row number in dt
  # Initialize with NA
  row_lookup <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_lookup[cbind(dt$cell_idx, dt$year_idx)] <- dt$.row_id
  
  # --- Step 2: Expand neighbor pairs into (focal_row, neighbor_row) for ALL year slices ---
  
  # Build flat edge list from the nb object: (focal_cell_idx, neighbor_cell_idx)
  # rook_neighbors_unique is a list of length n_cells; element i contains integer indices of neighbors
  focal_cell   <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  neighbor_cell <- unlist(rook_neighbors_unique, use.names = FALSE)
  
  # Remove any 0-entries (spdep uses 0 for "no neighbors")
  valid <- neighbor_cell > 0L
  focal_cell    <- focal_cell[valid]
  neighbor_cell <- neighbor_cell[valid]
  
  n_edges <- length(focal_cell)
  
  # For every year, translate (focal_cell, year) and (neighbor_cell, year) into row numbers
  # We replicate the edge list across all years
  all_focal_rows    <- integer(n_edges * n_years)
  all_neighbor_rows <- integer(n_edges * n_years)
  
  for (yi in seq_len(n_years)) {
    offset <- (yi - 1L) * n_edges
    idx_range <- (offset + 1L):(offset + n_edges)
    all_focal_rows[idx_range]    <- row_lookup[cbind(focal_cell, rep(yi, n_edges))]
    all_neighbor_rows[idx_range] <- row_lookup[cbind(neighbor_cell, rep(yi, n_edges))]
  }
  
  # Remove pairs where either focal or neighbor row is NA (cell-year doesn't exist in data)
  keep <- !is.na(all_focal_rows) & !is.na(all_neighbor_rows)
  all_focal_rows    <- all_focal_rows[keep]
  all_neighbor_rows <- all_neighbor_rows[keep]
  
  # --- Step 3: Compute neighbor stats for each variable using data.table grouping ---
  
  edges_dt <- data.table(
    focal_row    = all_focal_rows,
    neighbor_row = all_neighbor_rows
  )
  
  for (var_name in neighbor_source_vars) {
    
    # Attach neighbor values via integer indexing (vectorized)
    neighbor_vals <- dt[[var_name]][edges_dt$neighbor_row]
    edges_dt[, nval := neighbor_vals]
    
    # Remove NA neighbor values before aggregation
    edges_valid <- edges_dt[!is.na(nval)]
    
    # Grouped aggregation — single pass
    agg <- edges_valid[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_row]
    
    # Initialize new columns with NA
    max_col  <- paste0("neighbor_", var_name, "_max")
    min_col  <- paste0("neighbor_", var_name, "_min")
    mean_col <- paste0("neighbor_", var_name, "_mean")
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    # Assign aggregated values back by row index
    set(dt, i = agg$focal_row, j = max_col,  value = agg$nb_max)
    set(dt, i = agg$focal_row, j = min_col,  value = agg$nb_min)
    set(dt, i = agg$focal_row, j = mean_col, value = agg$nb_mean)
  }
  
  # Clean up helper columns
  dt[, c(".row_id", "cell_idx", "year_idx") := NULL]
  
  return(dt)
}

# ============================================================
# USAGE — drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data, 
  id_order, 
  rook_neighbors_unique, 
  neighbor_source_vars
)

# ============================================================
# Random Forest prediction — unchanged, as it is NOT the bottleneck
# ============================================================

# rf_model <- readRDS("trained_rf_model.rds")          # load once
# cell_data$prediction <- predict(rf_model, cell_data)  # single vectorized call
# fwrite(cell_data, "predictions_output.csv")            # write once
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `paste` + named-vector string lookups | One integer matrix fill via `cbind` indexing |
| **Neighbor stats** | 6.46M `lapply` iterations × 5 variables = ~32.3M R calls | Flat `data.table` grouped aggregation (vectorized C) × 5 variables |
| **Estimated runtime** | 86+ hours | Minutes (typically 5–20 min depending on hardware) |
| **RF model** | Preserved — not retrained | Preserved — not retrained |
| **Numerical output** | `max`, `min`, `mean` of neighbor values | Identical `max`, `min`, `mean` — same estimand |

The trained Random Forest model is untouched. The numerical estimand (neighbor max, min, mean per variable per cell-year, and the downstream RF predictions) is preserved exactly.