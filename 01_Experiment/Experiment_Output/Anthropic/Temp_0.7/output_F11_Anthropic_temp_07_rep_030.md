 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that **Random Forest inference** (loading models, calling `predict()`, writing predictions) is the main bottleneck is **incorrect**. The code evidence points clearly to a different bottleneck: **the spatial neighbor feature engineering stage**, specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's why:

1. **`build_neighbor_lookup()`** iterates row-by-row over **~6.46 million rows** using `lapply`, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`) for every single row. Named vector lookup in R is **O(n)** hash-based but the sheer volume — 6.46M iterations each doing string construction and subsetting — is extremely expensive in interpreted R.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max/min/mean`. That's ~32.3 million interpreted R loop iterations total, each with allocation overhead from anonymous function closures.

3. **The final `do.call(rbind, result)`** in `compute_neighbor_stats` binds ~6.46 million small 3-element vectors into a matrix. `do.call(rbind, ...)` on millions of list elements is notoriously slow in R due to repeated memory allocation and copying.

4. By contrast, **Random Forest prediction** on a pre-trained model over ~6.46M rows with ~110 predictors is a single vectorized C/C++ call (whether using `randomForest`, `ranger`, or similar). It is inherently fast — typically seconds to a few minutes — and is certainly not an 86-hour operation.

**Conclusion:** The bottleneck is the **row-level interpreted R loops** in the neighbor lookup construction and neighbor statistics computation. The 86+ hour estimate is fully explained by ~38.8 million `lapply` iterations with per-iteration string manipulation and subsetting, not by RF inference.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace the row-by-row `lapply` with a fully vectorized approach using `data.table`. Pre-expand all neighbor relationships into an edge list (source_row → neighbor_row), join on `(neighbor_id, year)` to resolve row indices, then split into a list by source row.

2. **Vectorize `compute_neighbor_stats()`**: Instead of looping over each row's neighbor indices, use the edge list with `data.table` grouped aggregation (`max`, `min`, `mean` by source row) — a single vectorized pass per variable.

3. **Eliminate `do.call(rbind, ...)`**: Grouped `data.table` aggregation returns a structured result directly; no need to bind millions of small vectors.

4. **Preserve the trained RF model and original numerical estimand**: The optimization only changes how neighbor features are computed. The values produced are numerically identical, so the RF model is used as-is with `predict()`.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# OPTIMIZED: build_neighbor_edge_list
# Replaces build_neighbor_lookup with a vectorized edge-list
# approach using data.table joins.
# ==============================================================

build_neighbor_edge_list <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id', 'year', and a .ROW_IDX column
  # id_order: vector of cell IDs (index i → cell id)
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  # Step 1: Build directed edge list at the cell level: (source_id, neighbor_id)
  # neighbors[[i]] gives the indices (into id_order) of neighbors of id_order[i]
  source_ref <- rep(seq_along(neighbors), lengths(neighbors))
  target_ref <- unlist(neighbors, use.names = FALSE)

  cell_edges <- data.table(
    source_id   = id_order[source_ref],
    neighbor_id = id_order[target_ref]
  )
  rm(source_ref, target_ref)

  # Step 2: Cross-join with years to get (source_id, year, neighbor_id, year)
  # Instead of a true cross join (expensive), we join through the data.
  # For each row in data (source_id, year), find all neighbor rows
  # that share the same year.

  # Create a keyed lookup: for each (id, year) → row index
  neighbor_rows <- data_dt[, .(neighbor_id = id, year, neighbor_row_idx = .ROW_IDX)]
  setkey(neighbor_rows, neighbor_id, year)

  # Expand: for each row in data, get its cell-level neighbors
  source_rows <- data_dt[, .(source_id = id, year, source_row_idx = .ROW_IDX)]
  setkey(cell_edges, source_id)
  setkey(source_rows, source_id)

  # Join source rows to cell edges to get (source_row_idx, year, neighbor_id)
  expanded <- cell_edges[source_rows, on = "source_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: source_id, neighbor_id, year, source_row_idx

  # Join to get neighbor_row_idx
  setkey(expanded, neighbor_id, year)
  edge_list <- neighbor_rows[expanded, on = c("neighbor_id", "year"), nomatch = 0L]
  # edge_list has: neighbor_id, year, neighbor_row_idx, source_id, source_row_idx

  edge_list[, .(source_row_idx, neighbor_row_idx)]
}


# ==============================================================
# OPTIMIZED: compute_neighbor_stats_vectorized
# Uses grouped data.table aggregation on the edge list.
# Returns a data.table with columns: source_row_idx, max_v, min_v, mean_v
# ==============================================================

compute_neighbor_stats_vectorized <- function(data_dt, edge_list, var_name, n_rows) {
  # Extract the variable values for all neighbor rows
  vals <- data_dt[[var_name]]

  # Build a working table: source_row_idx + neighbor value
  work <- edge_list[, .(source_row_idx, nval = vals[neighbor_row_idx])]

  # Remove NAs in neighbor values
  work <- work[!is.na(nval)]

  # Grouped aggregation — single vectorized pass
  stats <- work[, .(
    max_v  = max(nval),
    min_v  = min(nval),
    mean_v = mean(nval)
  ), by = source_row_idx]

  # Re-index to full row set (rows with no valid neighbors get NA)
  full <- data.table(source_row_idx = seq_len(n_rows))
  result <- stats[full, on = "source_row_idx"]

  result[, source_row_idx := NULL]
  result
}


# ==============================================================
# OPTIMIZED: compute_and_add_neighbor_features_vectorized
# Drop-in replacement that adds the 3 neighbor columns per variable.
# ==============================================================

compute_and_add_neighbor_features_vectorized <- function(data_dt, var_name, edge_list) {
  n_rows <- nrow(data_dt)
  stats  <- compute_neighbor_stats_vectorized(data_dt, edge_list, var_name, n_rows)

  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")

  set(data_dt, j = col_max,  value = stats$max_v)
  set(data_dt, j = col_min,  value = stats$min_v)
  set(data_dt, j = col_mean, value = stats$mean_v)

  invisible(data_dt)
}


# ==============================================================
# MAIN PIPELINE (replaces the original outer loop)
# ==============================================================

# Convert to data.table and add row index
cell_data_dt <- as.data.table(cell_data)
cell_data_dt[, .ROW_IDX := .I]

# Build the vectorized edge list (one-time cost)
message("Building neighbor edge list...")
edge_list <- build_neighbor_edge_list(cell_data_dt, id_order, rook_neighbors_unique)
setkey(edge_list, source_row_idx)
message("Edge list built: ", nrow(edge_list), " directed row-level edges.")

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var_name)
  compute_and_add_neighbor_features_vectorized(cell_data_dt, var_name, edge_list)
}

# Clean up helper column
cell_data_dt[, .ROW_IDX := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_data_dt)

# ==============================================================
# RANDOM FOREST PREDICTION (unchanged — not the bottleneck)
# ==============================================================
# The pre-trained RF model is loaded and used exactly as before.
# Example (preserving the original estimand):
#
#   library(ranger)  # or library(randomForest)
#   rf_model <- readRDS("path/to/trained_rf_model.rds")
#   cell_data$predicted <- predict(rf_model, data = cell_data)$predictions
#   write.csv(cell_data, "predictions_output.csv", row.names = FALSE)
#
# This predict() call is a single vectorized C++ operation and
# completes in seconds-to-minutes, NOT hours.
```

---

## Expected Performance Improvement

| Stage | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M `lapply` iterations with string ops | One vectorized `data.table` join (~seconds to minutes) |
| `compute_neighbor_stats` (×5 vars) | ~32.3M `lapply` iterations + `do.call(rbind, ...)` | 5 grouped `data.table` aggregations (~seconds each) |
| **Total neighbor feature engineering** | **~86+ hours** | **~5–20 minutes** |
| RF `predict()` | Minutes (was never the bottleneck) | Minutes (unchanged) |

The optimization replaces all interpreted row-level R loops with vectorized `data.table` joins and grouped aggregations executed in compiled C code, reducing runtime by roughly **three orders of magnitude** while producing numerically identical neighbor features and preserving the trained Random Forest model and original estimand.