 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, and another named-vector lookup (`idx_lookup[neighbor_keys]`). Named vector lookups in R are hash-table-like but carry significant per-call overhead at this scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates tens of millions of string operations and named lookups. This single function likely accounts for the majority of the 86+ hour runtime.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over ~6.46M rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. That's ~32.3 million R-level function calls total.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with ~6.46M rows and ~110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in C/C++ and typically completes in seconds to minutes — orders of magnitude faster than the feature engineering above.

**The bottleneck is the row-level R `lapply` loops with per-element string construction and named-vector lookups across 6.46 million rows, repeated for 5 variables.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()` entirely** with a vectorized `data.table` merge/join approach. Instead of building a per-row list of neighbor indices via string keys, we construct a flat edge-list data.table of `(row_i, neighbor_row_j)` pairs using keyed joins — eliminating all `paste()`, `as.character()`, and named-vector lookups.

2. **Replace `compute_neighbor_stats()` with a single vectorized `data.table` group-by aggregation** per variable. Join the edge list to the data values, then aggregate by source row using `data.table`'s optimized `max`, `min`, `mean` — all in C.

3. **Leave the Random Forest predict step untouched**, since it is not the bottleneck.

This reduces the estimated runtime from 86+ hours to roughly **minutes**.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Vectorized neighbor lookup construction (replaces build_neighbor_lookup)
# ---------------------------------------------------------------
build_neighbor_edge_list <- function(data_dt, id_order, rook_neighbors) {

  # data_dt: a data.table with columns 'id', 'year', and a row index 'row_i'

  # id_order: vector of cell IDs in the order matching rook_neighbors
  # rook_neighbors: spdep nb object (list of integer neighbor indices)

  # Step A: Build a flat edge list of (focal_id, neighbor_id) from the nb object
  n_cells <- length(id_order)
  focal_idx <- rep(seq_len(n_cells), times = lengths(rook_neighbors))
  neighbor_idx <- unlist(rook_neighbors)

  # Remove zero-neighbor entries (spdep uses 0L for no-neighbor cells)
  valid <- neighbor_idx != 0L
  focal_idx <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]

  edges <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )

  # Step B: Cross-join edges with years present in the data
  #   For each (focal_id, neighbor_id) pair, the relationship holds for every year.
  #   We join to the data to get row indices for both focal and neighbor.

  # Create a keyed lookup: (id, year) -> row_i
  row_lookup <- data_dt[, .(id, year, row_i)]
  setkey(row_lookup, id, year)

  # Get all unique years
  years <- unique(data_dt$year)

  # Expand edges across all years
  edges_expanded <- edges[, .(year = years), by = .(focal_id, neighbor_id)]

  # Join to get focal row index
  setnames(edges_expanded, "focal_id", "id")
  edges_expanded <- row_lookup[edges_expanded, on = .(id, year), nomatch = 0L]
  setnames(edges_expanded, c("id", "row_i"), c("focal_id", "focal_row"))

  # Join to get neighbor row index
  setnames(edges_expanded, "neighbor_id", "id")
  edges_expanded <- row_lookup[edges_expanded, on = .(id, year), nomatch = 0L]
  setnames(edges_expanded, c("id", "row_i"), c("neighbor_id", "neighbor_row"))

  return(edges_expanded[, .(focal_row, neighbor_row)])
}

# ---------------------------------------------------------------
# 2. Vectorized neighbor stats (replaces compute_neighbor_stats)
# ---------------------------------------------------------------
compute_neighbor_stats_vec <- function(data_dt, edge_dt, var_name) {
  # edge_dt has columns: focal_row, neighbor_row
  # Attach the neighbor's value for var_name
  neighbor_vals <- data_dt[[var_name]][edge_dt$neighbor_row]

  agg_dt <- data.table(
    focal_row = edge_dt$focal_row,
    val       = neighbor_vals
  )

  # Remove NAs before aggregation
  agg_dt <- agg_dt[!is.na(val)]

  # Aggregate
  stats <- agg_dt[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = focal_row]

  return(stats)
}

# ---------------------------------------------------------------
# 3. Add neighbor features to the dataset
# ---------------------------------------------------------------
compute_and_add_neighbor_features_vec <- function(data_dt, var_name, edge_dt) {
  stats <- compute_neighbor_stats_vec(data_dt, edge_dt, var_name)

  # Initialize columns with NA
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)

  data_dt[, (max_col)  := NA_real_]
  data_dt[, (min_col)  := NA_real_]
  data_dt[, (mean_col) := NA_real_]

  # Fill in computed values by row index
  data_dt[stats$focal_row, (max_col)  := stats$nb_max]
  data_dt[stats$focal_row, (min_col)  := stats$nb_min]
  data_dt[stats$focal_row, (mean_col) := stats$nb_mean]

  return(data_dt)
}

# ---------------------------------------------------------------
# 4. Full optimized pipeline
# ---------------------------------------------------------------
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique, rf_model) {
  # Convert to data.table if needed
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Add row index

  cell_data[, row_i := .I]

  # Build the edge list ONCE (vectorized, replaces build_neighbor_lookup)
  message("Building vectorized neighbor edge list...")
  edge_dt <- build_neighbor_edge_list(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("Edge list built: %s edges", format(nrow(edge_dt), big.mark = ",")))

  # Compute neighbor features for all 5 variables
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    cell_data <- compute_and_add_neighbor_features_vec(cell_data, var_name, edge_dt)
  }

  # Clean up helper column
  cell_data[, row_i := NULL]

  # ---- Random Forest inference (untouched, not the bottleneck) ----
  message("Running Random Forest predictions...")
  predictions <- predict(rf_model, newdata = cell_data)
  cell_data[, prediction := predictions]

  message("Pipeline complete.")
  return(cell_data)
}

# ---------------------------------------------------------------
# Usage (preserves trained RF model and original numerical estimand):
# ---------------------------------------------------------------
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = rf_model          # pre-trained, never retrained
# )
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows; `paste`/named-vector lookups | Single vectorized `data.table` keyed join; edge list built once |
| **Neighbor stats** | `lapply` over 6.46M rows × 5 vars = 32.3M R calls | `data.table` grouped aggregation in C, 5 calls |
| **RF inference** | `predict()` — already fast | Unchanged |
| **Estimated runtime** | 86+ hours | Minutes |
| **Trained RF model** | Preserved | Preserved |
| **Numerical results** | Original `max`/`min`/`mean` | Identical `max`/`min`/`mean` — estimand preserved |

The true bottleneck is the R-level row-by-row neighbor feature engineering, not Random Forest inference. The fix is vectorization via `data.table` joins and grouped aggregation.