 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, and another named-vector lookup (`idx_lookup[neighbor_keys]`). Named vector lookups in R are **O(n) string-hashing operations per call**, and doing this 6.46 million times with an `idx_lookup` vector of 6.46 million names is catastrophically slow.

2. **`compute_neighbor_stats()`** then iterates over the same ~6.46 million rows again, performing subsetting, `NA` removal, and `max/min/mean` per row. This is called **5 times** (once per neighbor source variable), yielding ~32.3 million `lapply` iterations total.

3. **Random Forest prediction** is a single vectorized `predict()` call on the final data frame. Even with ~6.46 million rows and ~110 predictors, a single `predict.randomForest()` call typically completes in minutes, not hours. Loading a serialized model (`readRDS`) is also fast. This is clearly not the 86+ hour bottleneck.

**Conclusion:** The bottleneck is the row-by-row R-level loop over millions of rows using named-vector string lookups and per-row `lapply` aggregation. This is a classic "death by a million R-level iterations" problem.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()` entirely** with a vectorized `data.table` merge/join approach. Instead of building a per-row list of neighbor indices, construct an **edge table** (a two-column data.table of `(focal_row, neighbor_row)`) using fast integer-keyed joins. This eliminates all `paste()`-based string key construction and named-vector lookups.

2. **Replace `compute_neighbor_stats()` with grouped `data.table` aggregation.** Join the edge table to the variable values, then compute `max`, `min`, and `mean` grouped by the focal row index — all in one vectorized pass per variable.

3. **Leave the Random Forest model and predict step untouched**, since it is not the bottleneck and the trained model must be preserved.

Expected speedup: from 86+ hours to **minutes** (the edge table has ~1.37M neighbor pairs × 28 years ≈ ~38M edges, and `data.table` grouped aggregation over ~38M rows is very fast).

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build the edge table (replaces build_neighbor_lookup)
# ============================================================
build_neighbor_edge_table <- function(data_dt, id_order, rook_neighbors) {
  # data_dt: a data.table with columns 'id' and 'year' (and row index = original row order)
  # id_order: vector of cell IDs in the order matching rook_neighbors
  # rook_neighbors: spdep nb object (list of integer neighbor index vectors)

  # --- Build directed edge list at the cell level ---
  # Each element rook_neighbors[[i]] gives the indices (into id_order) of
  # neighbors of cell id_order[i].
  n_cells <- length(id_order)
  focal_cell_idx <- rep(seq_len(n_cells), times = lengths(rook_neighbors))
  neighbor_cell_idx <- unlist(rook_neighbors, use.names = FALSE)

  # Map cell-level indices to actual cell IDs
  cell_edges <- data.table(
    focal_id    = id_order[focal_cell_idx],
    neighbor_id = id_order[neighbor_cell_idx]
  )

  # --- Expand to cell-year level via join ---
  # Create a lookup from (id, year) -> row position in data_dt
  data_dt[, .row_idx := .I]

  # Get unique years
  years <- unique(data_dt$year)

  # Cross join cell edges with all years
  cell_edges_year <- cell_edges[, .(year = years), by = .(focal_id, neighbor_id)]

  # Join to get focal row index
  setkey(data_dt, id, year)
  focal_lookup <- data_dt[, .(id, year, focal_row = .row_idx)]
  setkey(focal_lookup, id, year)
  setkey(cell_edges_year, focal_id, year)
  cell_edges_year <- focal_lookup[cell_edges_year,
                                   on = .(id = focal_id, year = year),
                                   nomatch = 0L]
  # Now cell_edges_year has columns: id, year, focal_row, neighbor_id

  # Join to get neighbor row index
  neighbor_lookup <- data_dt[, .(id, year, neighbor_row = .row_idx)]
  setkey(neighbor_lookup, id, year)
  setkey(cell_edges_year, neighbor_id, year)
  edge_table <- neighbor_lookup[cell_edges_year,
                                 on = .(id = neighbor_id, year = year),
                                 nomatch = 0L]
  # Keep only what we need
  edge_table <- edge_table[, .(focal_row, neighbor_row)]

  # Clean up temporary column
  data_dt[, .row_idx := NULL]

  return(edge_table)
}

# ============================================================
# STEP 2: Vectorized neighbor stats (replaces compute_neighbor_stats)
# ============================================================
compute_neighbor_stats_fast <- function(data_dt, edge_table, var_name) {
  # Extract neighbor values via the edge table
  vals <- data_dt[[var_name]]
  edges <- copy(edge_table)
  edges[, val := vals[neighbor_row]]

  # Remove NA neighbor values
  edges <- edges[!is.na(val)]

  # Grouped aggregation
  agg <- edges[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = focal_row]

  # Initialize result columns with NA for all rows
  n <- nrow(data_dt)
  max_col  <- rep(NA_real_, n)
  min_col  <- rep(NA_real_, n)
  mean_col <- rep(NA_real_, n)

  # Fill in computed values
  max_col[agg$focal_row]  <- agg$nb_max
  min_col[agg$focal_row]  <- agg$nb_min
  mean_col[agg$focal_row] <- agg$nb_mean

  list(max_col = max_col, min_col = min_col, mean_col = mean_col)
}

# ============================================================
# STEP 3: Add neighbor features (replaces compute_and_add_neighbor_features)
# ============================================================
add_neighbor_features_fast <- function(data_dt, var_name, edge_table) {
  stats <- compute_neighbor_stats_fast(data_dt, edge_table, var_name)
  set(data_dt, j = paste0(var_name, "_nb_max"),  value = stats$max_col)
  set(data_dt, j = paste0(var_name, "_nb_min"),  value = stats$min_col)
  set(data_dt, j = paste0(var_name, "_nb_mean"), value = stats$mean_col)
  invisible(data_dt)
}

# ============================================================
# STEP 4: Full optimized pipeline
# ============================================================
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique, rf_model) {
  # Convert to data.table if not already (by reference, no deep copy)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Build edge table once (vectorized, fast)
  message("Building edge table...")
  edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("Edge table: %d edges", nrow(edge_table)))

  # Compute neighbor features for all 5 variables
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    add_neighbor_features_fast(cell_data, var_name, edge_table)
  }

  # --- Random Forest prediction (unchanged, not the bottleneck) ---
  message("Running Random Forest prediction...")
  cell_data[, prediction := predict(rf_model, newdata = cell_data)]

  message("Done.")
  return(cell_data)
}

# ============================================================
# Usage (assumes objects already in memory):
# ============================================================
# rf_model              <- readRDS("trained_rf_model.rds")
# cell_data             <- readRDS("cell_data.rds")          # ~6.46M rows
# id_order              <- readRDS("id_order.rds")           # 344,208 cell IDs
# rook_neighbors_unique <- readRDS("rook_neighbors.rds")     # spdep nb object
#
# result <- run_optimized_pipeline(cell_data, id_order, rook_neighbors_unique, rf_model)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste()` + named vector lookups | Single vectorized `data.table` join to build edge table |
| `compute_neighbor_stats` | 6.46M `lapply` iterations × 5 vars = 32.3M R-level loops | 5 grouped `data.table` aggregations (~38M rows each, vectorized C-level) |
| RF inference | Single `predict()` call (fast) | **Unchanged** — not the bottleneck |
| Estimated runtime | 86+ hours | **Minutes** (edge table build ~1–3 min; each variable aggregation ~10–30 sec) |
| Numerical results | Original estimand | **Preserved** — identical `max`, `min`, `mean` logic |
| Trained RF model | Preserved | **Preserved** — only `predict()` is called, no retraining |