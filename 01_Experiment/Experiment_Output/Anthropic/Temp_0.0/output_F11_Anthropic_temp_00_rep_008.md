 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M iterations, this alone is extremely expensive.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level loop iterations total.

3. **Random Forest inference** (`predict()` on a pre-trained model) is a single vectorized call over the dataset. Even with ~6.46M rows and ~110 predictors, a single `predict()` call on a `ranger` or `randomForest` object is internally optimized in C/C++ and typically completes in seconds to minutes — orders of magnitude faster than tens of millions of R-level `lapply` iterations.

The estimated 86+ hour runtime is consistent with the R-level loop overhead in the neighbor engineering, not with a single vectorized prediction call.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a vectorized `data.table` join approach. Instead of looping row-by-row, explode the neighbor list into an edge table `(cell_id, neighbor_id)`, join it with a `(cell_id, year) → row_index` mapping table, and produce the lookup as a grouped structure — all using `data.table` merge/join operations with no R-level per-row loop.

2. **Replace `compute_neighbor_stats()`** with a single vectorized `data.table` grouped aggregation. Using the edge table, join in the variable values, then compute `max`, `min`, and `mean` grouped by the focal row index — entirely in C-level `data.table` code.

3. **Leave the Random Forest predict() call untouched**, since it is not the bottleneck.

This reduces the complexity from ~32M+ R-level `lapply` iterations to a handful of `data.table` joins and grouped aggregations, bringing the estimated runtime from 86+ hours down to minutes.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Vectorized build of the neighbor edge table (replaces build_neighbor_lookup)
# ==============================================================================

build_neighbor_edges <- function(cell_data_dt, id_order, rook_neighbors_unique) {

# cell_data_dt: a data.table with columns 'id', 'year', and a row index
# id_order: vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an nb object (list of integer index vectors)

  # --- Explode the nb object into a directed edge list of (focal_id, neighbor_id) ---
  n_cells <- length(id_order)
  focal_indices <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  neighbor_indices <- unlist(rook_neighbors_unique)

  # Remove zero-neighbor entries (spdep uses integer(0) for islands)
  valid <- !is.na(neighbor_indices) & neighbor_indices != 0L
  focal_indices <- focal_indices[valid]
  neighbor_indices <- neighbor_indices[valid]

  edges <- data.table(
    focal_cell_id    = id_order[focal_indices],
    neighbor_cell_id = id_order[neighbor_indices]
  )

  # --- Build a row-index lookup: (id, year) -> row position in cell_data_dt ---
  cell_data_dt[, row_idx := .I]

  # --- Cross-join edges with years via merge ---
  # For each focal row, find its neighbors in the same year.
  # Merge edges with focal rows to get the year and focal row_idx
  focal_key <- cell_data_dt[, .(focal_cell_id = id, year, focal_row_idx = row_idx)]
  setkey(focal_key, focal_cell_id)
  setkey(edges, focal_cell_id)

  # Inner join: each edge gets every year the focal cell appears in
  edge_year <- merge(edges, focal_key, by = "focal_cell_id", allow.cartesian = TRUE)

  # Now find the row_idx of the neighbor in the same year
  neighbor_key <- cell_data_dt[, .(neighbor_cell_id = id, year, neighbor_row_idx = row_idx)]
  setkey(neighbor_key, neighbor_cell_id, year)
  setkey(edge_year, neighbor_cell_id, year)

  edge_full <- merge(edge_year, neighbor_key, by = c("neighbor_cell_id", "year"),
                     nomatch = NULL)

  # Return the essential columns
  edge_full[, .(focal_row_idx, neighbor_row_idx)]
}

# ==============================================================================
# STEP 2: Vectorized neighbor stats (replaces compute_neighbor_stats)
# ==============================================================================

compute_neighbor_stats_fast <- function(cell_data_dt, edge_dt, var_name) {
  # edge_dt has columns: focal_row_idx, neighbor_row_idx
  # Pull the variable values for each neighbor
  edge_dt[, val := cell_data_dt[[var_name]][neighbor_row_idx]]

  # Remove NAs
  agg <- edge_dt[!is.na(val),
                 .(nb_max  = max(val),
                   nb_min  = min(val),
                   nb_mean = mean(val)),
                 by = focal_row_idx]

  # Allocate full-length result (NA for rows with no valid neighbors)
  n <- nrow(cell_data_dt)
  result <- data.table(
    nb_max  = rep(NA_real_, n),
    nb_min  = rep(NA_real_, n),
    nb_mean = rep(NA_real_, n)
  )
  result[agg$focal_row_idx, `:=`(
    nb_max  = agg$nb_max,
    nb_min  = agg$nb_min,
    nb_mean = agg$nb_mean
  )]

  # Clean up temporary column
  edge_dt[, val := NULL]

  # Name columns to match the variable
  setnames(result,
           c("nb_max", "nb_min", "nb_mean"),
           paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))
  result
}

# ==============================================================================
# STEP 3: Full pipeline (replaces the outer loop)
# ==============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  cell_data_dt <- as.data.table(cell_data)

  message("Building neighbor edge table (vectorized)...")
  edge_dt <- build_neighbor_edges(cell_data_dt, id_order, rook_neighbors_unique)
  setkey(edge_dt, focal_row_idx)
  message(sprintf("  Edge table: %s rows", format(nrow(edge_dt), big.mark = ",")))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_neighbor_stats_fast(cell_data_dt, edge_dt, var_name)
    cell_data_dt <- cbind(cell_data_dt, stats)
  }

  # Remove helper column
  cell_data_dt[, row_idx := NULL]

  cell_data_dt
}

# ==============================================================================
# STEP 4: Run and then predict (RF model untouched)
# ==============================================================================

# --- Feature engineering (the actual bottleneck, now fast) ---
cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)

# --- Random Forest inference (NOT the bottleneck, unchanged) ---
# Assuming `rf_model` is the pre-trained model loaded from disk:
# rf_model <- readRDS("trained_rf_model.rds")
# cell_data$prediction <- predict(rf_model, data = cell_data)$predictions
# (or for randomForest: predict(rf_model, newdata = cell_data))
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level `lapply` iterations with `paste`/named-vector lookups | Single vectorized `data.table` merge producing an edge table |
| `compute_neighbor_stats` | ~6.46M × 5 = ~32.3M `lapply` iterations | 5 × one `data.table` grouped aggregation (C-level) |
| Estimated runtime | 86+ hours | Minutes (typically 5–15 min on 16 GB laptop) |
| RF `predict()` | Untouched | Untouched |
| Trained RF model | Preserved | Preserved |
| Numerical estimand | Preserved (same `max`, `min`, `mean` over same neighbor sets) | Preserved identically |

The true bottleneck is the row-level R loop over millions of rows in the neighbor feature engineering, not the Random Forest inference.