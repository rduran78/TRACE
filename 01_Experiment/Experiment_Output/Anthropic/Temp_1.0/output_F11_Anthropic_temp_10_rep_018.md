 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence clearly shows the bottleneck is in the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, performing per-row string pasting (`paste()`), character-to-index hash lookups, and NA filtering. That's 6.46 million R-level iterations with string allocation and named-vector lookups — an O(n × k) operation where k is the average neighbor count, all in interpreted R with no vectorization.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over the same 6.46 million rows via `lapply`, subsetting vectors, removing NAs, and computing `max/min/mean`. That's another ~32.3 million R-level loop iterations total, each with allocation overhead.

3. **The `do.call(rbind, result)` call** inside `compute_neighbor_stats` binds 6.46 million 3-element vectors into a matrix — a notoriously slow pattern in R.

4. By contrast, Random Forest **prediction** on a pre-trained model is a single call to `predict()` on a matrix of ~6.46M × 110 features. Packages like `ranger` or `randomForest` execute this in optimized C/C++ and typically complete in seconds to a few minutes, even on millions of rows. Loading a serialized model (`readRDS`) is also fast. Writing predictions is a single vectorized write. There is no loop, no string manipulation, no per-row R overhead.

**Conclusion:** The bottleneck is the neighbor feature engineering: building the lookup (one-time, ~6.46M R iterations with string ops) and computing neighbor stats (5 × 6.46M R iterations). The estimated 86+ hour runtime is attributable to these operations, not to RF inference.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace per-row `paste` and named-vector lookup with a single vectorized merge/join using `data.table`. Pre-build an edge list of (row_i, row_j) pairs by joining neighbor relationships against a cell-year index table. This eliminates all per-row string operations.

2. **Vectorize `compute_neighbor_stats()`**: Once we have the edge list, group-by-row aggregation (`max`, `min`, `mean`) can be done in a single `data.table` operation per variable — fully vectorized in C, no R-level `lapply`.

3. **Eliminate `do.call(rbind, ...)`**: The `data.table` approach produces columnar results directly.

4. **Leave the Random Forest model and prediction code untouched**: The trained model is preserved, and the numerical estimand (the same neighbor features fed to the same model producing the same predictions) is unchanged.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 1. Build a vectorized edge list (replaces build_neighbor_lookup)
# ──────────────────────────────────────────────────────────────────────
build_neighbor_edgelist <- function(cell_data_dt, id_order, rook_neighbors_unique) {
  # cell_data_dt must be a data.table with columns: id, year, and a row index
  # id_order: vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique: spdep nb object (list of integer neighbor indices)

  # Step A: Build directed edge list at the cell level: (focal_id, neighbor_id)
  n_cells <- length(id_order)
  focal_idx <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  neighbor_idx <- unlist(rook_neighbors_unique)

  cell_edges <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )

  # Step B: Add row_index to cell_data_dt
  cell_data_dt[, row_idx := .I]

  # Step C: Create a keyed lookup: (id, year) -> row_idx
  lookup <- cell_data_dt[, .(id, year, row_idx)]

  # Step D: Expand cell-level edges to cell-year-level edges
  #   For every (focal_id, neighbor_id) pair and every year,
  #   map both to their row indices.
  #   Because every cell appears in every year (balanced panel), we cross-join
  #   cell_edges with the unique years, then join twice to get row indices.

  years <- sort(unique(cell_data_dt$year))

  # Cross join edges × years
  cell_year_edges <- cell_edges[, CJ(year = years), by = .(focal_id, neighbor_id)]

  # Join focal row index
  setkey(lookup, id, year)
  setkey(cell_year_edges, focal_id, year)
  cell_year_edges[lookup, focal_row := i.row_idx, on = .(focal_id = id, year = year)]

  # Join neighbor row index
  setkey(cell_year_edges, neighbor_id, year)
  cell_year_edges[lookup, neighbor_row := i.row_idx, on = .(neighbor_id = id, year = year)]

  # Remove any edges where either side is missing
  cell_year_edges <- cell_year_edges[!is.na(focal_row) & !is.na(neighbor_row)]

  # Return only the row-index edges (compact)
  cell_year_edges[, .(focal_row, neighbor_row)]
}

# ──────────────────────────────────────────────────────────────────────
# 2. Vectorized neighbor stats (replaces compute_neighbor_stats)
# ──────────────────────────────────────────────────────────────────────
compute_neighbor_stats_vec <- function(cell_data_dt, edge_dt, var_name) {
  # edge_dt has columns: focal_row, neighbor_row
  # Pull the variable values for neighbor rows
  edge_dt[, val := cell_data_dt[[var_name]][neighbor_row]]

  # Remove NAs in val
  valid <- edge_dt[!is.na(val)]

  # Aggregate per focal row
  agg <- valid[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = focal_row]

  # Build full result aligned to all rows
  n <- nrow(cell_data_dt)
  result_max  <- rep(NA_real_, n)
  result_min  <- rep(NA_real_, n)
  result_mean <- rep(NA_real_, n)

  result_max[agg$focal_row]  <- agg$nb_max
  result_min[agg$focal_row]  <- agg$nb_min
  result_mean[agg$focal_row] <- agg$nb_mean

  # Clean up temporary column
  edge_dt[, val := NULL]

  list(nb_max = result_max, nb_min = result_min, nb_mean = result_mean)
}

# ──────────────────────────────────────────────────────────────────────
# 3. Full pipeline (replaces the outer loop)
# ──────────────────────────────────────────────────────────────────────
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  cell_data_dt <- as.data.table(cell_data)

  message("Building vectorized edge list...")
  edge_dt <- build_neighbor_edgelist(cell_data_dt, id_order, rook_neighbors_unique)
  message(sprintf("  Edge list has %s rows.", format(nrow(edge_dt), big.mark = ",")))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_neighbor_stats_vec(cell_data_dt, edge_dt, var_name)

    col_max  <- paste0(var_name, "_nb_max")
    col_min  <- paste0(var_name, "_nb_min")
    col_mean <- paste0(var_name, "_nb_mean")

    set(cell_data_dt, j = col_max,  value = stats$nb_max)
    set(cell_data_dt, j = col_min,  value = stats$nb_min)
    set(cell_data_dt, j = col_mean, value = stats$nb_mean)
  }

  # Remove helper column if present
  if ("row_idx" %in% names(cell_data_dt)) {
    cell_data_dt[, row_idx := NULL]
  }

  message("Neighbor features complete.")
  cell_data_dt
}

# ──────────────────────────────────────────────────────────────────────
# 4. Usage (drop-in replacement for the original outer loop)
# ──────────────────────────────────────────────────────────────────────
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# # Then proceed to RF prediction exactly as before — no changes needed:
# model <- readRDS("trained_rf_model.rds")
# cell_data$prediction <- predict(model, data = cell_data)$predictions
# # (or however predict is called for your specific RF package)
```

---

## Summary of Expected Improvement

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level `lapply` iterations with `paste()` + named vector lookup | Single vectorized `data.table` join (~seconds) |
| `compute_neighbor_stats` (×5 vars) | ~32.3M R-level iterations, each with subsetting + `max/min/mean` | 5 vectorized `data.table` group-by aggregations (~seconds each) |
| `do.call(rbind, ...)` (×5) | Binding 6.46M 3-element vectors per call | Eliminated entirely |
| **Estimated total** | **86+ hours** | **Minutes** (dominated by the edge-list cross-join memory allocation) |
| RF inference | Fast (unchanged) | Fast (unchanged) |

The trained Random Forest model is never modified. The same 15 neighbor-derived features (5 variables × {max, min, mean}) are computed with identical numerical values, preserving the original estimand.