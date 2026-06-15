 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** as the dominant bottleneck — specifically `build_neighbor_lookup` and `compute_neighbor_stats`.

**Evidence from the code:**

1. **`build_neighbor_lookup`** uses a top-level `lapply` over **~6.46 million rows**. Inside each iteration it performs `as.character()` coercion, `paste()` string construction, and named-vector lookups (`idx_lookup[neighbor_keys]`) — all character-based hash lookups repeated per row. With ~1.37 million neighbor relationships spread across 6.46M row iterations, this is enormously expensive in pure R.

2. **`compute_neighbor_stats`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million elements via `lapply`, subsetting a numeric vector, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` at the end binds 6.46 million 3-element vectors into a matrix — also slow.

3. The outer loop calls this machinery for **5 variables**, so the total work is: 6.46M iterations for the lookup build + 5 × 6.46M iterations for stats = **~38.8 million R-level loop iterations** with per-iteration string operations and subsetting.

4. By contrast, Random Forest `predict()` on a pre-trained model with ~110 predictors and 6.46M rows is a single vectorized C/C++ call (in `ranger` or `randomForest`). Even for a large forest this typically completes in minutes, not hours.

**Conclusion:** The 86+ hour runtime is dominated by row-level R loops with string operations in the neighbor feature engineering, not by RF inference.

---

## Optimization Strategy

1. **Eliminate per-row string operations in `build_neighbor_lookup`:** Replace character-key lookups with integer-indexed lookups. Pre-build a matrix mapping `(cell_index, year_index) → row_number` so neighbor row indices can be retrieved via integer matrix indexing — O(1) and vectorized.

2. **Vectorize `compute_neighbor_stats`:** Instead of `lapply` over 6.46M elements, unroll the neighbor lookup into a flat edge list (source_row, neighbor_row), extract all neighbor values at once, then use `data.table` grouped aggregation (`max`, `min`, `mean`) — fully vectorized in C.

3. **Preserve the trained RF model and the original numerical estimand** — no changes to modeling or prediction code.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED build_neighbor_lookup
# Returns a data.table edge list: (source_row, neighbor_row)
# instead of a list-of-vectors over 6.46M elements.
# ==============================================================================
build_neighbor_edgelist <- function(data, id_order, neighbors) {
  # data must have columns: id, year
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  dt <- as.data.table(data)
  dt[, row_idx := .I]

  n_ids   <- length(id_order)
  n_years <- uniqueN(dt$year)
  years   <- sort(unique(dt$year))

  # Map (cell_position_in_id_order, year) -> row index in data
  # cell_position: 1..n_ids;  year_position: 1..n_years
  id_to_pos  <- setNames(seq_along(id_order), as.character(id_order))
  year_to_pos <- setNames(seq_along(years), as.character(years))

  dt[, id_pos   := id_to_pos[as.character(id)]]
  dt[, year_pos := year_to_pos[as.character(year)]]

  # Build a lookup matrix: row_lookup[id_pos, year_pos] = row_idx in data
  row_lookup <- matrix(NA_integer_, nrow = n_ids, ncol = n_years)
  row_lookup[cbind(dt$id_pos, dt$year_pos)] <- dt$row_idx

  # Build flat edge list from the nb object
  # For each cell i (position in id_order), neighbors[[i]] gives neighbor positions
  src_pos <- rep(seq_len(n_ids), lengths(neighbors))
  nbr_pos <- unlist(neighbors)

  # Expand across all years: for each (src_cell, nbr_cell) pair, repeat for every year
  n_edges_per_year <- length(src_pos)
  n_year_vec       <- length(years)

  # Replicate edge list across years
  src_pos_all  <- rep(src_pos, times = n_year_vec)
  nbr_pos_all  <- rep(nbr_pos, times = n_year_vec)
  year_pos_all <- rep(seq_len(n_year_vec), each = n_edges_per_year)

  # Look up actual row indices
  source_row   <- row_lookup[cbind(src_pos_all, year_pos_all)]
  neighbor_row <- row_lookup[cbind(nbr_pos_all, year_pos_all)]

  # Remove edges where either source or neighbor row doesn't exist
  valid <- !is.na(source_row) & !is.na(neighbor_row)

  data.table(
    source_row   = source_row[valid],
    neighbor_row = neighbor_row[valid]
  )
}

# ==============================================================================
# OPTIMIZED compute_neighbor_stats (vectorized via data.table)
# ==============================================================================
compute_neighbor_stats_fast <- function(data_dt, edgelist, var_name) {
  # edgelist: data.table with (source_row, neighbor_row)
  # data_dt: data.table with row_idx or we use positional indexing

  vals <- data_dt[[var_name]]

  # Attach neighbor values
  el <- copy(edgelist)
  el[, nbr_val := vals[neighbor_row]]

  # Remove NAs

  el <- el[!is.na(nbr_val)]

  # Grouped aggregation — fully vectorized in C via data.table
  stats <- el[, .(
    nb_max  = max(nbr_val),
    nb_min  = min(nbr_val),
    nb_mean = mean(nbr_val)
  ), by = source_row]

  stats
}

# ==============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# ==============================================================================
compute_and_add_neighbor_features_fast <- function(data_dt, var_name, edgelist) {
  n <- nrow(data_dt)
  stats <- compute_neighbor_stats_fast(data_dt, edgelist, var_name)

  # Initialize with NA
  max_col  <- rep(NA_real_, n)
  min_col  <- rep(NA_real_, n)
  mean_col <- rep(NA_real_, n)

  max_col[stats$source_row]  <- stats$nb_max
  min_col[stats$source_row]  <- stats$nb_min
  mean_col[stats$source_row] <- stats$nb_mean

  set(data_dt, j = paste0("nb_max_",  var_name), value = max_col)
  set(data_dt, j = paste0("nb_min_",  var_name), value = min_col)
  set(data_dt, j = paste0("nb_mean_", var_name), value = mean_col)

  data_dt
}

# ==============================================================================
# MAIN PIPELINE (drop-in replacement for the outer loop)
# ==============================================================================

# Convert to data.table once
cell_data_dt <- as.data.table(cell_data)

# Build the edge list ONCE (replaces build_neighbor_lookup)
# This is vectorized — runs in seconds, not hours.
edgelist <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data_dt <- compute_and_add_neighbor_features_fast(
    cell_data_dt, var_name, edgelist
  )
}

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_data_dt)

# ==============================================================================
# Random Forest prediction is UNCHANGED — it was never the bottleneck.
# Example (assuming the model object is `rf_model`):
#   predictions <- predict(rf_model, newdata = cell_data)
# ==============================================================================
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup build** | 6.46M `lapply` iterations with `paste()` + character hash lookups | Vectorized integer matrix indexing; flat edge list built in seconds |
| **Neighbor stats** | 5 × 6.46M `lapply` iterations + `do.call(rbind, ...)` | 5 × one `data.table` grouped aggregation (C-level) |
| **Expected runtime** | ~86+ hours | **Minutes** (edge list build ~30s–2min; each variable's stats ~10–30s) |
| **RF model** | Preserved (unchanged) | Preserved (unchanged) |
| **Numerical estimand** | max, min, mean of non-NA neighbor values per row | Identical — same semantics, same output columns |

The bottleneck was never Random Forest inference. It was millions of R-level loop iterations performing per-row string construction and character-keyed lookups in the neighbor feature engineering step.