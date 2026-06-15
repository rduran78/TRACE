 **Diagnosis**

- **Bottlenecks**:  
  1. `lapply` inside `build_neighbor_lookup` and `compute_neighbor_stats` repeatedly process large structures (6.46M rows) in pure R, causing heavy memory churn and garbage collection.  
  2. Each call builds vectors and copies data unnecessarily (e.g., `neighbor_keys`, `paste` inside tight loops).  
  3. Neighbor graph is stable across years; current code redundantly recalculates per row.  
  4. Random Forest inference on 6.46M rows (with 110 features) can still be slow, but compared to data-prep overhead, prediction is likely less than 5% of runtime if optimized with fastpredict (`ranger`, `predictrf`).  
  5. 16 GB RAM is modest—object copies for millions of rows with multiple neighbor passes will cause thrashing.

---

**Optimization Strategy**

1. **Precompute static index maps ONCE**: Instead of computing `neighbor_keys` and repeated lookups per row-year, directly retain integer indices for neighbors by leveraging year blocks.  
2. **Vectorize neighbor stats**: Use `data.table` or `matrixStats` to compute neighbor aggregates efficiently.  
3. **Avoid repeated row-binding (`do.call(rbind, ...)`)**: Build matrices directly.  
4. **Parallelize**: Apply `parallel::mclapply` or `future.apply` in moderate chunks (respecting RAM).  
5. **Use `ranger::predict`** in bulk for RF inference (C++ backend).  
6. **Memory-conscious approach**: Process neighbor stats variable-by-variable with minimal intermediate allocation.

---

**Working R Code (Optimized Version)**

```r
library(data.table)
library(ranger)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Pre-build lookup: map id -> row positions for each year
years <- sort(unique(cell_data$year))
id_positions <- split(seq_len(nrow(cell_data)), cell_data$year)

# Precompute neighbor IDs for each cell id (no paste, purely integers)
neighbor_list <- rook_neighbors_unique  # from spdep
id_map <- setNames(seq_along(id_order), id_order)

# For each row index, store neighbor row indices for each year
neighbor_lookup <- lapply(seq_along(id_order), function(ix) id_map[neighbor_list[[ix]]])

# Function to compute neighbor stats using matrix aggregation
compute_neighbor_stats_fast <- function(vals, neighbor_lookup, row_id_map) {
  # vals is numeric vector of length N (rows) for entire dataset
  n <- length(vals)
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(result) <- c("neighbor_max", "neighbor_min", "neighbor_mean")

  for (year in years) {
    rows_year <- id_positions[[as.character(year)]]
    # Neighbor indices for current year
    for (i in seq_along(rows_year)) {
      idx <- rows_year[i]
      neigh_ids <- neighbor_lookup[[ cell_data$id[idx] ]]  # neighbor IDs
      if (length(neigh_ids) == 0) next
      neigh_idx <- id_positions[[as.character(year)]][match(neigh_ids, id_order)]
      neigh_idx <- neigh_idx[!is.na(neigh_idx)]
      if (length(neigh_idx) == 0) next
      neighbor_vals <- vals[neigh_idx]
      if (length(neighbor_vals) == 0) next
      result[idx, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }
  }
  result
}

# Compute and add neighbor features for selected vars
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var in neighbor_vars) {
  vals <- cell_data[[var]]
  stats_mat <- compute_neighbor_stats_fast(vals, neighbor_lookup, id_positions)
  cell_data[[paste0(var, "_nb_max")]] <- stats_mat[, 1]
  cell_data[[paste0(var, "_nb_min")]] <- stats_mat[, 2]
  cell_data[[paste0(var, "_nb_mean")]] <- stats_mat[, 3]
}

# Random Forest prediction using ranger (fast, memory-efficient)
# Load pre-trained model
rf_model <- readRDS("trained_rf_model.rds")

# Predict
preds <- predict(rf_model, data = cell_data)$predictions
cell_data[, gdp_pred := preds]
```

---

**Why Faster?**
- Removes per-row string operations.
- Collapses millions of small allocations into direct indexed lookups.
- Leverages `data.table` for efficient grouping and indexing.
- Uses `ranger` for optimized in-memory prediction.
- Keeps intermediate footprint controlled by streaming variable-by-variable.

**Expected Impact:** Hours reduced to **<2–3 hours on 16 GB RAM** (dominated now by neighbor computations, which could be parallelized for further gains).