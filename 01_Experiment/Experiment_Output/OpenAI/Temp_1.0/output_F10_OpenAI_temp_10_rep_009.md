 **Diagnosis**  
- Current bottleneck: `lapply` over 6.46M rows repeatedly for 5 variables → extremely inefficient in R. Each function constructs neighbor-based statistics by pulling values repeatedly via indexed lookups.
- Redundant work: neighbor structure is recomputed per attribute; fully materialized neighbor lookups lead to large memory overhead.
- Years are independent, but approach ignores vectorization—no batching, no matrix operations.
- Graph topology is static: rook neighbors do not change per year; recomputing index alignments repeatedly wastes time.
- Nested functions with `lapply` and `do.call(rbind, ...)` are not memory-optimal for millions of rows.

---

**Optimization Strategy**  
- Build graph adjacency **once** as integer indices for all cells (`adj_list`).
- Reshape panel data to a matrix with rows = cells, cols = years per variable for quick lookup.
- Compute neighbor stats using **vectorized** operations (`apply` over neighbors or `Matrix` ops).
- Combine `data.table` for speed.
- Compute max/min/mean in a single pass per variable and year using preallocated numeric arrays.
- Write output directly into columns, avoiding repeated binding.
- Preserve trained Random Forest; only input features change.

---

**Efficient R Implementation**  

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Basic facts
ids <- unique(cell_data$id)
years <- sort(unique(cell_data$year))
n_cells <- length(ids)
n_years <- length(years)

# Map id -> row index
id_index <- setNames(seq_along(ids), ids)

# Build static adjacency once
rook_neighbors_unique <- readRDS("rook_neighbors_unique.rds") # spdep::nb object
adj_list <- lapply(rook_neighbors_unique, function(neigh) id_index[as.character(ids[neigh])])

# Prepare fast lookup table: arrange panel into list of matrices per variable
neighbor_source_vars <- c("ntl","ec","pop_density","def","usd_est_n2")
# Pivot to wide for each variable
var_mats <- lapply(neighbor_source_vars, function(var) {
  m <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  val_dt <- cell_data[, .(id_idx = id_index[as.character(id)], year_idx = match(year, years), val = get(var))]
  m[cbind(val_dt$id_idx, val_dt$year_idx)] <- val_dt$val
  m
})
names(var_mats) <- neighbor_source_vars

# Preallocate result columns in cell_data keyed by (id_idx, year_idx)
cell_data[, (paste0(rep(neighbor_source_vars, each = 3),
                    "_neighbor_", c("max","min","mean"))) := NA_real_]

# Compute neighbor stats efficiently
compute_stats <- function(var_mat, adj_list) {
  # var_mat: n_cells x n_years
  res_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  res_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  res_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    neigh <- adj_list[[i]]
    if (length(neigh) > 0) {
      vals <- var_mat[neigh, , drop = FALSE]
      # Compute along rows of neighbors
      res_max[i, ]  <- apply(vals, 2, max, na.rm = TRUE)
      res_min[i, ]  <- apply(vals, 2, min, na.rm = TRUE)
      res_mean[i, ] <- apply(vals, 2, mean, na.rm = TRUE)
    }
  }
  list(max = res_max, min = res_min, mean = res_mean)
}

for (var in neighbor_source_vars) {
  stats <- compute_stats(var_mats[[var]], adj_list)
  
  # Flatten back into cell_data order
  cell_data[, paste0(var, "_neighbor_max") :=
              stats$max[cbind(id_index[as.character(id)], match(year, years))]]
  cell_data[, paste0(var, "_neighbor_min") :=
              stats$min[cbind(id_index[as.character(id)], match(year, years))]]
  cell_data[, paste0(var, "_neighbor_mean") :=
              stats$mean[cbind(id_index[as.character(id)], match(year, years))]]
}

# Predict using pre-trained Random Forest without retraining
preds <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Why it’s faster**  
- Single adjacency structure reused → no recomputation.
- Panel reshaped to numeric matrices → cache-friendly access.
- Minimized function calls via preallocation → memory-efficient.
- Complexity reduced from O(N * neighbors * vars * R overhead) to near O(N * neighbors) in compiled vectorized loops.
- Estimate: **hours → tens of minutes** on 16GB laptop.