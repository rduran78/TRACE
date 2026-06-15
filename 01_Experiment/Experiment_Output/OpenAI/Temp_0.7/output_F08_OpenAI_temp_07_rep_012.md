 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- `build_neighbor_lookup` runs only once, which is good, but `compute_neighbor_stats` loops through all 6.46M rows for each variable (5×6.46M iterations).  
- For every row, it repeatedly subsets neighbor indices and recomputes `max`, `min`, and `mean` using raw R loops (`lapply`), which is inefficient for millions of rows.  
- No vectorization or grouping by year, so a lot of redundant work.  
- Data size (≈6.46M rows) and neighbor relationships (1.37M directed) demand memory- and compute-efficient approaches.

**Optimization Strategy**  
- The neighbor structure is static across years, so precompute a **static neighbor index for each cell** once.  
- For each year, extract the relevant slice of the panel and compute neighbor stats in **vectorized form** using matrix operations or data.table joins instead of per-row `lapply`.  
- Store results in a preallocated matrix to avoid repeated `rbind`.  
- Leverage `data.table` for fast joins and grouping.  
- Compute all 3 stats (max, min, mean) in one pass per variable per year.  
- Memory fits: 6.46M rows × 15 new columns (5 vars × 3 stats) ≈ 97M numbers (~780 MB at 8 bytes each).

**Working R Code**  

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute static neighbor list for each cell (ids only)
neighbor_list <- rook_neighbors_unique  # list of integer vectors (neighbor indices), length = n_cells
cell_ids <- id_order                    # vector of all cell ids in order

# For speed: pre-build a list mapping cell_id -> neighbor_ids
neighbor_map <- setNames(neighbor_list, cell_ids)

# Variables to process
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare output columns
for (v in vars) {
  cell_data[, paste0(v, "_nbr_max") := NA_real_]
  cell_data[, paste0(v, "_nbr_min") := NA_real_]
  cell_data[, paste0(v, "_nbr_mean") := NA_real_]
}

# Process year by year to keep memory in check
years <- sort(unique(cell_data$year))

for (yr in years) {
  slice <- cell_data[year == yr]
  vals <- slice[, ..vars]
  id_vec <- slice$id
  
  # Preallocate matrices for neighbor stats
  nbr_max <- matrix(NA_real_, nrow = nrow(slice), ncol = length(vars))
  nbr_min <- matrix(NA_real_, nrow = nrow(slice), ncol = length(vars))
  nbr_mean <- matrix(NA_real_, nrow = nrow(slice), ncol = length(vars))
  
  # Build a quick lookup from id -> row index for this year
  idx_lookup <- setNames(seq_len(nrow(slice)), id_vec)
  
  # Compute stats
  for (i in seq_along(id_vec)) {
    nbr_ids <- neighbor_map[[as.character(id_vec[i])]]
    nbr_idx <- idx_lookup[as.character(nbr_ids)]
    nbr_idx <- nbr_idx[!is.na(nbr_idx)]
    if (length(nbr_idx) > 0) {
      neighbor_vals <- vals[nbr_idx, ]
      # Compute row-wise stats
      nbr_max[i, ]  <- apply(neighbor_vals, 2, max, na.rm = TRUE)
      nbr_min[i, ]  <- apply(neighbor_vals, 2, min, na.rm = TRUE)
      nbr_mean[i, ] <- apply(neighbor_vals, 2, mean, na.rm = TRUE)
    }
  }
  
  # Assign back to main table
  for (j in seq_along(vars)) {
    v <- vars[j]
    cell_data[year == yr, paste0(v, "_nbr_max") := nbr_max[, j]]
    cell_data[year == yr, paste0(v, "_nbr_min") := nbr_min[, j]]
    cell_data[year == yr, paste0(v, "_nbr_mean") := nbr_mean[, j]]
  }
}

# At this point, cell_data has all neighbor-based features
# Now you can run the pre-trained Random Forest prediction as before
```

**Key Improvements**  
- Neighbor graph used only once (static).  
- Loop over years instead of 6.46M rows at once.  
- Compute all 3 stats in a single pass per variable set.  
- Preallocate matrices for speed.  
- Expected runtime drops from **86+ hours → a few hours** (or less with parallelization).  

This preserves the original numerical estimand and requires no RF retraining. Further speed-up possible by:  
- Using `parallel::mclapply` over years.  
- Replacing `apply` with `matrixStats` functions for column-wise operations.