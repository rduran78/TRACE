 **Diagnosis:**  
The bottleneck is the repeated `lapply` over 6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation (`paste`) and hash lookups, which are extremely costly at this scale. The neighbor relationships are static across years, but the code redundantly recomputes neighbor indices for every cell-year. This results in ~6.46M × average neighbors operations, which is prohibitively expensive.

---

**Optimization Strategy:**  
1. **Precompute neighbor indices once per cell (not per cell-year)** since spatial neighbors do not change over time.
2. **Vectorize neighbor stats computation** using `data.table` or `dplyr` joins instead of per-row `lapply`.
3. **Avoid string concatenation for keys**; use integer indexing.
4. **Compute neighbor stats in a grouped manner**: reshape data to wide or use rolling joins so that for each year, neighbor values are merged efficiently.
5. **Parallelize where possible** using `data.table` or `future.apply`.

---

**Optimized R Code (data.table approach):**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor list as integer indices (once)
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_idx_list <- lapply(rook_neighbors_unique, function(nbs) id_to_idx[nbs])

# Add an integer index for each cell
cell_data[, cell_idx := id_to_idx[id]]

# For speed, set keys
setkey(cell_data, cell_idx, year)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(dt, var_name, neighbor_idx_list) {
  # Extract relevant columns
  vals <- dt[[var_name]]
  
  # Prepare result matrix
  res <- matrix(NA_real_, nrow = nrow(dt), ncol = 3)
  
  # Process by year to reduce memory footprint
  years <- unique(dt$year)
  
  for (yr in years) {
    # Subset for this year
    idx_year <- which(dt$year == yr)
    vals_year <- vals[idx_year]
    cell_idx_year <- dt$cell_idx[idx_year]
    
    # Map from cell_idx to position in this year's slice
    pos_map <- match(seq_along(id_order), cell_idx_year)
    
    # Compute neighbor stats
    for (i in seq_along(idx_year)) {
      nbs <- neighbor_idx_list[[cell_idx_year[i]]]
      nb_pos <- pos_map[nbs]
      nb_pos <- nb_pos[!is.na(nb_pos)]
      if (length(nb_pos) > 0) {
        nb_vals <- vals_year[nb_pos]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) > 0) {
          res[idx_year[i], ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
        }
      }
    }
  }
  
  colnames(res) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  res
}

# Apply for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
results_list <- lapply(neighbor_source_vars, function(v) compute_neighbor_stats_fast(cell_data, v, neighbor_idx_list))

# Combine results
neighbor_features <- do.call(cbind, results_list)
cell_data <- cbind(cell_data, neighbor_features)
```

---

**Expected Gains:**  
- Eliminates repeated string operations and hash lookups.
- Reduces complexity from O(n × neighbors × years) string-based to integer-based lookups.
- Memory-efficient by processing year-by-year.
- On a 16 GB laptop, this should cut runtime from 86+ hours to a few hours (or less with parallelization).

**Preserves:**  
- Original Random Forest model.
- Numerical estimand (same neighbor stats).
- Panel structure and neighbor relationships.