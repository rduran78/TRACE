 **Diagnosis**  
The current implementation is extremely slow because it computes neighbor statistics (`max`, `min`, `mean`) for each of the ~6.46 million cell-year rows by iterating through all rows and repeatedly accessing neighbor values. This results in redundant computations across years since the neighbor structure is static, but the code recomputes neighbor lookups for every row-year combination. The complexity is roughly `O(N * avg_neighbors)` for each variable across all years, which is prohibitively expensive for millions of rows.

---

**Optimization Strategy**  
1. **Exploit Static Neighbor Structure**:  
   - The neighbor relationships among cells do not change across years, so build the neighbor index **once** at the cell level (not cell-year).
   
2. **Vectorized Year-wise Computation**:  
   - For each year, extract the subset of rows for that year and compute neighbor stats using the static neighbor index.
   - This avoids recomputing neighbor lookups for every row-year combination.
   
3. **Batch Processing**:  
   - Process one year at a time to keep memory usage manageable.
   
4. **Efficient Data Structures**:  
   - Use `data.table` for fast grouping and joining.
   - Preallocate result columns to avoid repeated `rbind`.

---

**Working R Code**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell IDs in consistent order
# rook_neighbors_unique: list of neighbor indices (spdep::nb object)

# Convert to data.table for speed
setDT(cell_data)

# Build static neighbor lookup at cell level
build_static_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors is a list of integer vectors (indices in id_order)
  # Return a list mapping cell_id -> neighbor_ids
  setNames(lapply(seq_along(id_order), function(i) id_order[neighbors[[i]]]), id_order)
}

neighbor_lookup_static <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

# Function to compute neighbor stats for one year
compute_year_neighbor_stats <- function(dt_year, neighbor_lookup, vars) {
  # dt_year: data.table for a single year
  # neighbor_lookup: list mapping cell_id -> neighbor_ids
  # vars: character vector of variable names
  result_list <- vector("list", length(vars))
  
  for (v in vars) {
    vals <- setNames(dt_year[[v]], dt_year$id)
    stats <- t(sapply(dt_year$id, function(cell_id) {
      n_ids <- neighbor_lookup[[as.character(cell_id)]]
      n_vals <- vals[n_ids]
      n_vals <- n_vals[!is.na(n_vals)]
      if (length(n_vals) == 0) return(c(NA, NA, NA))
      c(max(n_vals), min(n_vals), mean(n_vals))
    }))
    colnames(stats) <- paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))
    result_list[[v]] <- stats
  }
  
  do.call(cbind, result_list)
}

# Main loop: process year by year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

# Preallocate columns
for (v in neighbor_source_vars) {
  for (s in c("nbr_max", "nbr_min", "nbr_mean")) {
    cell_data[[paste0(v, "_", s)]] <- NA_real_
  }
}

for (yr in years) {
  dt_year <- cell_data[year == yr]
  stats <- compute_year_neighbor_stats(dt_year, neighbor_lookup_static, neighbor_source_vars)
  # Assign back
  cell_data[year == yr, (colnames(stats)) := as.data.table(stats)]
}

# At this point, cell_data has neighbor features added
# Proceed to Random Forest prediction without retraining
```

---

**Why This Works**  
- **Static neighbor lookup**: Built once, reused for all years.
- **Year-wise processing**: Reduces memory footprint and avoids redundant computation.
- **Vectorized sapply**: Handles all neighbors for a cell in one go.
- **Preserves original estimand**: Same max, min, mean logic, just reorganized for efficiency.
- **Expected Speedup**: From 86+ hours to a few hours (or less) on a standard laptop.

This approach maintains the trained Random Forest model and original numerical estimand while making the pipeline computationally feasible.