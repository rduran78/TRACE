 **Diagnosis**  
The current pipeline is extremely slow (86+ hours) because `compute_neighbor_stats` is applied for every variable across all 6.46M rows using a per-row `lapply`. This creates massive overhead:  
- **Redundant computation**: For each year and variable, the same neighbor relationships are recalculated repeatedly.  
- **High memory usage and poor vectorization**: Each row is processed individually rather than in grouped or matrix form.  
- **Static neighbor structure ignored**: The neighbor graph does not change over time, but the code recomputes neighbor indices repeatedly by concatenating `id` and `year`.  

---

**Optimization Strategy**  
1. **Precompute a static neighbor index by `id` (cell)**, not by row. This is possible because neighbor relationships are constant across years.  
2. **Process one year at a time in a vectorized manner**: For each year, subset data and compute neighbor stats for all rows in that year using fast matrix operations.  
3. **Avoid repeated string concatenation and lookups**: Use integer indexing and pre-built maps.  
4. **Write results directly into columns** rather than rebuilding data frames in loops.  
5. **Use `data.table` for efficient grouping and assignment**.  

This reduces complexity from `O(N_rows × neighbors)` across the whole panel to `O(N_year × (N_cells + neighbors))` per variable.  

---

**Working R Code**  

```r
library(data.table)

# Assume: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer neighbor indices (1-based) matching id_order
# id_order: vector of cell IDs in the same order as rook_neighbors_unique

setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor indices by position
id_to_pos <- setNames(seq_along(id_order), id_order)
neighbor_pos_list <- lapply(rook_neighbors_unique, function(neigh_ids) {
  id_to_pos[as.character(neigh_ids)]
})

# Prepare new columns for neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var in neighbor_source_vars) {
  cell_data[, paste0(var, "_nbr_max") := NA_real_]
  cell_data[, paste0(var, "_nbr_min") := NA_real_]
  cell_data[, paste0(var, "_nbr_mean") := NA_real_]
}

# Compute per year in-memory
years <- sort(unique(cell_data$year))

for (yr in years) {
  cat("Processing year:", yr, "\n")
  # Subset this year's data in id_order
  year_dt <- cell_data[year == yr][match(id_order, id)]
  
  for (var in neighbor_source_vars) {
    vals <- year_dt[[var]]
    
    # Compute neighbor stats using precomputed neighbor positions
    nbr_stats <- lapply(neighbor_pos_list, function(nbr_idx) {
      nbr_vals <- vals[nbr_idx]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) == 0) return(c(NA, NA, NA))
      c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
    })
    
    nbr_stats_mat <- do.call(rbind, nbr_stats)
    
    # Assign back by id/year
    cell_data[year == yr, paste0(var, "_nbr_max") := nbr_stats_mat[, 1]]
    cell_data[year == yr, paste0(var, "_nbr_min") := nbr_stats_mat[, 2]]
    cell_data[year == yr, paste0(var, "_nbr_mean") := nbr_stats_mat[, 3]]
  }
}

# At this point, cell_data has neighbor-based features ready for RF prediction
```

---

**Why This Works**  
- **Static neighbor graph leveraged**: We build neighbor indices once, not per row-year.  
- **Year-based batching**: Each year runs in-memory for 344,208 rows rather than all 6.46M rows at once.  
- **Vectorized assignment**: Uses `data.table` for speed and memory efficiency.  
- **Preserves original estimand**: The computed stats (max, min, mean of neighbors per variable) remain unchanged.  
- **No retraining**: Random Forest step uses the enriched dataset as before.  

---

**Expected Performance Gain**  
From 86+ hours to a few hours on a standard 16 GB laptop (dominated by 28 passes × 344k rows, fully memory-based).  
Further gains possible by parallelizing the outer loop over years using `parallel::mclapply` or `future.apply`.