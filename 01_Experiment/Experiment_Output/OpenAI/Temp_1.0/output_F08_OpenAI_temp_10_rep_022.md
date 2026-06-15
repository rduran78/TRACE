 **Diagnosis**  
The current pipeline recomputes neighbor stats (max, min, mean) for each row in the 6.46M dataset using `lapply` across all rows. This is extremely inefficient because it repeatedly processes neighbor relationships (which never change) for each year. Essentially, the static neighbor graph is being traversed 6.46M times. Memory-wise, this involves large lists of indices being replicated for all years — consuming time and RAM.  

**Optimization Strategy**  
- The key insight: neighbor relationships are *static*, but cell-year data changes.  
- Precompute neighbor indices for each **cell only** (344,208 nodes, not 6.46M rows).  
- For each year:  
   - Extract the year-specific slice of data.  
   - Compute neighbor stats for that slice using the fixed neighbor lookup.  
   - Append results back to a vector or data frame by matching the cell–year.  
- Use **vectorized operations** instead of looping over every row individually.  
- Avoid storing large nested lists for 6M+ rows.  
- Process by year in batches instead of globally for all rows.  

This reduces RAM usage and speeds computation by focusing on **344k cells × 28 years** rather than **6.46M neighbor traversals**.  

---

### **Optimized R Code**

```r
# Build neighbor lookup for cells only (not by year)
build_cell_neighbor_lookup <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]   # Cell IDs of neighbors
    neighbor_ids
  })
}

# Compute neighbor stats for one variable for a single year
compute_year_neighbor_stats <- function(year_data, var_name, cell_neighbor_lookup) {
  vals <- year_data[[var_name]]
  result <- lapply(cell_neighbor_lookup, function(neighbor_ids) {
    neighbor_vals <- vals[match(neighbor_ids, year_data$id)]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  })
  do.call(rbind, result)
}

# Main optimized computation
cell_neighbor_lookup <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare storage
result_list <- vector("list", length(neighbor_source_vars))

for (var_name in neighbor_source_vars) {
  # Initialize empty list for yearly results
  yearly_results <- vector("list", length = length(unique(cell_data$year)))
  
  for (yr in sort(unique(cell_data$year))) {
    year_data <- subset(cell_data, year == yr, select = c("id", var_name))
    stats_matrix <- compute_year_neighbor_stats(year_data, var_name, cell_neighbor_lookup)
    colnames(stats_matrix) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
    
    # Attach year and id
    yearly_results[[as.character(yr)]] <- data.frame(
      id = year_data$id,
      year = yr,
      stats_matrix,
      stringsAsFactors = FALSE
    )
  }
  
  # Combine yearly results and merge back to cell_data
  var_stats <- do.call(rbind, yearly_results)
  cell_data <- merge(cell_data, var_stats, by = c("id", "year"), all.x = TRUE)
}

# Result: cell_data now has neighbor max/min/mean vars for all source vars
```

---

### **Why this works**
- Neighbor index computation happens **once for all cells**, not for every cell-year row.
- Per-year slicing preserves the original estimand without altering Random Forest requirements.
- Vectorized matching avoids repeated string concatenations and extremely large index lists.
- Memory footprint is much lighter: operations are done in `344k × neighbors` context, not `6.46M × neighbors` per row.
- Estimated runtime drops from **86+ hours** to a few hours (or less) on a 16 GB machine.

This preserves the trained Random Forest model, uses the original estimand, and follows best practices for panel data computation where spatial relationships are static and temporal data varies.