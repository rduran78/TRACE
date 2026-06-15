 **Diagnosis**  
- Current design computes neighbor statistics for **each cell-year row** using `lapply` over 6.46M rows.  
- Repeated lookups on a large list (`neighbor_lookup`) and repeated filtering cause massive overhead.  
- Neighbor relationships are **static** across years, but the code recomputes neighbor stats row by row.  
- This results in ~6.46M × 5 variables × 3 stats operations, which is very slow and memory-inefficient.  

---

**Optimization Strategy**  
1. **Precompute neighbor indices once** (already done by `build_neighbor_lookup`).  
2. **Exploit panel structure**:  
   - For each year, take the subset of data for that year.  
   - Compute neighbor stats **vectorized** using the static neighbor relationships.  
   - Append results back to the yearly subset.  
3. Use `matrix` operations or `vapply` instead of row-wise `lapply`.  
4. Process year by year to control memory footprint.  
5. Avoid re-allocating large objects repeatedly.  

---

**Optimized R Code**

```r
# Precompute static neighbor index list (same as before)
neighbor_lookup_static <- build_neighbor_lookup(
  data = data.frame(id = id_order, year = 1), # dummy year
  id_order = id_order,
  neighbors = rook_neighbors_unique
)

# Function to compute neighbor stats for a numeric vector given neighbor lookup
compute_neighbor_stats_vectorized <- function(vals, neighbor_lookup) {
  n <- length(vals)
  maxs <- numeric(n)
  mins <- numeric(n)
  means <- numeric(n)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) {
      maxs[i] <- NA; mins[i] <- NA; means[i] <- NA
    } else {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) == 0) {
        maxs[i] <- NA; mins[i] <- NA; means[i] <- NA
      } else {
        maxs[i] <- max(v)
        mins[i] <- min(v)
        means[i] <- mean(v)
      }
    }
  }
  cbind(maxs, mins, means)
}

# Main optimized loop: process by year
years <- sort(unique(cell_data$year))
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

result_list <- list()

for (yr in years) {
  subset_idx <- which(cell_data$year == yr)
  subset_data <- cell_data[subset_idx, ]
  
  # Ensure subset is sorted by id_order for consistent indexing
  subset_data <- subset_data[match(id_order, subset_data$id), ]
  
  for (var_name in neighbor_source_vars) {
    vals <- subset_data[[var_name]]
    stats_mat <- compute_neighbor_stats_vectorized(vals, neighbor_lookup_static)
    colnames(stats_mat) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
    subset_data <- cbind(subset_data, stats_mat)
  }
  
  result_list[[as.character(yr)]] <- subset_data
}

# Combine all years back
cell_data_enhanced <- do.call(rbind, result_list)

# Pass cell_data_enhanced to the pre-trained Random Forest model
# prediction <- predict(rf_model, newdata = cell_data_enhanced)
```

---

**Why This is Faster**  
- **Static neighbor list** used directly for all years — no recomputation per row.  
- **Year-by-year processing** keeps memory usage manageable.  
- Inner computation uses **simple loops over ~344K cells**, not 6.46M rows × neighbors repeatedly.  
- Removes expensive string concatenation and repeated lookup in `idx_lookup`.  

Estimated runtime drops from **86+ hours to a few hours** on a 16GB laptop.