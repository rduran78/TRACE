 **Diagnosis:**  
The current pipeline is extremely slow because `build_neighbor_lookup` is recomputing neighbor indices for every cell-year row (≈6.46M rows) and `compute_neighbor_stats` repeatedly scans large vectors. This results in massive redundant work across 28 years. The neighbor structure is static across years, so recomputing neighbor relationships per row is unnecessary.  

**Optimization Strategy:**  
- Build a reusable adjacency table **once** at the cell level (344,208 rows) instead of per cell-year row.
- For each year, join cell attributes to this adjacency table and compute neighbor statistics using vectorized operations or `data.table` joins.
- Avoid repeated `lapply` over millions of rows; instead, precompute neighbor pairs and aggregate efficiently.
- Keep the Random Forest model unchanged and preserve the estimand.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor pairs (cell-level, static)
# rook_neighbors_unique: list of integer vectors (neighbors for each cell)
neighbor_pairs <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Ensure keys for fast joins
setkey(neighbor_pairs, neighbor_id)
setkey(cell_data, id, year)

# Function to compute neighbor stats for a variable
compute_neighbor_stats_fast <- function(var_name) {
  # Join neighbor attributes for each year
  merged <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  merged <- neighbor_pairs[merged, on = "neighbor_id", allow.cartesian = TRUE]
  
  # Aggregate by focal cell and year
  stats <- merged[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(cell_id, year)]
  
  # Rename columns
  setnames(stats, c("cell_id", "year", "max_val", "min_val", "mean_val"),
           c("id", "year",
             paste0(var_name, "_nbr_max"),
             paste0(var_name, "_nbr_min"),
             paste0(var_name, "_nbr_mean")))
  
  stats
}

# Compute neighbor features for all variables and merge
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

neighbor_features <- lapply(neighbor_source_vars, compute_neighbor_stats_fast)
neighbor_features <- Reduce(function(x, y) merge(x, y, by = c("id", "year"), all = TRUE),
                            neighbor_features)

# Merge back into main dataset
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)

# Now cell_data has neighbor stats ready for Random Forest prediction
# Predict using existing model
predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

### **Why This is Faster**
- **Single adjacency table:** Built once for 344k cells instead of 6.46M rows.
- **Vectorized joins:** Uses `data.table` for efficient joins and aggregation.
- **Year-wise grouping:** Avoids per-row loops; aggregates in bulk.
- **Memory efficiency:** Handles 6.46M rows in chunks via joins, not nested loops.

**Expected runtime:** From 86+ hours down to a few hours (or less) on a 16 GB RAM laptop.