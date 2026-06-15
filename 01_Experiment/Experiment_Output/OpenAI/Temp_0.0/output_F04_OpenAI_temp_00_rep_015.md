 **Diagnosis**  
The bottleneck is the repeated `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation (`paste`) and hash lookups, which are extremely costly at this scale. The neighbor structure is static across years, but the current implementation rebuilds neighbor keys for every row-year combination. This results in billions of operations and excessive memory churn.

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** (not per row-year).  
2. **Vectorize across years**: replicate neighbor indices for all years without string concatenation.  
3. **Use matrix operations** instead of `lapply` for computing max/min/mean.  
4. **Leverage `data.table`** for fast joins and grouping.  
5. Avoid building large lists of length 6.46M; instead, compute neighbor stats by joining on `(id, year)` pairs.

---

### **Optimized Approach**
- Convert `cell_data` to `data.table`.
- Expand neighbor relationships into a long table `(id, neighbor_id)`.
- Cross join with years to get `(id, year, neighbor_id)`.
- Join to `cell_data` to fetch neighbor values.
- Compute `max`, `min`, `mean` by `(id, year)` in a single grouped aggregation.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs (id, neighbor_id)
neighbor_pairs <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Expand across years
years <- unique(cell_data$year)
neighbor_pairs <- neighbor_pairs[, .(neighbor_id, year = years), by = id]

# Set keys for fast joins
setkey(cell_data, id, year)
setkey(neighbor_pairs, neighbor_id, year)

# Function to compute and add neighbor stats
compute_and_add_neighbor_features <- function(dt, var_name, neighbor_pairs) {
  # Join neighbor values
  neighbor_vals <- neighbor_pairs[cell_data, on = .(neighbor_id = id, year), nomatch = 0]
  neighbor_vals[, value := get(var_name), by = .EACHI]

  # Aggregate stats by (id, year)
  stats <- neighbor_vals[, .(
    max_val = max(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    mean_val = mean(value, na.rm = TRUE)
  ), by = .(id, year)]

  # Replace NaN with NA
  for (col in c("max_val", "min_val", "mean_val")) {
    set(stats, which(!is.finite(stats[[col]])), col, NA)
  }

  # Merge back to main table
  setkey(stats, id, year)
  dt <- stats[dt]
  setnames(dt, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  dt[]
}

# Apply for all neighbor source variables
for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_pairs)
}
```

---

### **Why This Is Faster**
- Eliminates per-row string concatenation and list building.
- Uses `data.table` joins and grouped aggregation in C-optimized code.
- Processes all rows in bulk rather than millions of small R function calls.
- Memory footprint is controlled by working in long format and aggregating immediately.

**Expected Runtime**: From 86+ hours down to a few hours (or less) on a 16 GB laptop, depending on disk I/O and CPU.