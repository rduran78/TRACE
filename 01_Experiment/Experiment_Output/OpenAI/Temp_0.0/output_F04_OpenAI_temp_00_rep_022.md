 **Diagnosis**  
The bottleneck is the repeated `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation (`paste`) and hash lookups, which are extremely costly at this scale. The neighbor structure is static across years, but the current implementation rebuilds neighbor indices for every row-year combination. This results in massive redundant computation and memory overhead.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** (not per row-year).  
2. **Vectorize across years**: replicate neighbor indices for all years without recomputing string keys.  
3. **Use `data.table` for fast joins and aggregation** instead of nested `lapply`.  
4. **Compute neighbor stats in a single grouped operation** rather than row-wise loops.  
5. Avoid `paste` and repeated hash lookups inside loops.

---

**Optimized Approach**  
- Convert `cell_data` to `data.table`.  
- Expand neighbor relationships into a long table of `(cell_id, year, neighbor_id)`.  
- Join to get neighbor values and compute `max`, `min`, `mean` by `(cell_id, year)`.  
- Repeat for each variable efficiently.

---

### **Working R Code**

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute neighbor pairs (static across years)
# rook_neighbors_unique: list of integer vectors, length = number of cells
neighbor_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Cross join with years to create full neighbor mapping
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# Set keys for fast joins
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)

# Function to compute and add neighbor stats for one variable
compute_neighbor_features <- function(var_name) {
  # Join neighbor values
  tmp <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]
  tmp[, neighbor_val := get(var_name)]
  
  # Aggregate stats by (id, year)
  stats <- tmp[, .(
    max_val = max(neighbor_val, na.rm = TRUE),
    min_val = min(neighbor_val, na.rm = TRUE),
    mean_val = mean(neighbor_val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Handle all-NA cases
  stats[!is.finite(max_val), c("max_val", "min_val", "mean_val") := NA]
  
  # Merge back to cell_data
  setkey(stats, id, year)
  cell_data[stats, paste0(var_name, "_nbr_max") := i.max_val]
  cell_data[stats, paste0(var_name, "_nbr_min") := i.min_val]
  cell_data[stats, paste0(var_name, "_nbr_mean") := i.mean_val]
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in neighbor_source_vars) compute_neighbor_features(v)
```

---

**Why This Is Faster**  
- Eliminates per-row string operations and repeated lookups.  
- Uses `data.table` joins and group aggregation in compiled C code.  
- Processes all years in bulk rather than looping over 6.46M rows repeatedly.  

**Expected Performance**  
On a 16 GB laptop, this approach should reduce runtime from 86+ hours to **under 1 hour**, depending on disk I/O and CPU speed, because the heavy lifting is now vectorized and memory-efficient.  

**Preserves**  
- Original estimand (same neighbor stats).  
- Trained Random Forest model (unchanged).  
- Full dataset integrity.