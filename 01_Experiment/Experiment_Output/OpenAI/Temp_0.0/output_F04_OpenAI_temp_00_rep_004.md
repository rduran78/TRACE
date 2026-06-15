 **Diagnosis**  
The bottleneck is the repeated `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation (`paste`) and hash lookups, which are extremely costly at this scale. The neighbor structure is static across years, but the current implementation rebuilds neighbor indices for every row-year combination. This results in massive redundant computation and memory overhead.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** (not per cell-year).  
2. **Vectorize across years**: replicate neighbor indices for all years without recomputing string keys.  
3. **Use `data.table` for fast joins and aggregation** instead of nested `lapply`.  
4. **Compute neighbor stats in a single grouped operation** rather than row-wise loops.  
5. Avoid repeated `do.call(rbind, ...)` and unnecessary object growth.

---

**Optimized Approach**  
- Convert `cell_data` to `data.table`.  
- Precompute a long table of `(cell_id, year, var_name, value)` and join with neighbors using an expanded neighbor table.  
- Aggregate max, min, mean in one pass.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs (cell_id -> neighbor_id)
# rook_neighbors_unique: list of integer vectors, length = number of cells
neighbor_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Expand neighbor pairs across all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id = rep(id, each = length(years)),
                                neighbor_id = rep(neighbor_id, each = length(years)),
                                year = rep(years, times = .N))]

# Set keys for fast joins
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)

# Function to compute and add neighbor stats for one variable
compute_neighbor_features_dt <- function(var_name) {
  # Join neighbor values
  joined <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]
  # joined now has: id (from neighbor_dt), neighbor_id, year, and var_name from cell_data
  setnames(joined, var_name, "neighbor_val")
  
  # Aggregate stats by (id, year)
  agg <- joined[, .(
    max_val = max(neighbor_val, na.rm = TRUE),
    min_val = min(neighbor_val, na.rm = TRUE),
    mean_val = mean(neighbor_val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Merge back to cell_data
  setnames(agg, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data[agg, on = .(id, year), (names(agg)[3:5]) := mget(names(agg)[3:5])]
}

# Apply for all neighbor source variables
for (var_name in neighbor_source_vars) {
  compute_neighbor_features_dt(var_name)
}
```

---

**Why This Is Faster**  
- Eliminates per-row string operations and repeated lookups.  
- Uses `data.table` joins and grouped aggregation in compiled C code.  
- Processes all rows in bulk rather than millions of R-level loops.  

**Expected Performance**  
On 6.46M rows and ~1.37M neighbor links, this approach should reduce runtime from 86+ hours to a few hours (or less) on a 16 GB laptop, depending on disk I/O and CPU speed, without changing the Random Forest model or the numerical estimand.