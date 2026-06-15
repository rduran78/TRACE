 **Diagnosis:**  
The main bottleneck is **not** Random Forest inference. Inference with a pre-trained Random Forest on ~6.46M rows and 110 features is relatively fast compared to the upstream neighbor feature construction. The provided code reveals heavy use of **`lapply()` over millions of rows** in `build_neighbor_lookup` and `compute_neighbor_stats`, which performs repeated key lookups, string concatenation (`paste`), and index filtering. This is highly inefficient in R for large datasets and dominates runtime.

**Correct bottleneck:**  
The neighbor-feature computation loop (especially `build_neighbor_lookup` and repeated `lapply` in `compute_neighbor_stats`) is the primary bottleneck, not Random Forest prediction.

---

### **Optimization Strategy**
- Avoid repeated `paste`/string operations and slow R list indexing.
- Vectorize neighbor lookups using **data.table** or matrix-based operations.
- Precompute neighbor indices once and store them in efficient structures.
- Compute neighbor stats in a **vectorized** manner rather than per-row `lapply`.
- Use `data.table` joins to aggregate neighbor values quickly.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
cell_dt <- as.data.table(cell_data)

# Create a unique key for cell-year
cell_dt[, key := paste(id, year, sep = "_")]

# Build neighbor lookup as a long table instead of list-of-lists
# rook_neighbors_unique: list of neighbors per id_order index
neighbor_dt <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      id = id_order[i],
      neighbor_id = id_order[rook_neighbors_unique[[i]]]
    )
  })
)

# Expand across years to match panel
years <- unique(cell_dt$year)
neighbor_dt <- neighbor_dt[, .(id = rep(id, length(years)),
                                neighbor_id = rep(neighbor_id, length(years)),
                                year = rep(years, each = .N))]

# Add cell-year keys for joining
neighbor_dt[, key := paste(neighbor_id, year, sep = "_")]

# Join with cell_dt to get neighbor values
setkey(cell_dt, key)
setkey(neighbor_dt, key)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Get neighbor values for var_name
  neighbor_dt[, val := cell_dt[neighbor_dt, get(var_name)]]
  
  # Compute stats by id-year
  stats_dt <- neighbor_dt[!is.na(val), .(
    neigh_max = max(val),
    neigh_min = min(val),
    neigh_mean = mean(val)
  ), by = .(id, year)]
  
  # Join back to cell_dt
  cell_dt[stats_dt, c(paste0(var_name, "_neigh_max"),
                      paste0(var_name, "_neigh_min"),
                      paste0(var_name, "_neigh_mean")) :=
             .(i.neigh_max, i.neigh_min, i.neigh_mean),
          on = .(id, year)]
}

# Drop helper columns
cell_dt[, key := NULL]
```

---

### **Why this is better**
- Eliminates millions of `lapply` calls and string operations.
- Uses `data.table` joins and group operations which are highly optimized in C.
- Reduces complexity from **hours to minutes** for 6.46M rows on a standard laptop.

**Preserves:**  
- Original Random Forest model (not retrained).
- Original numerical estimand and neighbor-based features.

**Estimated improvement:**  
From **86+ hours → under 1 hour** depending on I/O and memory.