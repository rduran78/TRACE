 **Diagnosis**  
The bottleneck arises because `build_neighbor_lookup()` creates a massive list of neighbor indices recomputed per cell-year (6.46M times), performing costly string concatenations and list indexing in R. Since neighbors are static across years, repeating this lookup for every year is unnecessary. Additionally, `compute_neighbor_stats()` uses nested `lapply` calls for all rows, adding to inefficiency.

**Optimization Strategy**  
1. Precompute a **neighbor mapping by cell (not cell-year)** once.  
2. Use **vectorized joins** or `data.table` aggregation instead of dense nested loops.  
3. Compute neighbor stats year by year using efficient keyed joins rather than recomputing character keys.  
4. Avoid repeated string operations and conversions.  
5. Stream or chunk by year to fit memory constraints.

---

### **Optimized R Code using `data.table`**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2

# Precompute neighbor lookup as a list indexed by id
neighbor_lookup <- rook_neighbors_unique  # list: cell_id -> vector of neighbor ids
cell_ids <- id_order
neighbor_dt <- rbindlist(lapply(seq_along(neighbor_lookup), function(i) {
  if (length(neighbor_lookup[[i]]) == 0) return(NULL)
  data.table(id = cell_ids[i], neighbor_id = cell_ids[neighbor_lookup[[i]]])
}))

# Key for fast joins
setkey(cell_data, id, year)
setkey(neighbor_dt, id)

# Expand neighbors with year
# For each id-year, attach all neighbor_ids
expand_neighbors <- function(year_slice) {
  y <- year_slice$year[1]
  merged <- neighbor_dt[year_slice, allow.cartesian = TRUE]
  merged[, year := y]
  merged
}

# Process year by year to control memory
years <- unique(cell_data$year)
var_list <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (yr in years) {
  cat("Processing year:", yr, "\n")
  year_slice <- cell_data[year == yr, .(id, year)]
  
  neighbor_expanded <- expand_neighbors(year_slice)
  
  # Join to get neighbor values
  neighbor_expanded <- neighbor_expanded[cell_data, on = .(neighbor_id = id, year), 
                                         nomatch = 0L]
  
  # Compute stats per id for each var
  for (var_name in var_list) {
    agg <- neighbor_expanded[, .(
      max_val = max(get(var_name), na.rm = TRUE),
      min_val = min(get(var_name), na.rm = TRUE),
      mean_val = mean(get(var_name), na.rm = TRUE)
    ), by = .(id, year)]
    
    # Handle all NA case
    agg[!is.finite(max_val), `:=`(max_val = NA, min_val = NA, mean_val = NA)]
    
    # Merge back
    setnames(agg, c("id", "year", paste0(var_name, "_nbr_max"),
                            paste0(var_name, "_nbr_min"),
                            paste0(var_name, "_nbr_mean")))
    cell_data[agg, on = .(id, year)] <- agg[, -c("id", "year")]
  }
}
```

---

### **Key Improvements**
- Neighbor relationships computed once (1.37M pairs) instead of 6.46M times.
- Year-by-year join avoids full 6.5M × neighbor expansions in memory.
- Uses `data.table` joins and aggregations (highly optimized in C).
- Preserves model and numerical estimand.

**Expected runtime reduction**: From 86+ hours to a few hours on 16 GB RAM laptop.  
**Further gains**: Parallelize year-level loop (e.g., `future.apply`) or precompute neighbor expansions once and cache on disk if disk I/O is fast.