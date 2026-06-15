 **Diagnosis**  
The current approach is extremely slow and memory-intensive because:  
- `build_neighbor_lookup` creates a large list (length = 6.46M rows) of integer vectors, which is massive in memory.  
- `compute_neighbor_stats` repeatedly traverses these lists and subsets vectors, causing heavy R-level overhead.  
- The outer loop repeats this process for each variable, multiplying cost by 5.  
- Everything is done in pure R with `lapply` and `do.call(rbind, ...)`, which is inefficient for millions of rows.  

**Optimization Strategy**  
1. **Avoid building a huge list:** Instead of creating a per-row neighbor lookup, compute neighbor statistics by joining data on `(id, year)` pairs using `data.table`.  
2. **Vectorize and batch operations:** Use `data.table` grouping to compute max, min, mean for all neighbors in one pass per variable.  
3. **Exploit panel structure:** Expand neighbor relationships across years once, then join with cell-year data.  
4. **Memory efficiency:** Work with integer keys and `data.table` joins instead of large nested lists.  

**Optimized Approach**  
- Convert `cell_data` to `data.table`.  
- Create a long table of `(id, neighbor_id)` pairs from `rook_neighbors_unique`.  
- Cross-join with years to get `(id, neighbor_id, year)`.  
- Join neighbor values for each variable and compute summary stats by `(id, year)`.  

---

### **Working R Code**

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Convert rook_neighbors_unique (list) to data.table of edges
edges <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Expand edges across all years
years <- unique(cell_data$year)
edges_expanded <- edges[, .(neighbor_id, year = years), by = id]

# Set keys for fast joins
setkey(cell_data, id, year)
setkey(edges_expanded, neighbor_id, year)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_dt <- function(var_name) {
  # Join neighbor values
  joined <- cell_data[edges_expanded, .(id, year, neighbor_val = get(var_name))]
  
  # Compute stats by id-year
  stats <- joined[, .(
    max = if (.N > 0) max(neighbor_val, na.rm = TRUE) else NA_real_,
    min = if (.N > 0) min(neighbor_val, na.rm = TRUE) else NA_real_,
    mean = if (.N > 0) mean(neighbor_val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Rename columns
  setnames(stats, c("id", "year", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  stats
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute and merge all stats
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_dt(var_name)
  cell_data <- stats[cell_data, on = .(id, year)]
}

# cell_data now includes neighbor features
```

---

**Why this is faster and feasible on 16 GB RAM:**  
- Avoids creating a 6.46M-length list of vectors.  
- Uses `data.table` joins and grouping, which are highly optimized in C.  
- Processes all years in one batch rather than looping row-by-row.  
- Memory footprint is manageable because we only store expanded edges (~1.37M × 28 ≈ 38M rows) temporarily, which is feasible in `data.table` with efficient column storage.  

**Expected runtime improvement:** Hours → Minutes (depending on disk speed and CPU).  

This preserves the trained Random Forest model and original numerical estimand.