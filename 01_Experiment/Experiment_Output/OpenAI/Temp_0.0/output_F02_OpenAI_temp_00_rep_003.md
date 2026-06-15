 **Diagnosis**  
The current implementation is extremely slow and memory-intensive because:  
- `build_neighbor_lookup` creates a large list of length 6.46M, each element being a vector of neighbor indices. This is expensive in both memory and time.  
- `compute_neighbor_stats` repeatedly traverses large lists and performs many small operations in R loops, which are inherently slow.  
- The entire process is done in pure R with `lapply` over millions of rows, which is not scalable on a laptop.  

**Optimization Strategy**  
1. **Avoid building a massive neighbor list for every row**. Instead, work with a long-format edge list (cell-year → neighbor-year) and join it to the data.  
2. **Vectorize aggregation** using `data.table` or `dplyr` rather than looping.  
3. **Process by year** to reduce memory footprint (28 chunks instead of all at once).  
4. **Precompute neighbor relationships at the cell level** and then expand by year.  
5. **Compute max, min, mean in one grouped aggregation** rather than per-row loops.  

This approach reduces complexity from O(N × neighbors) in R loops to efficient grouped operations in C-backed data.table.  

---

### **Optimized R Code (data.table approach)**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor edge list at cell level
# rook_neighbors_unique: list of integer vectors (spdep nb object)
edges <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Process year by year to keep memory manageable
years <- sort(unique(cell_data$year))

# Function to compute neighbor stats for one year
compute_year_stats <- function(yr, data, edges, vars) {
  dt <- data[year == yr, .(id, year, (vars)), with = FALSE]
  setkey(dt, id)

  # Join edges to get neighbor values
  edge_dt <- edges[, .(id = from, neighbor = to)]
  edge_dt <- edge_dt[dt, on = .(id), nomatch = 0]
  edge_dt <- edge_dt[dt, on = .(neighbor = id), nomatch = 0, 
                     allow.cartesian = TRUE]

  # edge_dt now has: id, neighbor, year, vars...
  # Compute stats by id
  res_list <- lapply(vars, function(v) {
    edge_dt[, .(
      max = max(get(v), na.rm = TRUE),
      min = min(get(v), na.rm = TRUE),
      mean = mean(get(v), na.rm = TRUE)
    ), by = id]
  })

  # Merge all results
  out <- Reduce(function(x, y) merge(x, y, by = "id"), res_list)
  out[, year := yr]
  out
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute in chunks
result_list <- lapply(years, compute_year_stats, data = cell_data, 
                      edges = edges, vars = neighbor_source_vars)

neighbor_stats <- rbindlist(result_list)

# Merge back to main data
cell_data <- merge(cell_data, neighbor_stats, by = c("id", "year"), all.x = TRUE)
```

---

**Why this works better:**  
- No giant list of 6.46M elements.  
- Uses `data.table` joins and grouped aggregation in C for speed.  
- Processes one year at a time to keep memory under control.  
- Preserves original estimand and trained model (only feature engineering changes).  

**Expected performance:**  
- From 86+ hours to a few hours (or less) on a 16 GB laptop.  
- Memory footprint reduced drastically because we never materialize a full neighbor lookup list.  

**Further improvements:**  
- Parallelize `lapply(years, ...)` with `future.apply` or `parallel::mclapply`.  
- If still too large, split by year and variable separately.  
- Consider writing intermediate results to disk if RAM is tight.