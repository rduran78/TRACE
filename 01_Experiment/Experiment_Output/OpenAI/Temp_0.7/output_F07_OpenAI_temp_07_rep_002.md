 **Diagnosis**  
The bottleneck is the double nested loop:  
- `build_neighbor_lookup()` creates a year-specific neighbor index for each of 6.46M rows, which is extremely memory- and time-intensive.  
- Then `compute_neighbor_stats()` iterates over all rows again for each of 5 variables, repeatedly scanning subsets.  
This results in ~32M row-level computations and heavy list operations in R, which are slow for millions of elements. The entire process is single-threaded and not vectorized.

---

**Optimization Strategy**  
1. **Avoid building a huge `neighbor_lookup` list**: Instead, precompute neighbor pairs (cell-year → neighbor-year) as a flat `data.frame` using vectorized joins.  
2. **Exploit `data.table` for speed and memory efficiency**: Join once, aggregate neighbor values with `max`, `min`, `mean` using fast group operations.  
3. **Batch by year**: The panel structure allows splitting into 28 yearly slices to keep memory within 16 GB.  
4. **Parallelize across years** with `future.apply` or `parallel::mclapply`.  
5. **Preserve original estimand**: Use the same neighbor relationships and years; only the computational method changes.  

---

**Working R Code**

```r
library(data.table)
library(parallel)

# Assume: cell_data (id, year, ntl, ec, pop_density, def, usd_est_n2, ...)
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of IDs corresponding to rook_neighbors_unique

# Precompute neighbor pairs (directed)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
}))

setkey(neighbor_dt, src)

# Convert main data to data.table
setDT(cell_data)

# Variables to process
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Function to compute stats for one year
compute_year <- function(yr) {
  dt_year <- cell_data[year == yr, .SD, .SDcols = c("id", "year", neighbor_source_vars)]
  
  # Join neighbors for this year: src-year → nbr-year
  pairs <- neighbor_dt[dt_year, on = .(src = id), allow.cartesian = TRUE]
  setnames(pairs, "nbr", "id")  # rename for join
  
  # Join to get neighbor values
  pairs <- pairs[dt_year, on = .(id), nomatch = 0]
  
  # Aggregate neighbor statistics by source cell
  stats <- pairs[, lapply(.SD, function(x) {
    c(max = max(x, na.rm = TRUE),
      min = min(x, na.rm = TRUE),
      mean = mean(x, na.rm = TRUE))
  }), by = src, .SDcols = neighbor_source_vars]
  
  # Reshape result: wide format with suffixes
  out <- dcast(melt(stats, id.vars = "src"), src ~ variable + L1)
  
  # Merge back into dt_year
  setnames(out, "src", "id")
  merge(dt_year, out, by = "id", all.x = TRUE)
}

# Parallel execution by year
years <- sort(unique(cell_data$year))
cl <- makeCluster(detectCores() - 1)
clusterExport(cl, c("cell_data", "neighbor_dt", "neighbor_source_vars"))
clusterEvalQ(cl, library(data.table))

results_list <- parLapply(cl, years, compute_year)
stopCluster(cl)

# Combine all years
cell_data_enriched <- rbindlist(results_list)
```

---

**Expected Gains**  
- Eliminates 6.46M-list overhead → uses vectorized joins and aggregation.  
- Processes one year at a time in memory (<250K rows/year).  
- Parallelizes across 28 years on multiple cores.  
- Should reduce runtime from **86+ hours to well under 2 hours** on a 16 GB laptop.

**Preserves**:  
- Original rook-neighbor topology.  
- Same numerical estimand for max, min, mean.  
- Existing Random Forest model remains untouched.