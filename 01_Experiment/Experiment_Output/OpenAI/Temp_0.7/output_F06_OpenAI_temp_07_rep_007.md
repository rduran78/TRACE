 **Diagnosis**  
The bottleneck is the `lapply` over ~6.46M rows in `compute_neighbor_stats`, which repeatedly subsets large vectors and performs aggregation. This results in massive R-level overhead and memory churn. The nested loops (outer loop × 5 variables) exacerbate the problem. With 344k cells × 28 years and ~1.37M neighbor links, the naive approach becomes prohibitively slow.

**Optimization Strategy**  
- **Precompute neighbor indices once** and reuse them (already done).
- **Vectorize aggregation**: Instead of looping over rows, convert neighbor relationships into a long edge list and use `data.table` or `dplyr` group operations to compute `max`, `min`, and `mean` in bulk.
- **Leverage fast joins**: Use `data.table` keyed joins for speed and memory efficiency.
- Avoid raster focal analogy since we must exactly preserve neighbor relationships across panel years, which raster kernels can't guarantee for irregular IDs.

**Optimized Approach**  
1. Expand neighbor relationships into `(cell_id, neighbor_id)` pairs for *each year*.
2. Join this long edge table with the value column for each neighbor variable.
3. Aggregate per `(cell_id, year)` using `data.table::frollapply`-like grouping to compute max, min, mean.
4. Merge results back to `cell_data`.

---

### **Working R Code**

```r
library(data.table)

# Convert cell_data to data.table for speed
setDT(cell_data)

# Build edge list of neighbors for all years
# id_order: vector of cell IDs in reference order
# rook_neighbors_unique: list of neighbor indices

edge_list <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    src = id_order[i],
    nbr = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand edge list to all years
years <- sort(unique(cell_data$year))
edge_dt <- edge_list[, .(year = years), by = .(src, nbr)]

# Set keys for fast joins
setkey(cell_data, id, year)
setkey(edge_dt, nbr, year)

compute_neighbor_stats_dt <- function(var_name) {
  # Join neighbor values
  joined <- edge_dt[cell_data, on = .(nbr = id, year), nomatch = 0]
  setnames(joined, var_name, "nbr_val")

  # Drop NAs
  joined <- joined[!is.na(nbr_val)]

  # Aggregate max, min, mean by src + year
  agg <- joined[, .(
    max_val = max(nbr_val),
    min_val = min(nbr_val),
    mean_val = mean(nbr_val)
  ), by = .(id = src, year)]

  # Merge back to cell_data
  setnames(agg,
           c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  agg
}

# Process all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg_list <- lapply(neighbor_source_vars, compute_neighbor_stats_dt)

# Merge all aggregated features
result_dt <- Reduce(function(x, y) merge(x, y, by = c("id", "year"), all = TRUE),
                    agg_list)

# Join with original data
cell_data <- merge(cell_data, result_dt, by = c("id", "year"), all.x = TRUE)
```

---

**Why This Works Faster**  
- Single pass per variable instead of millions of tiny loops.
- `data.table` uses highly optimized C-level aggregation.
- Memory footprint is reduced by working with a long edge table rather than repeatedly slicing a huge data.frame.

**Expected Performance**  
On a 16 GB laptop, this approach should reduce runtime from 86+ hours to under 1 hour (varies by I/O overhead and CPU threads).

**Preserves**  
- Original Random Forest model.
- Exact neighbor-based statistics (max, min, mean).
- Panel structure across years.