 **Diagnosis:**  
The major bottleneck is `build_neighbor_lookup`, which currently iterates over all ~6.46M cell-year rows, performing string operations (`paste`) and repeated list indexing in R's interpreter. This results in huge overhead. The inner `compute_neighbor_stats` then iterates again. With 344k cells × 28 years and 5 features, this becomes extremely slow. Random Forest inference is not the main issue; neighbor feature construction dominates runtime.

---

**Optimization Strategy:**  
- **Precompute neighbor indices once at cell level, not per row.**
- Use **vectorized operations** and **data.table** join instead of looping with `lapply`.  
- Build a **long table of neighbor relationships expanded over years**, then compute the stats with `data.table` group aggregations, which are highly optimized in C.  
- Avoid repeated string concatenation; use integer keys for join.

---

**Optimized R Code:**

```r
library(data.table)

# Assume cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute static neighbor pairs at cell level
# rook_neighbors_unique: list where rook_neighbors_unique[[i]] are neighbor indices of id_order[i]
id_order <- as.integer(id_order)  # ensure numeric if possible
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id       = id_order[i],
    neighbor = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Cartesian join with years to get all id-year-neighbor-year combos
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor)]

# Melt cell_data for wide to long features
features <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare lookups for fast join
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor, year)

# Join neighbor feature values
neighbor_dt <- neighbor_dt[cell_data, on = .(neighbor = id, year), nomatch = 0L]
# neighbor_dt now has id, neighbor, year, and feature columns from neighbor

# Compute summary stats (max, min, mean) per id-year
out_list <- list()
for (var_name in features) {
  agg <- neighbor_dt[, .(
    max = max(get(var_name), na.rm = TRUE),
    min = min(get(var_name), na.rm = TRUE),
    mean = mean(get(var_name), na.rm = TRUE)
  ), by = .(id, year)]
  # Rename columns
  setnames(agg, c("id","year","max","min","mean"),
           c("id","year",
             paste0(var_name,"_nb_max"),
             paste0(var_name,"_nb_min"),
             paste0(var_name,"_nb_mean")))
  out_list[[var_name]] <- agg
}

# Merge all feature summaries back to cell_data
for (agg in out_list) {
  cell_data <- agg[cell_data, on = .(id, year)]
}

# cell_data now includes neighbor-based features efficiently
```

---

**Why This Works:**  
- Eliminates the 6.46M inner loops; all neighbor expansions and stats are computed in C via `data.table`.
- Memory footprint: neighbor_dt will have roughly (#neighbors × #years) ≈ 1.37M × 28 ≈ 38M rows, but joins and aggregation in `data.table` are efficient and feasible in 16 GB RAM.
- This reduces runtime from **86+ hours to minutes**, depending on I/O overhead.

---

**Preserves:**  
- Trained Random Forest model unchanged.  
- Original numerical estimand: max, min, mean of neighbor variables per cell-year identical to original logic.