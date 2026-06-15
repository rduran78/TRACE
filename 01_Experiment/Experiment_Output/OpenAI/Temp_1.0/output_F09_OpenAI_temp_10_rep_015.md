 **Diagnosis**  
The current pipeline recalculates neighbor relationships for all 6.46M rows in every step. This is extremely inefficient because spatial adjacency is static across years. With 28 yearly snapshots, you're redundantly generating and mapping neighbors 28 times inside `compute_and_add_neighbor_features`. Additionally, repeated `lapply` and `do.call(rbind, ...)` over millions of rows amplifies overhead.

**Optimization Strategy**  
1. **Precompute a reusable neighbor lookup for all cell-years:**  
   Expand static rook neighbors across years once.  
2. **Detach neighbor linkage from variable computation:**  
   Use the same lookup for all 5 neighbor source variables.  
3. **Vectorize / data.table operations:**  
   Avoid repeated full-looping R structures; apply joins for max/min/mean per year.  
4. **Apply incremental joins:**  
   Join yearly subsets against precomputed long-form neighbor pairs to compute summary stats efficiently.  

---

### **Optimized R Code**

```r
library(data.table)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: precomputed spdep::nb object
# id_order: vector of cell IDs in same order as rook_neighbors_unique

setDT(cell_data)

# ---- 1. Build reusable adjacency table (cell-year to neighbor-year) ----
build_adjacency_table <- function(id_order, neighbors, years) {
  adj_list <- lapply(seq_along(id_order), function(i) {
    if (length(neighbors[[i]]) == 0) return(NULL)
    data.table(
      id      = id_order[i],
      neigh_id = id_order[neighbors[[i]]]
    )
  })
  adj_dt <- rbindlist(adj_list)
  
  # Cross join with all years
  adj_dt <- adj_dt[, .(id = rep(id, each = length(years)),
                       year = years,
                       neigh_id = rep(neigh_id, each = length(years)))]
  adj_dt
}

years <- sort(unique(cell_data$year))
adj_dt <- build_adjacency_table(id_order, rook_neighbors_unique, years)

# ---- 2. For each neighbor variable, compute aggregated stats efficiently ----
compute_neighbor_stats_dt <- function(cell_data, adj_dt, var) {
  # Extract variable and prepare neighbor dataset
  var_dt <- cell_data[, .(neigh_id = id, year, value = get(var))]
  
  # Join adjacency with neighbor values
  joined <- adj_dt[var_dt, on = .(neigh_id, year)]
  
  # Aggregate max, min, mean by id-year
  stats <- joined[, .(
    paste0(var, "_nmax") := max(value, na.rm = TRUE),
    paste0(var, "_nmin") := min(value, na.rm = TRUE),
    paste0(var, "_nmean") := mean(value, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Handle empty neighbor cases (NaN -> NA)
  for (col in names(stats)[-(1:2)]) {
    set(stats, i = which(is.nan(stats[[col]])), j = col, value = NA_real_)
  }
  
  stats
}

# ---- 3. Loop through the 5 variables and merge results ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  stats <- compute_neighbor_stats_dt(cell_data, adj_dt, var)
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}

# cell_data now has neighbor max/min/mean for each source variable
```

---

**Performance Gains**
- The heavy, repeated neighbor index lookup is eliminated.
- Adjacency table built once (≈1.3M pairs × 28 years ≈ 38M rows) processed with `data.table` joins.
- Expected runtime reduction from 86+ hours to a few hours or less on a 16 GB laptop (likely 10–20× faster).

**Preserves:**  
- Original estimand (neighbor-based features computed on same panel).
- Trained Random Forest model remains unchanged.