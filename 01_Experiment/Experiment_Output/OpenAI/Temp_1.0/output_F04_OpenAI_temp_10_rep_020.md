 **Diagnosis**  
The primary bottleneck is `build_neighbor_lookup()`. It performs an `lapply` over **6.46 million rows**, creating repeated string concatenations (`paste`) and hash lookups for each cell-year combination. `compute_neighbor_stats()` also calls an inner `lapply` on the same scale. These are highly inefficient for such a large dataset.

Instead of expanding neighbor indices per **cell-year**, precompute a **cell-level neighbor mapping once**, and then use **vectorized aggregation over time**. Avoid constructing keys and repeated list traversals inside nested loops.

---

**Optimization Strategy**  
1. Create a **long-format neighbor table** once at the cell level using the `nb` object.  
2. Expand it over years using joins, not loops.  
3. Use `data.table` for fast aggregation (vectorized) instead of millions of `lapply`s.  
4. Compute summary stats (max, min, mean) using grouped operations.  

This avoids per-row list building and uses efficient join + aggregation.

---

**Optimized R Code**

```r
library(data.table)

# Convert to data.table
cell_data <- as.data.table(cell_data)

# Step 1: Precompute neighbor pairs (directed)
neighbor_pairs <- data.table(
  cell_id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Step 2: Expand over years via join
years <- unique(cell_data$year)
# Cartesian expand neighbor pairs x years
neighbor_pairs_expanded <- neighbor_pairs[CJ(year = years), on = .(dummy = NULL)]
setnames(neighbor_pairs_expanded, c("cell_id", "neighbor_id", "year"))

# Step 3: Prepare lookup tables for cell-year rows
setkey(cell_data, id, year)
setkey(neighbor_pairs_expanded, neighbor_id, year)

# Step 4: Attach neighbor values for each variable and aggregate
compute_neighbor_features_dt <- function(var_name) {
  tmp <- merge(
    neighbor_pairs_expanded,
    cell_data[, .(neighbor_id = id, year, val = get(var_name))],
    by = c("neighbor_id", "year"),
    allow.cartesian = TRUE
  )
  tmp[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(cell_id, year)]
}

# Step 5: Loop over variables and merge back efficiently
for (var_name in neighbor_source_vars) {
  stats_dt <- compute_neighbor_features_dt(var_name)
  setnames(stats_dt, c("cell_id", "year", 
                       paste0(var_name, "_nbr_max"),
                       paste0(var_name, "_nbr_min"),
                       paste0(var_name, "_nbr_mean")))
  cell_data <- merge(cell_data, stats_dt, by.x = c("id", "year"),
                     by.y = c("cell_id", "year"), all.x = TRUE)
}
```

---

**Why It’s Faster**
- Avoids 6.46M repetitive loops and string ops.
- Uses `data.table` parallelized and vectorized aggregation.
- `merge` + grouped `mean()` scales far better than repeated `lapply`.

**Expected Efficiency**
- Drops runtime from 86+ hours to a few hours (or less) on 16GB RAM.
- Still preserves:
  - All years (panel structure).
  - Original neighbor relationships.
  - Original Random Forest estimand (feature definitions unchanged).

This approach transforms the slow per-row neighbor computation into a **join-and-aggregate pipeline**, optimal for large datasets.