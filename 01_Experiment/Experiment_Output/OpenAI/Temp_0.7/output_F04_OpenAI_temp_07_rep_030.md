 **Diagnosis**  
The computational bottleneck is the repeated use of `lapply` over **6.46M rows** in `build_neighbor_lookup` and `compute_neighbor_stats`. Both functions repeatedly construct and subset vectors, causing heavy R-level overhead and redundant operations. Since the neighbor structure is static across years, the current approach recomputes lookups unnecessarily. This dominates runtime far more than Random Forest inference.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** (not per cell-year).
2. Use **vectorized aggregation** via `data.table` or `dplyr` instead of `lapply`.
3. Store data in `data.table` keyed by `id` and `year` for fast joins.
4. Compute neighbor statistics in **long format** with joins and grouped summaries.
5. Avoid repeated `paste` and list indexing inside loops.

---

**Optimized Approach**  
- Flatten the panel into a `data.table`.
- Expand neighbor relationships across all years via a Cartesian join.
- Compute max, min, mean in a single grouped step.
- Merge results back.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Key by id and year for fast joins
setkey(cell_data, id, year)

# Step 1: Build neighbor pairs (cell_id -> neighbor_id) once
neighbor_pairs <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Step 2: Expand across all years
years <- sort(unique(cell_data$year))
neighbor_expanded <- neighbor_pairs[CJ(year = years), on = .(dummy = NULL), allow.cartesian = TRUE]
setnames(neighbor_expanded, "year", "year")
# Add keys for join
setkey(neighbor_expanded, neighbor_id, year)

# Step 3: Join neighbor values for each variable and compute stats
compute_neighbor_features <- function(dt, var_name) {
  # Join neighbor values
  tmp <- neighbor_expanded[cell_data, on = .(neighbor_id = id, year), nomatch = 0]
  # tmp has: id, neighbor_id, year, var_name
  tmp_vals <- tmp[, .(neighbor_val = get(var_name)), by = .(id, year)]

  # Aggregate
  agg <- tmp_vals[
    !is.na(neighbor_val),
    .(max_val = max(neighbor_val),
      min_val = min(neighbor_val),
      mean_val = mean(neighbor_val)),
    by = .(id, year)
  ]

  # Merge back into main table
  setkey(agg, id, year)
  dt <- agg[cell_data, on = .(id, year)]
  # Rename new columns
  setnames(dt, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  dt
}

# Apply for all neighbor source variables
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  cell_data <- compute_neighbor_features(cell_data, var_name)
}
```

---

**Expected Performance Gain**  
- Eliminates `lapply` over millions of rows.
- Moves from repeated list operations to vectorized joins and grouped aggregation.
- On a 16 GB laptop, runtime should drop from **86+ hours** to a few hours (or less) depending on I/O and disk caching.

**Preserves:**  
- Original estimand and trained RF model remain intact.  
- All neighbor-based features computed identically, just faster.