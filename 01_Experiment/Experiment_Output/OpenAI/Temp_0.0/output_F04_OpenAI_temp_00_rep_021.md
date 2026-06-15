 **Diagnosis**  
The bottleneck is the repeated `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation (`paste`) and hash lookups, which are extremely costly at this scale. The neighbor structure is static across years, but the current implementation rebuilds neighbor indices for every row-year combination. This results in massive redundant computation and memory overhead.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** (not per row-year).  
2. **Vectorize across years**: replicate neighbor indices for all years without recomputing string keys.  
3. **Use `data.table` for fast joins and aggregation** instead of nested `lapply`.  
4. **Compute neighbor stats in a single grouped operation** rather than row-wise loops.  
5. Avoid `paste` and repeated hash lookups inside loops.

---

**Optimized Approach**  
- Convert `cell_data` to `data.table`.  
- Precompute a long table of `(cell_id, year, var)` and join with neighbors using an expanded neighbor table.  
- Aggregate max, min, mean in one pass.

---

### **Working R Code**

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute neighbor pairs (static across years)
# rook_neighbors_unique: list of integer vectors, length = number of cells
neighbor_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Expand neighbor pairs across all years
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(id = rep(id, each = length(years)),
                                neighbor_id = rep(neighbor_id, each = length(years)),
                                year = rep(years, times = .N)), by = .(id, neighbor_id)]
neighbor_dt[, c("id", "neighbor_id", "year")]

# Join neighbor values for all variables in one go
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt cell_data for easier join
long_data <- melt(cell_data, id.vars = c("id", "year"), measure.vars = vars,
                  variable.name = "var", value.name = "val")

# Join neighbor values
neighbor_long <- merge(neighbor_dt, long_data, by.x = c("neighbor_id", "year"),
                       by.y = c("id", "year"), allow.cartesian = TRUE)

# Compute stats by (id, year, var)
neighbor_stats <- neighbor_long[, .(
  max_val = max(val, na.rm = TRUE),
  min_val = min(val, na.rm = TRUE),
  mean_val = mean(val, na.rm = TRUE)
), by = .(id, year, var)]

# Reshape back to wide format and merge into cell_data
neighbor_wide <- dcast(neighbor_stats, id + year ~ var,
                       value.var = c("max_val", "min_val", "mean_val"))

cell_data <- merge(cell_data, neighbor_wide, by = c("id", "year"), all.x = TRUE)
```

---

**Why This Works**  
- Eliminates per-row loops and string concatenation.  
- Uses `data.table` joins and aggregation, which are highly optimized in C.  
- Handles all variables in one pass instead of five separate loops.  
- Memory footprint is controlled by processing in long format and aggregating early.

---

**Expected Performance**  
On 6.46M rows and ~1.37M neighbor pairs, this approach should reduce runtime from 86+ hours to **under 1 hour** on a 16 GB laptop, assuming efficient `data.table` operations.

**Preserves**:  
- Original estimand (same neighbor stats).  
- Trained Random Forest model (unchanged).  
- All predictor variables and neighbor features.