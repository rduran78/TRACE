 **Diagnosis**  
The bottleneck is the nested `lapply` approach over ~6.46M rows. For each row, `compute_neighbor_stats` iterates through neighbors, repeatedly subsetting vectors and recomputing stats. This is extremely inefficient in R due to high interpretation overhead and memory churn. The complexity is roughly *O(N × avg_neighbors)* with heavy R function call overhead. With millions of rows and multiple variables, 86+ hours is expected.

**Optimization Strategy**  
1. **Precompute neighbor lookup once** as integer indices (already done).
2. **Vectorize aggregation**:
   - Flatten neighbor relationships into a long table: `(source_row, neighbor_row)`.
   - Join neighbor values in vectorized form.
   - Use `data.table` or `collapse` for grouped aggregation (`max`, `min`, `mean`).
3. **Compute all neighbor variables in one pass** if possible.
4. Avoid loops in R for each row; rely on efficient grouping.

This keeps the same estimand (max, min, mean per cell-year among rook neighbors) but runs in minutes instead of days.

---

### **Working R Code**

```r
library(data.table)

# Assume: cell_data (data.frame) has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# neighbor_lookup: list of integer vectors (neighbors by row index)

# Convert to data.table for speed
setDT(cell_data)
cell_data[, row_id := .I]  # unique row index

# Build long table of (row_id, neighbor_id)
edge_list <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(row_id = i, neighbor_id = neighbor_lookup[[i]])
  })
)

# Merge neighbor values for all variables in one go
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Bring neighbor values
edge_list <- merge(edge_list,
                   cell_data[, c("row_id", neighbor_vars), with = FALSE],
                   by.x = "neighbor_id", by.y = "row_id",
                   all.x = TRUE)

# Compute stats grouped by row_id
agg <- edge_list[, .(
  ntl_max = max(ntl, na.rm = TRUE),
  ntl_min = min(ntl, na.rm = TRUE),
  ntl_mean = mean(ntl, na.rm = TRUE),
  ec_max = max(ec, na.rm = TRUE),
  ec_min = min(ec, na.rm = TRUE),
  ec_mean = mean(ec, na.rm = TRUE),
  pop_density_max = max(pop_density, na.rm = TRUE),
  pop_density_min = min(pop_density, na.rm = TRUE),
  pop_density_mean = mean(pop_density, na.rm = TRUE),
  def_max = max(def, na.rm = TRUE),
  def_min = min(def, na.rm = TRUE),
  def_mean = mean(def, na.rm = TRUE),
  usd_est_n2_max = max(usd_est_n2, na.rm = TRUE),
  usd_est_n2_min = min(usd_est_n2, na.rm = TRUE),
  usd_est_n2_mean = mean(usd_est_n2, na.rm = TRUE)
), by = row_id]

# Replace NaN with NA
for (j in names(agg)) set(agg, which(!is.finite(agg[[j]])), j, NA)

# Merge back into cell_data
cell_data <- merge(cell_data, agg, by = "row_id", all.x = TRUE)

# Drop helper column
cell_data[, row_id := NULL]
```

---

**Why This Is Fast**  
- Single `data.table` join and grouped aggregation.
- Eliminates millions of small R function calls.
- Processes all variables in one grouped aggregation step.
- Uses efficient C-level loops internally.

**Expected Runtime**  
On 6.46M rows and ~1.37M neighbor edges, this method should run in a few minutes on a 16GB laptop instead of 86+ hours. Memory footprint is manageable because the edge list (~1.37M rows) is far smaller than the full panel size.

**Preserves Original Estimand**  
- Same neighbors, same year alignment.
- Same max, min, mean logic.
- No retraining of the Random Forest model required.

**Next Step**: Feed updated `cell_data` into the existing Random Forest prediction pipeline.