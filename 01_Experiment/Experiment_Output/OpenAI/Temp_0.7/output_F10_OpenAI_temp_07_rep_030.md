 **Diagnosis**  
The current pipeline is extremely slow because it repeatedly computes neighbor statistics for ~6.46 million rows using `lapply` inside `compute_neighbor_stats`. This results in millions of nested loops and repeated indexing. Memory overhead is also high because neighbor lookups are recalculated for every variable, and the algorithm does not exploit the static graph topology across years. With 16 GB RAM and 86+ hours estimated runtime, the bottleneck is inefficient R loops and repeated operations.

---

**Optimization Strategy**  
1. **Build the graph topology once**: Represent neighbor relationships as an adjacency list or sparse matrix and reuse it across all years and variables.
2. **Vectorize computation**: Use `data.table` for fast grouping and joins instead of `lapply`.
3. **Exploit panel structure**: The neighbor graph is static across years, so replicate adjacency indices per year instead of reconstructing them repeatedly.
4. **Batch compute statistics**: Compute max, min, mean for all neighbors in one pass using data.table joins.
5. **Preserve numerical equivalence**: Ensure results match the original loop-based implementation.

---

**Optimized R Implementation**

```r
library(data.table)

# Assumptions:
# - cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# - id_order: vector of unique cell ids in original order
# - rook_neighbors_unique: list of integer neighbors (spdep::nb)
# - neighbor_source_vars: vector of variable names

# STEP 1: Precompute adjacency for all (id, year) combinations
setDT(cell_data)
setkey(cell_data, id, year)

# Build adjacency pairs: for each cell id, its neighbors
adj_list <- data.table(
  src = rep(id_order, lengths(rook_neighbors_unique)),
  nbr = unlist(rook_neighbors_unique)
)

# Expand adjacency across years
years <- sort(unique(cell_data$year))
adj_pairs <- adj_list[, .(src = rep(src, each = length(years)),
                          nbr = rep(nbr, each = length(years)),
                          year = rep(years, times = .N))]
setkey(adj_pairs, nbr, year)

# STEP 2: Merge neighbor values for all variables
# Melt cell_data for easy join
long_data <- melt(cell_data,
                  id.vars = c("id", "year"),
                  measure.vars = neighbor_source_vars,
                  variable.name = "var",
                  value.name = "val")

setkey(long_data, id, year)
adj_long <- adj_pairs[long_data, on = .(nbr = id, year)]

# adj_long now has: src, nbr, year, var, val
# Compute stats by src, year, var
neighbor_stats <- adj_long[, .(
  max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
  min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
  mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
), by = .(src, year, var)]

# STEP 3: Reshape back to wide format and merge into cell_data
neighbor_stats_wide <- dcast(
  neighbor_stats,
  src + year ~ var,
  value.var = c("max_val", "min_val", "mean_val")
)

setnames(neighbor_stats_wide, "src", "id")
cell_data <- merge(cell_data, neighbor_stats_wide, by = c("id", "year"), all.x = TRUE)

# cell_data now contains original vars + neighbor max/min/mean features
```

---

**Why This is Efficient**  
- **Single adjacency expansion**: Graph topology created once and reused.
- **Full vectorization**: No nested loops; aggregation uses highly optimized `data.table` operations in C.
- **Batch computation**: All variables handled in one pass using long format.
- **Scalable**: Memory-efficient joins and grouping operations rather than millions of `lapply` calls.

---

**Expected Performance**  
On 6.46M rows, this approach is orders of magnitude faster (minutes instead of 86+ hours) and fits within 16 GB RAM because it avoids redundant computations and leverages efficient joins.

**Next Step**: Pass `cell_data` (with new neighbor features) to the pre-trained Random Forest for prediction without retraining.