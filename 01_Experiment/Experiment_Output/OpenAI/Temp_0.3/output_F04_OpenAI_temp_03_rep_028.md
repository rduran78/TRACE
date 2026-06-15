 **Diagnosis**  
The main bottleneck is the repeated use of `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. These functions repeatedly perform string concatenation, named lookups, and list indexing in R’s interpreter, which is extremely slow at this scale. The neighbor relationships are static across years, but the current code rebuilds neighbor indices for every row-year combination, multiplying work unnecessarily.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** (not per row-year) and reuse them across all years.
2. **Vectorize computations**: Instead of looping through 6.46M rows, use `data.table` or `dplyr` joins and group operations.
3. **Avoid string concatenation for keys**: Use integer indices for fast lookups.
4. **Compute neighbor stats in a single grouped operation**: Reshape data to long format, join with neighbor mapping, and aggregate.
5. **Parallelize if possible**: Use `data.table` for efficient in-memory operations on 16 GB RAM.

---

**Optimized R Code (data.table approach)**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor mapping: cell_id -> neighbor_id
neighbor_dt <- data.table(
  cell_id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Melt cell_data to long format for neighbor vars
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
long_dt <- melt(cell_data, id.vars = c("id", "year"), measure.vars = vars,
                variable.name = "var", value.name = "val")

# Join neighbors: replicate rows for each neighbor
long_dt <- neighbor_dt[long_dt, on = .(cell_id = id), allow.cartesian = TRUE]

# Bring in neighbor values
setkey(long_dt, neighbor_id, year, var)
long_dt[, neighbor_val := long_dt[.SD, on = .(id = neighbor_id, year, var), x.val]]

# Compute stats by cell-year-var
agg_dt <- long_dt[!is.na(neighbor_val),
                  .(max_val = max(neighbor_val),
                    min_val = min(neighbor_val),
                    mean_val = mean(neighbor_val)),
                  by = .(cell_id, year, var)]

# Reshape back to wide and merge into original
agg_wide <- dcast(agg_dt, cell_id + year ~ var, value.var = c("max_val", "min_val", "mean_val"))
setnames(agg_wide, old = "cell_id", new = "id")

cell_data <- merge(cell_data, agg_wide, by = c("id", "year"), all.x = TRUE)
```

---

**Why This Works**  
- **One neighbor join instead of millions of lookups**: We leverage `data.table` joins and grouping, which are implemented in C and highly optimized.
- **No repeated string concatenation**: Integer joins replace expensive key construction.
- **Single pass aggregation**: All neighbor stats for all variables computed in one grouped operation.
- **Memory efficiency**: 16 GB RAM can handle this with `data.table` if processed in chunks if needed.

**Expected Speedup**: From 86+ hours to under 1 hour on a standard laptop (with possible further gains using `setDTthreads()` for parallelism).