 **Diagnosis**  
The bottleneck is the repeated neighbor feature computation across ~6.46M rows. The current implementation uses `lapply` with repeated string concatenation and list indexing, which is highly inefficient in R for large datasets. The neighbor lookup is rebuilt once, but `compute_neighbor_stats` iterates row-wise for each variable, causing redundant work. Memory overhead from lists and repeated `do.call(rbind, ...)` also adds cost.

**Optimization Strategy**  
- Precompute neighbor indices as integer vectors (avoid string keys).
- Use `data.table` for fast grouping and joins.
- Vectorize neighbor statistics computation by flattening neighbor relationships into a long table and aggregating.
- Compute all neighbor stats in one pass rather than looping per variable.
- Avoid repeated NA filtering inside tight loops; handle NA during aggregation.

**Optimized R Code**

```r
library(data.table)

# Convert data to data.table for efficiency
setDT(cell_data)

# Precompute neighbor relationships as a long table
# id_order: vector of cell IDs in reference order
# rook_neighbors_unique: list of neighbor indices
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand to panel by joining on years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id = rep(id, length(years)),
                                neighbor_id = rep(neighbor_id, length(years)),
                                year = rep(years, each = .N)), by = .(id, neighbor_id)]

# Join with cell_data to get neighbor values for all source vars
# Melt cell_data for easier aggregation
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
long_data <- melt(cell_data, id.vars = c("id", "year"), measure.vars = vars,
                  variable.name = "var_name", value.name = "value")

# Join neighbor_dt with long_data on neighbor_id/year
neighbor_values <- merge(neighbor_dt, long_data,
                         by.x = c("neighbor_id", "year"),
                         by.y = c("id", "year"),
                         allow.cartesian = TRUE)

# Aggregate neighbor stats: max, min, mean per id-year-var
agg_stats <- neighbor_values[, .(
  neigh_max = max(value, na.rm = TRUE),
  neigh_min = min(value, na.rm = TRUE),
  neigh_mean = mean(value, na.rm = TRUE)
), by = .(id, year, var_name)]

# Reshape back to wide format
agg_wide <- dcast(agg_stats, id + year ~ var_name,
                  value.var = c("neigh_max", "neigh_min", "neigh_mean"))

# Merge with original cell_data
cell_data <- merge(cell_data, agg_wide, by = c("id", "year"), all.x = TRUE)
```

**Why This Works**  
- Eliminates per-row loops; uses efficient joins and aggregation.
- Computes all neighbor stats in one pass.
- Scales better for millions of rows using `data.table`.
- Preserves original estimand and Random Forest model (no retraining).

**Expected Performance**  
On a 16 GB laptop, this approach should reduce runtime from 86+ hours to a few hours or less, depending on disk I/O and CPU.