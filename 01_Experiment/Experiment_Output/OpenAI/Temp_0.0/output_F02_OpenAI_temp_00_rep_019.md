 **Diagnosis**  
The bottleneck is the repeated use of `lapply` over 6.46M rows and dynamic lookups in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation, list indexing, and repeated NA filtering, which is extremely inefficient for large panel data. Memory overhead is high because `neighbor_lookup` stores large lists of integer vectors, and repeated `do.call(rbind, ...)` adds further cost.  

**Optimization Strategy**  
- Avoid per-row string operations and repeated list traversals.
- Precompute a fast integer-based mapping of `(id, year)` to row index using `data.table` keys.
- Use `data.table` joins instead of `lapply` for neighbor feature aggregation.
- Compute all neighbor stats in a single grouped operation rather than looping over variables.
- Keep everything in memory-efficient integer vectors and avoid large intermediate lists.

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table for fast joins
setDT(cell_data)

# Precompute mapping: (id, year) -> row index
cell_data[, row_id := .I]

# Expand neighbor relationships across years
# rook_neighbors_unique: list of integer vectors (neighbors per cell)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Cartesian join with years to create full neighbor-year mapping
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id, neighbor_id, year = years), by = .(id, neighbor_id)]

# Map to row indices
neighbor_dt[, row_id := cell_data[.SD, on = .(id, year), x.row_id]]
neighbor_dt[, neighbor_row_id := cell_data[.SD, on = .(id = neighbor_id, year), x.row_id]]

# Remove missing mappings
neighbor_dt <- neighbor_dt[!is.na(row_id) & !is.na(neighbor_row_id)]

# Compute neighbor stats for all variables in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt cell_data for fast aggregation
long_data <- melt(cell_data[, c("row_id", neighbor_source_vars), with = FALSE],
                  id.vars = "row_id", variable.name = "var_name", value.name = "val")

# Join neighbor relationships to values
neighbor_vals <- neighbor_dt[, .(row_id, neighbor_row_id)][
  long_data, on = .(neighbor_row_id = row_id)]

# Aggregate stats by (row_id, var_name)
agg_stats <- neighbor_vals[, .(
  max_val = max(val, na.rm = TRUE),
  min_val = min(val, na.rm = TRUE),
  mean_val = mean(val, na.rm = TRUE)
), by = .(row_id, var_name)]

# Reshape back to wide format
agg_wide <- dcast(agg_stats, row_id ~ var_name, value.var = c("max_val", "min_val", "mean_val"))

# Merge back to cell_data
cell_data <- merge(cell_data, agg_wide, by = "row_id", all.x = TRUE)

# Drop helper columns
cell_data[, row_id := NULL]
```

**Why This Works**  
- Eliminates per-row loops and string concatenation.
- Uses `data.table` joins and grouping for vectorized operations.
- Computes all neighbor stats in one grouped aggregation instead of multiple passes.
- Memory footprint is reduced because intermediate structures are compact and numeric.

**Expected Performance**  
This approach should reduce runtime from 86+ hours to a few hours (or less) on a 16 GB laptop, depending on disk I/O and CPU speed, because it leverages efficient joins and aggregation rather than millions of R-level loops.