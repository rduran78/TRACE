 **Diagnosis**  
The bottleneck is the repeated neighbor feature computation across ~6.46M rows. The current implementation uses `lapply` for every row and variable, causing excessive R-level loops and memory overhead. `build_neighbor_lookup` creates a large list of integer vectors, and `compute_neighbor_stats` iterates over it multiple times. This is highly inefficient for such a large dataset.

**Optimization Strategy**  
- Avoid repeated `lapply` calls for each variable; compute all neighbor stats in one pass.
- Use `data.table` for fast grouping and joins.
- Flatten neighbor relationships into a long table and aggregate with vectorized operations.
- Precompute neighbor stats per year and join back to the main table.
- Keep memory footprint low by processing year-by-year instead of all 6.46M rows at once.

**Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Flatten neighbor relationships
# id_order and rook_neighbors_unique assumed available
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Process year-by-year to reduce memory usage
years <- unique(cell_data$year)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (yr in years) {
  # Subset for the year
  year_dt <- cell_data[year == yr, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  
  # Join neighbors for this year
  joined <- merge(neighbor_dt, year_dt, by.x = "neighbor_id", by.y = "id", allow.cartesian = TRUE)
  
  # Compute stats for all variables in one aggregation
  agg <- joined[, lapply(.SD, function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(x), min(x), mean(x))
  }), by = .(id), .SDcols = neighbor_source_vars]
  
  # Reshape aggregated results
  agg_long <- melt(agg, id.vars = "id", variable.name = "var", value.name = "vals")
  agg_long[, c("max", "min", "mean") := tstrsplit(vals, " ", fixed = TRUE)]
  agg_long[, vals := NULL]
  
  # Cast back to wide format
  agg_wide <- dcast(agg_long, id ~ var, value.var = c("max", "min", "mean"))
  
  # Merge back to cell_data
  cell_data[year == yr, (names(agg_wide)[-1]) := agg_wide[.SD, on = .(id), mget(names(agg_wide)[-1])]]
}

# cell_data now contains neighbor features efficiently
```

**Why This Works**  
- Eliminates per-row loops by using vectorized aggregation.
- Processes year-by-year to fit in memory.
- Computes all neighbor stats in one pass per year.
- Preserves original estimand and Random Forest model.

This approach should reduce runtime from 86+ hours to a few hours or less on a standard laptop.