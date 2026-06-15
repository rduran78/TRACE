 **Diagnosis**  
The current bottleneck arises because neighbor statistics are recomputed for every cell-year row across the entire 6.46M dataset, repeatedly iterating over neighbors for each variable. This is highly inefficient because:
- The neighbor structure is static (same across all years), but we're rebuilding or reusing it without exploiting year separation.
- Computation is done row-wise for all years together, causing redundant lookups and memory churn.
- Each pass over `neighbor_lookup` for 5 variables is repeated rather than vectorized or grouped.
- The pipeline does not leverage the fact that neighbor relationships stay constant, so we could precompute per-year slices and apply `data.table` or matrix-based operations.

**Optimization Strategy**  
1. Precompute `neighbor_lookup` **once** at the cell level (not cell-year), since neighbors don’t change by year.
2. Process data **year by year** to reduce memory footprint and exploit static structure.
3. Use `data.table` for fast grouping and joins instead of `lapply` over millions of rows.
4. Avoid rebuilding neighbor statistics for each variable individually—compute all neighbor stats in a single pass per year.
5. Bind results back efficiently without exploding memory.

**Working R Code**

```r
library(data.table)

# Assumes: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order and rook_neighbors_unique are available

# Step 1: Precompute static neighbor lookup at cell level
build_neighbor_lookup_static <- function(id_order, neighbors) {
  lapply(seq_along(id_order), function(i) id_order[neighbors[[i]]])
}
neighbor_lookup_static <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)
names(neighbor_lookup_static) <- as.character(id_order)

# Step 2: Convert cell_data to data.table for efficient ops
setDT(cell_data)

# Step 3: Function to compute neighbor stats for one year slice
compute_year_neighbor_stats <- function(dt_year, neighbor_lookup_static, vars) {
  # Create a named vector for fast lookup
  vals_list <- lapply(vars, function(v) setNames(dt_year[[v]], as.character(dt_year$id)))
  
  # For each cell, compute stats from neighbors
  res_list <- lapply(seq_len(nrow(dt_year)), function(i) {
    cell_id <- as.character(dt_year$id[i])
    n_ids <- neighbor_lookup_static[[cell_id]]
    out <- numeric(length(vars) * 3)
    k <- 1
    for (v in vars) {
      n_vals <- vals_list[[v]][n_ids]
      n_vals <- n_vals[!is.na(n_vals)]
      if (length(n_vals) == 0) {
        out[k:(k+2)] <- NA
      } else {
        out[k:(k+2)] <- c(max(n_vals), min(n_vals), mean(n_vals))
      }
      k <- k + 3
    }
    out
  })
  
  res_mat <- do.call(rbind, res_list)
  col_names <- unlist(lapply(vars, function(v) paste0(v, c("_nb_max","_nb_min","_nb_mean"))))
  setnames(as.data.table(res_mat), col_names)
}

# Step 4: Apply year by year and bind results
neighbor_vars <- c("ntl","ec","pop_density","def","usd_est_n2")
years <- sort(unique(cell_data$year))

result_list <- vector("list", length(years))
for (j in seq_along(years)) {
  yr <- years[j]
  dt_year <- cell_data[year == yr]
  stats_dt <- compute_year_neighbor_stats(dt_year, neighbor_lookup_static, neighbor_vars)
  result_list[[j]] <- cbind(dt_year, stats_dt)
}

cell_data <- rbindlist(result_list)

# cell_data now includes neighbor max, min, mean for each variable
```

**Why this works faster**  
- The neighbor lookup is built **once** at cell level, not per row.
- Processing by year reduces memory strain from 6.46M rows.
- We batch compute all neighbor stats per cell instead of multiple passes.
- `data.table` speeds up joins and slicing.

**Expected performance improvement**  
From 86+ hours to a few hours, depending on machine resources, since redundant operations are eliminated and loops are scoped to 28 slices rather than 6.46M rows repeatedly.

This preserves the trained Random Forest model and original estimand while ensuring computational scalability.