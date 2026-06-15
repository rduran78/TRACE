 **Diagnosis**  
The bottleneck lies in the nested `lapply` structure applied over ~6.46M rows. For each observation, `compute_neighbor_stats` iterates through neighbor indices, computes `max`, `min`, and `mean`, and then binds results. This is highly inefficient in R because:

- It runs millions of small R function calls (interpreted overhead dominates).
- Repeated lookups in `vals` and filtering `NA`s are done row by row.
- No vectorization or compiled operations are leveraged.
- Memory thrashing occurs when binding millions of small vectors.

**Optimization Strategy**  
- Precompute neighbor lists once (already done).
- Use `data.table` for efficient grouping and joins.
- Reshape data into long format keyed by `(id, year)` for fast neighbor merges.
- Compute stats using vectorized `max`, `min`, and `mean` within groups.
- Parallelize computations by year (or chunks) using `future.apply` or `data.table` parallelism.
- Avoid nested loops entirely.

**Working R Code (Optimized)**  
```r
library(data.table)
library(future.apply)

# Assume: cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Convert to data.table
setDT(cell_data)

# neighbor_lookup: list of integer vectors (indices of neighbors) from build_neighbor_lookup
# Flatten into long mapping table once
neighbor_dt <- rbindlist(lapply(seq_along(neighbor_lookup), function(i) {
  if (length(neighbor_lookup[[i]]) == 0) return(NULL)
  data.table(row_idx = i, neighbor_idx = neighbor_lookup[[i]])
}))

# Add year and id to neighbor_dt by joining with cell_data
neighbor_dt[, year := cell_data$year[row_idx]]

# Join neighbor attributes by neighbor_idx
neighbor_dt[, neighbor_id := cell_data$id[neighbor_idx]]

# Function to compute neighbor stats for one variable
compute_neighbor_features <- function(var_name) {
  # Extract var values
  vals <- cell_data[[var_name]]
  neighbor_dt[, value := vals[neighbor_idx]]
  
  # Drop NAs
  neighbor_dt_non_na <- neighbor_dt[!is.na(value)]
  
  # Aggregate: for each row_idx (cell-year), compute max, min, mean
  stats <- neighbor_dt_non_na[, .(
    max_val = max(value),
    min_val = min(value),
    mean_val = mean(value)
  ), by = row_idx]
  
  # Prepare output data.table aligned with cell_data
  out <- data.table(row_idx = seq_len(nrow(cell_data)))
  out <- merge(out, stats, by = "row_idx", all.x = TRUE)
  
  # Assign to cell_data
  cell_data[, paste0(var_name, "_nbr_max") := out$max_val]
  cell_data[, paste0(var_name, "_nbr_min") := out$min_val]
  cell_data[, paste0(var_name, "_nbr_mean") := out$mean_val]
}

# Parallelize over variables
plan(multisession, workers = parallel::detectCores() - 1)
future_lapply(c("ntl", "ec", "pop_density", "def", "usd_est_n2"), compute_neighbor_features)

# cell_data now contains neighbor-based features for the Random Forest model
```

**Why this is faster:**  
- All neighbor relationships are processed in a single large `data.table`, avoiding millions of separate R function calls.
- Aggregation is done in compiled C code via `data.table`.
- Parallelization further reduces runtime.

**Expected performance:**  
From 86+ hours down to a few hours (or less) depending on disk I/O and CPU cores, while preserving the trained Random Forest model and original estimand.