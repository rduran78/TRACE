 **Diagnosis**  
- The bottleneck is the repeated use of `lapply` over ~6.46M rows, combined with frequent string concatenation (`paste`) and lookups in named vectors.  
- `build_neighbor_lookup` constructs neighbor indices for every row, creating a massive list in memory. For 6.46M rows, this is prohibitive.  
- `compute_neighbor_stats` iterates over the same list multiple times for each variable, multiplying cost.  
- The design is highly row-centric; this prevents vectorization and efficient joins.  

**Optimization Strategy**  
- Avoid building a full row-level neighbor lookup list. Instead, compute neighbor stats in a **vectorized and grouped manner** using `data.table` or `dplyr`.  
- Store data as `data.table` for fast keyed joins.  
- Expand neighbor relationships once per year, join to original data to fetch neighbor values, then aggregate stats grouped by `id, year`.  
- This eliminates huge lists and repeated looping, reducing memory footprint and compute time drastically.  

**Working R Code (optimized):**

```r
library(data.table)

# Assume cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Convert neighbors to a data.table of directed pairs
# rook_neighbors_unique: list of neighbors for each cell id in id_order
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    src_id = id_order[i],
    nbr_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand for all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(src_id, nbr_id)]

# Join neighbor values for each variable and aggregate
compute_neighbor_features <- function(var_name) {
  # Join neighbor values
  tmp <- merge(
    neighbor_dt,
    cell_data[, .(id, year, val = get(var_name))],
    by.x = c("nbr_id", "year"), by.y = c("id", "year"),
    all.x = TRUE
  )
  
  # Aggregate stats by src_id, year
  tmp <- tmp[!is.na(val), .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(src_id, year)]
  
  # Rename columns
  setnames(tmp, c("src_id", "year", "max_val", "min_val", "mean_val"),
           c("id", "year",
             paste0(var_name, "_nbr_max"),
             paste0(var_name, "_nbr_min"),
             paste0(var_name, "_nbr_mean")))
  tmp
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute and merge for all variables
feature_tables <- lapply(neighbor_source_vars, compute_neighbor_features)
all_features <- Reduce(function(x, y) merge(x, y, by = c("id", "year"), all = TRUE), feature_tables)

# Merge back to cell_data
cell_data <- merge(cell_data, all_features, by = c("id", "year"), all.x = TRUE)
```

**Why This Works**  
- Handles neighbor expansion once, then uses fast joins and group aggregation.
- Reduces memory by avoiding 6.46M-element lists.
- `data.table` grouping is highly optimized for large datasets.
- Preserves original estimand (max, min, mean of neighbors).
- Compatible with existing Random Forest model (feature names unchanged except suffixes).

**Expected Impact**  
- Moves from ~86 hours to a few hours on 16 GB RAM laptop.
- Scales efficiently without retraining the model or altering the data semantics.