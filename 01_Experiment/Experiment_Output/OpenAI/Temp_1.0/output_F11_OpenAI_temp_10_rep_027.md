 The evidence shows the main bottleneck is **neighbor feature computation**, not Random Forest inference. The `build_neighbor_lookup` and especially `compute_neighbor_stats` use deeply nested `lapply()` calls over 6.46 million rows × 5 variables. This creates massive R list processing overhead and repeated handling of vectors. In contrast, `predict()` on a trained Random Forest for ~6.5M rows and 110 predictors is fast relative to 86 hours.

### Diagnosis
- **Root cause**: For each row, extracting neighbor indices and computing summaries uses pure R loops and repeated vector slicing. This is highly inefficient at large scale.
- Random Forest inference on 6.46 M rows is typically minutes, not days, even on a laptop.

### Optimization Strategy
- Precompute neighbor lookup **once**, and store as an integer matrix or compressed list.
- Replace the per-row `lapply()` with **vectorized aggregation** using `data.table` or `dplyr`.
- Leverage joins keyed by `(id, year)` to compute neighbor statistics in batch rather than row-by-row iteration.

### Efficient R Implementation (using `data.table`)
```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Assume: cell_data has columns id, year, and all covariates
# rook_neighbors_unique: list of integer vectors (directed neighbors for each id)

# Create mapping for year expansion in a flat table form
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id        = id_order[i],
    neighbor  = id_order[rook_neighbors_unique[[i]]]
  )
}))

setkey(neighbor_dt, id)         # For fast joining
cell_data[, key := paste(id, year, sep = "_")]

# Expand neighbor-year relationships
expanded <- neighbor_dt[cell_data, on = .(id), allow.cartesian = TRUE]
expanded[, year := i.year]      # Borrow year from cell_data join

# Join neighbor values per variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  val_dt <- cell_data[, .(neighbor = id, year, val = get(var_name))]
  expanded_vals <- expanded[val_dt, on = .(neighbor, year)]
  
  stats <- expanded_vals[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Merge stats back
  setnames(stats, c("id","year","max_val","min_val","mean_val"),
           c("id","year",
             paste0(var_name,"_nb_max"),
             paste0(var_name,"_nb_min"),
             paste0(var_name,"_nb_mean")))
  
  cell_data <- stats[cell_data, on = .(id, year)]
}

# After this, cell_data has new neighbor features efficiently
```

### Why this works
- Eliminates millions of R function calls (`lapply` per row).
- Uses `data.table` joins and group aggregation in C, which is orders of magnitude faster.
- Scales well to millions of rows on modest hardware.

**Expected gain**: Reduce 86+ hours to well under an hour, dominated by disk I/O and join aggregation.