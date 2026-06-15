 **Diagnosis**  
The current pipeline repeatedly rebuilds neighbor relationships for every variable and every year, which is extremely inefficient for a dataset of ~6.46M rows. The function `build_neighbor_lookup` is called only once, but `compute_neighbor_stats` is applied in a loop for each variable, iterating over millions of rows with `lapply` and repeatedly subsetting large vectors. This incurs heavy R-level overhead and memory thrashing. Additionally, the neighbor lookup is computed in row order, which is inefficient for joins.

**Optimization Strategy**  
1. **Precompute a reusable adjacency table**: Build a single long-format data frame mapping each row to its neighbors (row indices), including year alignment. This avoids repeated lapply calls.
2. **Vectorized aggregation**: Use `data.table` joins to compute max, min, mean for all variables at once instead of looping millions of times in R.
3. **Join-based computation**: Perform neighbor summaries by grouping on the focal row after joining neighbor rows.
4. **Preserve model and estimand**: Only change feature generation; do not retrain.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Step 1: Build adjacency table once
# id_order: vector of unique cell IDs in same order as rook_neighbors_unique
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- cell_data[, .I, by = .(id, year)] # row index lookup

adj_list <- lapply(seq_along(id_order), function(i) {
  focal_id <- id_order[i]
  nbr_ids  <- id_order[rook_neighbors_unique[[i]]]
  if (length(nbr_ids) == 0) return(NULL)
  data.table(id = focal_id, neighbor_id = nbr_ids)
})

adj_dt <- rbindlist(adj_list)
rm(adj_list)

# Step 2: Expand adjacency across years (Cartesian join)
years <- sort(unique(cell_data$year))
adj_dt <- adj_dt[CJ(year = years), on = .(year)]
setcolorder(adj_dt, c("id", "neighbor_id", "year"))

# Step 3: Map to row indices for join
adj_dt <- merge(adj_dt, idx_lookup, by = c("id", "year"), all.x = TRUE)
setnames(adj_dt, "I", "row_id")
adj_dt <- merge(adj_dt, idx_lookup, by.x = c("neighbor_id", "year"), by.y = c("id", "year"), all.x = TRUE)
setnames(adj_dt, "I", "neighbor_row_id")

# Drop if neighbor not present for year
adj_dt <- adj_dt[!is.na(neighbor_row_id)]

# Step 4: Compute neighbor stats for all vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  temp <- cell_data[, .(neighbor_val = get(var), neighbor_row_id = .I)]
  stats <- merge(adj_dt[, .(row_id, neighbor_row_id)], temp, by = "neighbor_row_id", allow.cartesian = TRUE)
  stats <- stats[, .(
    max = max(neighbor_val, na.rm = TRUE),
    min = min(neighbor_val, na.rm = TRUE),
    mean = mean(neighbor_val, na.rm = TRUE)
  ), by = row_id]

  # Add to cell_data
  setnames(stats, c("max", "min", "mean"), paste0(var, "_nbr_", c("max", "min", "mean")))
  cell_data <- merge(cell_data, stats, by.x = ".I", by.y = "row_id", all.x = TRUE)
}

# cell_data now has neighbor features added efficiently
```

---

**Why This Is Faster**  
- The neighbor relationships are computed once and reused.
- `data.table` joins and grouped aggregation are implemented in C, dramatically reducing overhead versus nested `lapply`.
- Avoids 6.46M × 5 full R loops; instead, performs efficient merge-aggregate operations.

**Expected Performance Gain**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on disk I/O and CPU.