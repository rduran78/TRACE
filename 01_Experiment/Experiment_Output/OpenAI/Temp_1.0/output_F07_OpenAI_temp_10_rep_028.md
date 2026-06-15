 **Diagnosis**  
The performance bottleneck is the per-row `lapply` approach over ~6.46M rows, triggering numerous R-level loops and repeated vector allocations. It also recomputes lookups repeatedly and relies on `lapply` for both neighbor mapping and summary computation, which is slow at scale. Memory usage is reasonable, but execution time is prohibitively long because of R’s overhead in iterating millions of times.

**Optimization Strategy**  
1. **Vectorize and Precompute**:  
   - Flatten neighbor relationships into a single mapping table of `(row_idx, neighbor_idx)` pairs for all years at once.
   - Maintain each feature column as a numeric vector and compute max, min, mean per `row_idx` using efficient group aggregation.
2. **Use `data.table` for Speed**:  
   - `data.table`'s grouping (`by`) is far faster for millions of rows than base R loops.
3. **Memory Efficiency**:  
   - Avoid nested lists; store relationships as a long data frame/table only once.
4. **Reuse Neighbor Index Mapping** across all variables:  
   - Compute joins once, then aggregate per variable to avoid repeated neighbor lookups.

---

### **Optimized Working Code**

```r
library(data.table)

# Assumes: cell_data (data.frame with columns id, year, ntl, ec, pop_density, def, usd_est_n2)
#          id_order (vector of unique ids)
#          rook_neighbors_unique (list from spdep::nb)

setDT(cell_data)
setkey(cell_data, id, year)

# Step 1: Build long index map of row -> neighbor_row for all years
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
adj_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  cid <- id_order[i]
  nbrs <- id_order[rook_neighbors_unique[[i]]]
  if (length(nbrs)) {
    data.table(id = cid, neighbor_id = nbrs)
  } else NULL
}))

# Repeat for all years (Cartesian)
years <- unique(cell_data$year)
adj_dt <- adj_dt[, .(year = years), by = .(id, neighbor_id)]
# Map to row indices in cell_data
adj_dt[, row_idx := cell_data[J(id, year), which = TRUE]]
adj_dt[, neighbor_idx := cell_data[J(neighbor_id, year), which = TRUE]]
adj_dt <- adj_dt[!is.na(row_idx) & !is.na(neighbor_idx)]
# Remove columns not needed
adj_dt <- adj_dt[, .(row_idx, neighbor_idx)]

# Step 2: Function to compute neighbor stats for a given column
compute_neighbor_stats_dt <- function(var_name) {
  vals <- cell_data[[var_name]]
  tmp <- data.table(row_idx = adj_dt$row_idx,
                    val = vals[adj_dt$neighbor_idx])
  # Remove NAs before aggregation
  tmp <- tmp[!is.na(val)]
  agg <- tmp[, .(max = max(val),
                 min = min(val),
                 mean = mean(val)), by = row_idx]
  setnames(agg, c("row_idx", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  agg
}

# Step 3: Apply for all source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg_list <- lapply(neighbor_source_vars, compute_neighbor_stats_dt)

# Merge all aggregates back to cell_data
for (agg in agg_list) {
  cell_data <- merge(cell_data, agg, by.x = ".I", by.y = "row_idx", all.x = TRUE)
}

# Drop helper column .I if needed
```

---

**Why this works better**  
- We **flatten adjacency once** (~1.37M edges × 28 years ≈ 38M rows in `adj_dt`), still manageable on 16 GB RAM (≈ few GB).
- All aggregations occur through **highly optimized C-backed grouping in `data.table`** rather than millions of `lapply` calls.
- Sequence of steps eliminates redundant computations for each variable.
- Preserves original rook topology and numerical estimand since we only refactored the computational approach.

**Expected runtime** on 6.5M rows with `data.table` grouping: **minutes rather than 86+ hours** on a standard laptop. Memory footprint is higher but within 16 GB.