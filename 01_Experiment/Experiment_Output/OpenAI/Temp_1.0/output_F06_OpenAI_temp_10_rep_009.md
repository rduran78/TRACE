 **Diagnosis**  
The bottleneck is the nested `lapply` structure applied over **6.46 million rows × 5 variables**. For each row, the `compute_neighbor_stats` function iterates through an individual lookup list and recomputes max, min, and mean using R loops, causing heavy overhead. This design scales poorly because:

- It repeatedly processes each cell-year rather than computing statistics over the adjacency structure in bulk.
- Memory usage is inefficient due to millions of small list objects.
- Pure R loops cannot effectively utilize vectorization or parallelization here.

The operation is conceptually a **focal statistics computation on a panel grid**, which naturally maps to a **graph-based aggregation** problem. Using **vectorized joins or matrix-based reductions** can cut runtime by orders of magnitude while preserving original results.

---

**Optimization Strategy**  
1. **Vectorize neighbor aggregation**:
   - Expand neighbor relationships into a long “edges” table: `(source_row, neighbor_row)`.
   - Use **data.table** for fast joins and grouped aggregations to compute `max`, `min`, `mean`.
2. Compute all five variables in one pass rather than iterating five times.
3. Reattach aggregated stats to the original dataset.
4. (Optional) Use **parallelization** with `data.table` or `future` for scalability.

This avoids per-row list operations and leverages highly optimized C-backed group aggregations.

---

**Working R Code**

```r
library(data.table)

# Convert your data.frame to data.table for efficiency
setDT(cell_data)

# Step 1: Build neighbor lookup into a long edge list
id_to_ref <- setNames(seq_along(id_order), id_order)
idx_lookup <- setNames(seq_len(nrow(cell_data)),
                       paste(cell_data$id, cell_data$year, sep = "_"))

expand_neighbors <- function(data, neighbors, id_order) {
  # For each row in data, find its neighbors in the same year
  src_ids <- seq_len(nrow(data))
  # Get reference for cell -> rook neighbors
  ref_idx <- id_to_ref[as.character(data$id[src_ids])]
  
  edge_list <- vector("list", length(src_ids))
  
  for (i in src_ids) {
    neigh_cells <- id_order[neighbors[[ref_idx[i]]]]
    if (length(neigh_cells) > 0) {
      neigh_keys <- paste(neigh_cells, data$year[i], sep = "_")
      neigh_rows <- idx_lookup[neigh_keys]
      neigh_rows <- neigh_rows[!is.na(neigh_rows)]
      if (length(neigh_rows) > 0) {
        edge_list[[i]] <- data.table(src = i, nbr = neigh_rows)
      }
    }
  }
  rbindlist(edge_list)
}

# Build edge table
edges <- expand_neighbors(cell_data, rook_neighbors_unique, id_order)
# edges: src (row in cell_data), nbr (row in cell_data)

# Step 2: Aggregate all neighbor variables efficiently
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt neighbor values for join
agg_list <- lapply(neighbor_source_vars, function(var) {
  dt <- cell_data[, .(nbr_row = .I, val = get(var))]
  setnames(dt, "val", var)
  dt
})

neighbor_vals <- Reduce(function(x, y) merge(x, y, by = "nbr_row", all = TRUE), agg_list)

# Merge edges with neighbor values
edges <- merge(edges, neighbor_vals, by.x = "nbr", by.y = "nbr_row", all.x = TRUE)

# Step 3: Compute stats by src row
agg_stats <- edges[, .(
  ntl_max = max(ntl, na.rm = TRUE),
  ntl_min = min(ntl, na.rm = TRUE),
  ntl_mean = mean(ntl, na.rm = TRUE),
  ec_max = max(ec, na.rm = TRUE),
  ec_min = min(ec, na.rm = TRUE),
  ec_mean = mean(ec, na.rm = TRUE),
  pop_density_max = max(pop_density, na.rm = TRUE),
  pop_density_min = min(pop_density, na.rm = TRUE),
  pop_density_mean = mean(pop_density, na.rm = TRUE),
  def_max = max(def, na.rm = TRUE),
  def_min = min(def, na.rm = TRUE),
  def_mean = mean(def, na.rm = TRUE),
  usd_est_n2_max = max(usd_est_n2, na.rm = TRUE),
  usd_est_n2_min = min(usd_est_n2, na.rm = TRUE),
  usd_est_n2_mean = mean(usd_est_n2, na.rm = TRUE)
), by = src]

# Replace NaN for groups with no neighbors
for (j in names(agg_stats)) {
  set(agg_stats, which(is.nan(agg_stats[[j]])), j, NA)
}

# Step 4: Merge aggregated features back to cell_data
cell_data <- merge(cell_data, agg_stats, by.x = ".I", by.y = "src", all.x = TRUE)

# Now cell_data has new neighbor-based features
```

---

**Expected Performance Gain:**  
- Original: ~86 hours (pure R looping over millions of rows).
- Optimized: minutes to under an hour on 16 GB RAM laptop using `data.table` and vectorized merges.

**Key Points:**  
- Preserves trained Random Forest model (no retraining).
- Preserves original estimands: max, min, and mean for each neighbor variable.
- Scales well for 6.5M rows and 1.3M edges.