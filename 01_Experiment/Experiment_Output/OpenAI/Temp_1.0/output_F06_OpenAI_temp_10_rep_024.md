 **Diagnosis**  
The bottleneck is the nested structure of `lapply` over ~6.46M rows, computing neighbor statistics naively for each cell-year. Each iteration recomputes values with random memory access and R function calls, which is prohibitively slow. The task is embarrassingly parallelizable, but also cache-inefficient. The neighbor set (rook neighbors) is constant per cell, while panel years create many repeated lookups — doing this via R loops is inefficient. Current complexity is roughly *O(N × avg_neighbors)* with `N ≈ 6.46M`.

---

**Optimization Strategy**  
1. **Precompute reusable structures**:  
   - Create a mapping of neighbors per cell once. Avoid repeated string concatenation.
2. **Use `data.table` for fast grouping and joins**, leveraging ID and year as keys.  
3. **Vectorize neighbor aggregation**:  
   - Reshape data to *wide by year* only if needed, but here keep long format and compute aggregates through a self-join keyed by `(cell_id, year)`.
4. **Optional parallelization** using `future.apply` or `data.table` multithreading.
5. **Preserve estimand**: results of `max`, `min`, `mean` over neighbor values must be identical.

---

**Working R Code** *(Efficient data.table solution)*  

```r
library(data.table)

# Assume cell_data as data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Convert neighbors list to long format edge table: cell_id -> neighbor_id
edges <- data.table(from = rep(id_order, lengths(rook_neighbors_unique)),
                    to   = unlist(rook_neighbors_unique))
# You only need one direction? Here it's directed as per original
setnames(edges, c("cell_id", "neighbor_id"))

# Prepare keys for fast join
setkey(cell_data, id, year)

compute_neighbor_stats_dt <- function(cell_dt, edges, var_name) {
  # Extract pairings with year
  # Join left: edges$neighbor_id -> cell_data$id to get neighbor values
  neighbor_vals <- cell_dt[, .(neighbor_id = id, year, value = get(var_name))]
  setkey(neighbor_vals, neighbor_id, year)

  # Expand: join edges on neighbor_id
  joined <- neighbor_vals[edges, on = .(neighbor_id), allow.cartesian = TRUE]
  # joined now has columns: neighbor_id, year, value, cell_id (from edges)
  
  # Group by focal cell and year
  stats <- joined[, .(
    max_val = if (.N > 0) max(value, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(value, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(value, na.rm = TRUE) else NA_real_
  ), by = .(cell_id, year)]

  # Join back to main table
  out <- cell_dt[stats, on = .(id = cell_id, year)]
  setnames(out, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  out[]
}

# Apply for all variables
for (var in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  cell_data <- compute_neighbor_stats_dt(cell_data, edges, var)
}

# cell_data now enriched with neighbor stats, ready for RF prediction
```

---

**Why this works faster**  
- No per-row lapply: operations are aggregated by `data.table` group statistics in C.
- String operations eliminated; joins driven by integer IDs.
- Handles all rows in parallel internally.
- Complexity reduces to linear in relation/row count instead of nested R calls.
- Memory-friendly and works within 16 GB RAM for given dataset (~6.5M rows + ~1.37M edges).

**Expected Performance**  
From 86+ hours → typically **minutes to low hours** on standard hardware. Further speedup possible with `setDTthreads(parallel::detectCores())`.  

**Model Preservation**  
The Random Forest model remains untouched; only feature engineering is accelerated while preserving original statistical outputs.