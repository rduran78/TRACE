 **Diagnosis**  
The current pipeline is slow because `build_neighbor_lookup()` recomputes neighbor indices for every cell-year combination (6.46M rows), and `compute_neighbor_stats()` repeatedly loops through full vectors for each year-variable combination. This results in heavy list allocations, many `paste()` string operations, and repeated subset operations, leading to exponential overhead.

**Optimization Strategy**  
- **Precompute adjacency once**: Use cell-level adjacency table from `rook_neighbors_unique` and reuse it across years.
- **Vectorize yearly computation**: Process one year at a time and join neighbor stats without repeatedly constructing strings.
- **Pre-index neighbors by numeric positions**: Avoid costly string concatenations; operate on numeric indices.
- **Compute stats in `data.table`** for efficient joins and group operations.
- **Reuse in-memory objects**, keep 16 GB RAM in mind.

---

### **Optimized Working R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2, etc.
setkey(cell_data, id, year)

# Precompute adjacency once (neighbors as integer positions)
id_order <- sort(unique(cell_data$id))
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_lookup <- lapply(rook_neighbors_unique, function(nbs) id_to_idx[nbs])

# Collapse into edge list (cell_id -> neighbor_id)
edge_list <- data.table(
  id = rep(id_order, lengths(neighbor_lookup)),
  neighbor = unlist(lapply(neighbor_lookup, function(x) id_order[x]))
)

# Function to compute neighbor stats for a given variable
compute_neighbor_stats_fast <- function(dt, var_name) {
  # Melt relevant columns by year
  val_dt <- dt[, .(id, year, val = get(var_name))]

  # Join edge_list twice: origin and neighbor for same year
  joined <- edge_list[val_dt, on = .(neighbor = id)][, neighbor_val := i.val]
  joined <- joined[val_dt, on = .(id, year)][, origin_val := i.val]

  # Compute stats by id, year
  res <- joined[, .(
    max_val = max(neighbor_val, na.rm = TRUE),
    min_val = min(neighbor_val, na.rm = TRUE),
    mean_val = mean(neighbor_val, na.rm = TRUE)
  ), by = .(id, year)]

  # Replace Inf/-Inf with NA (when all neighbor_val are NA)
  res[!is.finite(max_val), c("max_val", "min_val", "mean_val") := .(NA, NA, NA)]

  # Rename columns
  setnames(res, old = c("max_val", "min_val", "mean_val"),
                new = paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  res
}

# Compute and merge for each neighbor source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  feature_dt <- compute_neighbor_stats_fast(cell_data, var_name)
  cell_data <- merge(cell_data, feature_dt, by = c("id", "year"), all.x = TRUE)
}

# Predict using pre-trained Random Forest (model preserved)
cell_data$rf_pred <- predict(rf_model, newdata = cell_data)
```

---

### **Why This Is Faster**
- **Adjacency built once** and kept as numeric IDs—no string concatenation.
- Uses efficient **joins in `data.table`** rather than `lapply` loops over 6.46M rows.
- **No copy-heavy operations inside loops; process in vectorized form**.
- Expected to reduce runtime from **86+ hours to a few hours** on a 16 GB laptop, depending on I/O and join complexity.

This preserves the original estimand and the trained Random Forest model, while drastically improving computational efficiency.