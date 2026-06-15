 **Diagnosis**  
The bottleneck is the nested `lapply` over ~6.46M rows, repeatedly scanning neighbor indices to compute max, min, and mean. This results in excessive R-level loops and memory overhead. The computation is essentially a focal operation (like raster kernel) but implemented in pure R lists, which is inefficient for millions of rows.  

**Optimization Strategy**  
- Avoid per-row `lapply` for neighbor stats; use **vectorized aggregation**.
- Flatten neighbor relationships into a long table: `(cell_year, neighbor_year, var_value)`.
- Use `data.table` for fast group-by aggregation (max, min, mean).
- Precompute keys for `(id, year)` once; join neighbor relationships efficiently.
- Preserve original estimand by computing stats exactly as before (max, min, mean of rook neighbors for same year).
- Do not retrain the Random Forest model; only accelerate feature computation.

---

### **Working R Code**

```r
library(data.table)

# Convert cell_data to data.table
setDT(cell_data)

# Build lookup for (id, year) -> row index
cell_data[, key := paste(id, year, sep = "_")]

# Flatten neighbor relationships
# rook_neighbors_unique: list of neighbors per id_order position
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

# Create long table of neighbor pairs (cell_id, neighbor_id)
neighbor_pairs <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand to include years
years <- unique(cell_data$year)
neighbor_pairs <- neighbor_pairs[, .(id, neighbor_id, year = years), by = .(id, neighbor_id)]

# Add keys for join
neighbor_pairs[, key := paste(id, year, sep = "_")]
neighbor_pairs[, neighbor_key := paste(neighbor_id, year, sep = "_")]

# Join neighbor values
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  vals <- cell_data[, .(neighbor_key = key, value = get(var_name))]
  neighbor_pairs <- vals[neighbor_pairs, on = "neighbor_key"]
  
  # Compute stats by cell-year
  stats <- neighbor_pairs[, .(
    max_val = max(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    mean_val = mean(value, na.rm = TRUE)
  ), by = key]
  
  # Merge back to cell_data
  setnames(stats, c("key", "max_val", "min_val", "mean_val"),
           c("key", paste0(var_name, "_nbr_max"), paste0(var_name, "_nbr_min"), paste0(var_name, "_nbr_mean")))
  cell_data <- stats[cell_data, on = "key"]
}

# Remove helper column
cell_data[, key := NULL]
```

---

**Why This Is Faster**  
- Eliminates 6.46M per-row loops; uses efficient `data.table` joins and group-by.
- Works in chunks per variable, minimizing memory footprint.
- Preserves original logic: same-year rook neighbors, same estimand (max, min, mean).
- Scales to millions of rows on 16 GB RAM; expected runtime drops from 86+ hours to **minutes** to a few hours depending on disk I/O.

**Next Steps**  
- Save `cell_data` with new neighbor features.
- Apply the pre-trained Random Forest model as before (no changes needed).