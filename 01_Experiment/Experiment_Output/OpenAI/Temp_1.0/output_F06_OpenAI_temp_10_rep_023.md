 **Diagnosis**  
The core bottleneck is the `lapply`-based per-row neighbor iteration over 6.46M rows across 28 years. This involves frequent list indexing, repeated lookups, and creation of small intermediate objects. With ~1.37M neighbor relations and multiple variables, the nested loops cause severe overhead in R's interpreter. The approach is memory-safe but extremely slow.  

**Optimization Strategy**  
- Represent neighbor relationships in long-form as an edge list with `(i_row, j_row)` pairs for all valid neighbors (including year alignment).
- Use `data.table` for fast joins and grouped calculations.
- Compute `max`, `min`, `mean` per `(i_row)` and `var_name` in vectorized batches rather than row-wise loops.
- Reattach summary stats back to `cell_data` efficiently.
- Avoid altering the trained Random Forest model; only optimize feature computation.
  
**Working R Code**  

```r
library(data.table)

# Convert cell_data to data.table
setDT(cell_data)  # assumes columns: id, year, ntl, ec, pop_density, def, usd_est_n2

# Step 1: Build long edge list (i_row -> j_row where neighbors in same year)
cell_data[, rowid := .I]

# Map id to index for fast lookup
id_to_idx <- setNames(seq_along(id_order), id_order)

# Expand rook neighbor list
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = id_order[i], nb = id_order[rook_neighbors_unique[[i]]])
}))

# Join with cell_data for all years
edges_expanded <- merge(cell_data[, .(year, src = id, i_row = rowid)],
                         edges, by = "src", allow.cartesian = TRUE)

edges_expanded <- merge(edges_expanded,
                        cell_data[, .(year, nb = id, j_row = rowid)],
                        by = c("year", "nb"), allow.cartesian = TRUE)

# edges_expanded columns: src, i_row, nb, j_row, year
# Drop unnecessary cols
edges_expanded[, c("src", "nb") := NULL]

# Step 2: For each var_name, compute neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Extract neighbor values
  vals <- cell_data[, .(rowid, val = get(var_name))]

  # Join neighbor values to edge list
  dt <- merge(edges_expanded, vals, by.x = "j_row", by.y = "rowid", all.x = TRUE)

  # Compute max, min, mean per i_row
  stats <- dt[!is.na(val),
              .(max_val = max(val), min_val = min(val), mean_val = mean(val)),
              by = i_row]

  # Merge with main table
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbmax", "_nbmin", "_nbmean")))
  cell_data <- merge(cell_data, stats, by.x = "rowid", by.y = "i_row", all.x = TRUE)
}

# Cleanup
cell_data[, rowid := NULL]
```

**Why This Works**  
- Eliminates per-row `lapply` loops by using one global edge expansion and grouped summaries.
- `data.table` provides vectorized aggregation for 6.46M rows efficiently in memory on a 16 GB machine.
- Preserves all estimands: neighbor stats for each variable across all years.
- Estimated runtime drops from ~86 hours to a few minutes (depending on disk I/O, likely under 1 hour).

**Next Steps**  
Once neighbor features are appended, you can safely feed `cell_data` into your pre-trained Random Forest model without retraining or altering its structure.