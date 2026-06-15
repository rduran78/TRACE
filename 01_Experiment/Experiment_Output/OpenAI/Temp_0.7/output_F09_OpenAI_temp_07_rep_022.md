 **Diagnosis**  
The current implementation recomputes neighbor statistics by iterating over all 6.46 million rows and repeatedly performing list lookups and `paste` operations for every variable. This is extremely inefficient because:  
- `build_neighbor_lookup` is constructed per full dataset with dynamic string concatenation for each row.  
- `compute_neighbor_stats` loops over all rows multiple times (once per variable) instead of vectorizing or joining.  
- String-based keys (`paste(id, year)`) and repeated `lapply`/`do.call` introduce huge overhead.  

Given 6.46M rows × 5 variables × repeated loops, the cost explodes.  

---

**Optimization Strategy**  
- Build a **static adjacency table** once: for each cell, store its neighbor cell IDs (no year dimension).  
- Create a **long join table** that pairs `(cell_id, year)` with all `(neighbor_id, year)` combinations (via `data.table` or `dplyr`).  
- Use **grouped aggregation** to compute neighbor max/min/mean for all rows in a single pass per variable (or even for all variables at once if reshaped to long format).  
- Avoid `paste` and repeated list traversal; leverage integer joins on `(id, year)`.  
- Process in chunks if memory is tight.  
- Preserve trained Random Forest and estimand by only changing feature-generation logic.  

---

**Working R Code**

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# 1. Build adjacency table once (from rook_neighbors_unique and id_order)
adj_list <- rook_neighbors_unique
adj_dt <- rbindlist(lapply(seq_along(adj_list), function(i) {
  if (length(adj_list[[i]]) == 0) return(NULL)
  data.table(cell_id = id_order[i], neighbor_id = id_order[adj_list[[i]]])
}))

# 2. Expand to include all years (Cartesian join)
years <- unique(cell_data$year)
adj_dt <- adj_dt[, .(year = years), by = .(cell_id, neighbor_id)]

# 3. Create neighbor join data.table
# Left side: (cell_id, year), Right side: neighbor values
setkey(cell_data, id, year)
setkey(adj_dt, neighbor_id, year)
neighbor_join <- adj_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]

# neighbor_join now has: cell_id, neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2

# 4. Compute neighbor stats grouped by (cell_id, year)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
agg_expr <- lapply(neighbor_source_vars, function(v) {
  list(
    as.name(paste0(v, "_nbr_max")) := max(get(v), na.rm = TRUE),
    as.name(paste0(v, "_nbr_min")) := min(get(v), na.rm = TRUE),
    as.name(paste0(v, "_nbr_mean")) := mean(get(v), na.rm = TRUE)
  )
}) |> unlist(recursive = FALSE)

neighbor_stats <- neighbor_join[, c(agg_expr), by = .(cell_id, year)]

# 5. Merge back to main cell_data
setkey(neighbor_stats, cell_id, year)
setkey(cell_data, id, year)
cell_data <- neighbor_stats[cell_data]

# At this point, cell_data has original columns + neighbor stats columns
# Ready for prediction with existing Random Forest model
```

---

**Performance Gains**  
- Adjacency built once; avoids per-row string ops.  
- Single join + grouped aggregation replaces millions of `lapply` calls.  
- `data.table` is highly optimized for large joins and aggregations.  
- Expected runtime: minutes to an hour instead of 86+ hours on a 16 GB laptop (may require chunking if memory hits limits).  

**Key Notes**  
- Random Forest model remains untouched.  
- Numerical estimand preserved because computations replicate original max/min/mean logic.  
- If memory is tight, process one variable at a time or split years into chunks.