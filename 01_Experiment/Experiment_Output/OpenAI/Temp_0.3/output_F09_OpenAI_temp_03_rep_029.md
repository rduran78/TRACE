 **Diagnosis:**  
The current pipeline recomputes neighbor relationships for every cell-year row repeatedly, which is extremely costly given 6.46M rows and 5 variables. The `build_neighbor_lookup` function already creates an index-based lookup, but the outer loop still iterates over all rows and variables, performing repeated expensive operations. The main bottleneck is redundant computation of neighbor indices and repeated scanning of large vectors.  

**Optimization Strategy:**  
- Precompute a reusable adjacency table (`neighbor_table`) mapping each cell-year row to its neighbors once.
- Join yearly attributes onto this table and compute neighbor statistics (max, min, mean) in a **vectorized** or grouped manner using `data.table` or `dplyr`.
- Avoid repeated `lapply` over millions of rows for each variable.
- Keep the Random Forest model unchanged; only optimize feature engineering.
- Memory-wise: store adjacency as integers and process in chunks if needed.

---

### **Optimized Approach with `data.table`**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor lookup once (cell_id -> neighbor_ids)
neighbor_lookup <- build_neighbor_lookup(cell_data[year == 1992], id_order, rook_neighbors_unique)
# This returns a list of neighbor indices for each cell_id (not year-specific)

# Build adjacency table for all years
years <- sort(unique(cell_data$year))
adj_list <- vector("list", length(years))

for (y in seq_along(years)) {
  yr <- years[y]
  idx <- which(cell_data$year == yr)
  # For each row in this year, map to neighbor indices
  adj_list[[y]] <- data.table(
    src = idx[rep(seq_along(idx), lengths(neighbor_lookup))],
    nbr = unlist(neighbor_lookup)
  )
}

neighbor_table <- rbindlist(adj_list)
rm(adj_list)  # free memory

# Join neighbor attributes for each variable and compute stats
compute_neighbor_stats_dt <- function(dt, neighbor_table, var_name) {
  # Join neighbor values
  tmp <- neighbor_table[, .(src, nbr_val = dt[[var_name]][nbr])]
  # Compute max, min, mean by src
  stats <- tmp[, .(
    paste0(var_name, "_nbr_max") := max(nbr_val, na.rm = TRUE),
    paste0(var_name, "_nbr_min") := min(nbr_val, na.rm = TRUE),
    paste0(var_name, "_nbr_mean") := mean(nbr_val, na.rm = TRUE)
  ), by = src]
  stats
}

# Apply for all variables
all_stats <- lapply(neighbor_source_vars, function(v) compute_neighbor_stats_dt(cell_data, neighbor_table, v))
all_stats_dt <- Reduce(function(x, y) merge(x, y, by = "src"), all_stats)

# Merge back to cell_data
cell_data <- cbind(cell_data, all_stats_dt[order(src)][, -"src"])
```

---

### **Why This is Faster**
- Neighbor relationships computed **once**.
- Aggregations done in **grouped vectorized operations** instead of nested loops.
- Avoids repeated `lapply` over millions of rows for each variable.
- Uses `data.table` for efficient joins and aggregations.

**Expected Impact:**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on I/O and CPU.  

**Preserves:**  
- Original estimand.
- Trained Random Forest model.
- All feature definitions (max, min, mean of neighbors).