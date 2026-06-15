 **Diagnosis:**  
The bottleneck is the nested `lapply` approach over ~6.46M rows, repeatedly computing neighbor stats row-by-row in R lists. Each call performs multiple lookups and vector allocations. This is highly inefficient in pure R given the massive panel size and 1.37M neighbor relationships. Memory pressure and interpreter overhead dominate runtime.

---

**Optimization Strategy:**  
1. **Avoid nested R loops and repeated lookups**: Replace per-row `lapply` with vectorized or grouped operations.
2. **Precompute neighbor relationships in a long (edge) table**: `(cell_id, year) → neighbor_id`.
3. **Join and aggregate** using `data.table` for speed and memory efficiency.
4. **Use keyed joins** to compute max, min, mean per focal cell-year in one pass per variable.
5. Preserve all results and Random Forest model without retraining.

---

**Efficient Implementation in R (`data.table`):**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor table
# id_order aligns rook_neighbors_unique to cell ids
neighbor_pairs <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Expand across years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_pairs[, .(id = from, neighbor_id = to)][
  , .(id = rep(id, each = length(years)),
      neighbor_id = rep(neighbor_id, each = length(years)),
      year = rep(years, times = .N))
]

# Prepare key fields
cell_data[, key := paste(id, year, sep = "_")]
neighbor_dt[, key := paste(neighbor_id, year, sep = "_")]

# Create lookup for neighbor values
setkey(cell_data, key)
setkey(neighbor_dt, key)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join neighbor values
  tmp <- neighbor_dt[cell_data, on = .(key), nomatch = 0]
  setnames(tmp, var_name, "neighbor_val")
  
  # Aggregate stats by focal id and year
  stats <- tmp[!is.na(neighbor_val),
               .(max_val = max(neighbor_val),
                 min_val = min(neighbor_val),
                 mean_val = mean(neighbor_val)),
               by = .(id, year)]
  
  # Merge back into cell_data
  setkey(stats, id, year)
  setkey(cell_data, id, year)
  cell_data <- stats[cell_data]
  
  # Rename columns appropriately
  setnames(cell_data,
           c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
}

# cell_data now includes neighbor features for all source vars
```

---

**Why This Works:**  
- Uses `data.table` for memory-efficient joins and aggregation.
- Eliminates per-row `lapply` overhead: aggregation happens in C backend.
- Single pass per variable, not millions of R function calls.
- Preserves original estimands and integrates seamlessly with existing Random Forest predictions.
- Expected runtime drops from **86+ hours** to **minutes** on 16 GB RAM laptop.

**Key Gains:**  
- Vectorization and grouping instead of iterative loops.
- Joins and aggregations leverage compiled `data.table` operations.