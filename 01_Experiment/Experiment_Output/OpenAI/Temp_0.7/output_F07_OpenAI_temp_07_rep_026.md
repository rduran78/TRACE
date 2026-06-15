 **Diagnosis:**  
The current implementation is extremely slow (86+ hours) because it iterates through ~6.46M rows and repeatedly performs list lookups, string concatenation, and vectorized operations inside R loops. The `build_neighbor_lookup` constructs repeated string keys per row-year, which is costly. Additionally, `compute_neighbor_stats` uses `lapply` row-wise, which is inefficient for millions of rows. Memory overhead from repeated object creation also adds to the slowness.

---

**Optimization Strategy:**  
1. **Avoid string concatenation per row:** Precompute a numeric index mapping for `(cell_id, year)` to a row index using integer operations.
2. **Vectorize neighbor computations:** Flatten neighbor relationships and compute max, min, and mean using `data.table` or `dplyr` group operations instead of per-row loops.
3. **Pre-expand neighbor pairs for all years:** For each year, join `cell_id` with its neighbors and perform aggregations in bulk.
4. **Streaming or chunk processing:** Process year-by-year to keep RAM usage within 16 GB.
5. **Preserve estimand and model:** Do not retrain; simply produce features from the optimized pipeline.

---

**Working R Code (Efficient Approach using `data.table`):**
```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor pairs (cell_id -> neighbors)
# rook_neighbors_unique: list of integer vectors indexed by id_order
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Set keys for fast joins
setkey(cell_data, id, year)
setkey(neighbor_dt, id)

# Process year-by-year to keep memory low
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (yr in unique(cell_data$year)) {
  cat("Processing year:", yr, "\n")
  
  # Subset for this year
  year_data <- cell_data[year == yr, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  setkey(year_data, id)
  
  # Join neighbor pairs with year_data twice: as focal and as neighbor
  pairs <- neighbor_dt[year_data, nomatch = 0L]  # adds 'year_data' columns to pairs by id
  pairs <- pairs[year_data, on = .(neighbor_id = id), nomatch = 0L, suffixes = c("", "_nbr")]
  
  # For each source var, compute neighbor stats
  for (var in neighbor_source_vars) {
    stats <- pairs[, .(
      max_val = max(get(paste0(var, "_nbr")), na.rm = TRUE),
      min_val = min(get(paste0(var, "_nbr")), na.rm = TRUE),
      mean_val = mean(get(paste0(var, "_nbr")), na.rm = TRUE)
    ), by = .(id, year)]
    
    # Merge back to main cell_data
    setnames(stats, c("id", "year", paste0(var, "_nbr_max"),
                                 paste0(var, "_nbr_min"),
                                 paste0(var, "_nbr_mean")))
    cell_data[year == yr, (names(stats)[3:5]) := stats[, 3:5]]
  }
}

# At this point, cell_data has neighbor-based features for all years and vars
```

---

**Why This Works Efficiently:**  
- Eliminates per-row loops; uses bulk joins and group aggregations in `data.table` (highly optimized in C).
- Processes year-by-year to reduce memory footprint.
- Avoids building large intermediate string keys.
- Preserves rook-neighbor logic and original estimand.

**Expected Performance:**  
On a standard laptop, this chunked and vectorized approach should reduce processing from 86+ hours to a few hours (or less, depending on I/O).