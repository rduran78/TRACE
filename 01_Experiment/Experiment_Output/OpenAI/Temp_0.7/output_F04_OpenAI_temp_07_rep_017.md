 **Diagnosis**  
The bottleneck is the repeated `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. These functions reconstruct neighbor indices and compute statistics in pure R loops for every cell-year row. This is extremely inefficient because:

- `build_neighbor_lookup` is invoked once but creates a massive list of length 6.46M, each element potentially several integers.
- `compute_neighbor_stats` applies another lapply over this huge list for each variable (5 times).
- String-based key lookups (`paste`, `setNames`) and repeated R-level loops cause memory thrashing and high overhead.

The Random Forest inference is not the problem; the pre-model feature construction is.

---

**Optimization Strategy**  
1. **Avoid per-row neighbor lookup**: Instead of creating a 6.46M-element list, join neighbor relationships in a long table and aggregate.
2. **Leverage `data.table` for fast joins and aggregations**:
   - Expand neighbor relationships to `(cell_id, neighbor_id)` pairs.
   - Cross join with years so both have `(id, year)`.
   - Merge once with source variable values, then compute `max`, `min`, `mean` in a grouped fashion.
3. **Vectorize computations**: Do all five variables in a single pass if possible, using `data.table` grouping.
4. **Memory management**: Process one variable at a time if memory is tight.

---

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Assume columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer neighbor IDs for each id in id_order

# Build neighbor pairs
neighbor_dt <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Expand with years (cartesian join)
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# Key for fast joins
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(var_name) {
  temp <- merge(neighbor_dt, cell_data[, .(neighbor_id = id, year, val = get(var_name))],
                by = c("neighbor_id", "year"), all.x = TRUE)
  
  # Aggregate by (id, year)
  temp_stats <- temp[, .(
    max_val = if (.N == 0 || all(is.na(val))) NA_real_ else max(val, na.rm = TRUE),
    min_val = if (.N == 0 || all(is.na(val))) NA_real_ else min(val, na.rm = TRUE),
    mean_val = if (.N == 0 || all(is.na(val))) NA_real_ else mean(val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Rename columns
  setnames(temp_stats, c("id", "year", "max_val", "min_val", "mean_val"),
           c("id", "year",
             paste0(var_name, "_nbr_max"),
             paste0(var_name, "_nbr_min"),
             paste0(var_name, "_nbr_mean")))
  
  temp_stats
}

# Compute for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (v in neighbor_source_vars) {
  stats_dt <- compute_neighbor_stats_fast(v)
  cell_data <- merge(cell_data, stats_dt, by = c("id", "year"), all.x = TRUE)
}
```

---

**Why this is faster**  
- Eliminates 6.46M-element nested lists.
- Uses efficient keyed joins and group aggregations in C via `data.table`.
- Reduces complexity from O(N * avg_neighbors) per variable in R loops to a single vectorized join and group.

**Expected performance gain**  
From 86+ hours to a few hours (or less) on a standard laptop, given `data.table`’s ability to handle millions of rows efficiently.

**Preserves**  
- Original estimand.
- Trained Random Forest model.
- Same neighbor-based features (max, min, mean per neighbor set).