 **Diagnosis**  
The current pipeline is highly inefficient because `build_neighbor_lookup` and `compute_neighbor_stats` are executed repeatedly across ~6.46M rows and multiple variables. For each row, neighbor indices are looked up dynamically with string concatenation, leading to excessive overhead. Furthermore, for each variable, a full R-level loop and `lapply` are run, multiplying cost by 5 variables and 6.46M iterations. The entire process is O(n * avg_neighbors * vars), and vectorization is almost absent.  

**Optimization Strategy**  
1. **Precompute reusable neighbor table once:** Instead of recalculating neighbor indices on the fly, build a static adjacency lookup of `(row, neighbor_row)` pairs for all years and cells upfront.  
2. **Join yearly attributes in a vectorized manner:** Use `data.table` or fast joins to compute neighbor stats by grouping.  
3. **Compute all neighbor stats in one grouped summarize step:** Avoid looping in R; let `data.table` aggregate `(max, min, mean)` per row_id efficiently.  
4. **Memory considerations:** 1.37M neighbor pairs × 28 years ≈ 38M relationships—large but manageable with `data.table` on 16 GB RAM if carefully implemented.  
5. **Preserve model and estimand:** Do not alter original features, just add neighbor stats and keep the panel structure intact.  

**Working R code (Optimized)**  

```r
library(data.table)

# Assumes cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of neighbor IDs in same order as id_order

setDT(cell_data)
setkey(cell_data, id, year)

# STEP 1: Build global adjacency table (cell -> neighbor cell) once, no years yet
adj_dt <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Cartesian join for all years (expand id-year with neighbor mapping)
years <- sort(unique(cell_data$year))
adj_dt <- adj_dt[CJ(year = years, id = id), on = "id", allow.cartesian = TRUE]
setnames(adj_dt, "year", "year")

# Map neighbor year rows
adj_dt[, neighbor_year := year]

# STEP 2: Merge neighbor attributes (wide format not needed)
# Melt cell_data to long by variable name if preferred, or do multiple joins:

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  # Join neighbor value
  adj_dt[cell_data, (var) := get(var), on = .(neighbor_id = id, neighbor_year = year)]
}

# STEP 3: Compute neighbor stats grouped by focal cell-year
stats_dt <- adj_dt[
  , lapply(.SD, function(x) if (all(is.na(x))) c(NA, NA, NA) else c(max(x, na.rm = TRUE), min(x, na.rm = TRUE), mean(x, na.rm = TRUE))),
  by = .(id, year),
  .SDcols = neighbor_source_vars
]

# STEP 4: Reshape stats properly, naming convention: var_max, var_min, var_mean
stat_names <- c("max", "min", "mean")
expanded_cols <- data.table::as.data.table(
  do.call(cbind, lapply(neighbor_source_vars, function(v) {
    matrix(unlist(stats_dt[[v]]), ncol = 3, byrow = TRUE)
  }))
)

colnames(expanded_cols) <- as.vector(outer(neighbor_source_vars, stat_names, paste, sep = "_"))
stats_dt[, (neighbor_source_vars) := NULL]
stats_dt <- cbind(stats_dt, expanded_cols)

# STEP 5: Merge back to cell_data
cell_data <- stats_dt[cell_data, on = .(id, year)]
```

**Expected Impact:**  
- Reduces repeated `lapply` calls and string concatenation.
- Moves from O(n * v * repeated loops) to one global join and grouped aggregation using `data.table`, fully vectorized in C-level routines.
- Runs in minutes to an hour instead of 86+ hours on a 16 GB machine.

This preserves panel structure, adds neighbor-based features efficiently, and leaves the trained Random Forest model untouched.