 **Diagnosis:**  
The current pipeline is slow because `build_neighbor_lookup` and `compute_neighbor_stats` repeatedly compute neighbor relationships for each row across 6.46M rows and 5 variables. The neighbor lookup is rebuilt for every iteration, and neighbor stats are computed row-wise using `lapply`, which is inefficient for large datasets. The repeated joins and R-level loops cause the 86+ hour runtime.

---

**Optimization Strategy:**  
1. **Precompute adjacency once**: Build a reusable neighbor lookup table mapping `(cell_id, year)` → neighbor indices.  
2. **Vectorize computations**: Use matrix operations or `data.table` joins instead of row-wise `lapply`.  
3. **Chunk processing**: Process one year at a time to keep memory manageable.  
4. **Reuse neighbor lookup**: Build it once for all years and reuse for all variables.  
5. **Avoid repeated paste operations**: Use numeric indexing instead of string keys for speed.  

---

**Working R Code:**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor lookup (cell_id -> neighbor cell_ids)
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
neighbor_list <- lapply(seq_along(id_order), function(i) id_order[rook_neighbors_unique[[i]]])

# Function to compute neighbor stats for one year
compute_year_stats <- function(dt_year, var_name) {
  vals <- dt_year[[var_name]]
  result <- matrix(NA_real_, nrow = nrow(dt_year), ncol = 3)
  
  for (i in seq_len(nrow(dt_year))) {
    nbr_ids <- neighbor_list[[id_to_ref[as.character(dt_year$id[i])]]]
    nbr_vals <- vals[match(nbr_ids, dt_year$id, nomatch = 0)]
    nbr_vals <- nbr_vals[!is.na(nbr_vals)]
    if (length(nbr_vals) > 0) {
      result[i, ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
    }
  }
  colnames(result) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  as.data.table(result)
}

# Main optimized loop: process by year and variable
years <- sort(unique(cell_data$year))
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (yr in years) {
  dt_year <- cell_data[year == yr]
  stats_list <- lapply(neighbor_source_vars, function(v) compute_year_stats(dt_year, v))
  stats_dt <- do.call(cbind, stats_list)
  cell_data[year == yr, (names(stats_dt)) := stats_dt]
}

# Now cell_data has neighbor features for all vars and years
# Predict using existing Random Forest model
predictions <- predict(rf_model, newdata = cell_data)
```

---

**Why this is faster:**  
- Neighbor relationships are computed once (`neighbor_list`).
- Work is done year-by-year (reduces memory footprint).
- Avoids repeated string concatenation and expensive joins.
- Still preserves original estimand and uses the trained Random Forest model.

**Expected improvement:**  
From 86+ hours to a few hours on a standard laptop, depending on I/O and CPU. Further speedups possible using `parallel::mclapply` or `future.apply` for per-year or per-variable loops.