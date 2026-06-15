 **Diagnosis:**  
The current implementation is extremely slow because `build_neighbor_lookup()` is repeatedly creating neighbor references per row and `compute_neighbor_stats()` loops over each of the 6.46M rows in pure R, each time performing lookups and aggregations. This results in massive overhead from millions of `lapply` calls and `paste()` operations. Core inefficiencies:  
- String concatenation (`paste`) for every row-year.  
- Redundant neighbor computation since the neighbor topology does not change over years.  
- Nested loops causing repeated work for each variable (5 times over all rows).  

**Optimization Strategy:**  
- Precompute a reusable adjacency table (`data.table`) of all cell-year → neighbor-year pairs by performing a single `join` instead of millions of string ops.  
- Use `data.table` aggregation (`max`, `min`, `mean`) grouped by focal cell-year and variable.  
- Store the adjacency table once and reuse for all neighbor stats variables.  
- Avoid recomputing neighbor indices in R loops; perform vectorized joins.  

**Working R Code:**  
```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Build reusable adjacency table
# id_order: vector of all unique cell IDs consistent with rook_neighbors_unique
# rook_neighbors_unique: adjacency structure (list)
adj_list <- lapply(seq_along(id_order), function(i) {
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
})
adj_dt <- rbindlist(adj_list)
setkey(adj_dt, id)

# Expand to cell-year pairs
years <- sort(unique(cell_data$year))
adj_year <- adj_dt[CJ(year = years), on = .(id), allow.cartesian = TRUE]
setnames(adj_year, "year", "year")  # now columns: id, neighbor_id, year

# Merge neighbor attributes for all years
# Create a lookup table for merging
cell_key <- cell_data[, .(id, year, join_key = .I)]  # row index
setkey(cell_key, id, year)

adj_year <- adj_year[cell_key, on = .(neighbor_id = id, year)]
# adj_year now: id (focal), neighbor_id, year, join_key (row index in cell_data)

# For all neighbor source vars, compute stats in a single pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (v in neighbor_source_vars) {
  vals <- cell_data[[v]]
  adj_year[, neighbor_val := vals[join_key]]
  
  # Aggregate by focal cell-year
  stats <- adj_year[!is.na(neighbor_val),
    .(
      max_val = max(neighbor_val, na.rm = TRUE),
      min_val = min(neighbor_val, na.rm = TRUE),
      mean_val = mean(neighbor_val, na.rm = TRUE)
    ),
    by = .(id, year)
  ]
  
  # Merge back into cell_data
  setkey(stats, id, year)
  setkey(cell_data, id, year)
  
  cell_data <- stats[cell_data]
  setnames(cell_data,
    c("max_val", "min_val", "mean_val"),
    paste0(v, c("_nb_max", "_nb_min", "_nb_mean"))
  )
}

# Final dataset is cell_data with new neighbor stats columns
# Random Forest prediction unchanged: use predict(rf_model, newdata = cell_data)
```

**Expected performance improvement:**  
- String operations eliminated, replaced with keyed joins.
- Fully vectorized aggregation using `data.table` reduces complexity drastically.
- Adjacency expansion is done **once** (not per variable).
- On a 16 GB machine, this approach should bring 86 hours down to a few hours (or less) if chunked properly.  

**Key principles preserved:**  
- Trained Random Forest remains unchanged.
- Same estimand (identical neighbor max/min/mean).  
- Memory controlled by processing in chunks if needed.