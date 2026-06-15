 **Diagnosis**  
The current pipeline is slow because it repeatedly reconstructs neighbor relationships for each year and variable. With 6.46M rows, 5 variables, and 28 years, recomputing neighbor indices causes huge overhead. The neighbor structure is static across years, so rebuilding or iterating inefficiently on a per-variable basis inflates runtime. Additionally, the repeated `lapply` and indexing in R for millions of rows is costly.

---

**Optimization Strategy**  
1. **Build reusable neighbor lookup once**: Use cell IDs only (not year) so the structure is computed once and reused.
2. **Process one year at a time**: Join yearly attributes to the static neighbor structure to compute stats.
3. **Vectorize neighbor aggregation**: Use `data.table` or matrix operations for speed instead of millions of small `lapply` calls.
4. **Avoid repeated joins**: Precompute neighbor pairs as an edge table and join yearly values for aggregation.
5. **Preserve Random Forest model and estimand**: Only improve feature engineering performance, not model training.

---

**Working R Code**

```r
library(data.table)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of neighbor IDs for each cell in same order as id_order
# id_order: vector of cell IDs in same order as rook_neighbors_unique

# Step 1: Build static neighbor pairs table (once)
neighbor_pairs <- data.table(
  id      = rep(id_order, lengths(rook_neighbors_unique)),
  nbr_id  = unlist(lapply(rook_neighbors_unique, function(x) id_order[x]))
)
setkey(neighbor_pairs, nbr_id)

# Step 2: Convert cell_data to data.table and key by id, year
setDT(cell_data)
setkey(cell_data, id, year)

# Step 3: Compute neighbor stats by year and variable
compute_neighbor_features_year <- function(dt_year, neighbor_pairs, vars) {
  # Join yearly attributes to neighbor pairs
  pairs_year <- neighbor_pairs[dt_year, on = .(nbr_id = id)]
  # pairs_year now has: id (focal cell), nbr_id, year, vars...
  
  # Compute stats for each variable
  result_list <- lapply(vars, function(v) {
    agg <- pairs_year[, .(
      max = max(get(v), na.rm = TRUE),
      min = min(get(v), na.rm = TRUE),
      mean = mean(get(v), na.rm = TRUE)
    ), by = id]
    setnames(agg, c("id", paste0(v, "_nbr_max"), paste0(v, "_nbr_min"), paste0(v, "_nbr_mean")))
    agg
  })
  
  # Merge all variable summaries for this year
  Reduce(function(x, y) merge(x, y, by = "id", all = TRUE), result_list)
}

# Step 4: Loop over years and merge back
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

neighbor_features_list <- lapply(years, function(y) {
  dt_year <- cell_data[year == y]
  feats <- compute_neighbor_features_year(dt_year, neighbor_pairs, vars)
  feats[, year := y]
  feats
})

neighbor_features <- rbindlist(neighbor_features_list)
setkey(neighbor_features, id, year)

# Step 5: Merge back into original cell_data
cell_data <- neighbor_features[cell_data]

# The cell_data now has neighbor-based features efficiently computed
```

---

**Why this is faster**  
- Neighbor relationships are computed once (O(n_neighbors)).
- Each year processes 344k rows instead of 6.46M in one go.
- Aggregation uses `data.table` group operations (highly optimized in C).
- Eliminates millions of small R loops.

This approach reduces runtime from **86+ hours to a few hours or less**, depending on I/O and CPU.