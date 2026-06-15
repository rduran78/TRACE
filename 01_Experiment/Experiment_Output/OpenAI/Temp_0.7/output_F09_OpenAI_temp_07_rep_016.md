 **Diagnosis**  
The current pipeline rebuilds neighbor relationships for every iteration and loops inefficiently over 6.46M rows and 28 years. For each neighbor source variable, `compute_neighbor_stats` iterates through every cell-year, repeatedly scanning vectors, which is extremely costly given the dataset size. This explains the 86+ hour runtime.

**Optimization Strategy**  
- Build a **single reusable adjacency table** mapping each cell to its neighbors once.
- Join yearly attributes to this static adjacency structure for all years, rather than recomputing neighbor relationships repeatedly.
- Use **vectorized joins with `data.table`** for speed and memory efficiency.
- Compute neighbor statistics in a grouped manner instead of per-row `lapply`.

---

### **Optimized Implementation in R**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Build adjacency table once
build_neighbor_table <- function(id_order, neighbors) {
  # Create a data.table of all neighbor pairs
  from_ids <- rep(id_order, lengths(neighbors))
  to_ids   <- unlist(neighbors)
  neighbor_table <- data.table(
    id = from_ids,
    neighbor_id = id_order[to_ids]
  )
  return(neighbor_table)
}

neighbor_table <- build_neighbor_table(id_order, rook_neighbors_unique)

# Expand adjacency table across all years
years <- unique(cell_data$year)
neighbor_table_expanded <- neighbor_table[
  , .(neighbor_id = neighbor_id, year = years), by = .(id)
]

# Prepare key for fast joins
setkey(cell_data, id, year)
setkey(neighbor_table_expanded, neighbor_id, year)

# Merge neighbor attributes for each variable and compute stats
compute_neighbor_features <- function(data, neighbor_table_expanded, var_name) {
  # Join neighbor attribute
  merged <- neighbor_table_expanded[data, on = .(neighbor_id = id, year), nomatch = 0]
  # merged now has: id (focal), neighbor_id, year, var_name
  merged <- merged[!is.na(get(var_name))]
  
  # Compute neighbor stats by focal id-year
  stats <- merged[, .(
    max_val = max(get(var_name), na.rm = TRUE),
    min_val = min(get(var_name), na.rm = TRUE),
    mean_val = mean(get(var_name), na.rm = TRUE)
  ), by = .(id, year)]
  
  # Rename columns
  setnames(stats,
           c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  
  # Join back to main data
  data <- stats[data, on = .(id, year)]
  return(data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_neighbor_features(cell_data, neighbor_table_expanded, var_name)
}

# cell_data now includes neighbor-based features for Random Forest prediction
```

---

### **Why This is Faster**
- **Single adjacency table** avoids recomputation.
- **Vectorized joins and grouped aggregation** replace millions of small `lapply` calls.
- `data.table` uses efficient hashing and in-memory operations, suitable for 6.46M rows on 16 GB RAM.
- Expected runtime reduction from 86+ hours to **well under an hour** (I/O and join dominated).

**Preserves**:
- The trained Random Forest model (unchanged).
- Original estimand and feature definitions.

You can now feed `cell_data` with the new features into `predict(trained_rf_model, newdata = cell_data)`.