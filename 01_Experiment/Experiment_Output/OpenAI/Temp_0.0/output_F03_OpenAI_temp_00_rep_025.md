 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small vectors and lists. This is extremely slow and memory-heavy.  
2. **Excessive object copying**: Each `compute_and_add_neighbor_features` call copies `cell_data`.  
3. **Inefficient neighbor lookup**: Current approach builds a large list of integer vectors for every row, which is costly in both time and memory.  
4. **Random Forest inference**: If predictions are done in a loop (row-by-row), this is catastrophic. `predict()` should be vectorized over the entire dataset or large chunks.  
5. **Memory pressure**: 6.46M rows × 110+ features is large but manageable if processed in chunks and without redundant copies.  

---

**Optimization Strategy**  
- **Precompute neighbor relationships in a long format** (edge list) and join instead of building a huge list.  
- **Use `data.table` for fast joins and aggregations**.  
- **Compute neighbor stats in a single grouped operation** rather than per-row `lapply`.  
- **Avoid repeated copying of `cell_data`**; compute all neighbor features in one pass.  
- **Batch Random Forest predictions**: Use `predict(model, newdata, type="response")` on the full dataset or in chunks (e.g., 500k rows).  
- **Keep everything in memory-efficient structures**.  

---

**Working R Code (Optimized)**  

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.frame), id_order, rook_neighbors_unique, rf_model loaded

# Convert to data.table
setDT(cell_data)

# Create a unique key for cell-year
cell_data[, cell_year := paste(id, year, sep = "_")]

# Build neighbor edge list (id-level)
edges <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Expand to cell-year level by joining on year
years <- unique(cell_data$year)
edges_expanded <- edges[, .(id = from, neighbor_id = to)][
  , .(id = rep(id, length(years)), neighbor_id = rep(neighbor_id, length(years)), year = years)
]

# Join to get cell_year keys
edges_expanded[, cell_year := paste(id, year, sep = "_")]
edges_expanded[, neighbor_cell_year := paste(neighbor_id, year, sep = "_")]

# Map neighbor indices
edges_expanded <- merge(edges_expanded, cell_data[, .(neighbor_cell_year = cell_year, row_idx = .I)], 
                        by = "neighbor_cell_year", all.x = TRUE)

# Drop NAs (neighbors that don't exist for that year)
edges_expanded <- edges_expanded[!is.na(row_idx)]

# Compute neighbor stats for all variables in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  # Map neighbor values
  edges_expanded[, val := cell_data[[var]][row_idx]]
  
  # Aggregate stats by focal cell_year
  stats <- edges_expanded[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = cell_year]
  
  # Merge back to cell_data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- merge(cell_data, stats, by = "cell_year", all.x = TRUE)
}

# Drop helper columns
cell_data[, cell_year := NULL]

# Random Forest prediction in chunks
chunk_size <- 500000
n <- nrow(cell_data)
preds <- numeric(n)

for (i in seq(1, n, by = chunk_size)) {
  idx <- i:min(i + chunk_size - 1, n)
  preds[idx] <- predict(rf_model, newdata = cell_data[idx])
}

cell_data[, gdp_pred := preds]
```

---

**Why this is faster**  
- Eliminates per-row `lapply` and repeated copying.  
- Uses `data.table` joins and group operations (highly optimized in C).  
- Computes all neighbor features in one pass per variable.  
- Predicts in large chunks instead of row-by-row.  

**Expected improvement**: From 86+ hours to a few hours (or less) on a 16 GB laptop.