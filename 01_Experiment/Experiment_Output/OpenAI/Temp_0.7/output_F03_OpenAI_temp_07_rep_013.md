 **Diagnosis**  
The primary bottlenecks are:  
- **Repeated R loops and `lapply` over millions of rows** in `build_neighbor_lookup` and `compute_neighbor_stats`. These are extremely slow for 6.46M rows.  
- **Inefficient neighbor-stat computation**: repeated `paste`, repeated lookups, and repeated object copying.  
- **Prediction loop**: Random Forest predictions are likely happening in chunks or row-by-row rather than vectorized.  
- **Memory overhead**: constructing large lists of neighbors and repeatedly appending to `data.frame` causes huge overhead.  

---

### **Optimization Strategy**
1. **Precompute neighbor relationships in a long format table** (cell-year → neighbor-year) using vectorized joins instead of loops.
2. **Use `data.table` for fast grouping and aggregation** instead of `lapply`.
3. **Compute all neighbor stats for all variables in one go** using melt/cast operations.
4. **Vectorize Random Forest predictions**: Pass the entire feature matrix (or big chunks) to `predict()`.
5. **Avoid repeated string concatenation (`paste`) and lookups**: use numeric indices for joining.
6. **Minimize copying**: work with `data.table` in place.

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.frame), id_order (vector), rook_neighbors_unique (list), rf_model loaded

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor pairs (id → neighbor_id)
neighbor_pairs <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Expand to panel by joining on years
years <- unique(cell_data$year)
neighbor_panel <- neighbor_pairs[CJ(year = years), on = .(dummy = NULL)]
setnames(neighbor_panel, c("id", "neighbor_id", "year"))

# Merge indices
neighbor_panel[cell_data, on = .(id = id, year = year), idx := .I]
neighbor_panel[cell_data, on = .(neighbor_id = id, year = year), n_idx := .I]

# Drop rows without valid neighbor-cell-year
neighbor_panel <- neighbor_panel[!is.na(idx) & !is.na(n_idx)]

# Compute neighbor stats for all variables at once
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt cell_data for neighbor vars
melted <- melt(cell_data, id.vars = c("id", "year"), measure.vars = neighbor_source_vars)
melted[, cell_row := .I]

# Join neighbors
neighbor_vals <- neighbor_panel[melted, on = .(n_idx = cell_row)]
# neighbor_vals: idx (target row), variable, value

# Aggregate stats
agg <- neighbor_vals[, .(
  n_max = max(value, na.rm = TRUE),
  n_min = min(value, na.rm = TRUE),
  n_mean = mean(value, na.rm = TRUE)
), by = .(idx, variable)]

# Reshape wide
agg_wide <- dcast(agg, idx ~ variable, value.var = c("n_max", "n_min", "n_mean"))

# Join back to cell_data
cell_data <- cbind(cell_data, agg_wide[order(idx)][, -"idx"])

# ---- Random Forest Prediction ----
# Prepare matrix of predictors
predictors <- as.data.frame(cell_data[, setdiff(names(cell_data), c("id", "year", "target_var")), with = FALSE])

# Predict in large chunks for memory safety
chunk_size <- 1e6
n <- nrow(predictors)
preds <- numeric(n)
for (i in seq(1, n, by = chunk_size)) {
  j <- min(i + chunk_size - 1, n)
  preds[i:j] <- predict(rf_model, newdata = predictors[i:j, ])
}

cell_data[, prediction := preds]
```

---

### **Why This Works**
- **`data.table` joins** eliminate costly per-row operations.
- **Single pass aggregation** dramatically reduces overhead.
- **Chunked prediction** avoids memory overflow while maximizing vectorization.
- **No retraining**: The existing Random Forest model is preserved.
- This approach can reduce runtime from **86+ hours to under a few hours** on a 16 GB laptop.

**Next step:** Save intermediate artifacts (`neighbor_panel`, melted data) if needed for incremental runs.