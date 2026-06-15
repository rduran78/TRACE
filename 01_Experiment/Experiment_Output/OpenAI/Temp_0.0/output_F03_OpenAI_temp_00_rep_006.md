 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small vectors and lists. This is extremely slow and memory-heavy.  
2. **Excessive object copying**: Each `compute_and_add_neighbor_features` likely copies `cell_data` repeatedly.  
3. **Inefficient neighbor lookup**: Current approach builds a full list of neighbors for every row, even though neighbors are static across years.  
4. **Random Forest inference**: If predictions are done row-by-row, this is very slow. `predict()` should be vectorized on the full data frame or large chunks.  
5. **Memory pressure**: 6.46M rows × 110+ predictors is large; avoid intermediate large lists.  

---

**Optimization Strategy**  
- **Precompute neighbor relationships once per cell (not per row)** and reuse across years.  
- **Vectorize neighbor feature computation** using `data.table` or `dplyr` joins instead of `lapply`.  
- **Chunked or full-batch prediction**: Use `predict(rf_model, newdata, type="response")` on the entire dataset or in large chunks.  
- **Avoid repeated copies**: Compute all neighbor stats in one pass and `cbind` results.  
- **Use `data.table` for speed and memory efficiency**.  

---

**Optimized R Code** (using `data.table`):

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.frame), id_order, rook_neighbors_unique, rf_model loaded

setDT(cell_data)  # convert to data.table
setkey(cell_data, id, year)

# Precompute neighbor relationships once
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand neighbor relationships across years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# Join neighbor values for all variables in one go
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt cell_data for easier join
long_data <- melt(cell_data, id.vars = c("id", "year"), measure.vars = vars,
                  variable.name = "var", value.name = "val")

# Join neighbors
neighbor_vals <- merge(neighbor_dt, long_data, by.x = c("neighbor_id", "year"),
                       by.y = c("id", "year"), allow.cartesian = TRUE)

# Compute stats by (id, year, var)
neighbor_stats <- neighbor_vals[, .(
  max_val = max(val, na.rm = TRUE),
  min_val = min(val, na.rm = TRUE),
  mean_val = mean(val, na.rm = TRUE)
), by = .(id, year, var)]

# Reshape wide
neighbor_stats_wide <- dcast(neighbor_stats, id + year ~ var,
                              value.var = c("max_val", "min_val", "mean_val"))

# Merge back to cell_data
cell_data <- merge(cell_data, neighbor_stats_wide, by = c("id", "year"), all.x = TRUE)

# Random Forest prediction in chunks (if memory is tight)
chunk_size <- 500000
n <- nrow(cell_data)
preds <- numeric(n)

for (i in seq(1, n, by = chunk_size)) {
  idx <- i:min(i + chunk_size - 1, n)
  preds[idx] <- predict(rf_model, newdata = cell_data[idx])
}

cell_data[, prediction := preds]
```

---

**Why this is faster**  
- Eliminates per-row `lapply` loops (millions of iterations → vectorized joins).  
- Computes all neighbor stats in one grouped aggregation using `data.table`.  
- Predicts in large chunks instead of row-by-row.  
- Avoids repeated copying of `cell_data`.  

**Expected improvement**: From 86+ hours to a few hours or less on a 16 GB laptop.