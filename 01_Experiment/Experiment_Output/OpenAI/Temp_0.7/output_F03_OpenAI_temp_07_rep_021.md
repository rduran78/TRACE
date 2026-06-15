 **Diagnosis**  
The main bottleneck is the repeated use of `lapply` and repeated object copying when building neighbor features for 6.46M rows. The current workflow:  
- `build_neighbor_lookup` constructs a massive nested list of integer indexes using `lapply` over millions of rows, which is very memory- and CPU-intensive.  
- `compute_neighbor_stats` repeatedly traverses these lists and performs dynamic allocations (`c()`, `rbind`) for each variable.  
- These steps dominate runtime before Random Forest inference even begins.  
- Additional overhead likely occurs during prediction if data are split into small chunks and model loading is repeated unnecessarily.  

**Optimization Strategy**  
1. **Precompute neighbor stats in a vectorized/data.table way** instead of per-row `lapply`.  
2. **Avoid building a huge list**; instead, melt the neighbor relationships into a long format (cell-year → neighbor-year) and join once.  
3. **Use `data.table` for grouping and aggregation** to compute max, min, mean efficiently.  
4. **Compute all neighbor features in one pass** rather than looping five times.  
5. **Keep the Random Forest model loaded only once** and predict in large batches.  
6. **Ensure numeric columns, avoid copies** by using `set()` in `data.table`.  

**Working R Code (Optimized)**  

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of neighbor IDs per cell (spdep::nb)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Expand neighbor relationships once
# Create a table of (id, neighbor_id)
neighbor_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = i, neighbor_id = rook_neighbors_unique[[i]])
  })
)

# Cross join with years to get full panel neighbor pairs
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# Merge source and neighbor values
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

long_dt <- merge(
  neighbor_dt,
  cell_data[, c("id", "year", vars), with = FALSE],
  by.x = c("neighbor_id", "year"),
  by.y = c("id", "year"),
  all.x = TRUE
)

# Aggregate neighbor stats for each (id, year)
agg_dt <- long_dt[, c(
  lapply(.SD, max, na.rm = TRUE),
  lapply(.SD, min, na.rm = TRUE),
  lapply(.SD, mean, na.rm = TRUE)
), by = .(id, year), .SDcols = vars]

# Rename columns
setnames(agg_dt,
         old = names(agg_dt)[-(1:2)],
         new = c(
           paste0(vars, "_nb_max"),
           paste0(vars, "_nb_min"),
           paste0(vars, "_nb_mean")
         ))

# Merge back into cell_data
cell_data <- merge(cell_data, agg_dt, by = c("id", "year"), all.x = TRUE)

# Replace NaN from empty neighbor sets with NA
for (col in names(cell_data)) {
  if (is.numeric(cell_data[[col]])) {
    set(cell_data, which(is.nan(cell_data[[col]])), col, NA_real_)
  }
}

# ---- Random Forest Prediction ----
# rf_model is pre-loaded Random Forest model
# Predict in chunks to manage memory
chunk_size <- 1e6
n <- nrow(cell_data)
pred <- numeric(n)
for (i in seq(1, n, by = chunk_size)) {
  j <- min(i + chunk_size - 1, n)
  pred[i:j] <- predict(rf_model, newdata = cell_data[i:j])
}
cell_data[, gdp_pred := pred]
```

**Key Gains:**  
- Eliminates per-row loops in favor of a single vectorized aggregation.  
- Avoids massive list creation (`neighbor_lookup`).  
- Computes all neighbor stats in one pass using `data.table`.  
- Loads model once; prediction in large batches minimizes overhead.  

**Expected improvement:** Hours → minutes (depending on disk I/O), with memory footprint manageable on 16 GB RAM.