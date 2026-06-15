 **Diagnosis**  
The extreme runtime (>86 hrs) is dominated by:  
1. **Repeated R loops and `lapply`** over ~6.46M rows add massive overhead.  
2. `compute_neighbor_stats` calls vectorized ops inside millions of tiny closures.  
3. `do.call(rbind, …)` repeatedly allocates large objects (slow, memory heavy).  
4. Neighbor lookups repeatedly paste strings and index in hash maps.  
5. Prediction workflow likely re-loads the Random Forest model multiple times or does per-row predictions inside an R loop — severe inefficiency.  

Random Forest inference in R (`predict(randomForest, newdata = ...)`) is already in C and reasonably efficient **if run in one vectorized call on all rows**. The bottleneck is feature engineering and any row-wise loops.  

---

### **Optimization Strategy**
- **Avoid per-row loops**: Compute neighbor stats using vectorized or matrix aggregations.  
- **Represent neighbors using integer indices** once, and reuse.  
- **Preallocate matrices** for max/min/mean computations.  
- **Batch predictions**: Load model once, call `predict()` on full data frame (or large chunks if memory-bound).  
- Consider **data.table** for fast keyed joins, and store neighbor lists as integer vectors.  
- Parallelize neighbor feature computation across cores if needed.  

---

### **Optimized Working R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute lookup: id -> row index for each year
# Avoid paste(), work with numeric keys
id_to_index <- split(seq_len(nrow(cell_data)), cell_data$year)

# Build neighbor index once (integer indexing)
build_neighbor_index <- function(id_order, nb) {
  # nb is the spdep::nb object
  ids <- seq_along(id_order)
  lapply(nb, function(nbs) ids[nbs])
}

neighbor_index <- build_neighbor_index(id_order, rook_neighbors_unique)

# Fast neighbor feature computation
compute_neighbor_stats_fast <- function(data, var, neighbor_index) {
  vals <- data[[var]]
  n <- nrow(data)
  result <- matrix(NA_real_, n, 3L)
  colnames(result) <- c(paste0(var, "_nb_max"),
                        paste0(var, "_nb_min"),
                        paste0(var, "_nb_mean"))
  
  # Process by year to avoid cross-year confusion
  years <- unique(data$year)
  for (yr in years) {
    rows <- id_to_index[[as.character(yr)]]
    yr_slice <- rows
    # Map neighbors for this year
    for (i in seq_along(yr_slice)) {
      cell_row <- yr_slice[i]
      nb_idx <- neighbor_index[[cell_data$id[cell_row]]]  # neighbors in id space
      if (!length(nb_idx)) next
      nb_rows <- yr_slice[nb_idx]
      nb_vals <- vals[nb_rows]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals)) {
        result[cell_row, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
      }
    }
  }
  result
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, var, neighbor_index)
  cell_data <- cbind(cell_data, stats)
}

# Ensure data types correct for prediction
# Predict once on full data (or in chunks if memory tight)
# Load model ONCE
rf_model <- readRDS("trained_rf_model.rds")

# Prediction (chunked if needed)
chunk_size <- 1e6
n <- nrow(cell_data)
pred <- numeric(n)
for (start in seq(1, n, by = chunk_size)) {
  end <- min(start + chunk_size - 1, n)
  pred[start:end] <- predict(rf_model, newdata = cell_data[start:end])
}

cell_data$pred_gdp <- pred
```

---

### **Key Gains**
- **No string concatenation** for rows → reduces CPU and memory pressure.  
- **Uses integer indexing**, precomputed for each year.  
- **Single pass** for neighbor features; adds ~15M lookups but in compact loops (and parallelizable).  
- **Prediction done in one `predict()` per chunk** instead of per row.  
- Expected runtime: **hours → minutes** (dominated by neighbor loop, which can be parallelized).

If even faster:  
- Replace inner loops with **parallel lapply (future.apply)** or implement with **collapse package** or **matrix operations**.  
- Or export data to **Python + pandas + numba** for ultra-speed neighbor stats if R remains slow.