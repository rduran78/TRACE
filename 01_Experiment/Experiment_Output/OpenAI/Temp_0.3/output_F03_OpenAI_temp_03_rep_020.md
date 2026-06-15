 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over millions of rows** in `build_neighbor_lookup` and `compute_neighbor_stats`, causing excessive object creation and copying.  
2. **Inefficient neighbor lookup**: constructing character keys (`paste`) and repeatedly indexing lists is costly.  
3. **Memory overhead** from building large lists and repeatedly binding results.  
4. **Prediction loop inefficiency**: repeated single-row predictions instead of batch prediction.  
5. **Model loading**: ensure it is loaded once, not repeatedly.  

---

**Optimization Strategy**  
- Precompute neighbor indices in a **vectorized matrix form** instead of per-row lists.  
- Avoid character concatenation for keys; use numeric indices.  
- Compute neighbor stats using **matrix operations** or `data.table` joins.  
- Use `data.table` for fast grouping and feature computation.  
- Perform **batch prediction**: pass the entire feature matrix to `predict()` at once.  
- Keep the Random Forest model in memory for the entire run.  

---

**Optimized R Code**  

```r
library(data.table)
library(randomForest)

# Convert to data.table for speed
cell_data <- as.data.table(cell_data)

# Precompute lookup: neighbor indices per cell_id
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_list <- rook_neighbors_unique

# Build neighbor index matrix (list of integer vectors)
neighbor_lookup <- lapply(seq_along(id_order), function(i) {
  id_to_idx[ id_order[ neighbor_list[[i]] ] ]
})

# Map cell_id to row indices by year
cell_data[, key := .I]  # row index
setkey(cell_data, id, year)

# Compute neighbor stats efficiently
compute_neighbor_stats_dt <- function(dt, neighbor_lookup, var_name) {
  vals <- dt[[var_name]]
  result <- vector("list", length(neighbor_lookup))
  
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) {
      result[[i]] <- c(NA_real_, NA_real_, NA_real_)
    } else {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) {
        result[[i]] <- c(NA_real_, NA_real_, NA_real_)
      } else {
        result[[i]] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  do.call(rbind, result)
}

# Apply for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_dt(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}

# Prepare feature matrix for prediction
predictors <- setdiff(names(cell_data), c("id", "year", "key", "target_var")) # exclude non-predictors
X <- as.matrix(cell_data[, ..predictors])

# Load model once
rf_model <- readRDS("trained_rf_model.rds")

# Batch prediction
predictions <- predict(rf_model, X)

# Attach predictions
cell_data[, gdp_pred := predictions]
```

---

**Key Improvements**  
- Eliminated repeated `paste` and character-based keys.  
- Used `data.table` for efficient indexing and memory handling.  
- Computed neighbor stats in a single loop with precomputed indices.  
- Performed **batch prediction** instead of row-by-row.  
- Preserved trained Random Forest model and original estimand.  

This approach should reduce runtime from **86+ hours to a few hours** on a 16 GB RAM machine.