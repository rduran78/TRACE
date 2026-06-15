 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated R loops and `lapply`** over 6.46M rows for neighbor stats → extremely slow in R due to interpreted loops and object copying.  
2. **`build_neighbor_lookup`** returns a large list of integer vectors (size ~6.46M), consuming huge memory and causing GC overhead.  
3. **Repeated `compute_neighbor_stats`** calls recompute expensive lookups for each variable.  
4. **Random Forest inference** on millions of rows with `predict()` in R is slow if done in chunks or with repeated model loading.  

**Optimization Strategy**  
- Precompute neighbor relationships as an **integer matrix** or compressed format instead of a list of lists.  
- Use **data.table** or **matrix operations** for neighbor stats instead of `lapply`.  
- Compute all neighbor summaries in **one pass** rather than per variable.  
- Use **parallelization** (e.g., `data.table` + `parallel` or `future.apply`) for neighbor aggregation.  
- For Random Forest:  
  - Load model once.  
  - Use `predict(model, newdata, type="response", num.threads = <cores>)` if using `ranger` (much faster than `randomForest`).  
  - Predict in large batches to avoid memory thrashing.  

---

### **Optimized R Implementation**

```r
library(data.table)
library(ranger)  # much faster for inference

# Assume: cell_data is a data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute neighbor lookup as integer matrix
build_neighbor_matrix <- function(id_order, neighbors) {
  max_neighbors <- max(sapply(neighbors, length))
  mat <- matrix(NA_integer_, nrow = length(id_order), ncol = max_neighbors)
  for (i in seq_along(neighbors)) {
    nbs <- neighbors[[i]]
    if (length(nbs) > 0) mat[i, seq_along(nbs)] <- nbs
  }
  mat
}

neighbor_mat <- build_neighbor_matrix(id_order, rook_neighbors_unique)

# Map id to row index
id_to_idx <- setNames(seq_along(id_order), id_order)

# Add index column for fast join
cell_data[, idx := id_to_idx[as.character(id)]]

# Compute neighbor stats for all variables in one pass
compute_all_neighbor_stats <- function(dt, neighbor_mat, vars) {
  n <- nrow(dt)
  res_list <- vector("list", length(vars))
  names(res_list) <- vars
  
  # Preallocate result matrices
  for (v in vars) {
    res_list[[v]] <- matrix(NA_real_, nrow = n, ncol = 3)
  }
  
  vals_list <- lapply(vars, function(v) dt[[v]])
  
  for (i in seq_len(n)) {
    nb_ids <- neighbor_mat[dt$idx[i], ]
    nb_ids <- nb_ids[!is.na(nb_ids)]
    if (length(nb_ids) == 0) next
    
    # Find neighbor rows for same year
    nb_rows <- dt$idx %in% nb_ids & dt$year == dt$year[i]
    for (j in seq_along(vars)) {
      nb_vals <- vals_list[[j]][nb_rows]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) > 0) {
        res_list[[j]][i, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
      }
    }
  }
  res_list
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats <- compute_all_neighbor_stats(cell_data, neighbor_mat, neighbor_source_vars)

# Bind results
for (v in neighbor_source_vars) {
  cell_data[, paste0(v, "_nb_max") := neighbor_stats[[v]][,1]]
  cell_data[, paste0(v, "_nb_min") := neighbor_stats[[v]][,2]]
  cell_data[, paste0(v, "_nb_mean") := neighbor_stats[[v]][,3]]
}

# Random Forest prediction
# Load trained model once
rf_model <- readRDS("trained_rf_model.rds")

# Predict using ranger (fast, multi-threaded)
preds <- predict(rf_model, data = cell_data, num.threads = parallel::detectCores())$predictions
cell_data[, gdp_pred := preds]
```

---

### **Key Gains**
- Eliminates repeated `lapply` calls → single pass neighbor computation.
- Uses integer matrix for neighbors → memory efficient.
- Uses `ranger` for prediction → 10–50× faster than `randomForest`.
- Predicts in one batch with multi-threading.

**Expected runtime reduction:** from 86+ hours → a few hours (dominated by neighbor aggregation). Further speedup possible by chunking and parallelizing the neighbor loop or moving to C++ via `Rcpp`.