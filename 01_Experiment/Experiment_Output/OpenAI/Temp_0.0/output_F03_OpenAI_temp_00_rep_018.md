 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small R objects. This is extremely slow and memory-inefficient.  
2. **Excessive object copying**: Each `compute_and_add_neighbor_features` likely copies `cell_data` repeatedly.  
3. **Inefficient neighbor lookup**: `lapply` with string concatenation (`paste`) for every row is costly.  
4. **Random Forest prediction**: If predictions are done in small chunks or repeatedly loading the model, this adds overhead.  
5. **Memory pressure**: 6.46M rows × 110+ variables is large; repeated intermediate objects exacerbate RAM usage.  

---

**Optimization Strategy**  
- **Precompute neighbor indices as integer vectors** (avoid string keys).  
- **Vectorize neighbor stats computation** using `data.table` or `matrix` operations instead of millions of `lapply` calls.  
- **Avoid repeated copies**: compute all neighbor features in one pass.  
- **Batch Random Forest predictions**: load model once, predict in large chunks.  
- **Use `data.table` for fast joins and memory efficiency**.  

---

**Optimized R Code**  

```r
library(data.table)
library(randomForest)

# Convert to data.table for speed
setDT(cell_data)

# Precompute a fast lookup: map (id, year) -> row index
cell_data[, key := .I]  # row index
id_year_key <- cell_data[, .(id, year, key)]
setkey(id_year_key, id)

# Build neighbor lookup as integer indices (no string concatenation)
build_neighbor_lookup_fast <- function(id_order, neighbors, id_year_key) {
  # neighbors: list of integer vectors (rook neighbors)
  # id_order: vector of ids in same order as neighbors
  id_to_idx <- setNames(seq_along(id_order), id_order)
  
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    # Return neighbor ids (not row indices yet)
    neighbor_ids
  })
}

neighbor_lookup_ids <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique, id_year_key)

# Compute all neighbor stats in one pass
compute_all_neighbor_stats <- function(cell_data, neighbor_lookup_ids, vars) {
  n <- nrow(cell_data)
  result_list <- vector("list", length(vars))
  
  for (v in seq_along(vars)) {
    var_name <- vars[v]
    vals <- cell_data[[var_name]]
    
    # Preallocate matrix: rows = n, cols = 3 (max, min, mean)
    mat <- matrix(NA_real_, n, 3)
    
    for (i in seq_along(neighbor_lookup_ids)) {
      # Get all rows for this id across years
      id_rows <- which(cell_data$id == id_order[i])
      for (r in id_rows) {
        year <- cell_data$year[r]
        # Find neighbors for same year
        neighbor_ids <- neighbor_lookup_ids[[i]]
        neighbor_rows <- id_year_key[J(neighbor_ids)][year == year]$key
        if (length(neighbor_rows) > 0) {
          neighbor_vals <- vals[neighbor_rows]
          neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
          if (length(neighbor_vals) > 0) {
            mat[r, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
          }
        }
      }
    }
    result_list[[v]] <- mat
  }
  result_list
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats <- compute_all_neighbor_stats(cell_data, neighbor_lookup_ids, neighbor_source_vars)

# Bind results efficiently
for (i in seq_along(neighbor_source_vars)) {
  var <- neighbor_source_vars[i]
  cell_data[, paste0(var, "_nbr_max") := neighbor_stats[[i]][, 1]]
  cell_data[, paste0(var, "_nbr_min") := neighbor_stats[[i]][, 2]]
  cell_data[, paste0(var, "_nbr_mean") := neighbor_stats[[i]][, 3]]
}

# Random Forest prediction in batches
load("trained_rf_model.RData")  # Assume model is in rf_model
batch_size <- 500000
n <- nrow(cell_data)
preds <- numeric(n)

for (start in seq(1, n, by = batch_size)) {
  end <- min(start + batch_size - 1, n)
  preds[start:end] <- predict(rf_model, newdata = cell_data[start:end])
}

cell_data[, gdp_pred := preds]
```

---

**Key Improvements**  
- Eliminated millions of `lapply` calls → replaced with indexed loops and preallocation.  
- Avoided repeated `paste` and string lookups.  
- Used `data.table` for fast joins and memory efficiency.  
- Batched predictions to avoid memory spikes.  

**Expected Impact**: From 86+ hours to a few hours (dominated by neighbor stats computation). Further speedup possible by parallelizing inner loops or using `Rcpp`.