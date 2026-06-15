 **Diagnosis**

The repeated construction of `neighbor_keys` and repeated lookups in `idx_lookup` inside the `lapply` loop is **not just a local inefficiency**—it is a symptom of a broader algorithmic pattern that recomputes string-based keys for every cell-year row. With ~6.46M rows and 1.37M neighbor relationships, this approach incurs:

- **String concatenation cost**: `paste(neighbor_cell_ids, data$year[i], sep = "_")` for every row.
- **Hash lookup cost**: `idx_lookup[neighbor_keys]` repeatedly for millions of keys.
- **Repeated work across variables**: The neighbor index structure is built once, but the heavy lifting happens in `compute_neighbor_stats` for each variable.

This is fundamentally an **O(N × k)** string-based lookup problem (N = 6.46M rows, k ≈ average neighbors), which is extremely inefficient in R.

---

### **Optimization Strategy**

1. **Precompute numeric indices instead of string keys**:
   - Convert `(id, year)` pairs to integer indices once.
   - Avoid `paste()` and hash lookups entirely.

2. **Vectorize neighbor relationships across all years**:
   - For each cell, neighbors are static across years.
   - Expand neighbor relationships across all years in a single numeric matrix.

3. **Compute neighbor stats in a fully vectorized way**:
   - Use matrix operations or `data.table` joins rather than millions of small `lapply` calls.

**Key idea**: Build a numeric matrix `neighbor_idx` where `neighbor_idx[i, ]` gives the row indices of neighbors for row `i` (or a list of integer vectors). This is computed once and reused for all variables.

---

### **Working R Code**

```r
library(data.table)

# Assume: cell_data has columns id, year, and predictor variables
# rook_neighbors_unique: list of integer vectors (spdep::nb)
# id_order: vector of unique cell IDs in the same order as rook_neighbors_unique

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mapping from (id, year) -> row index
id_to_idx <- cell_data[, .I, by = .(id, year)]

# Build neighbor index lookup once
build_neighbor_index <- function(cell_data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), id_order)
  n <- nrow(cell_data)
  row_ids <- seq_len(n)
  
  # Map each row to its neighbor rows numerically
  neighbor_lookup <- vector("list", n)
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[[ as.character(cell_data$id[i]) ]]
    neighbor_ids <- id_order[ neighbors[[ref_idx]] ]
    # Join to get neighbor row indices for the same year
    yr <- cell_data$year[i]
    idx <- id_to_idx[J(neighbor_ids, yr), I]
    neighbor_lookup[[i]] <- idx[!is.na(idx)]
  }
  
  neighbor_lookup
}

neighbor_lookup <- build_neighbor_index(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats efficiently
compute_neighbor_stats <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  max_vals <- numeric(n)
  min_vals <- numeric(n)
  mean_vals <- numeric(n)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) {
      max_vals[i] <- min_vals[i] <- mean_vals[i] <- NA_real_
    } else {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) == 0) {
        max_vals[i] <- min_vals[i] <- mean_vals[i] <- NA_real_
      } else {
        max_vals[i] <- max(v)
        min_vals[i] <- min(v)
        mean_vals[i] <- mean(v)
      }
    }
  }
  
  cbind(max_vals, min_vals, mean_vals)
}

# Apply for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Is Faster**

- **No string concatenation**: All lookups are integer-based.
- **Neighbor index computed once**: Reused across all variables.
- **Data.table joins**: Efficient mapping from `(id, year)` to row index.
- **Memory efficient**: Only stores integer vectors for neighbors.

---

### **Expected Performance Gain**

- Eliminates millions of `paste()` and hash lookups.
- Reduces complexity from repeated string operations to numeric lookups.
- On 6.46M rows, this should cut runtime from **86+ hours to a few hours** on a 16 GB laptop.

---

**Preserves**:
- Original Random Forest model (unchanged).
- Original numerical estimand (same neighbor stats, just computed faster).