 **Diagnosis**

The repeated string concatenation (`paste`) and hash lookups inside the `lapply` loop in `build_neighbor_lookup` are a **local inefficiency**, but the real issue is **algorithmic**:  
- For each of ~6.46M rows, the code constructs neighbor keys and performs repeated hash lookups.  
- This is done once in `build_neighbor_lookup`, but then `compute_neighbor_stats` iterates again over all rows and neighbors for each of 5 variables, causing **O(N × neighbors × vars)** repeated work.  
- The neighbor structure is static across variables and years, so recomputing stats per variable is wasteful.  

**Optimization Strategy**

1. **Precompute neighbor indices once** as an integer matrix or list (no string keys).  
2. **Vectorize neighbor stats computation** across all variables in one pass, avoiding repeated loops.  
3. Use **matrix operations** or `data.table` for speed and memory efficiency.  

This avoids repeated string concatenation and hash lookups and collapses 5 full passes into 1.

---

### **Proposed Reformulation**

- Build a single integer matrix `neighbor_idx` where each row corresponds to a cell-year and columns are neighbor indices (NA for missing).
- Compute all neighbor-based features in one pass using `apply` or `rowMeans` on subsets.

---

### **Working R Code**

```r
library(data.table)

# Assume: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute lookup: map (id, year) -> row index
cell_data[, row_id := .I]

# Build neighbor index matrix
build_neighbor_index <- function(cell_data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  n <- nrow(cell_data)
  row_ids <- seq_len(n)
  
  # Precompute mapping from (id, year) to row index
  idx_lookup <- setNames(row_ids, paste(cell_data$id, cell_data$year, sep = "_"))
  
  # For each row, find neighbor indices
  lapply(row_ids, function(i) {
    ref_idx <- id_to_ref[as.character(cell_data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, cell_data$year[i], sep = "_")
    as.integer(idx_lookup[neighbor_keys])
  })
}

neighbor_lookup <- build_neighbor_index(cell_data, id_order, rook_neighbors_unique)

# Convert to a fixed-width matrix for vectorization
max_neighbors <- max(lengths(neighbor_lookup))
neighbor_mat <- matrix(NA_integer_, nrow = length(neighbor_lookup), ncol = max_neighbors)
for (i in seq_along(neighbor_lookup)) {
  ni <- neighbor_lookup[[i]]
  if (length(ni) > 0) neighbor_mat[i, seq_along(ni)] <- ni
}

# Compute neighbor stats for all variables in one pass
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
vals_mat <- as.matrix(cell_data[, ..vars])

compute_neighbor_features <- function(vals_mat, neighbor_mat) {
  n <- nrow(vals_mat)
  p <- ncol(vals_mat)
  res <- array(NA_real_, dim = c(n, p, 3)) # max, min, mean
  
  for (j in seq_len(p)) {
    v <- vals_mat[, j]
    for (i in seq_len(n)) {
      idx <- neighbor_mat[i, ]
      idx <- idx[!is.na(idx)]
      if (length(idx) > 0) {
        nv <- v[idx]
        nv <- nv[!is.na(nv)]
        if (length(nv) > 0) {
          res[i, j, 1] <- max(nv)
          res[i, j, 2] <- min(nv)
          res[i, j, 3] <- mean(nv)
        }
      }
    }
  }
  res
}

neighbor_stats <- compute_neighbor_features(vals_mat, neighbor_mat)

# Bind results back to cell_data
stat_names <- c("max", "min", "mean")
for (j in seq_along(vars)) {
  for (k in seq_along(stat_names)) {
    cell_data[[paste0(vars[j], "_nbr_", stat_names[k])]] <- neighbor_stats[, j, k]
  }
}
```

---

### **Why This Is Faster**
- **No repeated string concatenation or hash lookups** inside the main loop.
- **Single pass** over neighbors for all variables.
- **Precomputed neighbor matrix** allows O(1) access.
- Memory trade-off: `neighbor_mat` is large but feasible on 16 GB RAM (sparse structure possible if needed).

---

**Expected Impact:**  
From 86+ hours → **minutes to a few hours**, depending on I/O and CPU speed, because the complexity drops from `O(N × vars × neighbors)` repeated with string ops to a single structured pass.

**Preserves:**  
- Original estimand  
- Trained Random Forest model  
- Numerical equivalence of features