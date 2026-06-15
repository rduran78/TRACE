 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation suffers from three critical bottlenecks:

### 1. **O(n²)-scale neighbor lookup construction**
`build_neighbor_lookup` iterates over all ~6.46 million rows, and for each row performs string concatenation (`paste`) and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector indexing in R is hash-based but still slow at this scale. The lookup is built once but costs heavily.

### 2. **Per-row `lapply` in `compute_neighbor_stats`**
For each of the ~6.46 million rows, an anonymous function extracts neighbor values, removes NAs, and computes `max/min/mean`. This is called 5 times (once per variable), yielding ~32.3 million R-level function calls with repeated subsetting.

### 3. **Redundant topology recomputation across years**
The rook-neighbor graph is **spatial only** — it doesn't change across years. Yet the current code builds a single monolithic lookup over all 6.46M cell-year rows, interleaving spatial topology with temporal matching via string keys. This is wasteful: the same 1.37M directed edges repeat identically for each of 28 years.

### Memory profile
The `neighbor_lookup` list of 6.46M integer vectors, plus the `idx_lookup` named vector of 6.46M entries, plus intermediate string vectors, likely consumes 8–12 GB and causes severe GC pressure on a 16 GB machine.

---

## Optimization Strategy

**Core insight:** Separate the spatial graph topology (344K nodes, ~1.37M edges) from the temporal dimension (28 years). Build the sparse adjacency structure once over cells, then for each year, use vectorized sparse-matrix multiplication to compute neighbor aggregates.

### Specific techniques:

1. **Build a sparse adjacency matrix once** from `rook_neighbors_unique` (344,208 × 344,208, ~1.37M nonzeros). This is tiny in memory (~20 MB as a `dgCMatrix`).

2. **For each year, extract the variable column as a dense vector over cells, then compute:**
   - `neighbor_sum = A %*% x` (sparse matrix–vector multiply)
   - `neighbor_count = A %*% (!is.na(x))` (count of non-NA neighbors)
   - `neighbor_mean = neighbor_sum / neighbor_count`
   - For `max` and `min`: use a grouped operation over the edge list (CSC column pointers).

3. **Vectorize max/min** by operating on the CSC structure of the sparse matrix directly, using `vapply` over columns (each column's nonzero entries are the neighbors). Alternatively, use `data.table` grouped operations on the edge list.

4. **Loop over 28 years × 5 variables = 140 iterations**, each operating on a 344K-length vector. Each iteration takes ~0.1–0.5 seconds → total ~1–2 minutes.

5. **Preserve numerical equivalence:** The sparse matrix–vector product computes exactly the same sums; dividing by the exact non-NA count gives the identical mean. Max and min over the same neighbor sets are identical.

---

## Optimized R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build sparse adjacency matrix from the spdep nb object (ONCE)
# ==============================================================================
build_adjacency_matrix <- function(nb_obj, n) {

  # nb_obj: list of length n, nb_obj[[i]] = integer vector of neighbor indices
  # Builds a sparse n x n adjacency matrix A where A[j, i] = 1 if j is a

  # neighbor of i. This way A %*% x gives the sum of neighbor values for

  # each node.
  
  # Build COO representation
  from_list <- lapply(seq_along(nb_obj), function(i) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) == 0L) return(NULL)
    data.frame(j = nbrs, i = rep.int(i, length(nbrs)))
  })
  
  edges <- rbindlist(from_list)
  
  # A[j, i] = 1 means j is a neighbor of i
  # So A %*% x[i] = sum of x[j] for all j that are neighbors of i
  # Wait — we want: for node i, aggregate over its neighbors j.
  # A[j, i] = 1 means column i has nonzeros at rows j (the neighbors of i).
  # Then (A^T %*% x)[i] = sum_{j neighbor of i} x[j]. 
  # OR: build A[i, j] = 1 if j is neighbor of i, then A %*% x directly works.
  
  # Let's build A where A[i, j] = 1 if j is a neighbor of i.
  # Then (A %*% x)[i] = sum of x[j] for j in neighbors(i).
  
  A <- sparseMatrix(
    i = edges$i,
    j = edges$j,
    x = 1,
    dims = c(n, n),
    repr = "C"   # CSC format
  )
  
  return(A)
}

# ==============================================================================
# STEP 2: Compute max and min using the sparse matrix structure
# ==============================================================================
# For max/min we cannot use matrix multiplication. We use the CSR structure.
# In CSR (dgRMatrix), row i's nonzero column indices are in
# j[p[i]+1 : p[i+1]], which correspond to the neighbors of node i.
# We iterate over rows using compiled C-level access via .Call or vectorized R.

compute_sparse_max_min <- function(A_csr, x) {
  # A_csr: dgRMatrix (CSR), x: numeric vector of length ncol(A_csr)
  # Returns matrix of (n x 2): col1 = max, col2 = min over neighbors
  
  n <- nrow(A_csr)
  p <- A_csr@p        # row pointers (0-based), length n+1
  j <- A_csr@j        # column indices (0-based)
  
  # Pre-allocate
  max_vals <- rep(NA_real_, n)
  min_vals <- rep(NA_real_, n)
  
  # Vectorized approach: build a data.table of (row_id, neighbor_value)
  # and do grouped max/min
  
  # Map each nonzero entry to its row
  # Row i (0-based) owns entries from p[i]+1 to p[i+1] (1-based: p[i+1]+1 to p[i+2])
  # Number of nonzeros per row:
  row_counts <- diff(p)  # length n
  
  if (sum(row_counts) == 0L) {
    return(cbind(max_vals, min_vals))
  }
  
  row_ids <- rep.int(seq_len(n), row_counts)
  col_ids <- j + 1L  # convert to 1-based
  
  neighbor_vals <- x[col_ids]
  
  dt <- data.table(row_id = row_ids, val = neighbor_vals)
  dt <- dt[!is.na(val)]
  
  if (nrow(dt) == 0L) {
    return(cbind(max_vals, min_vals))
  }
  
  agg <- dt[, .(mx = max(val), mn = min(val)), by = row_id]
  
  max_vals[agg$row_id] <- agg$mx
  min_vals[agg$row_id] <- agg$mn
  
  cbind(max_vals, min_vals)
}

# ==============================================================================
# STEP 3: Main pipeline
# ==============================================================================
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # Convert to data.table for speed
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  n_cells <- length(id_order)
  cat("Number of cells:", n_cells, "\n")
  
  # --- Build adjacency matrix ONCE ---
  cat("Building sparse adjacency matrix...\n")
  A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
  cat("  Adjacency matrix:", nrow(A), "x", ncol(A), 
      "with", length(A@x), "nonzeros\n")
  
  # Also build CSR version for max/min
  A_csr <- as(A, "RsparseMatrix")
  
  # --- Build cell ID to matrix-row mapping ---
  # id_order[k] is the cell ID for matrix row/col k
  id_to_matidx <- setNames(seq_along(id_order), as.character(id_order))
  
  # --- Get unique years, sorted ---
  years <- sort(unique(cell_data$year))
  cat("Years:", min(years), "-", max(years), "(", length(years), "years)\n")
  
  # --- Map each row of cell_data to its matrix index ---
  cell_data[, mat_idx := id_to_matidx[as.character(id)]]
  
  # --- Ensure cell_data is keyed for fast subsetting ---
  setkey(cell_data, year)
  
  # --- Source variables ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  # Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)
  }
  
  # --- Main loop: iterate over years, then variables ---
  cat("Computing neighbor statistics...\n")
  t0 <- proc.time()
  
  for (yr in years) {
    # Get row indices for this year
    yr_rows <- which(cell_data$year == yr)
    
    # Get the matrix indices for these rows (which cell each row maps to)
    yr_mat_idx <- cell_data$mat_idx[yr_rows]
    
    # Build a dense vector for each variable: x[mat_idx] = value
    # Only cells present in this year get filled; others stay NA
    
    for (var_name in neighbor_source_vars) {
      # Build dense vector over all cells for this year
      x <- rep(NA_real_, n_cells)
      x[yr_mat_idx] <- cell_data[[var_name]][yr_rows]
      
      # --- Neighbor mean via sparse matrix multiply ---
      # Handle NAs: replace NA with 0 for sum, track counts separately
      x_nona <- x
      x_nona[is.na(x_nona)] <- 0
      not_na <- as.numeric(!is.na(x))
      
      neighbor_sum   <- as.numeric(A %*% x_nona)    # length n_cells
      neighbor_count <- as.numeric(A %*% not_na)     # length n_cells
      
      neighbor_mean <- ifelse(neighbor_count > 0, 
                              neighbor_sum / neighbor_count, 
                              NA_real_)
      
      # --- Neighbor max and min via CSR grouped aggregation ---
      maxmin <- compute_sparse_max_min(A_csr, x)
      neighbor_max <- maxmin[, 1]
      neighbor_min <- maxmin[, 2]
      
      # --- Write results back to cell_data for the rows of this year ---
      col_max  <- paste0("neighbor_max_", var_name)
      col_min  <- paste0("neighbor_min_", var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      
      set(cell_data, i = yr_rows, j = col_max,  value = neighbor_max[yr_mat_idx])
      set(cell_data, i = yr_rows, j = col_min,  value = neighbor_min[yr_mat_idx])
      set(cell_data, i = yr_rows, j = col_mean, value = neighbor_mean[yr_mat_idx])
    }
    
    cat("  Year", yr, "done\n")
  }
  
  elapsed <- (proc.time() - t0)[3]
  cat("Neighbor features computed in", round(elapsed / 60, 1), "minutes\n")
  
  # Clean up helper column
  cell_data[, mat_idx := NULL]
  
  return(cell_data)
}

# ==============================================================================
# STEP 4: Run pipeline and predict with existing Random Forest
# ==============================================================================

# --- Load pre-trained model and data ---
# rf_model          <- readRDS("path/to/trained_rf_model.rds")
# cell_data         <- readRDS("path/to/cell_data.rds")
# id_order          <- readRDS("path/to/id_order.rds")
# rook_neighbors_unique <- readRDS("path/to/rook_neighbors_unique.rds")

# --- Compute neighbor features ---
cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)

# --- Predict using the EXISTING pre-trained Random Forest (no retraining) ---
# Ensure column names match what the model expects
# predictions <- predict(rf_model, newdata = cell_data)
# cell_data$prediction <- predictions
```

---

## Why This Is Correct and Numerically Equivalent

| Statistic | Original method | Optimized method | Equivalence |
|-----------|----------------|-----------------|-------------|
| **Mean** | `mean(vals[neighbors])` after removing NAs | `(A %*% x_nona)[i] / (A %*% not_na)[i]` | Identical: same sum divided by same count |
| **Max** | `max(vals[neighbors])` after removing NAs | Grouped `max` over CSR row entries, NAs excluded | Identical: same set of values |
| **Min** | `min(vals[neighbors])` after removing NAs | Grouped `min` over CSR row entries, NAs excluded | Identical: same set of values |
| **NA handling** | Rows with no valid neighbors → `NA` | `neighbor_count == 0` → `NA`; empty groups → `NA` | Identical |

## Performance Analysis

| Aspect | Original | Optimized | Speedup |
|--------|----------|-----------|---------|
| **Lookup build** | ~6.46M string pastes + hash lookups | One `sparseMatrix()` call on 344K nodes | ~1000× |
| **Mean computation** | 6.46M × 5 R function calls | 28 × 5 = 140 sparse mat-vec multiplies | ~500× |
| **Max/Min** | 6.46M × 5 R function calls | 140 `data.table` grouped aggregations | ~200× |
| **Memory** | ~8–12 GB (string keys + list of vectors) | ~200 MB (sparse matrix + temp vectors) | ~40× |
| **Estimated wall time** | 86+ hours | **2–5 minutes** | ~1000× |

The sparse matrix is ~20 MB. Each year's dense vector is ~2.6 MB. The `data.table` grouped aggregation for max/min processes ~1.37M edges per iteration. Total: 140 iterations × ~0.5s ≈ ~70 seconds, plus overhead. The trained Random Forest model is never modified.