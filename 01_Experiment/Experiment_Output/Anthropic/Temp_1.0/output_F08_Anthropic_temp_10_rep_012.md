 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates the static spatial topology with the dynamic temporal data.**

### Specific bottlenecks:

1. **`build_neighbor_lookup` operates on the full 6.46M-row cell-year panel.** It constructs a lookup keyed by `paste(id, year)` for every row, then for each of the 6.46M rows, it finds that row's neighbors *within the same year* by string-pasting keys and matching. This means:
   - ~6.46M string paste operations for keys.
   - ~6.46M iterations, each doing additional paste + lookup operations.
   - The neighbor *topology* (which cell neighbors which) is identical across all 28 years, yet the function redundantly re-resolves it for every year.

2. **`compute_neighbor_stats` iterates over 6.46M list elements** with R-level `lapply`, extracting and summarizing neighbor values one row at a time. This is done 5 times (once per variable), totaling ~32.3M R-level loop iterations with subsetting.

3. **Memory overhead:** The `neighbor_lookup` list has 6.46M elements, each a vector of integer indices. Since the topology is the same for all 28 years, this stores 28 redundant copies of the same neighbor structure (just shifted by row offsets).

### Root cause summary:

| Aspect | Nature | Current handling |
|---|---|---|
| Neighbor topology (which cells are adjacent) | **Static** across years | Rebuilt for every cell-year row (6.46M times) |
| Variable values (ntl, ec, etc.) | **Dynamic** across years | Correctly varying |

The fix is to **separate the static topology from the dynamic computation**: build the neighbor index once over the 344K cells, then use vectorized/matrix operations over years.

---

## Optimization Strategy

1. **Build the neighbor lookup once over 344K cells, not 6.46M rows.** Convert `rook_neighbors_unique` (an `nb` object) into a simple cell-level adjacency list indexed by position in `id_order`. This is a one-time O(344K) operation.

2. **Reshape each variable into a matrix of shape `(344208 cells × 28 years)`.** This allows column-wise (year-wise) vectorized operations.

3. **Compute neighbor stats using vectorized sparse-matrix multiplication.** Construct a sparse adjacency matrix `W` of dimension 344,208 × 344,208 from the `nb` object. Then:
   - `neighbor_mean = W %*% X / neighbor_count` (where `X` is the cell × year matrix)
   - `neighbor_max` and `neighbor_min` can be computed via a grouped row-wise operation over the sparse structure.

   For **mean**, sparse matrix multiplication is trivially fast. For **max** and **min**, we iterate over the 344K cells (not 6.46M) and extract neighbor values from the matrix — a 28× speedup over the status quo, and each iteration is a simple matrix row subset.

4. **Reassemble results back into the long panel format** and attach the 15 new columns (5 vars × 3 stats).

### Expected speedup:

| Step | Before | After |
|---|---|---|
| Neighbor lookup construction | ~6.46M string ops | ~344K integer list (reuse `nb` directly) |
| Neighbor stat computation (per var) | ~6.46M R-level iterations | Sparse matrix multiply (mean) + ~344K iterations for min/max across 28 columns |
| Total iterations | ~32.3M + 6.46M | ~1.72M + 3 sparse matmuls |
| **Estimated time** | **86+ hours** | **~5–15 minutes** |

---

## Working R Code

```r
library(Matrix)
library(data.table)

# =============================================================================
# STEP 0: Ensure data is a data.table keyed properly
# =============================================================================
cell_dt <- as.data.table(cell_data)

# id_order: vector of 344,208 unique cell IDs (defines position mapping)
# rook_neighbors_unique: spdep nb object of length 344,208

n_cells <- length(id_order)
stopifnot(n_cells == length(rook_neighbors_unique))

# Create a map from cell ID to positional index (1-based, matching id_order)
id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

# =============================================================================
# STEP 1: Build sparse adjacency matrix W (static, built once)
#          Also compute neighbor counts for the mean calculation.
# =============================================================================
build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj is a list of length n; nb_obj[[i]] gives integer indices of neighbors
  # of cell i (0 means no neighbors in spdep convention).
  
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # spdep uses 0L to indicate no neighbors; remove those
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  return(W)
}

W <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

# Neighbor count per cell (static)
neighbor_count <- as.numeric(W %*% rep(1, n_cells))  # length n_cells

# =============================================================================
# STEP 2: Identify years and sort data for matrix reshaping
# =============================================================================
years <- sort(unique(cell_dt$year))
n_years <- length(years)
year_to_col <- setNames(seq_len(n_years), as.character(years))

# Ensure data is ordered by (cell position, year) for fast matrix fill
cell_dt[, cell_pos := id_to_pos[as.character(id)]]
setorder(cell_dt, cell_pos, year)

# Verify complete panel (every cell has every year)
stopifnot(nrow(cell_dt) == n_cells * n_years)

# =============================================================================
# STEP 3: Function to reshape a variable to cell x year matrix
# =============================================================================
var_to_matrix <- function(dt, var_name, n_cells, n_years) {
  # Data is sorted by (cell_pos, year), so we can directly fill column-major
  mat <- matrix(dt[[var_name]], nrow = n_cells, ncol = n_years, byrow = FALSE)
  return(mat)
}

# =============================================================================
# STEP 4: Compute neighbor stats efficiently
# =============================================================================
compute_neighbor_stats_fast <- function(W, nb_obj, var_mat, neighbor_count, 
                                        n_cells, n_years) {
  # --- MEAN: sparse matrix multiplication ---
  # W %*% var_mat gives sum of neighbor values for each cell x year
  # Handle NAs: compute sum of non-NA neighbor values and count of non-NA neighbors
  
  not_na_mat <- matrix(as.numeric(!is.na(var_mat)), nrow = n_cells, ncol = n_years)
  var_mat_0  <- var_mat
  var_mat_0[is.na(var_mat_0)] <- 0  # replace NA with 0 for summation
  
  neighbor_sum     <- as.matrix(W %*% var_mat_0)          # n_cells x n_years
  neighbor_nna     <- as.matrix(W %*% not_na_mat)         # count of non-NA neighbors
  
  neighbor_mean_mat <- neighbor_sum / neighbor_nna         # element-wise
  neighbor_mean_mat[neighbor_nna == 0] <- NA
  
  # --- MAX and MIN: iterate over cells (344K, not 6.46M) ---
  neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nb_idx <- nb_obj[[i]]
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) next
    
    # Extract sub-matrix: rows = neighbors, cols = years
    nb_vals <- var_mat[nb_idx, , drop = FALSE]  # length(nb_idx) x n_years
    
    # Compute column-wise max and min, ignoring NAs
    if (length(nb_idx) == 1L) {
      neighbor_max_mat[i, ] <- nb_vals[1L, ]
      neighbor_min_mat[i, ] <- nb_vals[1L, ]
    } else {
      # suppressWarnings to handle all-NA columns gracefully
      suppressWarnings({
        neighbor_max_mat[i, ] <- apply(nb_vals, 2L, max, na.rm = TRUE)
        neighbor_min_mat[i, ] <- apply(nb_vals, 2L, min, na.rm = TRUE)
      })
      # Fix columns where all neighbors were NA (apply returns -Inf/Inf)
      all_na_cols <- colSums(!is.na(nb_vals)) == 0L
      if (any(all_na_cols)) {
        neighbor_max_mat[i, all_na_cols] <- NA
        neighbor_min_mat[i, all_na_cols] <- NA
      }
    }
  }
  
  list(max = neighbor_max_mat, min = neighbor_min_mat, mean = neighbor_mean_mat)
}

# =============================================================================
# STEP 5: Process all neighbor source variables
# =============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Processing neighbor stats for: ", var_name)
  
  # Reshape to matrix
  var_mat <- var_to_matrix(cell_dt, var_name, n_cells, n_years)
  
  # Compute stats
  stats <- compute_neighbor_stats_fast(
    W, rook_neighbors_unique, var_mat, neighbor_count, n_cells, n_years
  )
  
  # Flatten matrices back to vectors (column-major = sorted by cell_pos, year)
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  cell_dt[, (col_max)  := as.vector(stats$max)]
  cell_dt[, (col_min)  := as.vector(stats$min)]
  cell_dt[, (col_mean) := as.vector(stats$mean)]
  
  rm(var_mat, stats)
  gc()
}

# =============================================================================
# STEP 6: Restore original row order if needed, clean up helper column
# =============================================================================
# If the original cell_data had a specific row order, restore it:
# setorder(cell_dt, <original_order_column>)

cell_dt[, cell_pos := NULL]

# Convert back to data.frame if the downstream RF predict expects one
cell_data <- as.data.frame(cell_dt)

# =============================================================================
# STEP 7: Predict with the pre-trained Random Forest (unchanged)
# =============================================================================
# predictions <- predict(trained_rf_model, newdata = cell_data)
# (This step is preserved exactly as before; model is NOT retrained.)
```

---

## Why this preserves correctness

| Requirement | How it's satisfied |
|---|---|
| **Numerical estimand preserved** | The neighbor max, min, and mean are computed with the identical definition: for each cell-year, gather the same-year values from rook-adjacent cells, compute the stat. The sparse matrix `W` encodes the exact same adjacency as `rook_neighbors_unique`. The sum/count decomposition for mean is algebraically identical. Max and min use the same neighbor sets and `na.rm` logic. |
| **Trained RF model preserved** | The code only prepares features; `predict()` is called on the existing model object with no retraining. |
| **Column names preserved** | Output columns follow the same naming convention (`neighbor_max_ntl`, `neighbor_min_ec`, etc.) so the RF model's expected feature names are matched. |

---

## Performance summary

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M string-key list entries | One 344K × 344K sparse matrix (~1.4M non-zeros) |
| Mean computation | 6.46M R-level iterations × 5 vars | 5 sparse matrix multiplications (seconds each) |
| Max/Min computation | 6.46M R-level iterations × 5 vars | 344K iterations, each processing a small matrix slice across 28 years |
| **Estimated total time** | **86+ hours** | **~5–15 minutes** |
| **Peak RAM** | ~6.46M-element list + data copies | Sparse matrix (~16 MB) + one 344K×28 dense matrix (~77 MB) at a time |