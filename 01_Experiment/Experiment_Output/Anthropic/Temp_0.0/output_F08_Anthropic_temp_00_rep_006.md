 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cell×year) neighbor indices.** It creates a list of ~6.46 million entries, each requiring string-pasting (`paste(id, year)`), hash-table lookups (`idx_lookup[neighbor_keys]`), and NA filtering. But the neighbor *relationships* between cells never change across years — only the *values* attached to cells change. This means the function does ~28× more work than necessary.

2. **`compute_neighbor_stats` iterates over ~6.46 million rows** with an `lapply` loop in R, extracting neighbor values one row at a time. This is inherently slow due to R's interpreted loop overhead.

3. **String-based key lookups** (`paste(id, year, sep="_")` → named vector indexing) are extremely expensive at this scale. With ~6.46M keys, each `paste` + lookup is O(n) in aggregate and memory-heavy.

4. **The neighbor lookup is year-aware but doesn't need to be.** Since cell `i`'s neighbors are the same in every year, we only need a cell-level adjacency list (344,208 entries), not a cell-year-level one (6.46M entries). The year dimension should be handled at the *value extraction* stage, not the *topology* stage.

### Summary

| Component | Current Scale | Required Scale | Waste Factor |
|---|---|---|---|
| Neighbor lookup | 6.46M entries | 344,208 entries | ~28× |
| String key construction | ~6.46M × avg_neighbors | 0 (use integer indexing) | ∞ |
| Stats computation loop | 6.46M R-level iterations | Vectorized matrix ops | Orders of magnitude |

---

## Optimization Strategy

### Core Insight

**Separate the static graph structure from the dynamic variable values.**

1. **Build the neighbor lookup once at the cell level (not cell×year).** This is a list of 344,208 elements, each containing integer indices into the cell-order vector. This is built once and reused for all years and all variables.

2. **Reshape each variable into a matrix: cells × years.** Each column is one year. This allows vectorized extraction of neighbor values for all years simultaneously.

3. **Compute neighbor stats using sparse matrix multiplication** (for mean/sum) and vectorized row operations (for max/min). Specifically:
   - Convert the adjacency list to a sparse matrix `A` (344,208 × 344,208).
   - **Neighbor mean** of variable `X` (a cells×years matrix) = `(A %*% X) / degree_vector` — a single sparse matrix multiply.
   - **Neighbor sum** = `A %*% X`.
   - **Neighbor max/min**: iterate over cells but use the sparse structure, or use a chunked approach with matrix indexing.

4. **Flatten results back** to the original cell-year row order and bind as new columns.

### Expected Speedup

- Neighbor lookup: from ~6.46M string ops to ~344K integer ops → **~28× faster + no string overhead**.
- Neighbor mean: from 6.46M R-loop iterations to one sparse matrix multiply → **~1000× faster**.
- Neighbor max/min: from 6.46M R-loop iterations to 344K iterations over a pre-indexed integer structure, repeated 28 times vectorized → **~50-100× faster**.
- Overall: from ~86 hours to **minutes**.

---

## Working R Code

```r
library(Matrix)

# ==============================================================================
# STEP 1: Build cell-level neighbor lookup ONCE (static topology)
# ==============================================================================
# rook_neighbors_unique: spdep nb object, indexed by position in id_order
# id_order: vector of 344,208 cell IDs in the order matching the nb object

build_cell_neighbor_lookup <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object: list of integer vectors (positional indices)
  # Already in the right form — each element gives neighbor positions in id_order.
  # We just ensure it's a clean integer list.
  n <- length(id_order)
  stopifnot(length(neighbors_nb) == n)
  
  cell_neighbors <- vector("list", n)
  for (i in seq_len(n)) {
    nb_i <- neighbors_nb[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    if (length(nb_i) == 1L && nb_i[1] == 0L) {
      cell_neighbors[[i]] <- integer(0)
    } else {
      cell_neighbors[[i]] <- as.integer(nb_i)
    }
  }
  cell_neighbors
}

# ==============================================================================
# STEP 2: Build sparse adjacency matrix from cell-level neighbor list
# ==============================================================================
build_adjacency_matrix <- function(cell_neighbors, n) {
  # Build COO triplets
  from <- integer(0)
  to   <- integer(0)
  
  # Pre-count total edges for pre-allocation
  total_edges <- sum(vapply(cell_neighbors, length, integer(1)))
  from <- integer(total_edges)
  to   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb <- cell_neighbors[[i]]
    k  <- length(nb)
    if (k > 0L) {
      from[pos:(pos + k - 1L)] <- i
      to[pos:(pos + k - 1L)]   <- nb
      pos <- pos + k
    }
  }
  
  A <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  A
}

# ==============================================================================
# STEP 3: Reshape a variable from long cell-year data to cells × years matrix
# ==============================================================================
reshape_to_matrix <- function(cell_data, id_order, years, var_name) {
  # cell_data must have columns: id, year, and var_name
  # Returns a matrix: n_cells rows × n_years columns
  # Row i corresponds to id_order[i], column j corresponds to years[j]
  
  n_cells <- length(id_order)
  n_years <- length(years)
  
  # Create fast lookup: cell_id -> row index in matrix
  id_to_row <- setNames(seq_along(id_order), as.character(id_order))
  # Create fast lookup: year -> column index in matrix
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  row_idx <- id_to_row[as.character(cell_data$id)]
  col_idx <- year_to_col[as.character(cell_data$year)]
  
  mat[cbind(row_idx, col_idx)] <- cell_data[[var_name]]
  mat
}

# ==============================================================================
# STEP 4: Compute neighbor stats using sparse matrix ops + vectorized max/min
# ==============================================================================
compute_neighbor_stats_fast <- function(A, cell_neighbors, var_matrix) {
  # A: sparse adjacency matrix (n_cells × n_cells)
  # cell_neighbors: list of integer neighbor indices (cell-level)
  # var_matrix: n_cells × n_years matrix of variable values
  
  n_cells <- nrow(var_matrix)
  n_years <- ncol(var_matrix)
  
  # --- Neighbor MEAN ---
  # neighbor_sum = A %*% var_matrix  (sparse matrix multiply, very fast)
  neighbor_sum <- A %*% var_matrix  # n_cells × n_years
  
  # degree = number of non-NA neighbors per cell per year
  # To handle NAs properly: replace NA with 0 in values, and count non-NA
  var_notna <- var_matrix
  var_notna[is.na(var_notna)] <- 0
  
  indicator <- matrix(1, nrow = n_cells, ncol = n_years)
  indicator[is.na(var_matrix)] <- 0
  
  neighbor_sum_clean <- A %*% var_notna       # sum of non-NA neighbor values
  neighbor_count     <- A %*% indicator        # count of non-NA neighbors
  
  # Convert to dense
  neighbor_sum_clean <- as.matrix(neighbor_sum_clean)
  neighbor_count     <- as.matrix(neighbor_count)
  
  neighbor_mean <- neighbor_sum_clean / neighbor_count
  neighbor_mean[neighbor_count == 0] <- NA_real_
  
  # --- Neighbor MAX and MIN ---
  # These cannot be done with matrix multiplication.
  # We iterate over cells (344K, not 6.46M) and vectorize across years.
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nb <- cell_neighbors[[i]]
    if (length(nb) == 0L) next
    
    # Extract sub-matrix: length(nb) × n_years
    nb_vals <- var_matrix[nb, , drop = FALSE]
    
    # suppressWarnings to handle all-NA columns gracefully
    suppressWarnings({
      neighbor_max[i, ] <- apply(nb_vals, 2, max, na.rm = TRUE)
      neighbor_min[i, ] <- apply(nb_vals, 2, min, na.rm = TRUE)
    })
    
    # Fix all-NA columns (max/min return -Inf/Inf)
    all_na_cols <- colSums(!is.na(nb_vals)) == 0L
    if (any(all_na_cols)) {
      neighbor_max[i, all_na_cols] <- NA_real_
      neighbor_min[i, all_na_cols] <- NA_real_
    }
  }
  
  list(
    neighbor_max  = neighbor_max,
    neighbor_min  = neighbor_min,
    neighbor_mean = neighbor_mean
  )
}

# ==============================================================================
# STEP 5: Flatten matrix back to long format matching cell_data row order
# ==============================================================================
flatten_matrix_to_long <- function(mat, cell_data, id_order, years) {
  id_to_row  <- setNames(seq_along(id_order), as.character(id_order))
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  row_idx <- id_to_row[as.character(cell_data$id)]
  col_idx <- year_to_col[as.character(cell_data$year)]
  
  mat[cbind(row_idx, col_idx)]
}

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

# --- Setup (done once) ---
years   <- sort(unique(cell_data$year))  # 1992:2019
n_cells <- length(id_order)              # 344,208

cat("Building cell-level neighbor lookup (static topology)...\n")
cell_neighbors <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)

cat("Building sparse adjacency matrix...\n")
A <- build_adjacency_matrix(cell_neighbors, n_cells)

# --- Compute neighbor features for each source variable ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))
  
  # Reshape to matrix (cells × years)
  var_mat <- reshape_to_matrix(cell_data, id_order, years, var_name)
  
  # Compute all three stats at once
  stats <- compute_neighbor_stats_fast(A, cell_neighbors, var_mat)
  
  # Flatten back to long format and attach to cell_data
  cell_data[[paste0("neighbor_max_",  var_name)]] <-
    flatten_matrix_to_long(stats$neighbor_max, cell_data, id_order, years)
  cell_data[[paste0("neighbor_min_",  var_name)]] <-
    flatten_matrix_to_long(stats$neighbor_min, cell_data, id_order, years)
  cell_data[[paste0("neighbor_mean_", var_name)]] <-
    flatten_matrix_to_long(stats$neighbor_mean, cell_data, id_order, years)
  
  # Free memory
  rm(var_mat, stats)
  gc()
  
  cat(sprintf("  Done: %s\n", var_name))
}

cat("All neighbor features computed. Proceeding to prediction...\n")

# --- Prediction with pre-trained Random Forest (unchanged) ---
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Complexity Comparison

| Metric | Original | Optimized |
|---|---|---|
| Neighbor lookup entries | 6,460,000 | 344,208 |
| String operations | ~25M+ `paste` + hash lookups | 0 |
| Stats loop iterations | 6.46M × 5 vars = 32.3M | 344K × 5 = 1.72M (max/min only) |
| Mean computation | 32.3M R-level iterations | 5 sparse matrix multiplies |
| Memory for lookup | ~6.46M-element list of vectors | ~344K-element list + one sparse matrix |
| Estimated runtime | ~86+ hours | **~5–15 minutes** |

## Key Guarantees

- **Trained Random Forest is untouched**: no retraining; the code only prepares input features and then calls `predict()`.
- **Numerical equivalence preserved**: the same `max`, `min`, and `mean` of the same non-NA neighbor values are computed; the `na.rm = TRUE` handling mirrors the original's explicit NA filtering.
- **16 GB RAM feasibility**: the largest single object is a 344,208 × 28 dense matrix (~77 MB); the sparse matrix has ~1.37M non-zeros (~33 MB). Total peak memory is well under 4 GB for this pipeline stage.