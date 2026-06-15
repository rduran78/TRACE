 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation suffers from two fundamental inefficiencies:

### 1. Redundant Neighbor Lookup Construction (The Core Problem)

`build_neighbor_lookup` builds a lookup of **row indices** in the full 6.46-million-row `cell_data` data frame. Because the lookup maps each row to its neighbors' rows *within the same year*, it iterates over all ~6.46M rows, performs string-pasting (`paste(id, year)`), and hash-table lookups for every single row — even though **the neighbor topology is identical across all 28 years**. The spatial adjacency graph has only 344,208 cells and ~1.37M directed edges. This structure is duplicated 28 times for no reason.

### 2. Redundant Neighbor Stats Computation

`compute_neighbor_stats` iterates over the 6.46M-element `neighbor_lookup` list, indexing into a values vector for each row. This is an R-level loop over millions of list elements, each containing a variable-length integer vector. It is slow due to R's interpreted overhead and poor cache behavior.

### 3. Combined Cost

- `build_neighbor_lookup`: ~6.46M iterations × string operations = **hours**.
- `compute_neighbor_stats`: called 5 times × 6.46M iterations = **hours more**.
- Total: the estimated 86+ hours.

---

## Optimization Strategy

**Key insight:** Separate the *static topology* (which cells neighbor which cells) from the *dynamic variables* (which change by year). Compute neighbor statistics using a **year-within-topology** approach:

1. **Build the topology once** — a simple list of length 344,208, mapping each cell's positional index to its neighbors' positional indices. This is essentially `rook_neighbors_unique` itself (an `nb` object), which is already available. No string hashing, no year dimension.

2. **For each variable, loop over years (28 iterations), not over cell-years (6.46M iterations).** Within each year, extract the variable as a numeric vector of length 344,208 (one value per cell), then compute neighbor max/min/mean using vectorized operations over the adjacency list. This reduces the inner loop from 6.46M to 344,208 per year, and there are only 28 years.

3. **Vectorize the stats computation** using `vapply` over the 344,208-element neighbor list, which is far faster than over a 6.46M-element list. Alternatively, convert to a sparse adjacency matrix and use matrix operations for mean (and row-wise operations for max/min).

4. **Sparse matrix acceleration for `mean`:** Construct a row-normalized sparse adjacency matrix `W` (344,208 × 344,208). Then `neighbor_mean = W %*% vals` is a single sparse matrix-vector multiply — essentially instantaneous. For `max` and `min`, use a loop over the 344,208-element `nb` list with `vapply`, which takes seconds.

### Expected Speedup

| Step | Before | After |
|---|---|---|
| Build lookup | ~6.46M string ops | Reuse existing `nb` (zero cost) |
| Stats per var per year | 6.46M list iterations | 344K `vapply` + 1 sparse matmul |
| Total iterations | 5 vars × 6.46M = 32.3M | 5 vars × 28 yrs × 344K = 48.2M but vectorized |
| Estimated time | 86+ hours | **~5–15 minutes** |

---

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 0: Ensure cell_data is a data.table keyed by (id, year) for fast ops.
#         Preserve original column order for downstream RF prediction.
# ==============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Store original column order to restore later
original_cols <- copy(names(cell_data))

# ==============================================================================
# STEP 1: Build static topology structures ONCE from the nb object.
#
#   - id_order:              vector of cell IDs in the order matching the nb object
#   - rook_neighbors_unique: the spdep::nb object (list of length N_cells)
#   - We build:
#       (a) A sparse row-normalized weight matrix W for computing neighbor means.
#       (b) The raw nb list is reused directly for max/min via vapply.
# ==============================================================================

N_cells <- length(id_order)  # 344,208

# --- Build sparse adjacency matrix and row-normalized version ---
# Each element of rook_neighbors_unique is an integer vector of neighbor indices
# (into id_order). spdep::nb objects use 0L to denote no neighbors.

build_sparse_weights <- function(nb_obj, n) {
  # Construct COO triplets
  from <- integer(0)
  to   <- integer(0)
  
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep convention: 0L means no neighbors
    if (length(nbrs) == 1L && nbrs[1L] == 0L) next
    from <- c(from, rep(i, length(nbrs)))
    to   <- c(to, nbrs)
  }
  
  # Binary adjacency matrix (sparse)
  A <- sparseMatrix(i = from, j = to, x = 1.0, dims = c(n, n))
  
  # Row-normalize: each row sums to 1 (for mean computation)
  row_sums <- rowSums(A)
  row_sums[row_sums == 0] <- 1  # avoid division by zero; those rows will be NA'd later
  W <- A / row_sums
  
  list(A = A, W = W, row_sums_orig = rowSums(sparseMatrix(i = from, j = to, x = 1.0, dims = c(n, n))))
}

message("Building sparse adjacency matrix (one-time cost)...")
sparse_info  <- build_sparse_weights(rook_neighbors_unique, N_cells)
W_mean       <- sparse_info$W        # row-normalized sparse matrix for neighbor mean
A_binary     <- sparse_info$A        # binary sparse matrix for detecting zero-neighbor cells
neighbor_cnt <- diff(A_binary@p)     # number of neighbors per cell (CSC, but for symmetric ~ same)

# More reliable neighbor count from the nb object directly:
neighbor_count <- vapply(rook_neighbors_unique, function(nbrs) {
  if (length(nbrs) == 1L && nbrs[1L] == 0L) 0L else length(nbrs)
}, integer(1))

has_neighbors <- neighbor_count > 0L

message("Sparse matrix built. Non-zero entries: ", length(A_binary@x))

# ==============================================================================
# STEP 2: Create a cell-index mapping.
#
#   For each row in cell_data, we need to know its position in id_order
#   (i.e., its index in the nb object / sparse matrix).
#   We also need to process data year-by-year.
# ==============================================================================

# Map cell IDs to their positional index in id_order
id_to_pos <- setNames(seq_len(N_cells), as.character(id_order))

# Add positional index to cell_data
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# Identify unique years
years <- sort(unique(cell_data$year))
n_years <- length(years)

message(sprintf("Processing %d cells x %d years = %d cell-years",
                N_cells, n_years, nrow(cell_data)))

# ==============================================================================
# STEP 3: For each neighbor source variable, compute neighbor max, min, mean
#          year by year, using the static topology.
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate output columns in cell_data
for (var_name in neighbor_source_vars) {
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  if (!max_col  %in% names(cell_data)) cell_data[, (max_col)  := NA_real_]
  if (!min_col  %in% names(cell_data)) cell_data[, (min_col)  := NA_real_]
  if (!mean_col %in% names(cell_data)) cell_data[, (mean_col) := NA_real_]
}

# Key cell_data by (year, cell_pos) for fast subsetting
setkey(cell_data, year, cell_pos)

compute_neighbor_max_min <- function(vals, nb_obj, n, has_nb) {
  # vals: numeric vector of length n (one per cell, ordered by id_order position)
  # Returns a 2-column matrix: [max, min], length n
  # Cells without neighbors get NA.
  
  result <- matrix(NA_real_, nrow = n, ncol = 2)
  
  # Only iterate over cells that have neighbors
  which_has <- which(has_nb)
  
  res <- vapply(which_has, function(i) {
    nv <- vals[nb_obj[[i]]]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) return(c(NA_real_, NA_real_))
    c(max(nv), min(nv))
  }, numeric(2))
  
  # res is 2 x length(which_has)
  result[which_has, 1] <- res[1, ]
  result[which_has, 2] <- res[2, ]
  
  result
}

compute_neighbor_mean_sparse <- function(vals, W, has_nb) {
  # vals: numeric vector of length n
  # W: row-normalized sparse matrix
  # Returns numeric vector of length n
  
  # Handle NAs in vals: replace with 0 for multiplication, then adjust
  # Actually, for correctness with NAs, we need:
  #   mean_i = sum(vals[neighbors_i], na.rm=TRUE) / count_non_na(neighbors_i)
  # This requires two sparse multiplications.
  
  not_na <- !is.na(vals)
  vals_clean <- vals
  vals_clean[is.na(vals_clean)] <- 0
  
  # Sum of neighbor values (non-NA treated as 0)
  # Use binary adjacency matrix, not row-normalized
  neighbor_sum   <- as.numeric(A_binary %*% vals_clean)
  

  # Count of non-NA neighbors
  neighbor_nonna <- as.numeric(A_binary %*% as.numeric(not_na))
  
  result <- ifelse(neighbor_nonna > 0, neighbor_sum / neighbor_nonna, NA_real_)
  result[!has_nb] <- NA_real_
  
  result
}

message("Computing neighbor statistics for ", length(neighbor_source_vars), " variables x ", n_years, " years...")

for (var_name in neighbor_source_vars) {
  
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  message(sprintf("  Variable: %s", var_name))
  t0 <- proc.time()
  
  for (yr in years) {
    
    # Extract this year's slice — already keyed by (year, cell_pos)
    yr_rows <- cell_data[.(yr), which = TRUE]
    
    # Build a full-length vector (N_cells) for this variable in this year
    # Some cells might be missing for a given year; those stay NA.
    vals_full <- rep(NA_real_, N_cells)
    
    pos_this_year <- cell_data$cell_pos[yr_rows]
    var_this_year <- cell_data[[var_name]][yr_rows]
    vals_full[pos_this_year] <- var_this_year
    
    # --- Compute neighbor mean via sparse matrix multiplication ---
    n_mean <- compute_neighbor_mean_sparse(vals_full, W_mean, has_neighbors)
    
    # --- Compute neighbor max and min via vapply over nb list ---
    maxmin <- compute_neighbor_max_min(vals_full, rook_neighbors_unique, N_cells, has_neighbors)
    
    # --- Write results back to cell_data ---
    # Map from cell_pos back to row indices in cell_data
    set(cell_data, i = yr_rows, j = max_col,  value = maxmin[pos_this_year, 1])
    set(cell_data, i = yr_rows, j = min_col,  value = maxmin[pos_this_year, 2])
    set(cell_data, i = yr_rows, j = mean_col, value = n_mean[pos_this_year])
  }
  
  elapsed <- (proc.time() - t0)[3]
  message(sprintf("    Done in %.1f seconds", elapsed))
}

# ==============================================================================
# STEP 4: Clean up helper column; restore data.frame if needed downstream.
# ==============================================================================
cell_data[, cell_pos := NULL]

# Restore original column order + new neighbor columns
new_cols <- setdiff(names(cell_data), original_cols)
setcolorder(cell_data, c(original_cols, new_cols))

message("Neighbor feature computation complete.")
message(sprintf("cell_data: %d rows x %d columns", nrow(cell_data), ncol(cell_data)))

# ==============================================================================
# STEP 5: Predict with the pre-trained Random Forest (UNCHANGED).
#
#   The trained RF model object and prediction call remain exactly as before.
#   For example:
#     predictions <- predict(rf_model, newdata = cell_data)
#
#   The numerical estimand is preserved because:
#     - The same neighbor max, min, mean formulas are used (max, min, arithmetic mean).
#     - The same variable names are produced.
#     - The same NA handling logic is preserved.
#     - The RF model is not retrained.
# ==============================================================================
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Redesigned |
|---|---|---|
| **Topology construction** | 6.46M string-paste + hash lookups | Reuse `nb` object directly (zero cost) |
| **Neighbor mean** | R loop over 6.46M list elements | Sparse matrix–vector multiply per year (344K × 344K, ~1.37M nonzeros) |
| **Neighbor max/min** | R loop over 6.46M list elements | `vapply` over 344K cells per year |
| **Total outer iterations** | 5 vars × 6.46M = 32.3M | 5 vars × 28 years = 140 year-slices |
| **Inner work per year-slice** | N/A (monolithic) | 1 sparse matmul + 1 vapply(344K) |
| **Estimated runtime** | 86+ hours | **~5–15 minutes** |
| **Numerical result** | `max`, `min`, `mean` with `na.rm` | Identical formulas, identical NA handling |
| **RF model** | Pre-trained | **Untouched — no retraining** |
| **Memory** | ~6.46M-element list of int vectors | Sparse matrix (~20 MB) + year-slice vectors (~2.6 MB each) |

The key architectural change is **factoring the static spatial topology out of the dynamic yearly computation**, exactly as the prompt's insight suggests. The adjacency structure is encoded once in a sparse matrix and a neighbor list, then reused across all 28 years with only the variable values changing.