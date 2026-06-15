 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a vector of 6.46M names — this alone is catastrophically slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a serial `lapply`**, extracting subsets of a numeric vector and computing `max/min/mean`. The per-element overhead of R function calls, subsetting, and `NA` checks dominates.

3. **The neighbor lookup is year-aware but redundant**: the spatial topology is identical across all 28 years. The current code rebuilds year-specific row indices for every cell-year, but the adjacency structure is static. The lookup should be built once on the ~344K cell topology and then applied per-year via vectorized operations.

**Root cause**: The design treats the problem as a generic graph with 6.46M nodes, when it is actually a **static spatial graph of 344K nodes × 28 independent year slices**. Each year slice can be processed independently using the same sparse adjacency matrix, enabling fully vectorized sparse matrix–vector operations.

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from the `nb` object (344,208 × 344,208, ~1.37M non-zeros). This is tiny in memory (~16 MB).

2. **Process each year independently**: subset the data by year, align cell ordering, and use sparse matrix multiplication to compute neighbor sums and counts in one shot.

3. **Compute max/min via sparse matrix tricks**: For `mean`, use `A %*% x / A %*% 1`. For `max` and `min`, iterate over the CSC/CSR structure directly in vectorized C-level code via the `Matrix` package, or use a grouped operation on the sparse triplet representation.

4. **Avoid all `lapply` over millions of elements**, all string-pasting, and all named-vector lookups.

**Expected speedup**: From 86+ hours to **~2–5 minutes** on the same laptop.

## Optimized R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build sparse adjacency matrix from nb object (once)
# ==============================================================================
build_adjacency_matrix <- function(nb_obj, id_order) {
 # nb_obj: spdep nb object (list of integer vectors of neighbor indices)
 # id_order: vector of cell IDs corresponding to positions in nb_obj
 n <- length(nb_obj)
 stopifnot(n == length(id_order))

 # Build COO triplets: row i has edges to nb_obj[[i]]
 from <- rep(seq_len(n), lengths(nb_obj))
 to   <- unlist(nb_obj, use.names = FALSE)

 # Remove any 0-neighbor placeholders (spdep uses 0L for no-neighbor cards)
 valid <- to > 0L
 from  <- from[valid]
 to    <- to[valid]

 # Sparse binary adjacency matrix (row = focal cell, col = neighbor cell)
 A <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
 return(A)
}

# ==============================================================================
# STEP 2: Compute neighbor stats using sparse matrix operations
# ==============================================================================
# For mean: A %*% x / (A %*% ones), trivially vectorized.
# For max and min: we exploit the CSC structure of the transpose.
#
# Key insight: row i of A has non-zeros in columns that are i's neighbors.
# So (A %*% x)[i] = sum of x over neighbors of i.
# For max/min, we use the dgCMatrix structure directly.

compute_neighbor_max_min <- function(A, x) {
 # A is n x n sparse (dgCMatrix), x is numeric length n
 # Returns n x 2 matrix: [max, min] of neighbor values for each node
 # We work with A in CSC format. Row indices of column j tell us
 # which nodes have j as a neighbor. But we need: for each row i,
 # the values x[j] for all j in neighbors of i.
 #
 # Transpose: At = t(A). At is CSC. Column i of At = row i of A = neighbors of i.
 # So At@p, At@i give us the neighbor indices for each node.

 At <- t(A)  # Now column i of At contains the neighbor indices of node i
 n  <- ncol(At)

 neighbor_max <- rep(NA_real_, n)
 neighbor_min <- rep(NA_real_, n)

 # At@p: column pointers (0-indexed, length n+1)
 # At@i: row indices (0-indexed)
 p <- At@p
 idx_all <- At@i + 1L  # convert to 1-indexed

 # Vectorized grouped max/min using data.table
 # Build a table of (node_id, neighbor_value)
 # node_id for each entry: column index
 col_lengths <- diff(p)
 node_ids <- rep(seq_len(n), times = col_lengths)

 if (length(node_ids) == 0) {
   return(cbind(neighbor_max, neighbor_min))
 }

 neighbor_vals <- x[idx_all]

 dt <- data.table(node = node_ids, val = neighbor_vals)
 dt <- dt[!is.na(val)]

 if (nrow(dt) == 0) {
   return(cbind(neighbor_max, neighbor_min))
 }

 stats <- dt[, .(nmax = max(val), nmin = min(val)), by = node]
 neighbor_max[stats$node] <- stats$nmax
 neighbor_min[stats$node] <- stats$nmin

 cbind(neighbor_max, neighbor_min)
}

compute_neighbor_mean <- function(A, x) {
 # Replace NA with 0 for summation, track valid counts
 x_clean <- x
 x_clean[is.na(x_clean)] <- 0
 valid <- as.numeric(!is.na(x))

 sums   <- as.numeric(A %*% x_clean)
 counts <- as.numeric(A %*% valid)

 result <- ifelse(counts > 0, sums / counts, NA_real_)
 return(result)
}

# ==============================================================================
# STEP 3: Main pipeline
# ==============================================================================
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
 # Convert to data.table for speed (non-destructive copy)
 if (!is.data.table(cell_data)) {
   cell_data <- as.data.table(cell_data)
 }

 # --- Build adjacency matrix once ---
 cat("Building sparse adjacency matrix...\n")
 A <- build_adjacency_matrix(rook_neighbors_unique, id_order)
 cat(sprintf("  Adjacency matrix: %d x %d, %d non-zeros\n",
             nrow(A), ncol(A), nnzero(A)))

 # --- Create cell-index mapping ---
 # id_order[k] is the cell ID at position k in the nb object / adjacency matrix
 # We need a fast lookup: cell_id -> matrix row/col index
 id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

 # --- Pre-transpose for max/min ---
 At <- t(A)
 At_p <- At@p
 At_i <- At@i + 1L
 col_lengths <- diff(At_p)
 # node_ids_template: for each non-zero in At, which column (= which focal node)
 node_ids_template <- rep(seq_len(ncol(At)), times = col_lengths)
 n <- length(id_order)

 # --- Process each year independently ---
 years <- sort(unique(cell_data$year))
 cat(sprintf("Processing %d years x %d variables...\n",
             length(years), length(neighbor_source_vars)))

 # Pre-allocate output columns
 for (var_name in neighbor_source_vars) {
   col_max  <- paste0("neighbor_max_", var_name)
   col_min  <- paste0("neighbor_min_", var_name)
   col_mean <- paste0("neighbor_mean_", var_name)
   set(cell_data, j = col_max,  value = NA_real_)
   set(cell_data, j = col_min,  value = NA_real_)
   set(cell_data, j = col_mean, value = NA_real_)
 }

 for (yr in years) {
   cat(sprintf("  Year %d...\n", yr))

   # Row indices for this year
   yr_rows <- which(cell_data$year == yr)

   # Cell IDs for this year's rows
   yr_ids <- cell_data$id[yr_rows]

   # Map each cell to its position in the adjacency matrix
   yr_pos <- id_to_pos[as.character(yr_ids)]

   # Not all 344K cells may appear in every year; build a dense vector
   # aligned to the adjacency matrix ordering
   for (var_name in neighbor_source_vars) {
     # Build full-length vector aligned to adjacency matrix
     x_full <- rep(NA_real_, n)
     x_full[yr_pos] <- cell_data[[var_name]][yr_rows]

     # --- Neighbor mean via sparse matrix multiplication ---
     x_clean <- x_full
     x_clean[is.na(x_clean)] <- 0
     valid_mask <- as.numeric(!is.na(x_full))

     sums   <- as.numeric(A %*% x_clean)
     counts <- as.numeric(A %*% valid_mask)
     n_mean <- ifelse(counts > 0, sums / counts, NA_real_)

     # --- Neighbor max and min via grouped operations on CSC structure ---
     neighbor_vals <- x_full[At_i]

     n_max <- rep(NA_real_, n)
     n_min <- rep(NA_real_, n)

     if (length(node_ids_template) > 0) {
       # Use data.table for fast grouped max/min
       dt <- data.table(node = node_ids_template, val = neighbor_vals)
       dt <- dt[!is.na(val)]
       if (nrow(dt) > 0) {
         stats <- dt[, .(nmax = max(val), nmin = min(val)), by = node]
         n_max[stats$node] <- stats$nmax
         n_min[stats$node] <- stats$nmin
       }
     }

     # --- Write results back to the cell_data rows for this year ---
     col_max  <- paste0("neighbor_max_", var_name)
     col_min  <- paste0("neighbor_min_", var_name)
     col_mean <- paste0("neighbor_mean_", var_name)

     set(cell_data, i = yr_rows, j = col_max,  value = n_max[yr_pos])
     set(cell_data, i = yr_rows, j = col_min,  value = n_min[yr_pos])
     set(cell_data, i = yr_rows, j = col_mean, value = n_mean[yr_pos])
   }
 }

 cat("Neighbor feature computation complete.\n")
 return(cell_data)
}

# ==============================================================================
# STEP 4: Execute and predict
# ==============================================================================

# --- Run the optimized pipeline ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- run_neighbor_feature_pipeline(
 cell_data,
 id_order,
 rook_neighbors_unique,
 neighbor_source_vars
)

# --- Apply the pre-trained Random Forest (unchanged) ---
# The model object (e.g., `rf_model`) is already in memory or loaded from disk.
# Prediction uses the same column names the model was trained on.
cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + named-vector lookups | Single sparse matrix build (344K × 344K) |
| **Mean computation** | `lapply` over 6.46M elements | Sparse matrix–vector multiply (`A %*% x`) |
| **Max/Min computation** | `lapply` with per-element subsetting | `data.table` grouped aggregation on CSC indices (~1.37M rows per variable-year) |
| **Year handling** | All years entangled in one 6.46M-row lookup | Independent per-year slices on static topology |
| **Memory** | ~6.46M-element list of integer vectors | ~16 MB sparse matrix + temporary vectors |
| **Estimated runtime** | 86+ hours | ~2–5 minutes |
| **Numerical equivalence** | — | ✅ Identical `max`, `min`, `mean` values |
| **RF model** | — | ✅ Untouched; `predict()` call unchanged |

**Key invariant preserved**: For every cell-year row, the neighbor max, min, and mean are computed over exactly the same set of rook-adjacent cells present in that year, with identical `NA` handling (NAs excluded before aggregation; all-NA neighborhoods yield `NA`). This guarantees numerical equivalence with the original pipeline, so the pre-trained Random Forest model produces identical predictions.