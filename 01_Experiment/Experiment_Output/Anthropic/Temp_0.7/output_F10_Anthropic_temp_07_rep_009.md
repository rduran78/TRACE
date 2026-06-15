 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each built via string-pasting and named-vector lookups. The `paste(id, year, sep="_")` key construction and `idx_lookup[neighbor_keys]` named-vector subsetting is O(n) per lookup in the worst case, and doing this 6.46M times is catastrophically slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a pure-R `lapply`**, extracting subsets of a vector by index, removing NAs, and computing three summary statistics. This is called 5 times (once per source variable), yielding ~32.3 million R-level loop iterations with per-element allocation overhead.

3. **The graph topology is year-invariant but the lookup is rebuilt entangling spatial topology with the temporal dimension.** Every cell has the same rook neighbors every year. The current code re-resolves neighbor identities across the full 6.46M-row panel, when it should resolve the ~344K spatial adjacency once and then broadcast across 28 years via vectorized matrix operations.

**Root cause:** The implementation treats the problem as a generic row-level operation on a long panel, rather than exploiting the fact that the adjacency graph is static across time and the aggregation (max, min, mean) can be expressed as sparse matrix–vector operations.

---

## Optimization Strategy

### Key Insight: Separate Space from Time

The rook adjacency is purely spatial (344,208 nodes, ~1.37M directed edges). The panel has 28 years. If the data is reshaped so that each variable is a **344,208 × 28 matrix** (cells × years), then neighbor aggregation becomes a sparse-matrix operation applied identically to each column (year).

### Concrete Steps

1. **Build a sparse adjacency matrix `A`** (344,208 × 344,208) from `rook_neighbors_unique` once. This is a binary CSC/CSR matrix using the `Matrix` package.

2. **Reshape each source variable into a dense matrix `V`** of dimension `n_cells × n_years`, where rows are ordered by `id_order`.

3. **Compute neighbor statistics via sparse matrix operations:**
   - **Mean:** `A %*% V / degree_vector` — a single sparse matrix multiply gives the sum of neighbor values; dividing by degree gives the mean.
   - **Max and Min:** These are not expressible as linear algebra. Use a grouped C++-level operation. With `dgCMatrix` column-compressed format, iterate over each row's non-zero entries in the adjacency matrix to extract neighbor values and compute max/min. This is implemented efficiently via `Rcpp` or, without compilation, via a chunked vectorized approach using the sparse matrix's slot structure.

4. **Flatten results back** to the long panel format and `cbind` to `cell_data`.

### Expected Speedup

| Component | Current | Optimized |
|---|---|---|
| Topology build | ~hours (6.46M string ops) | ~seconds (sparse matrix from nb) |
| Mean aggregation (per var) | ~hours (6.46M lapply) | ~1–2 sec (sparse matrix multiply) |
| Max/Min aggregation (per var) | embedded in above | ~30–60 sec (vectorized sparse slot ops) |
| **Total for 5 vars** | **86+ hours** | **~5–10 minutes** |

---

## Optimized R Code

```r
# ==============================================================================
# Optimized Neighbor Aggregation Pipeline
# Preserves numerical equivalence with original compute_neighbor_stats output.
# Requires: Matrix, data.table
# ==============================================================================

library(Matrix)
library(data.table)

# --------------------------------------------------------------------------
# STEP 0: Ensure cell_data is a data.table keyed by (id, year)
# --------------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# --------------------------------------------------------------------------
# STEP 1: Build the spatial adjacency sparse matrix (once)
#
# rook_neighbors_unique: an nb object (list of integer vectors) of length
#   n_cells = length(id_order) = 344,208.
# id_order: the cell IDs in the order corresponding to the nb object.
# --------------------------------------------------------------------------
build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj[[i]] contains integer indices of neighbors of node i.
  # spdep nb objects use 0L to indicate no neighbors.
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  
  # Remove zero entries (spdep convention for no-neighbor nodes)
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

n_cells <- length(id_order)
A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

# Degree vector (number of neighbors per cell) — used for mean computation
degree_vec <- as.numeric(rowSums(A))  # length n_cells

# --------------------------------------------------------------------------
# STEP 2: Establish mapping from id_order to row indices of the matrix,
#          and from cell_data rows to matrix positions.
# --------------------------------------------------------------------------

# Map: cell_id -> matrix row index (1..n_cells)
id_to_row <- setNames(seq_along(id_order), as.character(id_order))

# Sorted unique years
years_all <- sort(unique(cell_data$year))
n_years   <- length(years_all)
year_to_col <- setNames(seq_along(years_all), as.character(years_all))

# Key cell_data for fast access
setkey(cell_data, id, year)

# Pre-compute the matrix-row index and matrix-col index for every row of cell_data
# This lets us scatter results back to the long panel efficiently.
cell_data[, `:=`(
  mat_row = id_to_row[as.character(id)],
  mat_col = year_to_col[as.character(year)]
)]

# --------------------------------------------------------------------------
# STEP 3: Function to reshape a variable from long panel to n_cells x n_years matrix
# --------------------------------------------------------------------------
var_to_matrix <- function(dt, var_name, n_cells, n_years, id_to_row, year_to_col) {
  # Initialize with NA (to handle missing cell-year combos)
  M <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  rows <- id_to_row[as.character(dt$id)]
  cols <- year_to_col[as.character(dt$year)]
  M[cbind(rows, cols)] <- dt[[var_name]]
  M
}

# --------------------------------------------------------------------------
# STEP 4: Compute neighbor max and min using sparse matrix slot structure
#
# For a dgCMatrix A (column-compressed), to get row-wise neighbor aggregates
# we work with t(A) which is also dgCMatrix, where column j of t(A) holds the
# neighbors of node j.
#
# But for row-wise operations on A, it's more natural to convert A to dgRMatrix
# or transpose and work column-wise on t(A).
#
# Strategy: Use t(A) as dgCMatrix. Column j contains the neighbor indices of
# node j. For each node j, extract V[neighbors_of_j, ] and compute
# colMaxs / colMins.
#
# To avoid a 344K-iteration R loop, we use a fully vectorized approach:
# Expand the sparse structure into (node, neighbor, year) triples and use
# data.table grouped aggregation.
# --------------------------------------------------------------------------

compute_neighbor_features_optimized <- function(A, V, degree_vec) {
  # A: n_cells x n_cells sparse adjacency (dgCMatrix)
  # V: n_cells x n_years dense matrix of variable values
  # Returns list with max_mat, min_mat, mean_mat (each n_cells x n_years)
  
  n_cells <- nrow(V)
  n_years <- ncol(V)
  
  # --- MEAN: sparse matrix multiply ---
  # sum_mat[i,y] = sum of V[j,y] for all neighbors j of i
  sum_mat <- A %*% V  # sparse %*% dense -> dense, very fast
  
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  has_neighbors <- degree_vec > 0
  mean_mat[has_neighbors, ] <- as.matrix(sum_mat[has_neighbors, ]) /
    degree_vec[has_neighbors]
  
  # --- MAX and MIN: vectorized via sparse structure expansion ---
  # Convert A to triplet form to get (from_node, to_node) pairs
  At <- as(A, "TsparseMatrix")  # or use summary()
  from_node <- At@i + 1L  # 1-indexed row (the node)
  to_node   <- At@j + 1L  # 1-indexed col (the neighbor)
  
  # For memory efficiency, process years in chunks
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Number of edges
  n_edges <- length(from_node)
  
  # Process in year chunks to control memory
  # Each chunk: expand edges x chunk_years into a data.table, group by from_node
  chunk_size <- 7L  # years per chunk; 1.37M edges * 7 years ~ 9.6M rows, manageable
  
  year_chunks <- split(seq_len(n_years),
                       ceiling(seq_len(n_years) / chunk_size))
  
  for (yc in year_chunks) {
    n_yc <- length(yc)
    
    # Gather neighbor values for all edges and years in this chunk
    # neighbor_vals[e, t] = V[to_node[e], yc[t]]
    neighbor_vals <- V[to_node, yc, drop = FALSE]  # n_edges x n_yc matrix
    
    # For each (from_node, year_col), compute max and min
    # Use data.table for grouped aggregation
    # Expand from_node across year columns
    
    for (t_idx in seq_len(n_yc)) {
      yr_col <- yc[t_idx]
      vals   <- neighbor_vals[, t_idx]
      
      # Build a data.table of (node, value), drop NAs, aggregate
      dt_agg <- data.table(node = from_node, val = vals)
      dt_agg <- dt_agg[!is.na(val)]
      
      if (nrow(dt_agg) > 0) {
        agg <- dt_agg[, .(mx = max(val), mn = min(val)), by = node]
        max_mat[agg$node, yr_col] <- agg$mx
        min_mat[agg$node, yr_col] <- agg$mn
      }
    }
  }
  
  list(max_mat = max_mat, min_mat = min_mat, mean_mat = mean_mat)
}

# --------------------------------------------------------------------------
# STEP 5: Main loop — process each source variable, attach results to cell_data
# --------------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-compute the linear index for scattering matrix results back to cell_data rows
# cell_data$mat_row and cell_data$mat_col are already set
scatter_idx <- cbind(cell_data$mat_row, cell_data$mat_col)

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor features for:", var_name, "\n")
  
  # Reshape to matrix
  V <- var_to_matrix(cell_data, var_name, n_cells, n_years, id_to_row, year_to_col)
  
  # Compute neighbor stats
  stats <- compute_neighbor_features_optimized(A, V, degree_vec)
  
  # Scatter back to long panel using pre-computed indices
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  cell_data[, (max_col)  := stats$max_mat[scatter_idx]]
  cell_data[, (min_col)  := stats$min_mat[scatter_idx]]
  cell_data[, (mean_col) := stats$mean_mat[scatter_idx]]
  
  # Free memory
  rm(V, stats)
  gc()
  
  cat("  Done:", var_name, "\n")
}

# Clean up helper columns
cell_data[, `:=`(mat_row = NULL, mat_col = NULL)]

# --------------------------------------------------------------------------
# STEP 6: Predict with the pre-trained Random Forest (unchanged)
# --------------------------------------------------------------------------
# The trained model object (e.g., rf_model) is loaded from disk and used as-is.
# Ensure predictor columns match the model's expected feature names.
#
# predictions <- predict(rf_model, newdata = cell_data)
#
# The Random Forest is NOT retrained. Only the feature engineering is optimized.
# --------------------------------------------------------------------------

cat("Neighbor feature engineering complete.\n")
cat("Rows:", nrow(cell_data), " Columns:", ncol(cell_data), "\n")
```

---

## Why This Preserves Numerical Equivalence

| Statistic | Original | Optimized | Equivalence |
|---|---|---|---|
| **Max** | `max(vals[idx])` after removing NAs | `data.table [, max(val)]` after removing NAs on the same neighbor set | ✅ Identical — same set, same `max()` |
| **Min** | `min(vals[idx])` after removing NAs | `data.table [, min(val)]` after removing NAs on the same set | ✅ Identical |
| **Mean** | `mean(vals[idx])` after removing NAs | `(A %*% V)[i,y] / degree` — but only when no NAs exist. When NAs exist, the sum and count from the data.table path match | ✅ See note below |

**NA-handling note for mean:** The sparse matrix multiply `A %*% V` treats `NA` as propagating (any NA neighbor makes the sum NA). To exactly match the original's `na.rm=TRUE` behavior for mean when some neighbors have NA values, we add a correction:

```r
# If NA values are possible in source variables, replace the mean computation:
# Instead of A %*% V / degree, compute:
#   V_nona <- V; V_nona[is.na(V_nona)] <- 0
#   sum_mat <- A %*% V_nona
#   count_mat <- A %*% (!is.na(V)) * 1.0
#   mean_mat <- ifelse(count_mat > 0, sum_mat / count_mat, NA)
```

Here is that drop-in replacement for the mean section inside `compute_neighbor_features_optimized` if NAs are present in any source variable:

```r
  # --- MEAN (NA-safe version) ---
  V_nona <- V
  V_nona[is.na(V_nona)] <- 0
  sum_mat   <- A %*% V_nona
  valid_mat <- matrix(as.numeric(!is.na(V)), nrow = n_cells, ncol = n_years)
  count_mat <- A %*% valid_mat
  
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  pos <- count_mat > 0
  mean_mat[pos] <- as.matrix(sum_mat)[pos] / as.matrix(count_mat)[pos]
```

This guarantees `mean_mat[i,y] = mean(V[neighbors_of_i, y], na.rm=TRUE)`, numerically identical to the original.

---

## Memory Budget

| Object | Size |
|---|---|
| Sparse matrix `A` (1.37M entries) | ~22 MB |
| Dense matrix `V` (344K × 28) | ~77 MB |
| Triplet expansion `from_node`, `to_node` | ~11 MB |
| `neighbor_vals` per chunk (1.37M × 7) | ~77 MB |
| `data.table` per year aggregation | ~22 MB |
| `cell_data` (~6.46M × 110 cols) | ~5.7 GB |
| **Peak total** | **~7–8 GB** ✅ fits in 16 GB |

**Estimated total runtime: 5–10 minutes** on a standard 16 GB laptop, down from 86+ hours.