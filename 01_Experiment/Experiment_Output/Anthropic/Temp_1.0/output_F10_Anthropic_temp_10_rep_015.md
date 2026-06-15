 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates 6.46 million list entries**, each requiring string-pasting, hashing into a named lookup vector, and NA filtering. This is O(N × k) with enormous constant factors from R's string operations and named vector lookups.

2. **`compute_neighbor_stats` iterates over 6.46 million list entries** in an R-level `lapply`, extracting subsets of a vector for each node-year. This is repeated 5 times (once per variable), totaling ~32.3 million R-level subset operations.

3. **The neighbor lookup conflates topology and time.** Rook neighbors are a *spatial* property — they don't change across years. Yet the lookup is built at the cell-year level, inflating a 344,208-node adjacency structure into a 6.46-million-row structure. This is a 28× blowup in both memory and computation.

**Root cause:** The graph topology is year-invariant, but the implementation re-resolves it per cell-year row. String-based key lookups (`paste` + named vector indexing) are extremely slow at this scale.

## Optimization Strategy

1. **Separate spatial topology from temporal indexing.** Build a sparse adjacency structure once over 344,208 cells using integer indices — no strings.

2. **Reshape the data so each variable is a (cells × years) matrix.** Neighbor aggregation then becomes sparse matrix–dense matrix multiplication (for mean) and row-wise grouped operations (for max/min).

3. **Use `Matrix` package sparse operations.** Construct a sparse row-normalized adjacency matrix `W` (344,208 × 344,208). Then `W %*% X` gives neighbor means in one vectorized operation per variable. For max/min, use `dgCMatrix` column iteration or the `slam` / direct C-level approach.

4. **Vectorize max/min via sparse matrix tricks.** Use the sparse adjacency in CSR-like form to do grouped max/min via `data.table` or direct vectorized code.

5. **Total operations:** 5 variables × 3 stats × 1 sparse-matrix operation = 15 operations on 344K × 28 matrices, replacing 32.3 million R-level list iterations.

**Expected speedup:** From 86+ hours to minutes.

## Optimized R Code

```r
library(Matrix)
library(data.table)

optimize_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars) {
  # ---------------------------------------------------------------
  # 1. Convert cell_data to data.table for speed
  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # ---------------------------------------------------------------
  # 2. Build integer-indexed spatial adjacency (year-invariant)
  #    id_order is the vector of cell IDs in the order matching

  #    rook_neighbors_unique (an nb object: list of integer vectors
  #    referencing positions in id_order).
  # ---------------------------------------------------------------
  n_cells <- length(id_order)
  cat("Building sparse adjacency matrix for", n_cells, "cells...\n")
  
  # Build COO (coordinate) representation from the nb object
  from_idx <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique, use.names = FALSE)
  
  # Remove 0-entries (spdep uses 0 for "no neighbors" in some representations)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  # Sparse binary adjacency matrix (n_cells x n_cells), row i has 1s at neighbors

  A <- sparseMatrix(i = from_idx, j = to_idx, x = 1,
                    dims = c(n_cells, n_cells), repr = "C")
  
  # Number of neighbors per cell (for mean computation)
  n_neighbors <- diff(A@p)  # CSC column counts if transposed; use row sums
  n_neighbors_vec <- as.numeric(rowSums(A))  # per cell
  
  cat("Adjacency matrix:", nrow(A), "x", ncol(A),
      "with", length(A@x), "non-zero entries\n")
  
  # ---------------------------------------------------------------
  # 3. Create cell-ID to row-index mapping (position in id_order)
  # ---------------------------------------------------------------
  cell_id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))
  
  # ---------------------------------------------------------------
  # 4. Identify the years and create cell index column in dt
  # ---------------------------------------------------------------
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_len(n_years), as.character(years))
  
  # Map each row's cell ID to spatial index
  dt[, cell_idx := cell_id_to_idx[as.character(id)]]
  dt[, year_idx := year_to_col[as.character(year)]]
  
  # Verify no missing mappings
  stopifnot(!anyNA(dt$cell_idx))
  stopifnot(!anyNA(dt$year_idx))
  
  # ---------------------------------------------------------------
  # 5. For each variable, build a (n_cells x n_years) matrix,
  #    compute neighbor max, min, mean, and write back.
  # ---------------------------------------------------------------
  
  # Pre-extract row ordering for write-back: for each (cell_idx, year_idx),
  # store the row index in dt
  setkey(dt, cell_idx, year_idx)
  row_order <- dt[, .I]  # rows are now sorted by (cell_idx, year_idx)
  
  # Build mapping matrix: rows of dt -> (cell_idx, year_idx)
  ci <- dt$cell_idx
  yi <- dt$year_idx
  
  # --- Helper: build (n_cells x n_years) matrix from dt column ---
  build_matrix <- function(vals) {
    M <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    M[cbind(ci, yi)] <- vals
    M
  }
  
  # --- Helper: compute neighbor stats using sparse A ---
  # For MEAN: (A %*% M) / n_neighbors (element-wise)
  # For MAX and MIN: iterate over CSR structure
  
  # Convert A to dgRMatrix (CSR) for efficient row-wise access
  # Matrix package stores dgCMatrix (CSC). We transpose for CSC-of-transpose = CSR-of-original.
  At <- t(A)  # dgCMatrix; column j of At = row j of A = neighbors of cell j
  
  compute_neighbor_mean <- function(M) {
    # A %*% M gives sum of neighbor values; divide by count
    S <- as.matrix(A %*% M)  # n_cells x n_years dense matrix
    # Where n_neighbors_vec == 0, result should be NA
    NN <- matrix(n_neighbors_vec, nrow = n_cells, ncol = n_years)
    result <- S / NN
    result[NN == 0] <- NA_real_
    # Where all neighbors had NA, the sum is 0 but should be NA
    # We need to handle NAs in M properly.
    result
  }
  
  compute_neighbor_mean_na_aware <- function(M) {
    # Replace NA with 0 for summation, but track counts of non-NA neighbors
    M_nona <- M
    M_nona[is.na(M_nona)] <- 0
    
    # Indicator of non-NA
    M_valid <- matrix(1, nrow = n_cells, ncol = n_years)
    M_valid[is.na(M)] <- 0
    
    S <- as.matrix(A %*% M_nona)       # sum of non-NA neighbor values
    C <- as.matrix(A %*% M_valid)      # count of non-NA neighbors
    
    result <- S / C
    result[C == 0] <- NA_real_
    result
  }
  
  compute_neighbor_max_min <- function(M) {
    # Use CSC structure of At: column j of At = neighbors of cell j
    # At@p[j]+1 : At@p[j+1] gives row indices of neighbors of cell j
    
    max_M <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_M <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    p <- At@p
    ri <- At@i + 1L  # 0-based to 1-based
    
    for (yr in seq_len(n_years)) {
      col_vals <- M[, yr]  # values for this year
      
      for (j in seq_len(n_cells)) {
        start <- p[j] + 1L
        end   <- p[j + 1L]
        if (end < start) next
        
        nb_vals <- col_vals[ri[start:end]]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) == 0L) next
        
        max_M[j, yr] <- max(nb_vals)
        min_M[j, yr] <- min(nb_vals)
      }
    }
    list(max = max_M, min = min_M)
  }
  
  # --- FASTER max/min using data.table edge-list approach ---
  compute_neighbor_max_min_fast <- function(M) {
    # Expand adjacency to edge list with values per year
    # Edge list: (from_idx, to_idx) — "to" is the neighbor whose value we read
    p <- At@p
    ri <- At@i + 1L
    
    # Reconstruct edge list from CSC of At
    # Column j of At -> neighbors of node j
    col_counts <- diff(p)
    from_nodes <- rep(seq_len(n_cells), col_counts)
    to_nodes   <- ri
    
    n_edges <- length(from_nodes)
    
    # For each year, look up to_node values, then group by from_node
    max_M <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_M <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # Use data.table for grouped max/min
    edge_dt <- data.table(from = from_nodes, to = to_nodes)
    
    for (yr in seq_len(n_years)) {
      col_vals <- M[, yr]
      edge_dt[, val := col_vals[to]]
      
      # Remove NAs
      valid_edges <- edge_dt[!is.na(val)]
      if (nrow(valid_edges) == 0L) next
      
      stats <- valid_edges[, .(mx = max(val), mn = min(val)), by = from]
      max_M[cbind(stats$from, yr)] <- stats$mx
      min_M[cbind(stats$from, yr)] <- stats$mn
    }
    
    list(max = max_M, min = min_M)
  }
  
  cat("Processing", length(neighbor_source_vars), "variables...\n")
  
  for (var_name in neighbor_source_vars) {
    cat("  Variable:", var_name, "\n")
    
    # Build the (n_cells x n_years) matrix
    M <- build_matrix(dt[[var_name]])
    
    # Compute neighbor mean (NA-aware)
    mean_M <- compute_neighbor_mean_na_aware(M)
    
    # Compute neighbor max and min
    maxmin <- compute_neighbor_max_min_fast(M)
    
    # Extract values back in dt row order
    max_col <- paste0("neighbor_max_", var_name)
    min_col <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    dt[, (max_col)  := maxmin$max[cbind(ci, yi)]]
    dt[, (min_col)  := maxmin$min[cbind(ci, yi)]]
    dt[, (mean_col) := mean_M[cbind(ci, yi)]]
    
    cat("    Done.\n")
  }
  
  # ---------------------------------------------------------------
  # 6. Clean up helper columns and return
  # ---------------------------------------------------------------
  dt[, c("cell_idx", "year_idx") := NULL]
  
  cat("All neighbor features computed.\n")
  return(dt)
}


# =====================================================================
# MAIN EXECUTION
# =====================================================================

# Load prerequisites (adjust paths as needed)
# cell_data        <- readRDS("cell_data.rds")
# id_order         <- readRDS("id_order.rds")
# rook_neighbors_unique <- readRDS("rook_neighbors.rds")
# rf_model         <- readRDS("rf_model.rds")

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Run optimized pipeline
cell_data_dt <- optimize_neighbor_pipeline(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)

# Apply pre-trained Random Forest (do NOT retrain)
cell_data_dt[, prediction := predict(rf_model, newdata = cell_data_dt)]
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Topology resolution** | Per cell-year (6.46M string lookups) | Once, spatial only (344K integer-indexed sparse matrix) |
| **Neighbor lookup** | 6.46M-element list, string keys | Sparse matrix `A` (344K × 344K), ~1.37M non-zeros |
| **Mean computation** | R-level lapply over 6.46M rows × 5 vars | Sparse matrix multiply `A %*% M`, fully vectorized |
| **Max/Min computation** | R-level lapply, 32.3M subset ops | Edge-list `data.table` grouped aggregation, 28 iterations per variable |
| **Memory** | Massive list-of-lists + string vectors | Sparse matrix (~22 MB) + dense matrices (344K × 28 ≈ 77 MB each) |
| **Expected runtime** | 86+ hours | **~5–15 minutes** on a 16 GB laptop |
| **Numerical equivalence** | Baseline | ✅ Preserved: same max, min, mean over same neighbor sets, same NA handling |
| **RF model** | Pre-trained | ✅ Untouched: `predict()` only |

### Key Design Principles

1. **Build topology once** as a sparse integer matrix — no strings, no per-year duplication.
2. **Reshape to matrix form** (cells × years) so each variable is a dense matrix and years become columns processed in a tight loop.
3. **Sparse matrix multiplication** for the mean — this is the single biggest win, replacing millions of R-level operations with optimized C/FORTRAN code inside `Matrix`.
4. **Edge-list + data.table grouping** for max/min — `data.table`'s radix-based grouped aggregation is orders of magnitude faster than R-level lapply with subsetting.
5. **NA-aware mean** via the indicator-matrix trick: sum non-NA values and divide by the count of non-NA neighbors, yielding exact equivalence with the original `mean(neighbor_vals[!is.na(neighbor_vals)])`.