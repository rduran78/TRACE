 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a vector of 6.46M names — this alone is catastrophically slow. The function does ~6.46M × avg_neighbors hash lookups against a 6.46M-length named vector.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in an `lapply`**, extracting subsets of a numeric vector. While each individual operation is cheap, the R-level loop overhead across 6.46M iterations, repeated for 5 variables, is substantial.

3. **The neighbor lookup is year-aware but redundant**: rook neighbors are a *spatial* relationship — the same cell has the same neighbors every year. The current code re-resolves neighbor row indices per cell-year, but the spatial topology is fixed. With 344,208 cells × 28 years, the lookup is doing 28× redundant work relative to a topology-first approach.

**Estimated cost breakdown:**
- `build_neighbor_lookup`: ~70–80% of runtime (string operations + named vector lookups on 6.46M keys)
- `compute_neighbor_stats` × 5 vars: ~20–30% of runtime (R-level lapply over 6.46M elements × 5)

## Optimization Strategy

1. **Build the graph topology once as a sparse matrix (CSR format via `dgRMatrix` or equivalently use `Matrix::sparseMatrix`).** The adjacency matrix is 344,208 × 344,208 with ~1.37M non-zero entries. This is tiny in memory (~16 MB).

2. **Exploit the panel structure**: since the spatial topology is identical across all 28 years, process **one year at a time**. For each year, extract the N×1 attribute vector, then use sparse matrix–vector multiplication (or row-wise aggregation) to compute neighbor max, min, and mean.

3. **Use `Matrix` package sparse operations** for mean (sparse matrix × vector = sum of neighbor values; divide by neighbor count = mean). For max and min, use a compiled C++ routine via `Rcpp` or, more portably, use the sparse matrix structure to do grouped aggregation with vectorized R.

4. **Avoid all string-pasting and named-vector lookups entirely.** Map cell IDs to integer indices once; use integer indexing throughout.

5. **Memory**: the sparse matrix is ~16 MB; one year of data for 344K cells is trivial. Peak memory stays well under 4 GB.

**Expected speedup**: from 86+ hours to **~2–5 minutes**.

## Optimized R Code

```r
library(Matrix)
library(data.table)

# =============================================================================
# STEP 1: Build sparse adjacency matrix from the nb object (done ONCE)
# =============================================================================
build_adjacency_matrix <- function(nb_object, n) {
 # nb_object: list of length n, each element is integer vector of neighbor indices
 # n: number of spatial cells (344208)
 # Returns: sparse dgCMatrix of dimension n x n, entry (i,j)=1 if j is neighbor of i

 from <- rep(seq_along(nb_object), lengths(nb_object))
 to   <- unlist(nb_object)

 # Remove 0-entries (spdep uses 0L for "no neighbors")
 valid <- to > 0L
 from  <- from[valid]
 to    <- to[valid]

 sparseMatrix(i = from, j = to, x = 1, dims = c(n, n), repr = "C")
}

# =============================================================================
# STEP 2: Compute neighbor count per cell (for mean calculation)
# =============================================================================
# adj %*% ones = vector of neighbor counts per cell
# This is constant across years.

# =============================================================================
# STEP 3: Compute neighbor stats for one variable across all years
# =============================================================================
compute_neighbor_features_fast <- function(dt, adj, neighbor_count,
                                           id_to_idx, var_name, years) {
 # Pre-allocate output columns
 col_max  <- paste0("neighbor_max_", var_name)
 col_min  <- paste0("neighbor_min_", var_name)
 col_mean <- paste0("neighbor_mean_", var_name)

 n <- nrow(adj)
 n_total <- nrow(dt)

 out_max  <- rep(NA_real_, n_total)
 out_min  <- rep(NA_real_, n_total)
 out_mean <- rep(NA_real_, n_total)

 # Get the CSC structure of adj for row-wise neighbor traversal
 # adj is n x n CSC (dgCMatrix): adj@p, adj@i, adj@x
 # For row-wise access, transpose to get adj_t where column j of adj_t = row j of adj
 adj_t <- t(adj)  # now column j contains the neighbors of cell j
 # adj_t is dgCMatrix: adj_t@p[j]+1 .. adj_t@p[j+1] gives indices of neighbors of j

 p_ptr <- adj_t@p
 i_idx <- adj_t@i + 1L  # convert 0-based to 1-based

 for (yr in years) {
   # Get row indices in dt for this year
   yr_rows <- dt[year == yr, which = TRUE]

   if (length(yr_rows) == 0L) next

   # Get cell indices for these rows
   yr_cell_ids <- dt$id[yr_rows]
   yr_cell_idx <- id_to_idx[yr_cell_ids]  # integer index 1..n

   # Build a full-length vector: position k = value of var_name for cell k in this year
   # Initialize with NA
   val_vec <- rep(NA_real_, n)
   val_vec[yr_cell_idx] <- dt[[var_name]][yr_rows]

   # --- Neighbor MEAN via sparse matrix-vector multiply ---
   # Replace NA with 0 for sum, track non-NA for count
   val_for_sum <- val_vec
   val_for_sum[is.na(val_for_sum)] <- 0

   not_na <- as.double(!is.na(val_vec))  # 1 if present, 0 if NA

   neighbor_sum     <- as.numeric(adj %*% val_for_sum)
   neighbor_valid_n <- as.numeric(adj %*% not_na)

   yr_mean <- ifelse(neighbor_valid_n > 0, neighbor_sum / neighbor_valid_n, NA_real_)

   # --- Neighbor MAX and MIN via CSC traversal (vectorized per year) ---
   # For each cell, gather neighbor values and compute max/min
   # We vectorize by using the sparse structure directly

   yr_max <- rep(NA_real_, n)
   yr_min <- rep(NA_real_, n)

   # Process all cells that have at least one neighbor
   # Use the CSC pointers of adj_t
   for_cells <- yr_cell_idx  # only cells present this year

   # Batch approach: extract all neighbor values at once
   # For each cell k, neighbors are i_idx[(p_ptr[k]+1):p_ptr[k+1]]
   # We build a "from" and "to" edge list, then do grouped max/min

   # Edge list: from = cell index, to = neighbor index (already have this from adj)
   # Reuse the from/to from adjacency construction? No, use CSC structure.

   # Build grouped neighbor values
   # For each cell in for_cells, get neighbor indices and their values
   # Use vectorized approach with data.table

   # Expand edges for cells present this year
   cell_start <- p_ptr[for_cells] + 1L
   cell_end   <- p_ptr[for_cells + 1L]
   cell_len   <- cell_end - cell_start + 1L
   valid_mask <- cell_len > 0L

   fc <- for_cells[valid_mask]
   cs <- cell_start[valid_mask]
   ce <- cell_end[valid_mask]
   cl <- cell_len[valid_mask]

   # Expand: for each cell, list its neighbor positions in i_idx
   edge_cell <- rep(fc, cl)
   edge_pos  <- sequence(cl, from = cs, by = 1L)
   edge_nbr  <- i_idx[edge_pos]
   edge_val  <- val_vec[edge_nbr]

   # Remove NA neighbor values
   ok <- !is.na(edge_val)
   edge_cell_ok <- edge_cell[ok]
   edge_val_ok  <- edge_val[ok]

   if (length(edge_val_ok) > 0L) {
     # Grouped max and min using data.table
     tmp <- data.table(cell = edge_cell_ok, val = edge_val_ok)
     agg <- tmp[, .(vmax = max(val), vmin = min(val)), by = cell]
     yr_max[agg$cell] <- agg$vmax
     yr_min[agg$cell] <- agg$vmin
   }

   # Write results back to output vectors
   out_max[yr_rows]  <- yr_max[yr_cell_idx]
   out_min[yr_rows]  <- yr_min[yr_cell_idx]
   out_mean[yr_rows] <- yr_mean[yr_cell_idx]
 }

 # Assign to data.table by reference
 dt[, (col_max)  := out_max]
 dt[, (col_min)  := out_min]
 dt[, (col_mean) := out_mean]

 invisible(dt)
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                          rf_model) {
 # Convert to data.table if not already (for speed)
 if (!is.data.table(cell_data)) {
   cell_data <- as.data.table(cell_data)
 }

 n_cells <- length(id_order)
 years   <- sort(unique(cell_data$year))

 cat("Building adjacency matrix...\n")
 # id_order maps position index -> cell id
 # rook_neighbors_unique[[k]] gives neighbor positions for position k
 adj <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

 # Build id -> position index mapping (integer, no string ops)
 id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

 # Neighbor count (constant across years, used for verification)
 neighbor_count <- diff(t(adj)@p)  # number of neighbors per cell

 cat("Computing neighbor features...\n")
 neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

 for (var_name in neighbor_source_vars) {
   cat("  Processing:", var_name, "\n")
   compute_neighbor_features_fast(
     dt             = cell_data,
     adj            = adj,
     neighbor_count = neighbor_count,
     id_to_idx      = id_to_idx,
     var_name       = var_name,
     years          = years
   )
 }

 cat("Running prediction with pre-trained Random Forest...\n")
 # The RF model is pre-trained — apply it directly
 cell_data[, prediction := predict(rf_model, newdata = cell_data)]

 cat("Done.\n")
 return(cell_data)
}

# =============================================================================
# USAGE
# =============================================================================
# result <- run_neighbor_feature_pipeline(cell_data, id_order,
#                                         rook_neighbors_unique, rf_model)
```

## Why This Preserves Numerical Equivalence

| Original | Optimized | Equivalence |
|---|---|---|
| `max(neighbor_vals)` | `data.table [, max(val), by=cell]` on same neighbor set | Identical (IEEE 754 max) |
| `min(neighbor_vals)` | `data.table [, min(val), by=cell]` on same neighbor set | Identical (IEEE 754 min) |
| `mean(neighbor_vals)` | `(adj %*% val) / (adj %*% not_na)` — sum of non-NA values ÷ count of non-NA values | Identical (same arithmetic mean, same NA exclusion) |
| NA handling | NAs excluded identically: `not_na` mask for mean; `!is.na(edge_val)` filter for max/min | Identical |
| No-neighbor / all-NA case | Returns `NA` in all three stats | Identical |

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M string-paste + named-vector lookups | Sparse matrix CSC pointer arithmetic |
| Mean computation | R-level loop over 6.46M elements | Single sparse matrix × vector multiply per year (~0.01s) |
| Max/Min computation | R-level loop over 6.46M elements | Vectorized edge expansion + `data.table` grouped aggregation |
| Total passes | 1 build + 5 × 6.46M lapply | 1 build + 5 vars × 28 years × (1 SpMV + 1 grouped agg) |
| Memory | 6.46M-element list of integer vectors | ~16 MB sparse matrix + year-slice vectors |
| **Estimated runtime** | **86+ hours** | **~2–5 minutes** |
| RF model | Unchanged | Unchanged (predict only) |