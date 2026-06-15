 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via expensive string-pasting and named-vector lookups (`paste` + `setNames` + named indexing). Named vector lookups in R are O(n) hash probes on a 6.46M-element character vector — repeated 6.46M times. This alone is O(n²)-like in practice.

2. **`compute_neighbor_stats` iterates over 6.46M list elements** with `lapply`, extracting subsets of a numeric vector by index, removing NAs, and computing max/min/mean one row at a time. The per-element R overhead (function call, subsetting, `is.na`, three aggregation calls) dominates — there is no vectorization.

3. **The neighbor lookup is monolithic across all years.** But the graph topology (which cell neighbors which cell) is *time-invariant*. The 344,208-cell rook adjacency is duplicated across 28 years in the lookup, inflating from ~1.37M edges to ~38.5M index pairs stored in nested lists of lists.

**Net effect:** ~86+ hours driven by R-level loop overhead on millions of small operations, plus massive memory pressure from redundant data structures on a 16 GB laptop.

---

## Optimization Strategy

### Key Insight: Separate Topology from Time

The rook adjacency graph is **static** — it does not change across years. We should:

1. **Build the sparse adjacency matrix once** (344,208 × 344,208) from the `nb` object — a `dgCMatrix` with ~1.37M nonzero entries (~22 MB).
2. **Reshape each variable into a 344,208 × 28 matrix** (cells × years).
3. **Compute neighbor aggregates via sparse matrix–dense matrix multiplication** and analogous sparse-max/sparse-min operations — fully vectorized, year-parallel.

### Specific Techniques

| Operation | Method |
|---|---|
| **Neighbor mean** | `A_norm %*% X` where `A_norm` is the row-normalized adjacency (each row sums to 1, or to 1/degree). This is a single sparse matrix × dense matrix multiply — O(nnz × 28). |
| **Neighbor sum & count** | `A %*% X` gives sum; row-degree gives count; mean = sum/count. Handles NA via a parallel mask matrix. |
| **Neighbor max / min** | Iterate over sparse rows using the CSC/CSR structure in compiled code. We use `data.table` grouped operations on the edge list for max and min, which is extremely fast. |

### Complexity Comparison

| | Original | Optimized |
|---|---|---|
| Neighbor lookup build | O(N_rows × k) with string ops | O(nnz) integer sparse matrix construction |
| Mean (per variable) | O(N_rows × k) R-level loops | O(nnz × T) single sparse matmul |
| Max/Min (per variable) | O(N_rows × k) R-level loops | O(nnz × T) vectorized `data.table` grouped ops |
| Total time estimate | 86+ hours | **~2–10 minutes** |

### Numerical Equivalence

- Mean is computed as `sum_of_non_NA_neighbors / count_of_non_NA_neighbors` — identical to the original.
- Max and min are computed over exactly the same neighbor sets with NA exclusion — identical to the original.
- The trained Random Forest model is loaded and applied with `predict()` — never retrained.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Sparse graph topology × dense attribute matrices
# Numerically equivalent to the original loop-based implementation
# =============================================================================

library(Matrix)
library(data.table)

# ---- 1. Build sparse adjacency matrix ONCE from the nb object ---------------

build_adjacency_matrix <- function(nb_obj, n) {
 # nb_obj: spdep nb object (list of integer vectors of neighbor indices)
 # n: number of spatial cells (length of nb_obj)
 # Returns: sparse dgCMatrix of dimension n x n (binary adjacency)

 from <- rep(seq_len(n), times = vapply(nb_obj, length, integer(1)))
 to   <- unlist(nb_obj, use.names = FALSE)

 # Remove any 0-length or out-of-range entries
 valid <- to >= 1L & to <= n
 from  <- from[valid]
 to    <- to[valid]

 sparseMatrix(i = from, j = to, x = 1, dims = c(n, n), repr = "C")
}

# ---- 2. Reshape panel data to cell × year matrices --------------------------

reshape_to_matrix <- function(cell_dt, id_order, years, var_name) {
 # cell_dt:  data.table with columns: id, year, <var_name>
 # id_order: integer vector of cell IDs defining row order
 # years:    sorted integer vector of years defining column order
 # Returns:  numeric matrix [n_cells x n_years]

 n_cells <- length(id_order)
 n_years <- length(years)

 # Map cell id -> row index, year -> col index
 id_map   <- setNames(seq_along(id_order), as.character(id_order))
 year_map <- setNames(seq_along(years), as.character(years))

 row_idx <- id_map[as.character(cell_dt$id)]
 col_idx <- year_map[as.character(cell_dt$year)]

 mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
 mat[cbind(row_idx, col_idx)] <- cell_dt[[var_name]]
 mat
}

# ---- 3. Compute neighbor MEAN via sparse matrix multiplication ---------------

compute_neighbor_mean_sparse <- function(A, X) {
 # A: binary sparse adjacency matrix [n x n]
 # X: dense matrix [n x T] (may contain NAs)
 # Returns: dense matrix [n x T] of neighbor means (NA where no valid neighbors)

 # Mask: 1 where X is not NA, 0 where NA
 notNA <- matrix(1, nrow = nrow(X), ncol = ncol(X))
 notNA[is.na(X)] <- 0

 # Replace NAs with 0 for summation
 X0 <- X
 X0[is.na(X0)] <- 0

 # Neighbor sums and counts via sparse matmul
 neighbor_sum   <- A %*% X0       # [n x T]
 neighbor_count <- A %*% notNA    # [n x T]

 # Convert to dense
 neighbor_sum   <- as.matrix(neighbor_sum)
 neighbor_count <- as.matrix(neighbor_count)

 # Mean = sum / count; NA where count == 0
 result <- neighbor_sum / neighbor_count
 result[neighbor_count == 0] <- NA_real_
 result
}

# ---- 4. Compute neighbor MAX and MIN via edge-list + data.table --------------

compute_neighbor_max_min_sparse <- function(A, X) {
 # A: binary sparse adjacency matrix [n x n] (dgCMatrix)
 # X: dense matrix [n x T]
 # Returns: list(max = [n x T], min = [n x T])

 n_cells <- nrow(X)
 n_years <- ncol(X)

 # Extract edge list from sparse matrix (1-indexed)
 A_t <- as(A, "TsparseMatrix")  # triplet form
 from_idx <- A_t@i + 1L
 to_idx   <- A_t@j + 1L
 n_edges  <- length(from_idx)

 # Pre-allocate result matrices
 max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
 min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

 # Process one year-column at a time to control memory
 # (each year: edge-list table with ~1.37M rows — trivial for data.table)
 for (t in seq_len(n_years)) {
   neighbor_vals <- X[to_idx, t]

   # Build edge table for this year
   dt <- data.table(
     from = from_idx,
     val  = neighbor_vals
   )

   # Remove edges where neighbor value is NA
   dt <- dt[!is.na(val)]

   if (nrow(dt) == 0L) next

   # Grouped max and min
   agg <- dt[, .(vmax = max(val), vmin = min(val)), by = from]

   max_mat[agg$from, t] <- agg$vmax
   min_mat[agg$from, t] <- agg$vmin
 }

 list(max = max_mat, min = min_mat)
}

# ---- 5. Write results back to the panel data.table --------------------------

write_matrix_to_dt <- function(cell_dt, mat, id_order, years, col_name) {
 # mat: [n_cells x n_years] result matrix
 # Writes values back into cell_dt by matching id and year

 id_map   <- setNames(seq_along(id_order), as.character(id_order))
 year_map <- setNames(seq_along(years), as.character(years))

 row_idx <- id_map[as.character(cell_dt$id)]
 col_idx <- year_map[as.character(cell_dt$year)]

 cell_dt[[col_name]] <- mat[cbind(row_idx, col_idx)]
 invisible(cell_dt)
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================

run_optimized_pipeline <- function(cell_data,
                                   id_order,
                                   rook_neighbors_unique,
                                   rf_model,
                                   neighbor_source_vars = c("ntl", "ec",
                                                            "pop_density",
                                                            "def",
                                                            "usd_est_n2")) {

 cat("Converting to data.table...\n")
 cell_dt <- as.data.table(cell_data)
 setkey(cell_dt, id, year)

 n_cells <- length(id_order)
 years   <- sort(unique(cell_dt$year))
 n_years <- length(years)

 cat(sprintf("Grid: %d cells x %d years = %d rows\n",
             n_cells, n_years, nrow(cell_dt)))

 # ---- Step 1: Build sparse adjacency matrix (once) -------------------------
 cat("Building sparse adjacency matrix...\n")
 A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
 cat(sprintf("Adjacency: %d nonzeros (directed edges)\n", nnzero(A)))

 # ---- Step 2: For each source variable, compute neighbor features ----------
 for (var_name in neighbor_source_vars) {
   cat(sprintf("Processing variable: %s\n", var_name))

   # Reshape to matrix
   X <- reshape_to_matrix(cell_dt, id_order, years, var_name)

   # Neighbor mean (sparse matmul)
   cat("  Computing neighbor mean (sparse matmul)...\n")
   mean_mat <- compute_neighbor_mean_sparse(A, X)

   # Neighbor max and min (edge-list + data.table)
   cat("  Computing neighbor max/min (data.table grouped ops)...\n")
   maxmin <- compute_neighbor_max_min_sparse(A, X)

   # Write back to data.table
   max_col  <- paste0(var_name, "_neighbor_max")
   min_col  <- paste0(var_name, "_neighbor_min")
   mean_col <- paste0(var_name, "_neighbor_mean")

   write_matrix_to_dt(cell_dt, maxmin$max, id_order, years, max_col)
   write_matrix_to_dt(cell_dt, maxmin$min, id_order, years, min_col)
   write_matrix_to_dt(cell_dt, mean_mat,   id_order, years, mean_col)

   # Free memory
   rm(X, mean_mat, maxmin)
   gc(verbose = FALSE)

   cat(sprintf("  Done: added %s, %s, %s\n", max_col, min_col, mean_col))
 }

 # ---- Step 3: Apply pre-trained Random Forest (no retraining) --------------
 cat("Applying pre-trained Random Forest model...\n")

 # Convert back to data.frame for predict() compatibility
 cell_data_out <- as.data.frame(cell_dt)

 predictions <- predict(rf_model, newdata = cell_data_out)
 cell_data_out$rf_prediction <- predictions

 cat("Pipeline complete.\n")
 cell_data_out
}

# =============================================================================
# USAGE EXAMPLE
# =============================================================================
#
# # Load pre-computed objects
# load("cell_data.RData")            # cell_data data.frame
# load("id_order.RData")             # id_order integer vector
# load("rook_neighbors.RData")       # rook_neighbors_unique (nb object)
# load("rf_model.RData")             # rf_model (trained randomForest)
#
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = rf_model
# )
```

---

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized | Equivalence |
|---|---|---|---|
| **Mean** | `mean(vals[!is.na(vals)])` = sum/count | `(A %*% X0) / (A %*% notNA)` where X0 has NAs replaced with 0 | Identical: same sum, same count per node-year |
| **Max** | `max(vals[!is.na(vals)])` | `data.table [!is.na(val), max(val), by=from]` | Identical: same neighbor set, same `max()` call |
| **Min** | `min(vals[!is.na(vals)])` | `data.table [!is.na(val), min(val), by=from]` | Identical: same neighbor set, same `min()` call |
| **NA rule** | Returns `c(NA,NA,NA)` when zero valid neighbors | Matrix cells default to `NA`; `count==0` → `NA`; no `agg$from` entry → row stays `NA` | Identical |

The Random Forest model object is passed through unchanged and only called with `predict()` — no retraining.

---

## Performance Estimates

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~hours (string paste on 6.46M rows) | <1 sec (sparse matrix from integer vectors) |
| Neighbor mean (×5 vars) | ~hours (6.46M lapply iterations ×5) | ~10 sec (5 sparse matmuls, each 1.37M nnz × 28 cols) |
| Neighbor max/min (×5 vars) | ~hours | ~2 min (5 vars × 28 years × 1.37M edge data.table groupby) |
| RF predict | same | same |
| **Total** | **86+ hours** | **~2–10 minutes** |

Memory peak: ~1.5 GB (one 344K×28 dense matrix ≈ 77 MB; sparse matrix ≈ 22 MB; edge data.table per year ≈ 22 MB; original data.frame ≈ ~5 GB with 110 columns).