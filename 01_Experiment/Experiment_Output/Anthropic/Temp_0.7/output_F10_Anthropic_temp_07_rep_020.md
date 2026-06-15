 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each requiring string-pasting (`paste(id, year, sep="_")`), named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. This is an O(N×K) operation done entirely in interpreted R with character hashing — extremely slow at this scale.

2. **`compute_neighbor_stats` iterates over 6.46M list elements** in `lapply`, extracting subsets of a vector by index, removing NAs, and computing max/min/mean. Each call processes one variable, so this is repeated 5 times — ~32.3 million interpreted R loop iterations total.

3. **The neighbor topology is year-invariant but the lookup is rebuilt as if it's year-specific.** Each cell has the same rook neighbors every year. The lookup fuses topology and temporal indexing into one monolithic structure, preventing reuse and vectorization.

**Root cause:** The design treats this as a per-row problem (6.46M iterations) rather than a sparse matrix–vector product (344K nodes × 28 years, with a fixed adjacency structure of ~1.37M edges).

---

## Optimization Strategy

### Key Insight
Neighbor aggregation (max, min, mean) over a fixed spatial graph with node-year attributes decomposes into:

- **A fixed sparse adjacency matrix** `A` (344,208 × 344,208, ~1.37M nonzeros) — built once.
- **A dense attribute matrix** `V` (344,208 × 28) for each variable — one column per year.
- **Aggregations** computed column-wise (per year) using sparse matrix operations or grouped vectorized operations.

### Plan

1. **Build the sparse adjacency matrix once** from `rook_neighbors_unique` (the `nb` object). Use `Matrix::sparseMatrix`.

2. **Reshape each variable into a 344,208 × 28 matrix** (cell × year), maintaining a consistent cell ordering.

3. **Compute neighbor statistics per year-column** using vectorized sparse operations:
   - **Mean:** `A_row_normalized %*% V_year` (sparse matrix–vector multiply).
   - **Max / Min:** Use the sparse structure to do grouped max/min via `data.table` or direct C-level iteration on the CSR representation.

4. **Map results back** into the original `cell_data` data.frame, preserving row order and column names.

5. **Feed into the pre-trained Random Forest** unchanged.

This reduces 6.46M interpreted iterations to ~28 sparse-matrix operations per variable (or a single vectorized grouped operation), cutting runtime from 86+ hours to **minutes**.

---

## Optimized R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 0: Ensure cell_data is a data.table with correct ordering
# ==============================================================================
cell_dt <- as.data.table(cell_data)

# Canonical cell ordering — must match rook_neighbors_unique index positions
# id_order is the vector of cell IDs in the order matching the nb object
stopifnot(length(id_order) == 344208L)

# Create integer mapping: cell id -> position in id_order (1-based)
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Add spatial index to data
cell_dt[, cell_pos := id_to_pos[as.character(id)]]

# Verify all cells matched
stopifnot(!anyNA(cell_dt$cell_pos))

# ==============================================================================
# STEP 1: Build sparse adjacency matrix ONCE from nb object
# ==============================================================================
build_adjacency <- function(nb_obj, n) {
  # nb_obj is a list of length n; nb_obj[[i]] is integer vector of neighbor

  # indices (1-based) for node i. 0L means no neighbors in spdep convention.
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)

  # Remove spdep's 0-neighbor sentinel

  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]

  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

n_cells <- length(id_order)
A <- build_adjacency(rook_neighbors_unique, n_cells)

# Precompute neighbor counts per cell (for mean computation)
neighbor_count <- diff(A@p)  # CSC column counts if transposed; use rowSums
neighbor_count_vec <- as.numeric(rowSums(A))  # length n_cells

# CSR representation for fast row-wise grouped operations
A_csr <- as(A, "RsparseMatrix")  # dgRMatrix: row-compressed

cat("Adjacency matrix built:", nnzero(A), "directed edges\n")

# ==============================================================================
# STEP 2: Vectorized neighbor aggregation using sparse structure
# ==============================================================================

# For max and min, we cannot use matrix multiplication. Instead, we operate
# on the CSR structure directly with vectorized R code per year-column.
#
# A_csr@p  : row pointers (length n_cells + 1), 0-based
# A_csr@j  : column indices (0-based)
# For row i (0-indexed), neighbors are at A_csr@j[(A_csr@p[i]+1):A_csr@p[i+1]]

# Precompute expanded row-id and neighbor-column-id vectors (ONCE, reuse for
# every variable and year). This is the "edge list in CSR-expanded form."
row_ptr <- A_csr@p  # length n_cells + 1, 0-based
col_idx <- A_csr@j  # 0-based column indices

# Expand to edge list: for each row, repeat row index for each nonzero
# This is equivalent to: from-node for every directed edge
edge_from <- rep(seq_len(n_cells), diff(row_ptr))  # 1-based row indices
edge_to   <- col_idx + 1L                           # 1-based column indices

n_edges <- length(edge_from)
cat("Edge list expanded:", n_edges, "edges\n")

# We'll use data.table's grouped operations on this edge list.
# Pre-allocate the edge data.table skeleton (no variable values yet).
edge_dt <- data.table(from = edge_from, to = edge_to)

# ==============================================================================
# STEP 3: Prepare year structure
# ==============================================================================
years <- sort(unique(cell_dt$year))
n_years <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))

# ==============================================================================
# STEP 4: For each source variable, compute neighbor max, min, mean
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Set key for fast lookups
setkey(cell_dt, cell_pos, year)

# Function to compute all neighbor stats for one variable
compute_neighbor_features <- function(dt, var_name, edge_dt, id_order_len,
                                      years, neighbor_count_vec) {

  cat("Processing variable:", var_name, "\n")

  # Build cell × year matrix: rows = cell_pos, cols = year index
  # Extract relevant columns
  sub <- dt[, .(cell_pos, year, val = get(var_name))]
  setkey(sub, cell_pos, year)

  # Create dense matrix (n_cells × n_years)
  val_mat <- matrix(NA_real_, nrow = id_order_len, ncol = length(years))
  row_idx <- sub$cell_pos
  col_idx_mat <- year_to_col[as.character(sub$year)]
  val_mat[cbind(row_idx, col_idx_mat)] <- sub$val

  # For each year, compute neighbor stats
  max_mat  <- matrix(NA_real_, nrow = id_order_len, ncol = length(years))
  min_mat  <- matrix(NA_real_, nrow = id_order_len, ncol = length(years))
  mean_mat <- matrix(NA_real_, nrow = id_order_len, ncol = length(years))

  for (yi in seq_along(years)) {
    # Get this year's values for all cells
    v <- val_mat[, yi]  # length n_cells

    # Look up neighbor values via edge list
    neighbor_vals <- v[edge_dt$to]  # value at each edge's target node

    # Use data.table grouped aggregation on the edge list
    # Only non-NA neighbor values contribute
    agg_dt <- data.table(from = edge_dt$from, nv = neighbor_vals)
    agg_dt <- agg_dt[!is.na(nv)]

    if (nrow(agg_dt) > 0) {
      stats <- agg_dt[, .(nmax = max(nv), nmin = min(nv), nsum = sum(nv),
                           ncnt = .N), by = from]

      max_mat[stats$from, yi]  <- stats$nmax
      min_mat[stats$from, yi]  <- stats$nmin
      mean_mat[stats$from, yi] <- stats$nsum / stats$ncnt
    }
  }

  # Map back to cell_dt rows
  ri <- cbind(dt$cell_pos, year_to_col[as.character(dt$year)])

  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  dt[, (max_col)  := max_mat[ri]]
  dt[, (min_col)  := min_mat[ri]]
  dt[, (mean_col) := mean_mat[ri]]

  invisible(dt)
}

# ==============================================================================
# STEP 5: Run for all 5 variables
# ==============================================================================
for (var_name in neighbor_source_vars) {
  compute_neighbor_features(cell_dt, var_name, edge_dt, n_cells,
                            years, neighbor_count_vec)
}

cat("All neighbor features computed.\n")

# ==============================================================================
# STEP 6: Convert back and predict with the pre-trained Random Forest
# ==============================================================================
# Remove helper column
cell_dt[, cell_pos := NULL]

# Convert back to data.frame if the RF model expects one
cell_data <- as.data.frame(cell_dt)

# Predict using the EXISTING pre-trained model (do NOT retrain)
# Assumes `rf_model` is already loaded in the environment
predictions <- predict(rf_model, newdata = cell_data)
```

---

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M `lapply` iterations with string ops | Sparse matrix built once (~1.37M entries) |
| **Per-variable aggregation** | 6.46M `lapply` iterations × 5 vars | 28 `data.table` grouped-aggs × 5 vars = 140 vectorized ops |
| **String operations** | ~19M `paste()` + named-vector lookups | Zero — integer indexing only |
| **Memory pattern** | 6.46M-element R list (high overhead) | Dense matrices (344K × 28 ≈ 77 MB each) + edge vectors (~11 MB) |
| **Estimated runtime** | 86+ hours | **~5–15 minutes** |
| **Numerical equivalence** | Baseline | ✅ Identical max/min/mean, same NA handling |
| **RF model** | Pre-trained | ✅ Unchanged, predict-only |

### Why This Is Numerically Equivalent

- The edge list `(edge_from, edge_to)` encodes exactly the same directed rook-neighbor relationships as `rook_neighbors_unique`.
- For each `(cell, year)`, the set of neighbor values gathered is identical.
- `max`, `min`, and `mean` are computed on the same non-NA subsets.
- NA propagation: cells with zero non-NA neighbors get `NA` for all three statistics, matching the original `if (length(neighbor_vals) == 0) return(c(NA, NA, NA))`.