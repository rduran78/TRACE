 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is in **`build_neighbor_lookup`**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) hashing per call, and doing it 6.46 million times with variable-length neighbor sets is catastrophically slow. The secondary bottleneck is **`compute_neighbor_stats`**, which loops over 6.46 million entries again per variable, extracting and summarizing neighbor values with per-element R-level iteration.

**Specific issues:**

1. **`build_neighbor_lookup`**: The `paste(id, year)` keying strategy creates a giant named character vector (~6.46M entries). Each of the 6.46M rows then does a character-key lookup into this vector for each of its neighbors. This is O(N × avg_neighbors) with high constant overhead from R's string operations and named-vector hashing. Estimated: billions of character operations.

2. **`compute_neighbor_stats`**: Calls `lapply` over 6.46M elements, each invoking `max`, `min`, `mean` on small vectors. The R interpreter overhead per iteration dominates. Doing this 5 times (once per variable) multiplies the cost.

3. **Memory**: The `neighbor_lookup` list of 6.46M integer vectors is itself large. Combined with the 6.46M × 110 data frame and intermediate objects, 16 GB RAM is tight.

---

## Optimization Strategy

### Key Insight: Exploit the Panel Structure

Every cell has the **same set of spatial neighbors in every year**. The neighbor graph is time-invariant. Instead of building a 6.46M-row lookup, we can:

1. **Build the neighbor lookup once at the cell level** (344K cells, not 6.46M cell-years).
2. **Reshape each variable into a matrix** of dimensions (344K cells × 28 years).
3. **Vectorize the neighbor aggregation** using sparse-matrix multiplication or direct indexed matrix operations — no R-level loops over millions of rows.

### Concrete Plan

- **Step A**: Create a sparse adjacency matrix `W` (344K × 344K) from `rook_neighbors_unique`, row-normalized or raw.
- **Step B**: For each variable, reshape into a (344K × 28) matrix `V`.
- **Step C**: Compute `neighbor_mean = W %*% V / degree` using sparse matrix–dense matrix multiplication (one call, fully vectorized in C via the `Matrix` package). For min and max, use a grouped approach with `data.table`.
- **Step D**: Reshape results back to the long panel and attach to `cell_data`.

**Expected speedup**: Sparse matrix multiplication for mean is O(nnz × 28) ≈ 1.37M × 28 ≈ 38M operations, done in compiled C code — seconds, not hours. Min/max require a `data.table` grouped operation over ~38M neighbor-pairs, also seconds.

**Memory**: The sparse matrix is ~1.37M non-zero entries (~11 MB). Each variable matrix is 344K × 28 doubles (~77 MB). Total overhead is well within 16 GB.

---

## Working R Code

```r
library(Matrix)
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars) {

  # ---------------------------------------------------------------
  # STEP 0: Convert to data.table for speed; record original order
  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, ..orig_row_order := .I]

  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)

  # Map cell id -> integer index (1..n_cells)
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Map year -> integer index (1..n_years)
  year_to_idx <- setNames(seq_along(years), as.character(years))

  # Add integer indices to data.table
  dt[, cell_idx := id_to_idx[as.character(id)]]
  dt[, year_idx := year_to_idx[as.character(year)]]

  # ---------------------------------------------------------------
  # STEP 1: Build sparse adjacency matrix from nb object
  # ---------------------------------------------------------------
  # rook_neighbors_unique is a list of length n_cells;
  # element i contains integer indices (into id_order) of neighbors of cell i.
  from_idx <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique)

  # Remove any zero-length / NA entries
  valid <- !is.na(to_idx)
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  # Sparse adjacency matrix (non-symmetric storage is fine; nb objects store directed pairs)
  W <- sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = 1,
    dims = c(n_cells, n_cells)
  )

  # Degree vector (number of neighbors per cell)
  degree_vec <- rowSums(W)  # dense vector of length n_cells

  # ---------------------------------------------------------------
  # STEP 2: For each variable, compute neighbor max, min, mean
  # ---------------------------------------------------------------
  for (var_name in neighbor_source_vars) {

    message("Processing neighbor features for: ", var_name)

    # --- Build (n_cells x n_years) matrix of variable values ---
    # Initialize with NA
    V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    V[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]

    # ============================================================
    # NEIGHBOR MEAN via sparse matrix multiplication
    # ============================================================
    # W %*% V gives, for each cell i and year t, the SUM of neighbor values.
    # Divide by degree to get mean.
    neighbor_sum  <- as.matrix(W %*% V)   # n_cells x n_years dense matrix
    neighbor_mean <- neighbor_sum / degree_vec  # recycling over columns
    # Cells with 0 neighbors -> NaN from 0/0; convert to NA
    neighbor_mean[degree_vec == 0, ] <- NA_real_

    # ============================================================
    # NEIGHBOR MIN and MAX via data.table grouped operations
    # ============================================================
    # Build an edge table: for every (from_cell, to_cell) pair,
    # look up the to_cell's value for every year.
    # This is efficient because we iterate over edges × years,
    # not over cell-years × variable-length neighbor lists in R.

    # Edge data.table: one row per directed edge
    edges_dt <- data.table(from_cell = from_idx, to_cell = to_idx)

    # For each edge, get the neighbor (to_cell) values across all years.
    # V[to_cell, ] gives a row vector of length n_years.
    # We vectorize: create a matrix of neighbor values indexed by edges.
    neighbor_vals_mat <- V[edges_dt$to_cell, , drop = FALSE]  # n_edges x n_years

    # Now group by from_cell and compute column-wise min and max.
    # Convert to data.table for grouped aggregation.
    # To avoid creating a huge long-format table, we process year by year
    # or use matrix split. Given ~1.37M edges × 28 years, a long table
    # would be ~38M rows — manageable.

    # Efficient approach: use rowgroup-based aggregation with data.table
    # We'll process all years at once using matrix columns.

    # Assign group key
    from_groups <- edges_dt$from_cell  # length = n_edges

    # For min and max, we use tapply-style operations via data.table
    # Process each year column
    neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (t in seq_len(n_years)) {
      vals_t <- neighbor_vals_mat[, t]
      # Only process non-NA values
      valid_t <- !is.na(vals_t)
      if (!any(valid_t)) next

      agg_dt <- data.table(g = from_groups[valid_t], v = vals_t[valid_t])
      agg_min <- agg_dt[, .(vmin = min(v)), by = g]
      agg_max <- agg_dt[, .(vmax = max(v)), by = g]

      neighbor_min_mat[agg_min$g, t] <- agg_min$vmin
      neighbor_max_mat[agg_max$g, t] <- agg_max$vmax
    }

    # ============================================================
    # STEP 3: Map results back to cell_data rows
    # ============================================================
    idx_mat <- cbind(dt$cell_idx, dt$year_idx)

    max_col  <- paste0("n_max_", var_name)
    min_col  <- paste0("n_min_", var_name)
    mean_col <- paste0("n_mean_", var_name)

    dt[, (max_col)  := neighbor_max_mat[idx_mat]]
    dt[, (min_col)  := neighbor_min_mat[idx_mat]]
    dt[, (mean_col) := neighbor_mean[idx_mat]]
  }

  # ---------------------------------------------------------------
  # STEP 4: Restore original row order, drop helper columns, return
  # ---------------------------------------------------------------
  setorder(dt, ..orig_row_order)
  dt[, c("..orig_row_order", "cell_idx", "year_idx") := NULL]

  # Return as data.frame if original was data.frame
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# ===================================================================
# USAGE (drop-in replacement for the original outer loop)
# ===================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Predictions proceed as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Works and What It Preserves

| Concern | Resolution |
|---|---|
| **Numerical equivalence** | `W %*% V / degree` computes the identical arithmetic mean. Min and max are computed over the identical neighbor sets. Results are bit-for-bit equivalent (up to floating-point associativity). |
| **Trained RF model** | No model retraining. The code only computes the same 15 feature columns (`n_max_*`, `n_min_*`, `n_mean_*` for 5 variables) that the model expects. |
| **Memory (16 GB)** | Peak: ~77 MB per variable matrix × 4 (V, sum, min, max) ≈ 308 MB per variable, plus the sparse matrix (~22 MB), plus `cell_data` (~5.4 GB for 6.46M × 110 float64). Total peak ≈ 8–9 GB. Fits in 16 GB. |
| **Speed** | Sparse `W %*% V` (28 SpMV operations internally): ~2–5 seconds per variable. The `data.table` grouped min/max loop over 28 years × 1.37M edges: ~30–60 seconds per variable. **Total estimated runtime: 3–6 minutes** (down from 86+ hours). |
| **`spdep::nb` compatibility** | The code reads `rook_neighbors_unique` as a standard list of integer index vectors, which is exactly what `spdep::nb` objects are. |

---

## Optional Further Optimization

If the 28-iteration inner loop for min/max is still a concern, you can replace it with a single long-format `data.table` aggregation:

```r
# Alternative: single-pass min/max (uses ~38M row data.table, ~600 MB)
long_dt <- data.table(
  from_cell = rep(from_groups, n_years),
  year_idx  = rep(seq_len(n_years), each = length(from_groups)),
  val       = as.vector(neighbor_vals_mat)
)
long_dt <- long_dt[!is.na(val)]
agg <- long_dt[, .(vmin = min(val), vmax = max(val)), by = .(from_cell, year_idx)]
# Then scatter back into matrices...
```

This trades ~600 MB of extra memory for eliminating the year loop entirely, reducing min/max computation to a single `data.table` grouped aggregation (~10–20 seconds total for all years of one variable).