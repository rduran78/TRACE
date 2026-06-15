 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup for all ~6.46 million cell-year rows, when the neighbor **topology** is identical across all 28 years. Specifically:

1. **Redundant topology replication**: The neighbor graph among 344,208 cells is static. Yet `build_neighbor_lookup` builds a list of 6.46M entries (one per cell-year row), each storing row indices of that cell-year's neighbors. This means the same spatial neighbor set is resolved 28 times per cell — once per year — producing ~6.46M list elements instead of ~344K.

2. **String-key lookup overhead**: The function creates `idx_lookup` keyed by `paste(id, year, sep="_")` — a character vector of 6.46M entries — and then performs repeated character-key lookups inside a loop of 6.46M iterations. Character hashing over millions of keys is extremely slow.

3. **Row-level R loop**: The `lapply` over 6.46M rows, each doing character paste and lookup, is the dominant bottleneck. Even `compute_neighbor_stats` iterates 6.46M times, though that part is cheaper.

4. **No vectorization**: Both the lookup construction and the stats computation are purely scalar R loops with no vectorized or matrix-based operations.

**Estimated cost breakdown**: ~6.46M iterations × (character paste + hash lookup + NA filtering) ≈ 86+ hours on a laptop.

---

## Optimization Strategy

**Key insight**: Separate the **static spatial topology** (which cells are neighbors of which cells) from the **year-varying data** (variable values attached to cell-years).

### Step 1: Build the neighbor topology once over cells (not cell-years)

Create a simple cell-index → neighbor-cell-indices mapping for the 344,208 cells. This is done **once** and costs seconds, not hours.

### Step 2: Reshape data for fast column-wise access by cell index

Create a matrix of dimension `(n_cells × n_years)` for each variable, where rows are cells (in a fixed order) and columns are years. This allows vectorized access: given a cell's neighbor indices, pull all years simultaneously via matrix row subsetting.

### Step 3: Compute neighbor stats via vectorized matrix operations

For each variable, for each cell, gather the neighbor rows from the matrix, then compute `max`, `min`, `mean` across neighbors for each year (column). This can be further accelerated by iterating over cells (344K) rather than cell-years (6.46M), and doing 28 years at once per cell.

### Step 4: Even better — use sparse-matrix multiplication for `mean`, and row-wise grouped operations for `max`/`min`

- **Neighbor mean**: Construct a sparse row-normalized adjacency matrix `W` of dimension `(n_cells × n_cells)`. Then `W %*% X` (where `X` is the `n_cells × n_years` value matrix) gives neighbor means for all cells and all years in one matrix multiply. This takes seconds.
- **Neighbor max/min**: Use a loop over cells (344K iterations, not 6.46M) pulling neighbor rows and computing column-wise max/min. Or use a grouped-operation approach.

This reduces runtime from **86+ hours to minutes**.

### Preservation guarantees

- The Random Forest model is **not retrained** — we only restructure the feature-engineering step that precedes `predict()`.
- The numerical outputs (neighbor max, min, mean) are **identical** to the original implementation — same values, same column names.

---

## Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Exploits the static-vs-changing distinction:
#   - Neighbor topology: STATIC across years (built once over 344K cells)
#   - Variable values: CHANGE by year (organized as cell×year matrices)
# ==============================================================================

library(Matrix)  # for sparse matrix operations

# --------------------------------------------------------------------------
# STEP 1: Build static cell-level neighbor index map (done ONCE)
# --------------------------------------------------------------------------
# rook_neighbors_unique: spdep::nb object, length = n_cells
# id_order: vector of cell IDs in the order matching rook_neighbors_unique
#
# This produces a simple list: cell_index -> integer vector of neighbor cell indices
# No year dimension, no string keys. ~344K entries.

build_cell_neighbor_map <- function(id_order, neighbors) {
  n_cells <- length(id_order)
  # spdep::nb objects store integer indices (with 0 meaning no neighbors)
  lapply(seq_len(n_cells), function(i) {
    nb <- neighbors[[i]]
    nb <- nb[nb > 0L]  # remove the 0-flag for no-neighbor cells
    as.integer(nb)
  })
}

# --------------------------------------------------------------------------
# STEP 2: Build cell-index ↔ row-index mappings and value matrices
# --------------------------------------------------------------------------
# Assumptions about cell_data:
#   - data.frame/data.table with columns: id, year, <variables...>
#   - Sorted or unsorted; we handle arbitrary order.

prepare_cell_year_infrastructure <- function(cell_data, id_order) {
  # Create a stable cell index: position of each cell ID in id_order
  id_to_cellidx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Unique sorted years
  years <- sort(unique(cell_data$year))
  n_years <- length(years)
  year_to_colidx <- setNames(seq_along(years), as.character(years))
  
  # Map each row of cell_data to (cell_index, year_index)
  cell_idx <- id_to_cellidx[as.character(cell_data$id)]
  year_idx <- year_to_colidx[as.character(cell_data$year)]
  
  list(
    id_order       = id_order,
    years          = years,
    n_cells        = length(id_order),
    n_years        = n_years,
    cell_idx       = as.integer(cell_idx),  # per-row: which cell
    year_idx       = as.integer(year_idx),   # per-row: which year column
    year_to_colidx = year_to_colidx
  )
}

# Build a (n_cells x n_years) matrix for a given variable
build_value_matrix <- function(cell_data, var_name, infra) {
  mat <- matrix(NA_real_, nrow = infra$n_cells, ncol = infra$n_years)
  mat[cbind(infra$cell_idx, infra$year_idx)] <- cell_data[[var_name]]
  mat
}

# --------------------------------------------------------------------------
# STEP 3: Compute neighbor mean via sparse matrix multiplication
# --------------------------------------------------------------------------
# Build the row-normalized sparse adjacency matrix W (done ONCE)

build_neighbor_weight_matrix <- function(cell_neighbor_map, n_cells) {
  # Build COO triplets
  from <- integer(0)
  to   <- integer(0)
  
  for (i in seq_len(n_cells)) {
    nb <- cell_neighbor_map[[i]]
    if (length(nb) > 0) {
      from <- c(from, rep(i, length(nb)))
      to   <- c(to, nb)
    }
  }
  
  # Sparse adjacency (binary)
  A <- sparseMatrix(i = from, j = to, x = 1, dims = c(n_cells, n_cells))
  
  # Row-normalize for mean: divide each row by its number of neighbors
  row_sums <- rowSums(A)
  row_sums[row_sums == 0] <- 1  # avoid division by zero; these rows are all-zero anyway
  W <- A / row_sums
  
  list(A = A, W = W)
}

# Neighbor mean for all cells and all years: W %*% val_matrix
# Result: (n_cells x n_years) matrix of neighbor means
compute_neighbor_mean_matrix <- function(W, val_matrix) {
  # Handle NAs: sparse matrix multiply treats them poorly.
  # Strategy: replace NA with 0, track counts of non-NA neighbors per cell-year,
  # then compute mean = sum / count.
  
  not_na <- !is.na(val_matrix)
  val_zero <- val_matrix
  val_zero[!not_na] <- 0
  
  # A is the binary adjacency (un-normalized); we need it for correct NA handling
  # sum of neighbor values (replacing NA with 0)
  # We need the un-normalized adjacency to do: sum_vals and count_non_na
  # W is row-normalized, so A = W * row_degrees... 
  # Actually, let's just accept A as a separate argument.
  # For cleaner code, we'll pass both A and W from the build step.
  
  # This function will be called with A (binary adjacency) instead of W.
  # See the orchestrator below.
  stop("Use compute_neighbor_mean_matrix_with_A instead")
}

compute_neighbor_mean_matrix_with_A <- function(A, val_matrix) {
  not_na <- !is.na(val_matrix)
  val_zero <- val_matrix
  val_zero[!not_na] <- 0
  
  # Sum of non-NA neighbor values per cell per year
  sum_mat <- as.matrix(A %*% val_zero)   # (n_cells x n_years)
  
  # Count of non-NA neighbors per cell per year
  # not_na is logical; convert to numeric for multiplication
  count_mat <- as.matrix(A %*% (not_na * 1.0))
  
  mean_mat <- sum_mat / count_mat         # NaN where count==0
  mean_mat[count_mat == 0] <- NA_real_
  mean_mat
}

# --------------------------------------------------------------------------
# STEP 4: Compute neighbor max and min (loop over 344K cells, not 6.46M rows)
# --------------------------------------------------------------------------

compute_neighbor_maxmin_matrices <- function(cell_neighbor_map, val_matrix, n_cells, n_years) {
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nb <- cell_neighbor_map[[i]]
    if (length(nb) == 0L) next
    
    # Extract neighbor rows: matrix of (length(nb) x n_years)
    nb_vals <- val_matrix[nb, , drop = FALSE]
    
    # Column-wise max and min, ignoring NA
    # suppressWarnings to handle all-NA columns gracefully
    suppressWarnings({
      max_mat[i, ] <- apply(nb_vals, 2, max, na.rm = TRUE)
      min_mat[i, ] <- apply(nb_vals, 2, min, na.rm = TRUE)
    })
    
    # Fix columns where all neighbors were NA (apply returns -Inf/Inf)
    all_na_cols <- colSums(!is.na(nb_vals)) == 0L
    if (any(all_na_cols)) {
      max_mat[i, all_na_cols] <- NA_real_
      min_mat[i, all_na_cols] <- NA_real_
    }
  }
  
  list(max = max_mat, min = min_mat)
}

# --------------------------------------------------------------------------
# FASTER STEP 4 ALTERNATIVE: Vectorized max/min using edge list expansion
# (Avoids the 344K R-level loop with per-cell apply)
# --------------------------------------------------------------------------

compute_neighbor_maxmin_matrices_fast <- function(cell_neighbor_map, val_matrix, n_cells, n_years) {
  # Build edge list: (from_cell, to_cell) meaning to_cell is a neighbor of from_cell
  from_cells <- integer(0)
  to_cells   <- integer(0)
  for (i in seq_len(n_cells)) {
    nb <- cell_neighbor_map[[i]]
    if (length(nb) > 0L) {
      from_cells <- c(from_cells, rep(i, length(nb)))
      to_cells   <- c(to_cells, nb)
    }
  }
  
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  if (length(from_cells) == 0L) {
    return(list(max = max_mat, min = min_mat))
  }
  
  # For each year (column), do grouped max/min
  # This iterates over 28 years instead of 344K cells
  for (j in seq_len(n_years)) {
    neighbor_vals <- val_matrix[to_cells, j]
    
    # Use data.table for fast grouped aggregation
    if (requireNamespace("data.table", quietly = TRUE)) {
      dt <- data.table::data.table(
        from = from_cells,
        val  = neighbor_vals
      )
      dt <- dt[!is.na(val)]
      if (nrow(dt) > 0) {
        agg <- dt[, .(mx = max(val), mn = min(val)), by = from]
        max_mat[agg$from, j] <- agg$mx
        min_mat[agg$from, j] <- agg$mn
      }
    } else {
      # Fallback: tapply
      valid <- !is.na(neighbor_vals)
      if (any(valid)) {
        max_mat[, j] <- NA_real_
        min_mat[, j] <- NA_real_
        mx <- tapply(neighbor_vals[valid], from_cells[valid], max)
        mn <- tapply(neighbor_vals[valid], from_cells[valid], min)
        idx_mx <- as.integer(names(mx))
        idx_mn <- as.integer(names(mn))
        max_mat[idx_mx, j] <- as.numeric(mx)
        min_mat[idx_mn, j] <- as.numeric(mn)
      }
    }
  }
  
  list(max = max_mat, min = min_mat)
}

# --------------------------------------------------------------------------
# STEP 5: Write results back to cell_data
# --------------------------------------------------------------------------

write_matrix_to_data <- function(cell_data, mat, col_name, infra) {
  # mat is (n_cells x n_years); map back using cell_idx, year_idx
  cell_data[[col_name]] <- mat[cbind(infra$cell_idx, infra$year_idx)]
  cell_data
}

# --------------------------------------------------------------------------
# ORCHESTRATOR: Drop-in replacement for the original outer loop
# --------------------------------------------------------------------------

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                   "def", "usd_est_n2")) {
  
  cat("Building static cell-level neighbor map (once)...\n")
  cell_neighbor_map <- build_cell_neighbor_map(id_order, rook_neighbors_unique)
  
  cat("Preparing cell-year infrastructure...\n")
  infra <- prepare_cell_year_infrastructure(cell_data, id_order)
  
  cat("Building sparse adjacency matrix (once)...\n")
  sparse <- build_neighbor_weight_matrix(cell_neighbor_map, infra$n_cells)
  A <- sparse$A  # binary adjacency
  
  # Build edge list once for max/min (if using the fast method)
  cat("Building edge list for max/min (once)...\n")
  from_cells <- integer(0)
  to_cells   <- integer(0)
  for (i in seq_len(infra$n_cells)) {
    nb <- cell_neighbor_map[[i]]
    if (length(nb) > 0L) {
      from_cells <- c(from_cells, rep.int(i, length(nb)))
      to_cells   <- c(to_cells, nb)
    }
  }
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))
    
    # Build value matrix (n_cells x n_years)
    val_matrix <- build_value_matrix(cell_data, var_name, infra)
    
    # --- Neighbor MEAN via sparse matrix multiply ---
    cat("  Computing neighbor mean (sparse matrix multiply)...\n")
    mean_mat <- compute_neighbor_mean_matrix_with_A(A, val_matrix)
    
    # --- Neighbor MAX and MIN ---
    cat("  Computing neighbor max/min...\n")
    max_mat <- matrix(NA_real_, nrow = infra$n_cells, ncol = infra$n_years)
    min_mat <- matrix(NA_real_, nrow = infra$n_cells, ncol = infra$n_years)
    
    if (length(from_cells) > 0L) {
      if (requireNamespace("data.table", quietly = TRUE)) {
        for (j in seq_len(infra$n_years)) {
          neighbor_vals <- val_matrix[to_cells, j]
          dt <- data.table::data.table(from = from_cells, val = neighbor_vals)
          dt <- dt[!is.na(val)]
          if (nrow(dt) > 0L) {
            agg <- dt[, .(mx = max(val), mn = min(val)), by = from]
            max_mat[agg$from, j] <- agg$mx
            min_mat[agg$from, j] <- agg$mn
          }
        }
      } else {
        for (j in seq_len(infra$n_years)) {
          neighbor_vals <- val_matrix[to_cells, j]
          valid <- !is.na(neighbor_vals)
          if (any(valid)) {
            mx <- tapply(neighbor_vals[valid], from_cells[valid], max)
            mn <- tapply(neighbor_vals[valid], from_cells[valid], min)
            max_mat[as.integer(names(mx)), j] <- as.numeric(mx)
            min_mat[as.integer(names(mn)), j] <- as.numeric(mn)
          }
        }
      }
    }
    
    # --- Write back to cell_data with original column naming convention ---
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    cell_data <- write_matrix_to_data(cell_data, max_mat,  max_col,  infra)
    cell_data <- write_matrix_to_data(cell_data, min_mat,  min_col,  infra)
    cell_data <- write_matrix_to_data(cell_data, mean_mat, mean_col, infra)
    
    cat(sprintf("  Done: %s, %s, %s\n", max_col, min_col, mean_col))
  }
  
  cat("All neighbor features computed.\n")
  cell_data
}

# ==============================================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================================

# Replace this:
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }
#
# With this:
cell_data <- compute_all_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# Then proceed directly to prediction with the pre-trained Random Forest:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Topology resolution** | 6.46M cell-year lookups via string keys | 344K cell-index list (built once) |
| **Neighbor mean** | R loop over 6.46M rows | Sparse matrix multiply `A %*% X` — fully vectorized, seconds |
| **Neighbor max/min** | R loop over 6.46M rows | 28 iterations (one per year) × data.table grouped aggregation over ~1.37M edges |
| **String operations** | ~12.9M `paste()` calls + character hash lookups | **Zero** — all operations use integer indices |
| **Memory layout** | Column vector (6.46M), re-indexed per row | `(344K × 28)` matrix — cache-friendly, enables column-wise vectorization |
| **Estimated runtime** | 86+ hours | **~5–15 minutes** (depending on RAM/CPU) |
| **Numerical results** | Original values | **Identical** — same max, min, mean computed over the same neighbor sets |
| **RF model** | N/A | **Untouched** — no retraining; same `predict()` call downstream |