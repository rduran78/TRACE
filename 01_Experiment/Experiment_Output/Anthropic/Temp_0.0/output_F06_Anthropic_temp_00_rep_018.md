 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For each of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's spatial neighbors (fast).
2. Constructs character keys by pasting neighbor IDs with the current row's year (slow — `paste` on millions of iterations).
3. Performs named-vector lookups via `idx_lookup[neighbor_keys]` (slow — repeated character hashing).

This produces a **list of length 6.46M**, each element being an integer vector of row indices. The character key construction and lookup inside a per-row `lapply` is the dominant cost. With ~1.37M directed neighbor relationships spread across 344K cells and 28 years, each row averages ~4 neighbors, meaning roughly **25.8 million** character-key constructions and lookups.

### Bottleneck B: `compute_neighbor_stats` — repeated per-row `lapply`

For each of the 5 variables, another `lapply` over 6.46M rows computes `max`, `min`, `mean` of neighbor values. That's 5 × 6.46M = **32.3 million** R-level function calls with subsetting.

### Why raster focal/kernel operations are not directly applicable

Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed kernel window. While the data are gridded, the neighbor structure is stored as an `spdep::nb` object (which can handle irregular boundaries, missing cells, coastal cells, etc.). Forcing this into a raster focal operation risks:
- Incorrectly including NA cells or cells outside the study area as neighbors.
- Altering the numerical results at boundaries.

The correct approach is to **keep the exact neighbor structure** but replace the R-level row-by-row loops with **vectorized sparse-matrix operations**.

### Summary of the problem

| Component | Current complexity | Root cause |
|---|---|---|
| `build_neighbor_lookup` | ~25.8M character ops | Per-row `paste` + named vector lookup |
| `compute_neighbor_stats` | ~32.3M R function calls | Per-row `lapply` × 5 variables |
| **Total estimated time** | **86+ hours** | R-level loops over millions of rows |

---

## 2. Optimization Strategy

### Core idea: Sparse matrix multiplication replaces both bottlenecks

1. **Build a sparse adjacency matrix `W`** of dimension (6.46M × 6.46M) where `W[i,j] = 1` if row `j` is a rook neighbor of row `i` *in the same year*. This matrix is extremely sparse (~25.8M non-zero entries out of ~41.7 trillion possible).

2. **Compute neighbor stats via sparse matrix operations:**
   - **Mean:** `W %*% x / (W %*% ones)` — one sparse matrix-vector multiply gives the sum; dividing by the count of neighbors gives the mean.
   - **Max and Min:** Use the `{Matrix}` package's sparse structure to iterate over rows in C (via `summary()` of the sparse matrix), or use a grouped operation with `data.table`.

3. **Avoid character key construction entirely** by building the sparse matrix using integer indexing: for each year, offset the spatial neighbor indices by `(year_index - 1) * n_cells`.

### Expected speedup

| Component | Before | After |
|---|---|---|
| Neighbor lookup construction | ~hours (character ops) | ~seconds (integer arithmetic + sparse matrix construction) |
| Stats for 5 variables | ~hours (R lapply) | ~seconds per variable (sparse mat-vec multiply + grouped row ops) |
| **Total** | **86+ hours** | **~1–5 minutes** |

### What is preserved
- The exact same set of rook neighbors per cell-year.
- The exact same `max`, `min`, `mean` numerical values (no approximation).
- The pre-trained Random Forest model is untouched.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# Requirements: data.table, Matrix
# Preserves: exact numerical results, trained RF model
# =============================================================================

library(data.table)
library(Matrix)

# ---- Step 1: Build the sparse neighbor matrix (once) -----------------------

build_sparse_neighbor_matrix <- function(cell_data, id_order, rook_neighbors) {
  # Convert to data.table for fast operations
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]
  
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)
  n_rows  <- nrow(dt)
  
  # Create a fast lookup: (id, year) -> row index
  # Use integer-keyed lookup instead of character paste
  id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # For each cell-year row, we need its spatial index and year index
  dt[, spatial_idx := id_to_spatial_idx[as.character(id)]]
  
  year_to_idx <- setNames(seq_along(years), as.character(years))
  dt[, year_idx := year_to_idx[as.character(year)]]
  
  # Build a 2D lookup matrix: spatial_idx × year_idx -> row_idx
  # This replaces all character key operations
  lookup_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  lookup_matrix[cbind(dt$spatial_idx, dt$year_idx)] <- dt$row_idx
  
  # Now build sparse matrix triplets (i, j) where j is neighbor of i
  # Pre-allocate vectors
  # Count total edges first
  total_edges <- 0L
  for (s in seq_len(n_cells)) {
    nb <- rook_neighbors[[s]]
    nb <- nb[nb > 0L]  # spdep::nb uses 0 for no-neighbor indicator
    total_edges <- total_edges + length(nb)
  }
  total_entries <- as.numeric(total_edges) * n_years  # upper bound
  
  cat("Building sparse matrix with up to", total_entries, "non-zero entries\n")
  
  # Vectorized construction: expand neighbor pairs across all years
  # First, collect all (spatial_from, spatial_to) pairs
  from_spatial <- integer(total_edges)
  to_spatial   <- integer(total_edges)
  pos <- 1L
  for (s in seq_len(n_cells)) {
    nb <- rook_neighbors[[s]]
    nb <- nb[nb > 0L]
    len <- length(nb)
    if (len > 0L) {
      from_spatial[pos:(pos + len - 1L)] <- s
      to_spatial[pos:(pos + len - 1L)]   <- nb
      pos <- pos + len
    }
  }
  # Trim if needed
  from_spatial <- from_spatial[1:(pos - 1L)]
  to_spatial   <- to_spatial[1:(pos - 1L)]
  
  cat("Spatial neighbor pairs:", length(from_spatial), "\n")
  
  # Now expand across years using the lookup_matrix
  # For each year, map spatial indices to row indices
  all_i <- integer(0)
  all_j <- integer(0)
  
  for (y in seq_len(n_years)) {
    row_from <- lookup_matrix[from_spatial, y]
    row_to   <- lookup_matrix[to_spatial, y]
    
    # Keep only pairs where both cells exist in this year
    valid <- !is.na(row_from) & !is.na(row_to)
    all_i <- c(all_i, row_from[valid])
    all_j <- c(all_j, row_to[valid])
  }
  
  cat("Total non-zero entries in W:", length(all_i), "\n")
  
  # Build sparse matrix
  W <- sparseMatrix(
    i = all_i,
    j = all_j,
    x = 1,
    dims = c(n_rows, n_rows),
    repr = "C"   # CSC format; we'll also need row-access
  )
  
  # Clean up temporary columns
  dt[, c("spatial_idx", "year_idx") := NULL]
  
  return(W)
}


# ---- Step 2: Compute neighbor stats using sparse matrix ---------------------

compute_neighbor_stats_sparse <- function(cell_data, W, var_name) {
  # Extract the variable as a numeric vector
  x <- cell_data[[var_name]]
  n <- length(x)
  
  # --- MEAN via sparse matrix-vector multiply ---
  # Replace NA with 0 for summation, track non-NA
  not_na <- as.numeric(!is.na(x))
  x_safe <- ifelse(is.na(x), 0, x)
  
  # Neighbor sum and neighbor count (of non-NA values)
  neighbor_sum   <- as.numeric(W %*% x_safe)
  neighbor_count <- as.numeric(W %*% not_na)
  
  # Also get total neighbor count (including NA neighbors) to detect isolated cells

  ones <- rep(1, n)
  total_neighbors <- as.numeric(W %*% ones)
  
  # Mean
  nb_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
  # If a cell has no neighbors at all, set to NA
  nb_mean[total_neighbors == 0] <- NA_real_
  
  # --- MAX and MIN via row-wise grouped operations on sparse matrix ---
  # Extract the sparse structure
  W_summary <- summary(W)  # returns data.frame with i, j, x columns
  
  # Get neighbor values
  neighbor_vals <- x[W_summary$j]
  
  # Use data.table for fast grouped max/min
  dt_edges <- data.table(
    row_i = W_summary$i,
    val   = neighbor_vals
  )
  
  # Remove edges where neighbor value is NA
  dt_edges <- dt_edges[!is.na(val)]
  
  # Grouped max and min
  stats <- dt_edges[, .(nb_max = max(val), nb_min = min(val)), by = row_i]
  
  # Map back to full vector
  nb_max <- rep(NA_real_, n)
  nb_min <- rep(NA_real_, n)
  nb_max[stats$row_i] <- stats$nb_max
  nb_min[stats$row_i] <- stats$nb_min
  
  # Return as a 3-column matrix matching original output format
  cbind(nb_max, nb_min, nb_mean)
}


# ---- Step 3: Add features to data (replaces outer loop) --------------------

compute_and_add_all_neighbor_features <- function(cell_data, W, neighbor_source_vars) {
  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor stats for:", var_name, "\n")
    t0 <- Sys.time()
    
    stats_mat <- compute_neighbor_stats_sparse(cell_data, W, var_name)
    
    cell_data[[paste0(var_name, "_nb_max")]]  <- stats_mat[, 1]
    cell_data[[paste0(var_name, "_nb_min")]]  <- stats_mat[, 2]
    cell_data[[paste0(var_name, "_nb_mean")]] <- stats_mat[, 3]
    
    cat("  Done in", round(difftime(Sys.time(), t0, units = "secs"), 1), "seconds\n")
  }
  return(cell_data)
}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

# --- Build sparse neighbor matrix (replaces build_neighbor_lookup) ---
cat("Building sparse neighbor matrix...\n")
t_start <- Sys.time()

W <- build_sparse_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)

cat("Sparse matrix built in",
    round(difftime(Sys.time(), t_start, units = "secs"), 1), "seconds\n")

# --- Compute all neighbor features (replaces the for loop) ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_and_add_all_neighbor_features(
  cell_data, W, neighbor_source_vars
)

# --- Apply the pre-trained Random Forest model (unchanged) ---
# predictions <- predict(rf_model, newdata = cell_data)

cat("All neighbor features computed. Ready for prediction.\n")
```

---

## 4. Why This Works and What It Preserves

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `mean` is computed as `sum / count` over the identical neighbor set. `max` and `min` are computed over the identical neighbor set. No approximation, no kernel substitution. |
| **Boundary / irregular cells** | The sparse matrix is built from the exact `spdep::nb` object, so coastal cells, edge cells, and cells with missing years are handled identically to the original. |
| **Trained RF model** | Untouched — the code only computes input features; the model is applied afterward with `predict()`. |
| **Memory** | The sparse matrix has ~25.8M non-zero entries × 16 bytes ≈ ~400 MB. The `summary()` data.table for max/min adds ~200 MB transiently. Well within 16 GB. |
| **Speed** | Sparse matrix construction: ~10–30 seconds. Each variable's stats: ~10–30 seconds (sparse mat-vec multiply + data.table grouped aggregation). **Total: ~2–5 minutes** vs. 86+ hours. |

The key insight is that the neighbor lookup and the row-by-row stat computation are both expressible as **sparse linear algebra** (for sum/count → mean) and **grouped aggregations on the sparse matrix's edge list** (for max/min), eliminating millions of R-level function calls and character operations entirely.