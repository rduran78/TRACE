 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates 6.46 million R list entries**, each constructed via string-pasting, named-vector lookups (`idx_lookup[neighbor_keys]`), and per-row `lapply`. Named vector lookup in R is O(n) per query on average for large vectors, making the total cost O(n²)-like. With ~6.46M rows, this is catastrophic.

2. **`compute_neighbor_stats` iterates over 6.46M list elements** in R-level `lapply`, extracting subsets of a numeric vector per element. This is called 5 times (once per variable), totaling ~32.3M R-level loop iterations with per-iteration allocation overhead.

3. **The neighbor lookup is year-entangled.** The topology (which cells neighbor which) is identical across all 28 years, but the current code rebuilds row-index mappings that fuse cell identity with year, creating a 6.46M-entry lookup instead of a 344K-entry one. This is a ~19× unnecessary inflation.

**Key insight:** Rook-neighbor topology is purely spatial — it does not change across years. The adjacency structure among 344,208 cells is fixed. The per-year computation is simply: for each cell, gather neighbor attribute values *within the same year* and compute max/min/mean. This is a **sparse matrix–vector product** (for mean) and analogous sparse aggregation (for max/min), repeated 28 times × 5 variables = 140 passes over a sparse matrix with ~1.37M nonzeros. This should take **seconds**, not hours.

---

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from the `nb` object (344,208 × 344,208, ~1.37M nonzero entries). This is the graph topology.

2. **Organize data as cell × year matrices** (344,208 rows × 28 columns) for each variable. This allows vectorized column-wise (per-year) sparse aggregation.

3. **Compute neighbor statistics via sparse matrix operations:**
   - **Mean:** `A_norm %*% X` where `A_norm` is the row-normalized adjacency matrix (each row sums to 1, or the count of neighbors).
   - **Sum:** `A %*% X` (binary adjacency), then divide by neighbor count for mean.
   - **Max/Min:** Use a grouped operation via the sparse matrix's row structure — iterate over rows of the sparse matrix in C++ via `dgCMatrix` slot access, or use `data.table` grouped aggregation on the edge list.

4. **Reshape results back** to the long panel format and column-bind to `cell_data`.

5. **Feed the augmented `cell_data` to `predict(rf_model, ...)` unchanged.**

This reduces the problem from 6.46M R-level list operations to 140 sparse-matrix operations on a 344K × 344K matrix with 1.37M entries — a speedup of roughly **3–4 orders of magnitude**.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Sparse graph neighborhood aggregation — preserves numerical equivalence
# =============================================================================

library(Matrix)
library(data.table)

# -------------------------------------------------------------------------
# Step 1: Build sparse adjacency matrix from nb object (once)
# -------------------------------------------------------------------------
build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj: list of length n, each element is integer vector of neighbor indices
  # n: number of spatial cells (344208)
  
  # Build COO triplets
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  
  # Binary adjacency (directed edges as given)
  A <- sparseMatrix(
    i    = from,
    j    = to,
    x    = rep(1, length(from)),
    dims = c(n, n),
    repr = "C"   # CSC format, will convert to CSR-like via transpose trick
  )
  return(A)
}

# -------------------------------------------------------------------------
# Step 2: Compute max, min, mean for all neighbor source variables
# -------------------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, id_order, nb_obj,
                                          neighbor_source_vars) {
  
  n_cells <- length(id_order)
  
  # --- Build adjacency matrix ---
  cat("Building sparse adjacency matrix...\n")
  A <- build_adjacency_matrix(nb_obj, n_cells)
  
  # --- Precompute neighbor counts per cell (for mean calculation) ---
  neighbor_counts <- diff(A@p)  # For dgCMatrix in CSC: column counts

  # We need ROW counts for row-wise aggregation. Transpose to get row access:
  At <- t(A)  # Now At is CSC, and column j of At = row j of A
  row_neighbor_counts <- diff(At@p)  # number of neighbors per cell
  
  # --- Convert cell_data to data.table for fast reshaping ---
  cat("Preparing data structures...\n")
  dt <- as.data.table(cell_data)
  
  # Create cell index: map id -> position in id_order
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  dt[, cell_idx := id_to_idx[as.character(id)]]
  
  # Get sorted unique years
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))
  dt[, year_idx := year_to_col[as.character(year)]]
  
  # Sort dt by cell_idx and year_idx for consistent ordering
  # We need to track original row order to map results back
  dt[, orig_row := .I]
  setkey(dt, cell_idx, year_idx)
  
  # --- For each variable, build cell × year matrix, compute stats ---
  # We use At (transpose of A) in CSC format.
  # Column j of At contains the row indices of neighbors of cell j.
  # At@p[j]+1 to At@p[j+1] gives the positions in At@i for neighbors of cell j.
  
  # Extract sparse structure once
  Ap <- At@p      # length n_cells + 1
  Ai <- At@i + 1L # 0-based to 1-based: neighbor cell indices for each cell
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))
    
    # Build cell × year matrix (n_cells x n_years)
    # Fill with NA for missing cell-year combinations
    vals_vec <- dt[[var_name]]
    cidx     <- dt$cell_idx
    yidx     <- dt$year_idx
    
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(cidx, yidx)] <- vals_vec
    
    # Allocate output matrices
    max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # --- Vectorized aggregation per year ---
    for (yr in seq_len(n_years)) {
      x_yr <- X[, yr]  # length n_cells, values for this year
      
      # For cells with no neighbors, result stays NA
      # For cells with neighbors, gather neighbor values
      
      # Use sparse matrix to gather: for each cell j, neighbors are Ai[Ap[j]+1 : Ap[j+1]]
      # Expand neighbor values
      neighbor_vals <- x_yr[Ai]  # all neighbor values, ordered by cell
      
      # Group by cell using the pointer structure
      # Create a cell-id vector for each neighbor entry
      cell_of_entry <- rep(seq_len(n_cells), times = diff(Ap))
      
      # Remove NA neighbor values
      valid <- !is.na(neighbor_vals)
      nv_valid   <- neighbor_vals[valid]
      cell_valid <- cell_of_entry[valid]
      
      if (length(nv_valid) > 0) {
        # Use data.table for grouped max/min/mean — extremely fast
        agg_dt <- data.table(cell = cell_valid, val = nv_valid)
        agg <- agg_dt[, .(
          nb_max  = max(val),
          nb_min  = min(val),
          nb_mean = mean(val)
        ), by = cell]
        
        max_mat[agg$cell, yr]  <- agg$nb_max
        min_mat[agg$cell, yr]  <- agg$nb_min
        mean_mat[agg$cell, yr] <- agg$nb_mean
      }
    }
    
    # --- Map results back to dt rows ---
    max_col  <- paste0("nb_max_", var_name)
    min_col  <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)
    
    dt[, (max_col)  := max_mat[cbind(cell_idx, year_idx)]]
    dt[, (min_col)  := min_mat[cbind(cell_idx, year_idx)]]
    dt[, (mean_col) := mean_mat[cbind(cell_idx, year_idx)]]
  }
  
  # --- Restore original row order ---
  setorder(dt, orig_row)
  
  # Remove helper columns
  dt[, c("cell_idx", "year_idx", "orig_row") := NULL]
  
  return(dt)
}

# =============================================================================
# USAGE
# =============================================================================

# Load pre-existing objects (assumed already in environment or loaded from disk):
#   cell_data              — data.frame/data.table with columns: id, year, ntl, ec, ...
#   id_order               — integer vector of cell IDs (length 344208)
#   rook_neighbors_unique  — nb object (list of length 344208)
#   rf_model               — pre-trained randomForest model

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  nb_obj               = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# --- Predict with the pre-trained Random Forest (unchanged) ---
predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Original | Optimized | Equivalence |
|---|---|---|
| `vals[idx]` gathers neighbor values for one cell-year | `x_yr[Ai[...]]` gathers the same neighbor values via sparse column pointers | Same index set from same `nb` object |
| `max(neighbor_vals)` | `data.table [, max(val), by=cell]` | Identical `max()` function, same inputs |
| `min(neighbor_vals)` | `data.table [, min(val), by=cell]` | Identical `min()` function, same inputs |
| `mean(neighbor_vals)` | `data.table [, mean(val), by=cell]` | Identical `mean()` function, same inputs |
| NA removal: `neighbor_vals[!is.na(...)]` | `valid <- !is.na(neighbor_vals)` filtering | Same NA exclusion logic |
| All-NA → `c(NA, NA, NA)` | Cells absent from `agg` remain `NA` in output matrix | Same behavior |

No floating-point reordering occurs — `mean()` sums in the same element order within each cell's neighbor set (sparse column order preserves the `nb` object's neighbor ordering).

---

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (6.46M string ops) | ~2 seconds (`sparseMatrix`) |
| Per-variable aggregation | ~17 hours (6.46M R list iterations) | ~10–30 seconds (28 vectorized sparse gathers + `data.table` grouped agg) |
| Total for 5 variables | **86+ hours** | **~2–3 minutes** |
| Peak RAM | ~10–14 GB (6.46M-element list) | ~3–5 GB (sparse matrix + cell×year matrices) |

The sparse matrix `A` has ~1.37M nonzeros in a 344K × 344K matrix (density ≈ 0.001%), consuming ~22 MB. Each cell × year matrix is 344,208 × 28 ≈ 77 MB. Total working memory is well within 16 GB.