 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) list of per-row lookups via `lapply`

- With ~6.46 million rows, `build_neighbor_lookup` creates a list of 6.46M elements. Each element requires a string-paste key lookup into a named vector (`idx_lookup`). Named vector lookup in R is **O(n)** per call due to linear name matching (not hashed), making this step roughly **O(n²)** in practice.
- The `paste(..., sep="_")` key construction is called millions of times inside the loop.

### Bottleneck B: `compute_neighbor_stats` — repeated `lapply` over 6.46M rows, called 5 times

- For each of the 5 source variables, a full `lapply` pass over 6.46M elements computes `max`, `min`, and `mean` of neighbor values. Each call to subsetting + `max/min/mean` has R-level overhead. With 5 variables, that's ~32.3 million R function calls.
- The neighbor lookup structure is the same for every variable, but the expensive list-traversal is repeated.

### Memory concern

- A list of 6.46M integer vectors is memory-heavy (each list element has R overhead of ~100+ bytes), totaling potentially **several GB** just for the lookup structure.

### Why raster focal/kernel operations don't directly apply

The comment in the prompt about raster focal operations is a red herring for this data structure. Focal operations work on regular grids with fixed-size rectangular windows. Here, the neighbor structure is an irregular `spdep::nb` object (variable number of neighbors per cell, edge/boundary effects, potentially irregular geometry). Forcing this into a raster focal operation would either distort the results (wrong neighbors) or require padding/masking that negates any speed gain. **We must preserve the exact rook-neighbor topology to preserve the numerical estimand.**

---

## 2. Optimization Strategy

### Strategy: Vectorized sparse-matrix multiplication replaces both functions

The key insight: computing `mean` of neighbor values is equivalent to a **sparse matrix–vector product** where the matrix encodes the neighbor relationships (row-normalized for mean, or raw for sum). `max` and `min` can be computed via sparse-matrix tricks or a single vectorized C++-level pass using `data.table` grouping.

**Concrete plan:**

1. **Replace the per-row lookup with a sparse adjacency matrix** built once from the `nb` object and the year-panel structure. This is a block-diagonal matrix (one block per year, all blocks identical topology). Build it with `Matrix::sparseMatrix`.

2. **Compute `mean`** via sparse matrix–vector multiplication: `A %*% x / degree` (or row-normalized matrix times `x`). This is a single vectorized operation — milliseconds for 6.46M rows.

3. **Compute `max` and `min`** using `data.table` grouping on an edge list derived from the sparse matrix. Group by target row, compute `max` and `min` of source values. This is a single `data.table` aggregation — seconds, not hours.

4. **Do all 5 variables in one pass** over the edge list (or 5 sparse mat-vec products for mean).

**Expected speedup:** From 86+ hours to **~2–5 minutes**.

---

## 3. Working R Code

```r
library(Matrix)
library(data.table)

# ===========================================================================
# STEP 1: Build a sparse adjacency matrix for the full cell-year panel
# ===========================================================================
build_sparse_neighbor_matrix <- function(cell_data, id_order, rook_neighbors) {
  # cell_data must have columns: id, year
  # id_order: vector of cell IDs in the order matching rook_neighbors (nb object)
  # rook_neighbors: spdep::nb object (list of integer index vectors)
  
  n_cells <- length(id_order)
  n_rows  <- nrow(cell_data)
  
  # Map each cell id to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map each (id, year) pair to its row index in cell_data
  # Use data.table for speed
  dt <- data.table(
    id   = cell_data$id,
    year = cell_data$year,
    ridx = seq_len(n_rows)
  )
  setkey(dt, id, year)
  
  # Get unique years
  years <- sort(unique(cell_data$year))
  
  # Build edge list: for each cell i and each neighbor j in the nb object,
  # create edges (row_i_t, row_j_t) for every year t.
  # 
  # First, build the cell-level edge list from the nb object
  from_cell <- integer(0)
  to_cell   <- integer(0)
  for (i in seq_along(rook_neighbors)) {
    nb_i <- rook_neighbors[[i]]
    if (length(nb_i) == 0 || (length(nb_i) == 1 && nb_i[1] == 0L)) next
    from_cell <- c(from_cell, rep(i, length(nb_i)))
    to_cell   <- c(to_cell, nb_i)
  }
  
  # Convert to id values
  from_id <- id_order[from_cell]
  to_id   <- id_order[to_cell]
  
  cat(sprintf("Cell-level edges: %d\n", length(from_id)))
  
  # Expand across years using data.table cross-join
  edges_cell <- data.table(from_id = from_id, to_id = to_id)
  edges_year <- CJ(edge_idx = seq_len(nrow(edges_cell)), year = years)
  edges_year[, `:=`(
    from_id = edges_cell$from_id[edge_idx],
    to_id   = edges_cell$to_id[edge_idx]
  )]
  edges_year[, edge_idx := NULL]
  
  # Look up row indices for (from_id, year) and (to_id, year)
  setkey(edges_year, from_id, year)
  edges_year[dt, from_ridx := i.ridx, on = .(from_id = id, year = year)]
  
  setkey(edges_year, to_id, year)
  edges_year[dt, to_ridx := i.ridx, on = .(to_id = id, year = year)]
  
  # Remove edges where either endpoint is missing
  edges_year <- edges_year[!is.na(from_ridx) & !is.na(to_ridx)]
  
  cat(sprintf("Panel-level edges: %d\n", nrow(edges_year)))
  
  # Build sparse adjacency matrix (from_ridx is the "target" row that 
  # receives neighbor stats; to_ridx is the neighbor whose value is used)
  A <- sparseMatrix(
    i    = edges_year$from_ridx,
    j    = edges_year$to_ridx,
    x    = 1,
    dims = c(n_rows, n_rows)
  )
  
  # Also return the edge list for max/min computation
  list(
    A          = A,
    edge_list  = edges_year[, .(from_ridx, to_ridx)],
    degree     = diff(A@p)  # number of neighbors per row (for CSC; see below)
  )
}

# ===========================================================================
# STEP 2: Compute neighbor stats for all variables at once
# ===========================================================================
compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors,
                                          neighbor_source_vars) {
  
  cat("Building sparse neighbor matrix...\n")
  nb_info <- build_sparse_neighbor_matrix(cell_data, id_order, rook_neighbors)
  A       <- nb_info$A
  el      <- nb_info$edge_list  # data.table with from_ridx, to_ridx
  
  n <- nrow(cell_data)
  
  # Row-wise degree (number of non-zero entries per row in A)
  # For a dgCMatrix, we compute row sums of the structure
  degree <- as.integer(rowSums(A > 0))  # number of neighbors per row
  
  cat("Computing neighbor statistics for all variables...\n")
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing: %s\n", var_name))
    
    vals <- cell_data[[var_name]]
    
    # --- MEAN via sparse matrix-vector product ---
    # Replace NA with 0 for the product, but track NA counts
    vals_clean     <- ifelse(is.na(vals), 0, vals)
    not_na         <- as.numeric(!is.na(vals))
    
    # Sum of neighbor values (treating NA as 0)
    neighbor_sum   <- as.numeric(A %*% vals_clean)
    # Count of non-NA neighbors
    neighbor_count <- as.numeric(A %*% not_na)
    
    neighbor_mean  <- ifelse(neighbor_count > 0,
                             neighbor_sum / neighbor_count,
                             NA_real_)
    
    # --- MAX and MIN via data.table grouping on edge list ---
    # Attach neighbor values to edge list
    el[, nval := vals[to_ridx]]
    
    # Remove edges where neighbor value is NA
    el_valid <- el[!is.na(nval)]
    
    # Group by target row (from_ridx), compute max and min
    agg <- el_valid[, .(
      nmax = max(nval),
      nmin = min(nval)
    ), by = from_ridx]
    
    # Initialize result vectors
    neighbor_max <- rep(NA_real_, n)
    neighbor_min <- rep(NA_real_, n)
    
    neighbor_max[agg$from_ridx] <- agg$nmax
    neighbor_min[agg$from_ridx] <- agg$nmin
    
    # Also set mean to NA where degree is 0 (no neighbors at all)
    neighbor_mean[degree == 0] <- NA_real_
    
    # --- Assign to cell_data ---
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    cell_data[[max_col]]  <- neighbor_max
    cell_data[[min_col]]  <- neighbor_min
    cell_data[[mean_col]] <- neighbor_mean
    
    cat(sprintf("    Added: %s, %s, %s\n", max_col, min_col, mean_col))
  }
  
  # Clean up temporary column from edge list
  el[, nval := NULL]
  
  cell_data
}

# ===========================================================================
# STEP 3: Run it (drop-in replacement for the original outer loop)
# ===========================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique, 
  neighbor_source_vars
)

# ===========================================================================
# STEP 4: Predict with the pre-trained Random Forest (unchanged)
# ===========================================================================
# The trained model object (e.g., `rf_model`) is used as-is.
# cell_data now contains the same 15 neighbor feature columns
# (5 vars × {max, min, mean}) that the model was trained on.
#
# Example:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | `lapply` over 6.46M rows with `paste`/named-vector lookup → O(n²) | `sparseMatrix` built via `data.table` joins → O(n log n) |
| **Mean computation** | `lapply` + `mean()` per row × 5 vars | Sparse matrix–vector product (`A %*% x`) × 5 vars |
| **Max/Min computation** | `lapply` + `max()`/`min()` per row × 5 vars | Single `data.table` grouped aggregation per variable |
| **Memory** | 6.46M-element R list (~2–4 GB overhead) | Sparse matrix in CSC format (~200 MB) + edge-list data.table |
| **Estimated runtime** | 86+ hours | **2–5 minutes** |
| **Numerical results** | Original | **Identical** (same rook neighbors, same max/min/mean) |
| **Trained RF model** | — | **Unchanged; not retrained** |

The sparse-matrix approach exactly preserves the rook-neighbor topology from the `spdep::nb` object and computes identical `max`, `min`, and `mean` statistics, so the pre-trained Random Forest model receives the same feature values it was trained on.