 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation suffers from three compounding bottlenecks:

1. **O(n²)-scale key-based lookups**: `build_neighbor_lookup` creates a named character vector `idx_lookup` of ~6.46 million entries keyed by `paste(id, year)`. For each of the ~6.46M rows, it performs character-based named-vector lookups (`idx_lookup[neighbor_keys]`), which are O(n) hash lookups on a massive character vector. This alone is catastrophically slow.

2. **Row-wise `lapply` over 6.46M rows**: `compute_neighbor_stats` iterates row-by-row in R, calling `max`, `min`, `mean` on small vectors. The per-call overhead of R function dispatch × 6.46M × 5 variables ≈ 32.3 million R-level loop iterations.

3. **Redundant topology recomputation per year**: The rook-neighbor graph is purely spatial (time-invariant). Yet the lookup embeds year into every key, rebuilding the mapping for every cell-year combination instead of exploiting the fact that neighbor relationships are identical across all 28 years.

**Why 86+ hours**: ~6.46M rows × ~4 neighbors/row × 5 variables × character hashing + R-level loop overhead. The algorithm is correct but the data structure and iteration strategy are asymptotically poor for this scale.

---

## Optimization Strategy

### Core Insight
The rook-neighbor graph is **time-invariant**. A cell's neighbors in 1992 are the same cells in 2019. Therefore:

1. **Build the sparse adjacency structure once** over the 344,208 cells (not 6.46M cell-years).
2. **Operate year-by-year** using vectorized sparse matrix–vector operations: for each year-slice, extract the variable column, then use the sparse adjacency matrix to compute neighbor sums, counts, max, and min in one shot.
3. **Use a sparse matrix (CSR/CSC)** from the `Matrix` package for sum/count/mean. For max and min (which are not linear), use `data.table` grouped operations on an edge-list representation.

### Specific Techniques

| Operation | Method | Complexity |
|-----------|--------|------------|
| Neighbor mean | Sparse matrix multiply: `A %*% x / degree` | O(nnz) per variable-year |
| Neighbor max/min | `data.table` join + grouped aggregation on edge list | O(nnz) per variable-year |
| Year slicing | `data.table` keyed subset | O(n/28) per year |

### Expected Speedup
- Eliminates all character-key hashing (~6.46M × 4 lookups).
- Replaces 32.3M R-level `lapply` calls with ~140 vectorized passes (5 vars × 28 years) over a sparse structure with ~1.37M edges.
- Estimated runtime: **2–5 minutes** on a 16 GB laptop.

### Numerical Equivalence
The sparse matrix `A %*% x` computes exactly `sum(neighbor_vals)`. Dividing by the count of non-NA neighbors gives the identical `mean`. Max and min via `data.table` grouped operations are identical to the original `max(neighbor_vals)` and `min(neighbor_vals)`. NA handling is replicated exactly.

---

## Optimized R Code

```r
# =============================================================================
# Optimized Neighbor Aggregation Pipeline
# Preserves numerical equivalence with original compute_neighbor_stats
# =============================================================================

library(data.table)
library(Matrix)

# ---- Step 1: Build time-invariant sparse adjacency structures (ONCE) --------

build_sparse_neighbor_structures <- function(id_order, rook_neighbors_unique) {
  # id_order: vector of 344,208 cell IDs in the order matching the nb object
  # rook_neighbors_unique: spdep nb object (list of integer neighbor indices)
  
  n_cells <- length(id_order)
  
  # Build edge list: from_ref -> to_ref (1-based indices into id_order)
  from_list <- vector("list", n_cells)
  to_list   <- vector("list", n_cells)
  
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nb_i <- nb_i[nb_i != 0L]
    if (length(nb_i) > 0L) {
      from_list[[i]] <- rep.int(i, length(nb_i))
      to_list[[i]]   <- nb_i
    }
  }
  
  from_ref <- unlist(from_list, use.names = FALSE)
  to_ref   <- unlist(to_list, use.names = FALSE)
  
  # Sparse adjacency matrix (rows = focal cells, cols = neighbor cells)
  # A[i,j] = 1 means j is a rook neighbor of i
  A <- sparseMatrix(
    i = from_ref,
    j = to_ref,
    x = 1,
    dims = c(n_cells, n_cells),
    repr = "C"   # CSC format, efficient for column operations
  )
  
  # Edge list as data.table for max/min operations
  edge_dt <- data.table(
    focal_ref = from_ref,
    neighbor_ref = to_ref
  )
  
  # Map from cell ID to reference index
  id_to_ref <- setNames(seq_len(n_cells), as.character(id_order))
  
  list(
    A = A,
    edge_dt = edge_dt,
    id_to_ref = id_to_ref,
    id_order = id_order,
    n_cells = n_cells
  )
}


# ---- Step 2: Compute neighbor stats for all variables -----------------------

compute_all_neighbor_features <- function(cell_data, neighbor_source_vars,
                                          id_order, rook_neighbors_unique) {
  # Convert to data.table for speed (non-destructive if already data.table)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # Build sparse structures once
  cat("Building sparse neighbor structures...\n")
  sp <- build_sparse_neighbor_structures(id_order, rook_neighbors_unique)
  A        <- sp$A
  edge_dt  <- sp$edge_dt
  id_to_ref <- sp$id_to_ref
  n_cells  <- sp$n_cells
  
  # Ensure cell_data has a reference index column
  # Map each row's cell ID to the reference index in id_order
  cell_data[, ref_idx := id_to_ref[as.character(id)]]
  
  # Get sorted unique years
  years <- sort(unique(cell_data$year))
  
  # Key by year + ref_idx for fast subsetting
  setkey(cell_data, year, ref_idx)
  
  # Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    cell_data[, (col_max)  := NA_real_]
    cell_data[, (col_min)  := NA_real_]
    cell_data[, (col_mean) := NA_real_]
  }
  
  cat(sprintf("Processing %d variables x %d years = %d passes...\n",
              length(neighbor_source_vars), length(years),
              length(neighbor_source_vars) * length(years)))
  
  for (yr in years) {
    # Extract the year-slice: all cells present in this year
    yr_rows <- cell_data[.(yr)]  # keyed lookup
    
    # Map: for each ref_idx present this year, what is its row index in cell_data?
    # We need the actual row indices in the full cell_data
    yr_row_indices <- cell_data[, .I[year == yr]]
    
    # Build a vector: for ref_idx 1..n_cells, what is the row index in cell_data?
    # (NA if that cell is not present this year)
    ref_to_row <- rep(NA_integer_, n_cells)
    refs_present <- cell_data$ref_idx[yr_row_indices]
    ref_to_row[refs_present] <- yr_row_indices
    
    for (var_name in neighbor_source_vars) {
      col_max  <- paste0("neighbor_max_", var_name)
      col_min  <- paste0("neighbor_min_", var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      
      # Build a full-length vector of values indexed by ref_idx
      # (NA for cells not present this year)
      x <- rep(NA_real_, n_cells)
      x[refs_present] <- cell_data[[var_name]][yr_row_indices]
      
      # --- MEAN via sparse matrix ---
      # Replace NA with 0 for summation, track non-NA counts
      x_nona <- x
      x_nona[is.na(x_nona)] <- 0
      not_na <- as.numeric(!is.na(x))
      
      neighbor_sum   <- as.numeric(A %*% x_nona)      # sum of neighbor values
      neighbor_count <- as.numeric(A %*% not_na)       # count of non-NA neighbors
      
      neighbor_mean <- ifelse(neighbor_count > 0,
                              neighbor_sum / neighbor_count,
                              NA_real_)
      
      # --- MAX and MIN via edge list + data.table ---
      # Attach neighbor values to edge list
      edge_work <- copy(edge_dt)
      edge_work[, nval := x[neighbor_ref]]
      
      # Remove edges where neighbor value is NA
      edge_work <- edge_work[!is.na(nval)]
      
      # Grouped aggregation
      if (nrow(edge_work) > 0) {
        agg <- edge_work[, .(nmax = max(nval), nmin = min(nval)),
                         by = focal_ref]
        
        neighbor_max_vec <- rep(NA_real_, n_cells)
        neighbor_min_vec <- rep(NA_real_, n_cells)
        neighbor_max_vec[agg$focal_ref] <- agg$nmax
        neighbor_min_vec[agg$focal_ref] <- agg$nmin
      } else {
        neighbor_max_vec <- rep(NA_real_, n_cells)
        neighbor_min_vec <- rep(NA_real_, n_cells)
      }
      
      # Also set mean to NA where count == 0 (no non-NA neighbors)
      # (already handled above)
      
      # Write results back to cell_data for rows present this year
      set(cell_data, i = yr_row_indices, j = col_max,
          value = neighbor_max_vec[refs_present])
      set(cell_data, i = yr_row_indices, j = col_min,
          value = neighbor_min_vec[refs_present])
      set(cell_data, i = yr_row_indices, j = col_mean,
          value = neighbor_mean[refs_present])
    }
    
    cat(sprintf("  Year %d done.\n", yr))
  }
  
  # Clean up helper column
  cell_data[, ref_idx := NULL]
  
  cat("Neighbor feature computation complete.\n")
  return(cell_data)
}


# =============================================================================
# USAGE
# =============================================================================

# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- compute_all_neighbor_features(
#   cell_data              = cell_data,
#   neighbor_source_vars   = neighbor_source_vars,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique
# )
#
# # Now cell_data has columns:
# #   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
# #   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
# #   ... etc.
#
# # Apply the pre-trained Random Forest (unchanged):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Further Optimization: Eliminate `copy(edge_dt)` with Pre-allocated Columns

The `copy(edge_dt)` inside the inner loop creates ~1.37M-row copies 140 times. We can eliminate this:

```r
# Replace the inner-loop edge_work section with this zero-copy version:

# Before the year loop, add a persistent value column to edge_dt:
edge_dt[, nval := NA_real_]

# Then inside the loop, replace:
#   edge_work <- copy(edge_dt)
#   edge_work[, nval := x[neighbor_ref]]
#   edge_work <- edge_work[!is.na(nval)]
# With:
set(edge_dt, j = "nval", value = x[edge_dt$neighbor_ref])
valid_mask <- !is.na(edge_dt$nval)

if (any(valid_mask)) {
  agg <- edge_dt[valid_mask, .(nmax = max(nval), nmin = min(nval)),
                 by = focal_ref]
  neighbor_max_vec <- rep(NA_real_, n_cells)
  neighbor_min_vec <- rep(NA_real_, n_cells)
  neighbor_max_vec[agg$focal_ref] <- agg$nmax
  neighbor_min_vec[agg$focal_ref] <- agg$nmin
} else {
  neighbor_max_vec <- rep(NA_real_, n_cells)
  neighbor_min_vec <- rep(NA_real_, n_cells)
}
```

---

## Complexity Comparison

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Neighbor lookup construction | O(6.46M × k) character hashes | O(344K × k) integer edge list, **once** |
| Per-variable-year aggregation | O(6.46M) R-level `lapply` calls | O(1.37M) sparse matrix multiply + `data.table` grouped agg |
| Total R-level loop iterations | ~32.3M | 140 vectorized passes |
| Memory pattern | 6.46M-entry named character vector | 1.37M-edge sparse matrix (< 20 MB) |
| **Estimated wall time** | **86+ hours** | **2–5 minutes** |

The trained Random Forest model is never touched—only the feature engineering pipeline is replaced. The output columns (`neighbor_max_*`, `neighbor_min_*`, `neighbor_mean_*`) are numerically identical to the original, preserving the estimand.