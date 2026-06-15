 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The repeated string-key work is **not merely a local inefficiency — it is a symptom of a broader O(N × K) algorithmic pattern** that pervades the entire pipeline.

### What's happening

Inside `build_neighbor_lookup`, for each of the **~6.46 million rows**, the code:

1. Converts `data$id[i]` to a character and looks it up in `id_to_ref` — **6.46M character conversions + named-vector lookups**.
2. Extracts neighbor cell IDs from the `nb` object.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — **creating ~6.46M × avg_neighbors string keys** (with ~1.37M directed relationships, that's roughly **~25–30 million paste operations**).
4. Looks each key up in the named vector `idx_lookup` (itself built from **6.46M paste operations**).

Then `compute_neighbor_stats` is called **5 times** (once per source variable), each time iterating over 6.46M entries in the lookup list. The lookup list itself is a list of 6.46M integer vectors, consuming substantial memory.

### The core insight

The neighbor structure is **purely spatial** — it does not change across years. Cell `i`'s neighbors in 1992 are the same cells as its neighbors in 2019. Yet the code rebuilds neighbor lookups at the **cell-year** level by pasting year into keys, effectively re-discovering the same spatial topology 28 times (once per year).

**This is the larger repeated-lookup pattern:** the algorithm operates at the cell-year grain when the neighbor topology only exists at the cell grain.

### Quantitative impact

| Operation | Current cost |
|---|---|
| Building `idx_lookup` (paste over 6.46M rows) | ~6.46M string ops |
| Inner `lapply` over 6.46M rows, each doing paste + named lookup | ~25-30M string ops + hash lookups |
| Storing `neighbor_lookup` as list of 6.46M integer vectors | ~200-400 MB |
| `compute_neighbor_stats` called 5× over 6.46M list entries | ~32M R-level iterations |
| **Total estimated wall time** | **86+ hours** |

---

## 2. Optimization Strategy

### Key principle: Separate spatial topology from temporal indexing

Since neighbors are defined **per cell, not per cell-year**, we should:

1. **Build the neighbor lookup once at the cell level** (344,208 cells, not 6.46M cell-years).
2. **For each year, extract the relevant slice of data** (a matrix/data.table operation), compute neighbor statistics **vectorized** using the cell-level adjacency structure.
3. **Use a sparse adjacency matrix** and matrix multiplication / grouped operations to compute `mean`, `max`, `min` across neighbors — entirely vectorized, no R-level loops over millions of rows.

### Algorithmic reformulation

- Represent the rook-neighbor topology as a **sparse row-normalized (or raw) adjacency matrix** `W` of dimension 344,208 × 344,208 (~1.37M non-zero entries).
- For each variable and each year, arrange the variable values into a **vector of length 344,208** (one entry per cell).
- **Neighbor mean** = sparse matrix-vector multiply `W %*% x` (with row-normalization), or equivalently, compute the sum and count per row.
- **Neighbor max/min** = one pass over the sparse matrix structure.

This replaces ~6.46M R-level list iterations with ~28 sparse matrix operations per variable, each touching only ~1.37M non-zero entries.

### Expected speedup

| Component | Current | Proposed |
|---|---|---|
| Neighbor lookup construction | ~6.46M string ops + hash lookups | One-time sparse matrix build (~1.37M entries) |
| Neighbor stats (per variable) | 6.46M R-level list iterations | 28 sparse mat-vec ops (~1.37M entries each) |
| Total for 5 variables | ~32.3M R iterations | 140 sparse mat-vec ops + 140 vectorized max/min passes |
| **Estimated wall time** | **86+ hours** | **~2–10 minutes** |

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# =============================================================================
# 
# Prerequisites:
#   - cell_data: data.frame/data.table with columns 'id', 'year', and the
#     neighbor source variables. ~6.46M rows.
#   - id_order: vector of cell IDs in the order matching rook_neighbors_unique.
#   - rook_neighbors_unique: an nb object (list of integer index vectors)
#     of length equal to length(id_order) = 344,208.
#
# Output:
#   - cell_data gains columns: {var}_neighbor_max, {var}_neighbor_min,
#     {var}_neighbor_mean for each var in neighbor_source_vars.
#
# Preserves: trained Random Forest model (no retraining), original numerical
#   estimand (max, min, mean of non-NA neighbor values, NA when no valid
#   neighbors exist).
# =============================================================================

library(data.table)
library(Matrix)

# ---- Step 0: Ensure cell_data is a data.table ----
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ---- Step 1: Build sparse adjacency matrix (once, ~1.37M entries) ----
# rook_neighbors_unique is an nb object: a list of length N_cells,
# where each element is an integer vector of neighbor indices (into id_order).
# Index 0 means "no neighbors" in spdep convention.

build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj: list of integer vectors (1-indexed neighbor indices, 0 = no neighbors)
  # n: number of cells (length of nb_obj)
  
  from <- integer(0)
  to   <- integer(0)
  
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs > 0L]  # remove 0s (spdep convention for no-neighbor)
    if (length(nbrs) > 0L) {
      from <- c(from, rep(i, length(nbrs)))
      to   <- c(to, nbrs)
    }
  }
  
  # Sparse logical adjacency matrix (rows = focal cell, cols = neighbor cell)
  sparseMatrix(i = from, j = to, x = rep(1, length(from)), dims = c(n, n))
}

n_cells <- length(id_order)
cat("Building sparse adjacency matrix for", n_cells, "cells...\n")
W <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
cat("Adjacency matrix:", nrow(W), "x", ncol(W), "with", nnzero(W), "non-zero entries\n")

# ---- Step 2: Create cell-index mapping ----
# Map each cell ID to its position in id_order (1..344208)
cell_id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))

# Add cell_idx column to cell_data (position in the spatial grid)
cell_data[, cell_idx := cell_id_to_idx[as.character(id)]]

# ---- Step 3: Vectorized neighbor stats computation ----
# For each year and each variable, we:
#   (a) Create a vector of length n_cells with the variable values.
#   (b) Use the sparse matrix to compute sum, count, max, min across neighbors.
#   (c) Write results back to the corresponding cell-year rows.

# Pre-extract the sparse matrix structure for max/min (CSR-like access)
# We use the dgCMatrix (compressed sparse column) and work column-wise,
# or convert to dgRMatrix for row-wise access.

# For max/min we need row-wise iteration over the sparse matrix.
# Convert W to a dgRMatrix (row-compressed) or use the column structure of t(W).
# Actually, we'll extract row pointers from the CSR representation.

W_row <- as(W, "RsparseMatrix")  # dgRMatrix: row-compressed

# Extract CSR components
row_ptr <- W_row@p    # length n_cells + 1, 0-indexed pointers
col_idx <- W_row@j    # 0-indexed column indices
# W_row@x are all 1s

compute_neighbor_stats_sparse <- function(cell_data, var_name, W_row, row_ptr,
                                          col_idx, n_cells, cell_id_to_idx) {
  # Initialize result columns
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
  
  years <- sort(unique(cell_data$year))
  
  for (yr in years) {
    # Extract rows for this year
    yr_mask <- cell_data$year == yr
    yr_data <- cell_data[yr_mask]
    
    # Build a full-length vector (n_cells) for this variable in this year.
    # Cells not present in the data for this year get NA.
    vals_full <- rep(NA_real_, n_cells)
    vals_full[yr_data$cell_idx] <- yr_data[[var_name]]
    
    # For each cell (row of W), gather neighbor values and compute stats.
    # Vectorized approach using the CSR structure:
    #   For row i: neighbors are col_idx[(row_ptr[i]+1):(row_ptr[i+1])] (0-indexed)
    
    # We'll compute this in C-like fashion via vapply over unique cells,
    # but since n_cells = 344K this is fast (not 6.46M).
    
    # Even faster: use Matrix operations for mean, and a compiled loop for max/min.
    # Mean via sparse matrix-vector multiply:
    
    # neighbor_sum = W %*% vals_full (treating NA as 0 is wrong; we need NA-aware)
    # So we need:
    #   neighbor_sum  = W %*% vals_no_na
    #   neighbor_count = W %*% (!is.na(vals_full))
    #   neighbor_mean = neighbor_sum / neighbor_count
    
    not_na <- !is.na(vals_full)
    vals_zero <- vals_full
    vals_zero[!not_na] <- 0
    
    neighbor_sum   <- as.numeric(W %*% vals_zero)
    neighbor_count <- as.numeric(W %*% as.numeric(not_na))
    neighbor_mean  <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # For max and min, we iterate over cells (344K, not 6.46M).
    # This is ~344K iterations, each accessing a handful of neighbors. Very fast.
    neighbor_max <- rep(NA_real_, n_cells)
    neighbor_min <- rep(NA_real_, n_cells)
    
    for (i in seq_len(n_cells)) {
      start <- row_ptr[i]       # 0-indexed
      end   <- row_ptr[i + 1L]  # 0-indexed, exclusive
      if (end > start) {
        # col_idx is 0-indexed, so add 1
        nbr_indices <- col_idx[(start + 1L):end] + 1L
        nbr_vals <- vals_full[nbr_indices]
        nbr_vals <- nbr_vals[!is.na(nbr_vals)]
        if (length(nbr_vals) > 0L) {
          neighbor_max[i] <- max(nbr_vals)
          neighbor_min[i] <- min(nbr_vals)
        }
      }
    }
    
    # Also set mean to NA where count == 0
    # (already handled above)
    
    # Write results back to cell_data for this year's rows.
    # Map from cell_idx back to the data rows.
    yr_cell_indices <- yr_data$cell_idx
    
    cell_data[yr_mask, (max_col)  := neighbor_max[yr_cell_indices]]
    cell_data[yr_mask, (min_col)  := neighbor_min[yr_cell_indices]]
    cell_data[yr_mask, (mean_col) := neighbor_mean[yr_cell_indices]]
  }
  
  return(cell_data)
}

# ---- Step 4: Run for all neighbor source variables ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features for", length(neighbor_source_vars), "variables",
    "across", length(unique(cell_data$year)), "years...\n")

for (var_name in neighbor_source_vars) {
  cat("  Processing:", var_name, "\n")
  t0 <- proc.time()
  cell_data <- compute_neighbor_stats_sparse(
    cell_data, var_name, W, row_ptr, col_idx, n_cells, cell_id_to_idx
  )
  elapsed <- (proc.time() - t0)[3]
  cat("    Done in", round(elapsed, 1), "seconds\n")
}

# ---- Step 5: Clean up helper column ----
cell_data[, cell_idx := NULL]

cat("All neighbor features computed.\n")
```

---

## 4. Further optimization: eliminate the R-level loop for max/min

The inner `for (i in seq_len(n_cells))` loop for max/min over 344K cells is already fast (~seconds per year), but if you want to eliminate it entirely, you can use a **data.table grouping approach** on the edge list:

```r
# ---- Alternative: fully vectorized max/min via data.table edge-list join ----

compute_neighbor_stats_dt <- function(cell_data, var_name, edges_dt, n_cells) {
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
  
  years <- sort(unique(cell_data$year))
  
  # Pre-build a lookup: cell_idx -> value, keyed by cell_idx, rebuilt per year
  for (yr in years) {
    yr_rows <- cell_data[year == yr]
    
    # Lookup table: cell_idx -> variable value
    val_lookup <- yr_rows[, .(cell_idx, val = get(var_name))]
    setkey(val_lookup, cell_idx)
    
    # Join neighbor values onto the edge list
    # edges_dt has columns: from_idx, to_idx
    # We want, for each from_idx, the values of all to_idx neighbors.
    edge_vals <- edges_dt[val_lookup, on = .(to_idx = cell_idx), nomatch = 0L]
    # edge_vals now has: from_idx, to_idx, val
    
    # Remove NA values
    edge_vals <- edge_vals[!is.na(val)]
    
    # Aggregate by from_idx
    stats <- edge_vals[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = from_idx]
    
    # Map back: from_idx -> cell_idx in yr_rows
    # We need to join stats back onto the year slice
    yr_rows_idx <- yr_rows[, .(row_pos = .I, cell_idx)]
    yr_rows_idx <- stats[yr_rows_idx, on = .(from_idx = cell_idx)]
    
    # Write back (using the positions within the year mask)
    yr_positions <- which(cell_data$year == yr)
    
    # Create a full-size vector and fill in
    set(cell_data, i = yr_positions, j = max_col,  value = yr_rows_idx$nb_max)
    set(cell_data, i = yr_positions, j = min_col,  value = yr_rows_idx$nb_min)
    set(cell_data, i = yr_positions, j = mean_col, value = yr_rows_idx$nb_mean)
  }
  
  return(cell_data)
}

# Build edge list (data.table) from the nb object — done once
build_edge_dt <- function(nb_obj, n) {
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0L) {
      from_list[[i]] <- rep(i, length(nbrs))
      to_list[[i]]   <- nbrs
    }
  }
  data.table(
    from_idx = unlist(from_list, use.names = FALSE),
    to_idx   = unlist(to_list, use.names = FALSE)
  )
}

edges_dt <- build_edge_dt(rook_neighbors_unique, n_cells)
setkey(edges_dt, to_idx)

for (var_name in neighbor_source_vars) {
  cat("  Processing:", var_name, "\n")
  cell_data <- compute_neighbor_stats_dt(cell_data, var_name, edges_dt, n_cells)
}
```

---

## 5. Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Grain of neighbor lookup** | Cell-year (6.46M) | Cell (344K) |
| **String key operations** | ~31M paste + hash lookups | **Zero** |
| **Neighbor lookup storage** | List of 6.46M int vectors (~300MB+) | Sparse matrix or edge table (~22MB) |
| **Stats computation** | R-level lapply over 6.46M × 5 vars | Sparse mat-vec (mean) + 344K loop or data.table group-by (max/min), × 28 years × 5 vars |
| **Estimated wall time** | 86+ hours | **2–10 minutes** |
| **RAM pressure** | High (list overhead) | Low (~22MB adjacency + working vectors) |
| **Numerical output** | max, min, mean of non-NA neighbor values | **Identical** |
| **Trained RF model** | — | **Untouched** |

The root cause was not just string manipulation — it was **failing to separate the spatial topology (constant across years) from the temporal dimension**, causing an O(N_cells × N_years × avg_neighbors) string-based lookup when an O(N_cells × avg_neighbors) sparse structure queried 28 times suffices.