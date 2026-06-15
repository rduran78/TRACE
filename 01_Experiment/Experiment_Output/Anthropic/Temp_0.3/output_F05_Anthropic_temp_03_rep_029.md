 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` — **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` — fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — **allocates a new character vector per row** (~4 neighbor keys × 6.46M rows ≈ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` — **named-vector lookup on a 6.46M-element character vector, repeated 6.46M times**.

The named-vector lookup `idx_lookup[neighbor_keys]` is O(n) per probe in the worst case (R's internal hashing helps, but the hash table has 6.46M entries and is probed 25.8M times total). The real killer is that **the entire pattern is unnecessary**: the neighbor relationships are spatial (time-invariant), and the year dimension is trivially handled by structure, not by string matching.

### Why It's a Broader Algorithmic Issue

The string-keyed lookup conflates two independent dimensions — **spatial neighbors** (fixed across years) and **time** (panel year) — into a single flat lookup. This means:

- The spatial neighbor graph is "rediscovered" via string matching for every cell-year row.
- The same neighbor set for cell `c` is recomputed 28 times (once per year), identically each time.
- `compute_neighbor_stats` is then called 5 times (once per variable), each time traversing the full 6.46M-element lookup list.

**Total redundant work**: 6.46M × (string alloc + hash probe) in `build_neighbor_lookup`, then 5 × 6.46M list traversals in the stats loop.

### The Key Insight

Since neighbors are purely spatial and the panel is balanced (every cell appears in every year), we can:

1. **Work in matrix form**: reshape each variable into a `cells × years` matrix.
2. **Vectorize the neighbor aggregation** using the spatial neighbor list (344K entries, not 6.46M) and matrix column operations.
3. **Eliminate all string operations entirely.**

This reduces the problem from ~6.46M string-keyed row lookups to ~344K integer-indexed spatial lookups, each operating on vectors of length 28 — a **~18× reduction in iterations** with **far cheaper per-iteration cost**.

---

## Optimization Strategy

| Aspect | Current | Proposed |
|---|---|---|
| Lookup structure | 6.46M-entry named character vector | Integer spatial neighbor list (344K entries) + matrix column indexing |
| Neighbor resolution | Per cell-year, via string paste + hash lookup | Per cell, via integer index into matrix rows |
| String operations | ~25.8M `paste()` calls | **Zero** |
| Stats computation | 5 × `lapply` over 6.46M-element list | 5 × `lapply` over 344K-element list, each doing matrix row subsetting |
| Estimated time | 86+ hours | ~5–15 minutes |
| RAM | Lookup list of 6.46M integer vectors | Matrix of 344K × 28 per variable (~77 MB per variable) |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# 
# Assumptions (preserved from original):
#   - cell_data: data.frame with columns 'id', 'year', and all predictor vars
#   - id_order: vector of unique cell IDs in the spatial grid order
#   - rook_neighbors_unique: spdep::nb object (length = number of cells = 344,208)
#   - cell_data is a balanced panel: every cell appears in every year (1992–2019)
#   - The trained Random Forest model is untouched; we only reconstruct the same
#     numerical features it expects.
# =============================================================================

build_neighbor_features_optimized <- function(cell_data, id_order, neighbors,
                                               source_vars) {
  # ------------------------------------------------------------------
  # 1. Establish cell ordering and year ordering
  # ------------------------------------------------------------------
  unique_years <- sort(unique(cell_data$year))
  n_years      <- length(unique_years)
  n_cells      <- length(id_order)
  
  # Map cell id -> spatial index (integer position in id_order / neighbors list)
  # This is the ONLY mapping we need.
  id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # ------------------------------------------------------------------
  # 2. Sort cell_data by (id, year) so we can reliably reshape to matrix
  #    We'll record the original row order to restore it at the end.
  # ------------------------------------------------------------------
  cell_data$.orig_row_order <- seq_len(nrow(cell_data))
  
  # Compute spatial index for each row (vectorized, one-time cost)
  cell_data$.spatial_idx <- id_to_spatial_idx[as.character(cell_data$id)]
  
  # Sort by spatial_idx then year for matrix reshaping
  sort_order <- order(cell_data$.spatial_idx, cell_data$year)
  cell_data  <- cell_data[sort_order, , drop = FALSE]
  
  # After sorting, rows are arranged as:
  #   cell_1/year_1, cell_1/year_2, ..., cell_1/year_28,
  #   cell_2/year_1, ..., cell_2/year_28, ...
  # So we can reshape any column into a (n_cells x n_years) matrix directly.
  
  # ------------------------------------------------------------------
  # 3. For each source variable, compute neighbor max/min/mean via matrices
  # ------------------------------------------------------------------
  # Pre-allocate result columns in the sorted cell_data
  for (var_name in source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    cell_data[[col_max]]  <- NA_real_
    cell_data[[col_min]]  <- NA_real_
    cell_data[[col_mean]] <- NA_real_
  }
  
  for (var_name in source_vars) {
    message("Processing neighbor stats for: ", var_name)
    
    # Reshape variable into (n_cells x n_years) matrix
    # Rows = cells (in id_order sequence), Columns = years (sorted)
    var_matrix <- matrix(cell_data[[var_name]], nrow = n_cells, ncol = n_years,
                         byrow = TRUE)
    
    # Allocate output matrices (n_cells x n_years)
    max_matrix  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_matrix  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mean_matrix <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # Iterate over cells (344K iterations, not 6.46M)
    for (ci in seq_len(n_cells)) {
      nb_indices <- neighbors[[ci]]
      
      if (length(nb_indices) == 0L) next
      # nb_indices are already integer indices into id_order (spdep::nb convention)
      
      # Extract neighbor values: a (num_neighbors x n_years) sub-matrix
      # Each column is one year; each row is one neighbor cell
      nb_vals <- var_matrix[nb_indices, , drop = FALSE]
      
      # Compute column-wise (i.e., per-year) stats
      # Using colMeans / apply for max/min — but we can be smarter:
      # For small neighbor counts (typically 2-4 for rook), direct is fine.
      n_nb <- nrow(nb_vals)
      
      if (n_nb == 1L) {
        # Single neighbor: max = min = mean = that value
        max_matrix[ci, ]  <- nb_vals[1L, ]
        min_matrix[ci, ]  <- nb_vals[1L, ]
        mean_matrix[ci, ] <- nb_vals[1L, ]
      } else {
        # For rook neighbors, n_nb is typically 2-4, so apply is fine
        # Handle NAs consistently with original code
        for (yr in seq_len(n_years)) {
          nv <- nb_vals[, yr]
          nv <- nv[!is.na(nv)]
          if (length(nv) == 0L) next
          max_matrix[ci, yr]  <- max(nv)
          min_matrix[ci, yr]  <- min(nv)
          mean_matrix[ci, yr] <- mean(nv)
        }
      }
    }
    
    # Flatten matrices back to vectors (byrow = TRUE matches our reshape)
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    
    cell_data[[col_max]]  <- as.vector(t(max_matrix))
    cell_data[[col_min]]  <- as.vector(t(min_matrix))
    cell_data[[col_mean]] <- as.vector(t(mean_matrix))
  }
  
  # ------------------------------------------------------------------
  # 4. Restore original row order and clean up helper columns
  # ------------------------------------------------------------------
  cell_data <- cell_data[order(cell_data$.orig_row_order), , drop = FALSE]
  cell_data$.orig_row_order <- NULL
  cell_data$.spatial_idx    <- NULL
  rownames(cell_data) <- NULL
  
  cell_data
}
```

### Even Faster: Vectorized Inner Loop with `matrixStats` (Optional)

The inner `for (yr in seq_len(n_years))` loop can be eliminated if you install `matrixStats`:

```r
# Drop-in replacement for the inner loop block (requires matrixStats):
# install.packages("matrixStats")

build_neighbor_features_vectorized <- function(cell_data, id_order, neighbors,
                                                source_vars) {
  require(matrixStats)
  
  unique_years <- sort(unique(cell_data$year))
  n_years      <- length(unique_years)
  n_cells      <- length(id_order)
  
  id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  cell_data$.orig_row_order <- seq_len(nrow(cell_data))
  cell_data$.spatial_idx    <- id_to_spatial_idx[as.character(cell_data$id)]
  
  sort_order <- order(cell_data$.spatial_idx, cell_data$year)
  cell_data  <- cell_data[sort_order, , drop = FALSE]
  
  for (var_name in source_vars) {
    cell_data[[paste0("neighbor_max_", var_name)]]  <- NA_real_
    cell_data[[paste0("neighbor_min_", var_name)]]   <- NA_real_
    cell_data[[paste0("neighbor_mean_", var_name)]] <- NA_real_
  }
  
  for (var_name in source_vars) {
    message("Processing neighbor stats for: ", var_name)
    
    var_matrix  <- matrix(cell_data[[var_name]], nrow = n_cells, ncol = n_years,
                          byrow = TRUE)
    max_matrix  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_matrix  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mean_matrix <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    for (ci in seq_len(n_cells)) {
      nb_idx <- neighbors[[ci]]
      if (length(nb_idx) == 0L) next
      
      nb_vals <- var_matrix[nb_idx, , drop = FALSE]
      
      if (nrow(nb_vals) == 1L) {
        max_matrix[ci, ]  <- nb_vals[1L, ]
        min_matrix[ci, ]  <- nb_vals[1L, ]
        mean_matrix[ci, ] <- nb_vals[1L, ]
      } else {
        # colMaxs/colMins/colMeans2 from matrixStats; na.rm handles NAs
        max_matrix[ci, ]  <- colMaxs(nb_vals, na.rm = TRUE)
        min_matrix[ci, ]  <- colMins(nb_vals, na.rm = TRUE)
        mean_matrix[ci, ] <- colMeans2(nb_vals, na.rm = TRUE)
      }
    }
    
    # Fix -Inf/Inf from colMaxs/colMins when all values are NA
    max_matrix[is.infinite(max_matrix)]  <- NA_real_
    min_matrix[is.infinite(min_matrix)]  <- NA_real_
    
    cell_data[[paste0("neighbor_max_", var_name)]]  <- as.vector(t(max_matrix))
    cell_data[[paste0("neighbor_min_", var_name)]]   <- as.vector(t(min_matrix))
    cell_data[[paste0("neighbor_mean_", var_name)]] <- as.vector(t(mean_matrix))
  }
  
  cell_data <- cell_data[order(cell_data$.orig_row_order), , drop = FALSE]
  cell_data$.orig_row_order <- NULL
  cell_data$.spatial_idx    <- NULL
  rownames(cell_data) <- NULL
  
  cell_data
}
```

### Usage (Drop-in Replacement for the Original Outer Loop)

```r
# Original code:
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# Optimized replacement (single call):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_optimized(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)

# The trained Random Forest model is unchanged — it receives the same
# feature columns with identical numerical values.
```

---

## Summary of Gains

| Metric | Original | Optimized |
|---|---|---|
| String allocations | ~25.8M | **0** |
| Hash table probes | ~25.8M into 6.46M-entry table | **0** |
| Loop iterations (neighbor lookup) | 6.46M | **344K** |
| Stats loop iterations | 5 × 6.46M | 5 × 344K (matrix ops on 28-element columns) |
| Estimated wall time | 86+ hours | **~5–15 minutes** |
| RAM overhead | 6.46M-element list of integer vectors | ~77 MB per variable matrix (385 MB peak for 5 vars) |
| Numerical output | Identical | **Identical** (same max/min/mean, same NA handling) |

The fundamental fix is **separating the spatial dimension from the temporal dimension**: neighbors are a spatial concept, years are an independent panel dimension, and conflating them via string keys was the root cause of the combinatorial blowup.