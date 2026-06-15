 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` — **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` — fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — **allocates a new character vector per row** (~4 neighbor keys × 6.46M rows ≈ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` — **named-vector lookup via string matching on a 6.46M-element named vector**, done 6.46M times.

The `idx_lookup` named vector has 6.46M entries. R's named-vector lookup is **O(N)** internally (hashing helps, but the constant factor on 6.46M names is brutal). Doing this 6.46M times gives effectively **O(N²)** behavior — hence the 86+ hour estimate.

### The Broader Pattern

The neighbor lookup is **year-invariant in structure**: cell A's rook neighbors are the same cells every year. The only thing that changes across years is which rows in the data correspond to those neighbor cells. This means:

- The `neighbors` structure (which cell IDs are neighbors) is **static across years**.
- The mapping from `(cell_id, year)` → row index is a **simple arithmetic mapping** if the data is sorted, or a one-time hash table build.

Yet the current code reconstructs string keys and performs string-based lookups **per row**, completely ignoring this separable structure.

### `compute_neighbor_stats` Is Efficient — But Fed by an Expensive Lookup

`compute_neighbor_stats` itself is fine (simple numeric indexing). The entire bottleneck is `build_neighbor_lookup`.

---

## Optimization Strategy

### Key Insight: Separate the Spatial Structure from the Temporal Indexing

Since rook neighbors are time-invariant, we can:

1. **Build a row-index matrix** of shape `(n_cells, n_years)` mapping `(cell_position, year_position)` → row number in `data`. This is a one-time O(N) operation.
2. **For each cell**, its neighbor row-indices in year `t` are simply `row_matrix[neighbor_positions, year_position]`. No strings, no hashing — pure integer indexing.
3. **Vectorize the neighbor stats** using matrix operations or `data.table` grouping instead of per-row `lapply`.

### Further: Vectorize Stats Computation with Sparse Matrix Multiplication

The "mean/max/min of neighbor values" can be computed as:
- **Mean**: sparse adjacency matrix × value vector (matrix-vector multiply), divided by neighbor counts.
- **Max/Min**: row-wise sparse operations.

This replaces the entire `lapply` over 6.46M rows with a single sparse matrix operation per variable — **O(nnz)** where nnz ≈ 1.37M edges × 28 years ≈ 38.5M, done in optimized C code.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# =============================================================================
# Prerequisites: data.table, Matrix, spdep (already available in your pipeline)
# 
# Inputs expected:
#   cell_data            — data.frame/data.table with columns: id, year, and
#                          the 5 neighbor source variables
#   id_order             — integer vector of cell IDs in the order matching
#                          rook_neighbors_unique
#   rook_neighbors_unique — nb object (list of integer index vectors)
#
# Preserves: all original column values, trained RF model (untouched),
#            and the original numerical estimand (max, min, mean of
#            non-NA neighbor values per cell-year).
# =============================================================================

library(data.table)
library(Matrix)

optimized_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # ------------------------------------------------------------------
  # 0. Convert to data.table for speed; keep original row order
  # ------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_order := .I]
  
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)
  
  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(dt)))
  
  # ------------------------------------------------------------------
  # 1. Build integer mappings (one-time, no strings)
  # ------------------------------------------------------------------
  # Map cell id -> position in id_order (1..n_cells)
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_len(n_cells)
  
  # Map year -> position (1..n_years)
  year_to_pos <- setNames(seq_len(n_years), as.character(years))
  
  # ------------------------------------------------------------------
  # 2. Build row-index matrix: row_idx_mat[cell_pos, year_pos] = row in dt
  #    This replaces ALL string-key lookups. O(N) one-time cost.
  # ------------------------------------------------------------------
  cell_pos_vec <- id_to_pos[dt$id]
  year_pos_vec <- year_to_pos[as.character(dt$year)]
  
  row_idx_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_idx_mat[cbind(cell_pos_vec, year_pos_vec)] <- seq_len(nrow(dt))
  
  cat("Row-index matrix built.\n")
  
  # ------------------------------------------------------------------
  # 3. Build sparse adjacency matrix from nb object (one-time)
  #    A[i,j] = 1 if cell j is a rook neighbor of cell i
  #    Dimensions: n_cells x n_cells
  # ------------------------------------------------------------------
  # Extract COO (coordinate) representation from nb object
  from_list <- lapply(seq_len(n_cells), function(i) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 1L && nb[1] == 0L) {
      # spdep convention: 0 means no neighbors
      return(data.table(i = integer(0), j = integer(0)))
    }
    data.table(i = rep(i, length(nb)), j = nb)
  })
  edges <- rbindlist(from_list)
  
  # Sparse adjacency matrix (n_cells x n_cells)
  adj <- sparseMatrix(
    i = edges$i,
    j = edges$j,
    x = 1,
    dims = c(n_cells, n_cells)
  )
  
  # Neighbor counts per cell (used for mean calculation)
  # This is the structural count; NA handling adjusts it per variable/year.
  cat(sprintf("Adjacency matrix: %d cells, %d directed edges.\n",
              n_cells, length(edges$i)))
  rm(edges, from_list)
  
  # ------------------------------------------------------------------
  # 4. For each variable, compute neighbor max, min, mean per cell-year
  #    using sparse matrix operations — one year-slice at a time to

  #    handle NAs correctly and compute max/min (which aren't linear).
  # ------------------------------------------------------------------
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  # Pre-allocate result columns
  for (var_name in neighbor_source_vars) {
    dt[, paste0(var_name, "_neighbor_max")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_min")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_mean") := NA_real_]
  }
  
  # We process year-by-year. For each year, we have a vector of length
  # n_cells (some NA where cells don't appear that year).
  # For MEAN: sparse mat-vec multiply handles the sum; we just need
  #           the count of non-NA neighbors.
  # For MAX/MIN: we use a trick with the sparse matrix.
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing variable: %s\n", var_name))
    
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")
    
    for (yi in seq_len(n_years)) {
      # Row indices in dt for this year
      row_indices <- row_idx_mat[, yi]  # length n_cells, some NA
      
      # Build value vector for this year (length n_cells)
      vals <- rep(NA_real_, n_cells)
      present <- !is.na(row_indices)
      vals[present] <- dt[[var_name]][row_indices[present]]
      
      # --- MEAN via sparse matrix-vector multiply ---
      # Replace NA with 0 for summation, track non-NA mask
      not_na <- !is.na(vals)
      vals_zero <- ifelse(not_na, vals, 0)
      
      # Sum of neighbor values (adj %*% vals_zero)[i] = sum of vals over neighbors of i
      neighbor_sum   <- as.numeric(adj %*% vals_zero)
      # Count of non-NA neighbors
      neighbor_count <- as.numeric(adj %*% as.numeric(not_na))
      
      neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
      
      # --- MAX and MIN via sparse iteration ---
      # For max: replace NA with -Inf, multiply, then fix up.
      # But matrix multiply gives SUM, not MAX. We need a different approach.
      #
      # Efficient approach: use the adjacency list structure we already have
      # (rook_neighbors_unique) but vectorized per year-slice.
      # Since we have vals (length n_cells), we iterate over cells using
      # the nb list — but this is only 344K iterations (not 6.46M).
      
      neighbor_max <- rep(NA_real_, n_cells)
      neighbor_min <- rep(NA_real_, n_cells)
      
      # Vectorized approach: expand neighbor pairs, compute, then aggregate
      # We already have adj in sparse format. Extract its structure once
      # (outside the year loop for efficiency — we'll restructure below).
      # For now, use the nb list directly — 344K iterations is fast.
      
      for (ci in seq_len(n_cells)) {
        nb_idx <- rook_neighbors_unique[[ci]]
        if (length(nb_idx) == 1L && nb_idx[1] == 0L) next
        nb_vals <- vals[nb_idx]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) == 0L) next
        neighbor_max[ci] <- max(nb_vals)
        neighbor_min[ci] <- min(nb_vals)
      }
      
      # Write results back to dt for rows present this year
      target_rows <- row_indices[present]
      cell_positions <- which(present)
      
      set(dt, i = target_rows, j = col_max,  value = neighbor_max[cell_positions])
      set(dt, i = target_rows, j = col_min,  value = neighbor_min[cell_positions])
      set(dt, i = target_rows, j = col_mean, value = neighbor_mean[cell_positions])
    }
  }
  
  # ------------------------------------------------------------------
  # 5. Restore original order and return
  # ------------------------------------------------------------------
  setorder(dt, .row_order)
  dt[, .row_order := NULL]
  
  return(as.data.frame(dt))
}
```

**Wait** — the inner `for (ci in seq_len(n_cells))` loop over 344K cells × 28 years × 5 variables is still ~48M R-level loop iterations for max/min. That's much better than 6.46M × 5 but still suboptimal. Let's fully vectorize max/min using the COO expansion approach:

```r
# =============================================================================
# FULLY VECTORIZED VERSION
# =============================================================================

library(data.table)
library(Matrix)

optimized_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {
  
  dt <- as.data.table(cell_data)
  dt[, .row_order := .I]
  
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)
  
  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(dt)))
  
  # ---- Integer mappings ----
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_len(n_cells)
  
  year_to_pos <- setNames(seq_len(n_years), as.character(years))
  
  # ---- Row-index matrix ----
  cell_pos_vec <- id_to_pos[dt$id]
  year_pos_vec <- year_to_pos[as.character(dt$year)]
  
  row_idx_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_idx_mat[cbind(cell_pos_vec, year_pos_vec)] <- seq_len(nrow(dt))
  
  # ---- Build edge table from nb object (one-time) ----
  from_vec <- integer(0)
  to_vec   <- integer(0)
  for (ci in seq_len(n_cells)) {
    nb <- rook_neighbors_unique[[ci]]
    if (length(nb) == 1L && nb[1] == 0L) next
    from_vec <- c(from_vec, rep(ci, length(nb)))
    to_vec   <- c(to_vec, nb)
  }
  # More memory-efficient construction:
  edge_dt <- data.table(from = from_vec, to = to_vec)
  rm(from_vec, to_vec)
  n_edges <- nrow(edge_dt)
  cat(sprintf("Edge table: %d directed edges.\n", n_edges))
  
  # ---- Sparse adjacency for mean (sum + count) ----
  adj <- sparseMatrix(
    i = edge_dt$from,
    j = edge_dt$to,
    x = 1,
    dims = c(n_cells, n_cells)
  )
  
  # ---- Pre-allocate output columns ----
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  for (var_name in neighbor_source_vars) {
    dt[, paste0(var_name, "_neighbor_max")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_min")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_mean") := NA_real_]
  }
  
  # ---- Process each variable × year ----
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Variable: %s ...\n", var_name))
    
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")
    
    for (yi in seq_len(n_years)) {
      
      # Value vector for this year (length n_cells, NA where absent)
      row_indices <- row_idx_mat[, yi]
      present <- !is.na(row_indices)
      
      vals <- rep(NA_real_, n_cells)
      vals[present] <- dt[[var_name]][row_indices[present]]
      
      # ---- MEAN via sparse mat-vec ----
      not_na     <- !is.na(vals)
      vals_zero  <- ifelse(not_na, vals, 0)
      nb_sum     <- as.numeric(adj %*% vals_zero)
      nb_count   <- as.numeric(adj %*% as.numeric(not_na))
      nb_mean    <- ifelse(nb_count > 0, nb_sum / nb_count, NA_real_)
      
      # ---- MAX / MIN via edge-table vectorization ----
      # Get neighbor values for all edges at once
      nb_vals <- vals[edge_dt$to]  # length n_edges
      
      # Build a data.table of (from_cell, neighbor_value), drop NAs, aggregate
      agg_dt <- data.table(from = edge_dt$from, val = nb_vals)
      agg_dt <- agg_dt[!is.na(val)]
      
      if (nrow(agg_dt) > 0) {
        agg <- agg_dt[, .(vmax = max(val), vmin = min(val)), by = from]
        
        nb_max <- rep(NA_real_, n_cells)
        nb_min <- rep(NA_real_, n_cells)
        nb_max[agg$from] <- agg$vmax
        nb_min[agg$from] <- agg$vmin
      } else {
        nb_max <- rep(NA_real_, n_cells)
        nb_min <- rep(NA_real_, n_cells)
      }
      
      # ---- Write back to dt ----
      target_rows    <- row_indices[present]
      cell_positions <- which(present)
      
      set(dt, i = target_rows, j = col_max,  value = nb_max[cell_positions])
      set(dt, i = target_rows, j = col_min,  value = nb_min[cell_positions])
      set(dt, i = target_rows, j = col_mean, value = nb_mean[cell_positions])
    }
  }
  
  setorder(dt, .row_order)
  dt[, .row_order := NULL]
  return(as.data.frame(dt))
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================
# cell_data <- optimized_neighbor_features(cell_data, id_order, rook_neighbors_unique)
#
# The trained Random Forest model is untouched — only the feature-construction
# pipeline is replaced. Output columns have identical names and identical
# numerical values (max, min, mean of non-NA rook-neighbor values per cell-year).
```

---

## Complexity Comparison

| Aspect | Original | Optimized |
|---|---|---|
| **String key constructions** | ~6.46M × ~4 neighbors = **~25.8M** `paste()` calls inside `lapply`, plus 6.46M keys for `idx_lookup` | **Zero** — pure integer indexing |
| **Named-vector lookups** | 6.46M lookups into a 6.46M-element named vector (**O(N) hash probe × N**) | **Zero** — replaced by matrix indexing `row_idx_mat[cell, year]` |
| **R-level loop iterations** | **6.46M** (one per cell-year row) | **0** for mean; max/min use `data.table` grouped aggregation over ~1.37M edges × 28 years = **~38.4M rows** but processed in **vectorized C** via `data.table` |
| **Sparse mat-vec for mean** | N/A | **28 years × 5 vars = 140** sparse matrix-vector multiplies, each O(nnz) ≈ O(1.37M) in compiled code |
| **Estimated wall time** | **86+ hours** | **~5–15 minutes** (sparse mat-vec: ~140 × <0.5s; data.table agg: ~140 × ~1s; overhead ~1 min) |
| **Peak RAM** | ~6.46M-element named character vector + lapply overhead | `row_idx_mat`: 344K × 28 × 4B ≈ 38 MB; sparse `adj`: ~5.5 MB; `edge_dt`: ~22 MB; `agg_dt` per slice: ~22 MB. **Total overhead: ~100–200 MB** — well within 16 GB |

## Why This Preserves the Original Estimand

For each cell-year `(c, t)`:
- **Original**: finds all rook neighbors of `c`, looks up their row in year `t`, extracts non-NA values of `var`, computes `max`, `min`, `mean`.
- **Optimized**: uses the same `rook_neighbors_unique` structure, the same year, the same variable column, drops NAs identically, and computes the same `max`, `min`, `mean`. The sparse matrix mean uses `sum / count_non_na`, which is algebraically identical. The `data.table` grouped `max`/`min` are identical to R's `max`/`min` on the same non-NA subset.

The trained Random Forest model is never touched — only the upstream feature matrix construction is accelerated.