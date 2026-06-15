 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` — **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` — fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — **allocates a new character vector per row** (~4 neighbor keys × 6.46M rows ≈ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` — **named-vector lookup via string matching on a 6.46M-element named vector**, which is O(N) per probe in the worst case (hash collisions aside, the sheer memory traffic is enormous).

This means the string-key construction and lookup is **O(N × K)** where N = 6.46M rows and K = average neighbor count (~4), yielding ~25.8 billion character comparisons in the worst case. Even with R's internal hashing of named vectors, the constant factors are brutal: each `paste` allocates, each lookup probes a 6.46M-entry hash table.

### The Deeper Structural Issue

The lookup conflates **two independent dimensions** (cell ID and year) into a single string key. But the neighbor structure is **time-invariant** — cell 42's neighbors are the same in 1992 and 2019. The year dimension is only needed to align rows. This means:

- The neighbor topology needs to be resolved **once per cell** (344K cells), not once per cell-year (6.46M rows).
- The year-to-row mapping is a simple structured index, not something requiring string hashing.

### `compute_neighbor_stats` Is Also Suboptimal

It loops over 6.46M entries in `neighbor_lookup`, each time subsetting a numeric vector by integer indices. This is acceptable but can be replaced with a single vectorized matrix operation.

---

## Optimization Strategy

**Principle: Separate the spatial dimension from the temporal dimension.**

1. **Build a cell-index → row-indices mapping** (344K cells × 28 years). Since the panel is balanced (or near-balanced), create a matrix where `row_matrix[cell_pos, year_pos]` gives the row number in `data`. This is O(N) to build, no strings.

2. **Build a neighbor-row-index matrix** by expanding the `nb` object once. For each cell, look up its neighbors' cell positions, then use the row matrix to get all (neighbor, year) row indices. This produces a **pre-expanded integer index list** — one per cell-year row — using only integer arithmetic.

3. **Vectorize `compute_neighbor_stats`** using the pre-built integer index list, or better yet, use a sparse-matrix multiplication approach: construct a sparse neighbor-weight matrix W (6.46M × 6.46M) where entry (i, j) = 1 if row j is a spatial neighbor of row i in the same year. Then neighbor means = `(W %*% x) / (W %*% 1)`, neighbor max/min via row-wise operations.

4. **For max/min**, sparse matrix multiplication doesn't directly help, but we can use `data.table` grouped operations or a chunked approach.

The **most practical approach** for a 16 GB laptop: use `data.table` to avoid the string-key pattern entirely, and compute neighbor stats via vectorized joins.

---

## Working R Code

```r
library(data.table)
library(Matrix)

# =============================================================================
# STEP 1: Build the neighbor lookup using integer arithmetic only
# =============================================================================

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert data to data.table if not already
  dt <- as.data.table(data)
  
  # Create integer mappings: cell_id -> cell_position (1-based in id_order)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Create year -> year_position mapping
  years_sorted <- sort(unique(dt$year))
  year_to_pos <- setNames(seq_along(years_sorted), as.character(years_sorted))
  
  # Add cell_pos and year_pos columns to data
  dt[, cell_pos := id_to_pos[as.character(id)]]
  dt[, year_pos := year_to_pos[as.character(year)]]
  
  # Build a matrix: row_matrix[cell_pos, year_pos] = row index in dt
  # This replaces the entire string-key lookup
  n_cells <- length(id_order)
  n_years <- length(years_sorted)
  
  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_matrix[cbind(dt$cell_pos, dt$year_pos)] <- seq_len(nrow(dt))
  
  list(
    dt = dt,
    row_matrix = row_matrix,
    id_to_pos = id_to_pos,
    year_to_pos = year_to_pos,
    n_cells = n_cells,
    n_years = n_years,
    years_sorted = years_sorted
  )
}

# =============================================================================
# STEP 2: Build sparse neighbor matrix (same-year neighbors only)
#          W is (N x N) where N = nrow(data), W[i,j] = 1 iff row j is a
#          spatial neighbor of row i AND they share the same year.
# =============================================================================

build_sparse_neighbor_matrix <- function(lookup, neighbors) {
  row_matrix <- lookup$row_matrix
  n_cells    <- lookup$n_cells
  n_years    <- lookup$n_years
  dt         <- lookup$dt
  N          <- nrow(dt)
  
  # Pre-calculate total number of non-zero entries for memory pre-allocation
  # For each cell, count neighbors; multiply by number of years it appears
  neighbor_counts <- vapply(neighbors, length, integer(1))  # length 344K
  
  # For each cell_pos, count how many years have non-NA rows
  years_per_cell <- rowSums(!is.na(row_matrix))  # length 344K
  
  # Total directed neighbor-year pairs (upper bound for nnz)
  total_nnz <- sum(as.numeric(neighbor_counts) * as.numeric(years_per_cell))
  cat("Estimated nnz in sparse matrix:", total_nnz, "\n")
  
  # Pre-allocate vectors for sparse matrix triplets
  row_i <- integer(total_nnz)
  col_j <- integer(total_nnz)
  ptr <- 0L
  
  # Iterate over cells (344K iterations, not 6.46M)
  for (c_pos in seq_len(n_cells)) {
    nb_positions <- neighbors[[c_pos]]
    if (length(nb_positions) == 0L) next
    
    # Get neighbor cell positions in id_order
    # nb_positions already indexes into id_order (spdep::nb convention)
    
    for (y_pos in seq_len(n_years)) {
      focal_row <- row_matrix[c_pos, y_pos]
      if (is.na(focal_row)) next
      
      # Get neighbor rows for the same year
      nb_rows <- row_matrix[nb_positions, y_pos]
      nb_rows <- nb_rows[!is.na(nb_rows)]
      if (length(nb_rows) == 0L) next
      
      idx_range <- (ptr + 1L):(ptr + length(nb_rows))
      row_i[idx_range] <- focal_row
      col_j[idx_range] <- nb_rows
      ptr <- ptr + length(nb_rows)
    }
  }
  
  # Trim to actual size
  row_i <- row_i[1:ptr]
  col_j <- col_j[1:ptr]
  
  W <- sparseMatrix(
    i = row_i, j = col_j, x = rep(1, ptr),
    dims = c(N, N)
  )
  
  return(W)
}

# =============================================================================
# STEP 3: Compute neighbor stats vectorized using sparse matrix
# =============================================================================

compute_neighbor_stats_sparse <- function(dt, W, var_name) {
  x <- dt[[var_name]]
  
  # Replace NA with 0 for matrix multiplication, but track validity
  not_na <- as.numeric(!is.na(x))
  x_clean <- ifelse(is.na(x), 0, x)
  
  # Number of non-NA neighbors per row
  n_valid <- as.vector(W %*% not_na)
  
  # Sum of neighbor values (only non-NA contribute)
  neighbor_sum <- as.vector(W %*% x_clean)
  
  # Mean
  neighbor_mean <- ifelse(n_valid > 0, neighbor_sum / n_valid, NA_real_)
  
  # For max and min, we need a different approach since sparse matmul

  # doesn't give us max/min directly. We use a chunked row-wise approach.
  # 
  # Key insight: W is sparse, so we iterate over its row structure.
  # With dgCMatrix (column-sparse), we transpose to get row access via columns.
  
  Wt <- t(W)  # Now columns of Wt correspond to rows of W
  
  neighbor_max <- rep(NA_real_, nrow(dt))
  neighbor_min <- rep(NA_real_, nrow(dt))
  
  # Process in chunks to manage memory
  chunk_size <- 50000L
  N <- nrow(dt)
  n_chunks <- ceiling(N / chunk_size)
  
  cat("Computing max/min for", var_name, "in", n_chunks, "chunks\n")
  
  for (ch in seq_len(n_chunks)) {
    start_row <- (ch - 1L) * chunk_size + 1L
    end_row   <- min(ch * chunk_size, N)
    rows      <- start_row:end_row
    
    # Extract the submatrix: columns of Wt for these rows
    Wt_sub <- Wt[, rows, drop = FALSE]
    
    # For each column (= each focal row), find non-zero entries
    # dgCMatrix: @p gives column pointers, @i gives row indices (0-based)
    p <- Wt_sub@p
    idx_all <- Wt_sub@i + 1L  # 1-based row indices
    
    for (k in seq_along(rows)) {
      col_start <- p[k] + 1L
      col_end   <- p[k + 1L]
      if (col_end < col_start) next  # no neighbors
      
      nb_idx <- idx_all[col_start:col_end]
      nb_vals <- x[nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      
      if (length(nb_vals) == 0L) next
      
      neighbor_max[rows[k]] <- max(nb_vals)
      neighbor_min[rows[k]] <- min(nb_vals)
    }
  }
  
  list(
    max  = neighbor_max,
    min  = neighbor_min,
    mean = neighbor_mean
  )
}

# =============================================================================
# STEP 4: Alternative — fully vectorized max/min using data.table edge list
#          This avoids the inner R loop entirely.
# =============================================================================

compute_neighbor_stats_dt <- function(dt, W, var_name) {
  x <- dt[[var_name]]
  
  # --- Mean via sparse matrix (fast) ---
  not_na  <- as.numeric(!is.na(x))
  x_clean <- ifelse(is.na(x), 0, x)
  n_valid <- as.vector(W %*% not_na)
  nb_sum  <- as.vector(W %*% x_clean)
  nb_mean <- ifelse(n_valid > 0, nb_sum / n_valid, NA_real_)
  
  # --- Max/Min via data.table grouped operations ---
  # Extract edge list from sparse matrix
  W_coo <- summary(W)  # gives i, j, x columns (COO format)
  
  edges <- data.table(
    focal    = W_coo$i,
    neighbor = W_coo$j
  )
  
  # Attach neighbor values
  edges[, nb_val := x[neighbor]]
  
  # Remove edges where neighbor value is NA
  edges <- edges[!is.na(nb_val)]
  
  # Grouped aggregation — fully vectorized in data.table
  agg <- edges[, .(
    nb_max = max(nb_val),
    nb_min = min(nb_val)
  ), by = focal]
  
  # Map back to full row set
  nb_max <- rep(NA_real_, length(x))
  nb_min <- rep(NA_real_, length(x))
  nb_max[agg$focal] <- agg$nb_max
  nb_min[agg$focal] <- agg$nb_min
  
  list(max = nb_max, min = nb_min, mean = nb_mean)
}

# =============================================================================
# STEP 5: Main pipeline — drop-in replacement
# =============================================================================

run_optimized_neighbor_features <- function(cell_data, id_order, 
                                             rook_neighbors_unique) {
  cat("=== Building integer lookup structures ===\n")
  t0 <- proc.time()
  
  lookup <- build_neighbor_lookup_fast(cell_data, id_order, 
                                        rook_neighbors_unique)
  
  cat("  Lookup built in", (proc.time() - t0)[3], "sec\n")
  
  cat("=== Building sparse neighbor matrix ===\n")
  t1 <- proc.time()
  
  W <- build_sparse_neighbor_matrix(lookup, rook_neighbors_unique)
  
  cat("  Sparse matrix built in", (proc.time() - t1)[3], "sec\n")
  cat("  Dimensions:", dim(W)[1], "x", dim(W)[2], 
      " nnz:", nnzero(W), "\n")
  
  # Extract edge list once (reused across all variables)
  cat("=== Extracting edge list ===\n")
  W_coo <- summary(W)
  edges_template <- data.table(
    focal    = W_coo$i,
    neighbor = W_coo$j
  )
  
  dt <- as.data.table(lookup$dt)
  
  # Precompute not_na-count vector (for mean denominator) — 
  # reuse W for each variable
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  cat("=== Computing neighbor features for", 
      length(neighbor_source_vars), "variables ===\n")
  
  for (var_name in neighbor_source_vars) {
    cat("  Processing:", var_name, "\n")
    tv <- proc.time()
    
    x <- dt[[var_name]]
    
    # --- Mean via sparse matrix multiplication ---
    not_na  <- as.numeric(!is.na(x))
    x_clean <- ifelse(is.na(x), 0, x)
    n_valid <- as.vector(W %*% not_na)
    nb_sum  <- as.vector(W %*% x_clean)
    nb_mean <- ifelse(n_valid > 0, nb_sum / n_valid, NA_real_)
    
    # --- Max/Min via data.table grouped aggregation ---
    edges <- copy(edges_template)
    edges[, nb_val := x[neighbor]]
    edges <- edges[!is.na(nb_val)]
    
    agg <- edges[, .(
      nb_max = max(nb_val),
      nb_min = min(nb_val)
    ), by = focal]
    
    nb_max <- rep(NA_real_, nrow(dt))
    nb_min <- rep(NA_real_, nrow(dt))
    nb_max[agg$focal] <- agg$nb_max
    nb_min[agg$focal] <- agg$nb_min
    
    # --- Assign to data.table (same column names as original) ---
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    dt[, (max_col)  := nb_max]
    dt[, (min_col)  := nb_min]
    dt[, (mean_col) := nb_mean]
    
    cat("    Done in", (proc.time() - tv)[3], "sec\n")
  }
  
  # Remove helper columns
  dt[, c("cell_pos", "year_pos") := NULL]
  
  cat("=== All neighbor features complete ===\n")
  return(as.data.frame(dt))
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================

# cell_data <- run_optimized_neighbor_features(
#   cell_data, id_order, rook_neighbors_unique
# )
#
# # The trained Random Forest model is unchanged — just predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Performance Analysis

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| **String key construction** | 6.46M × `paste()` calls for lookup + 6.46M × K × `paste()` inside `lapply` | **Zero** string operations | ∞ |
| **Lookup structure** | 6.46M-entry named character vector (hash table with string keys) | Integer matrix `[344K × 28]` — direct indexing | ~100× |
| **Neighbor resolution** | 6.46M R-level `lapply` iterations, each doing string paste + hash probe | 344K × 28 iterations building sparse matrix (integer arithmetic only) | ~10-50× |
| **Neighbor mean** | R-level loop over 6.46M entries, subsetting + `mean()` each | Single sparse matrix multiplication `W %*% x` | ~500-1000× |
| **Neighbor max/min** | R-level loop over 6.46M entries | `data.table` grouped aggregation on edge list | ~50-100× |
| **Per-variable cost** | Rebuilds nothing but still loops 6.46M times | Sparse matmul + one `data.table` group-by | ~100× |
| **Total 5 variables** | ~86+ hours estimated | **~5-15 minutes expected** | ~350-1000× |

### Why This Works on 16 GB RAM

- **Sparse matrix W**: ~25.8M non-zeros (6.46M rows × ~4 neighbors) × 12 bytes ≈ **310 MB** in CSC format.
- **Edge list**: ~25.8M rows × 3 columns × 8 bytes ≈ **620 MB** (temporary, freed per variable).
- **Row matrix**: 344K × 28 × 4 bytes ≈ **39 MB**.
- **Data itself**: 6.46M × 110 columns × 8 bytes ≈ **5.7 GB**.
- **Total peak**: ~7-8 GB — fits in 16 GB with headroom.

### Numerical Equivalence

The optimized code computes **exactly the same `max`, `min`, and `mean`** over the same set of non-NA neighbor values for each row. The sparse matrix `W` encodes the identical neighbor-in-same-year relationship that the original string-key lookup resolved. The `data.table` aggregation and sparse matrix multiplication produce the same arithmetic results. The trained Random Forest model is never touched.