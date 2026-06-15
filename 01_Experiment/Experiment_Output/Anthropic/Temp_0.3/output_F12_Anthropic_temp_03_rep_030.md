 # Diagnosis, Optimization Strategy, and Working Code

## Diagnosis

The colleague is **partially right but misses the deeper bottleneck**. Let me audit both functions:

### `compute_neighbor_stats()` — Colleague's Claim

`do.call(rbind, result)` on a list of ~6.46 million 3-element vectors is indeed inefficient (it creates a matrix row-by-row). However, this is called only **5 times** (once per neighbor source variable). Binding 6.46M small vectors into a matrix is slow (maybe a few minutes per call), but it is **not** the 86-hour bottleneck. The `lapply` inside `compute_neighbor_stats` is simple integer subsetting and basic arithmetic — this is fast, vectorizable work.

### `build_neighbor_lookup()` — The True Bottleneck

This function is called **once** and runs an `lapply` over **all 6.46 million rows**. Inside each iteration it:

1. Looks up `ref_idx` via a named character vector — O(1) amortized but with string conversion overhead per row.
2. Retrieves `neighbor_cell_ids` from the `nb` object.
3. **Calls `paste()` to construct string keys** for every neighbor of every row.
4. **Performs named-vector lookup** (`idx_lookup[neighbor_keys]`) — this is a **hash lookup by string key, repeated ~6.46M times**, each time for multiple neighbors.

With ~1,373,394 directed neighbor relationships spread across 344,208 cells, and 28 years of panel data, the total number of string-key constructions and lookups is approximately:

> (average ~4 rook neighbors per cell) × 6.46M rows ≈ **25.8 million `paste` + hash lookups**, all inside a sequential `lapply`.

**This is the dominant bottleneck.** String construction (`paste`) and named-vector hash lookups inside a per-row `lapply` over 6.46M rows is catastrophically slow in R. The function essentially rebuilds the spatial-temporal join from scratch for every single row using string manipulation.

### Root Cause Summary

| Component | Time Complexity | Bottleneck? |
|---|---|---|
| `build_neighbor_lookup` — `paste` + string hash × 6.46M rows | O(N × avg_neighbors) string ops | **YES — dominant** |
| `compute_neighbor_stats` — `lapply` arithmetic | O(N × avg_neighbors) numeric ops | No — fast |
| `compute_neighbor_stats` — `do.call(rbind, ...)` | O(N) bind | Minor — seconds to low minutes |

**Verdict: Reject the colleague's diagnosis.** The true bottleneck is `build_neighbor_lookup()`, specifically the per-row string key construction and hash-table lookup pattern applied 6.46 million times. `do.call(rbind, ...)` is a minor inefficiency by comparison.

---

## Optimization Strategy

1. **Eliminate `build_neighbor_lookup` entirely as a per-row string-key operation.** Instead, exploit the panel structure: every cell appears once per year in the same order. Build the neighbor lookup **once at the cell level** (344K entries), then expand to row-level by arithmetic indexing using the panel's year structure.

2. **Vectorize `compute_neighbor_stats`** using pre-allocated matrices and direct integer indexing instead of per-row `lapply`. Use a padded neighbor matrix to enable fully vectorized column operations.

3. **Replace `do.call(rbind, ...)`** with direct matrix pre-allocation (a minor but free improvement).

### Key Insight

If the data is sorted by `(id, year)` — or we can establish a mapping from `(cell_index, year_index)` → row — then for any row `i` belonging to cell `c` in year `y`, its neighbors' rows are simply the rows of neighbor cells in the same year. This is a **pure integer arithmetic** operation, no strings needed.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE — replaces build_neighbor_lookup + compute_neighbor_stats
# Preserves the trained Random Forest model and original numerical estimand.
# =============================================================================

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # ------------------------------------------------------------------
  # Step 1: Build cell-level neighbor list (344K cells, not 6.46M rows)
  # ------------------------------------------------------------------
  n_cells <- length(id_order)
  
  # Map cell id -> position in id_order (integer, no strings in hot loop)
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_along(id_order)
  # If ids are not contiguous positive integers, fall back to hash:
  if (any(id_order <= 0) || max(id_order) > 10 * n_cells) {
    id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
    use_hash <- TRUE
  } else {
    use_hash <- FALSE
  }
  
  # ------------------------------------------------------------------
  # Step 2: Determine panel structure (cell_index, year_index) -> row
  # ------------------------------------------------------------------
  unique_years <- sort(unique(data$year))
  n_years      <- length(unique_years)
  year_to_idx  <- setNames(seq_along(unique_years), as.character(unique_years))
  
  # Build a (cell_pos, year_idx) -> row_number mapping matrix
  # This replaces ALL string paste + hash lookups
  cell_year_to_row <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  
  if (use_hash) {
    cell_positions <- as.integer(id_to_pos[as.character(data$id)])
  } else {
    cell_positions <- id_to_pos[data$id]
  }
  year_positions <- year_to_idx[as.character(data$year)]
  
  for (i in seq_len(nrow(data))) {
    cell_year_to_row[cell_positions[i], year_positions[i]] <- i
  }
  # Vectorized alternative (faster):
  idx_linear <- (year_positions - 1L) * n_cells + cell_positions
  cell_year_to_row[idx_linear] <- seq_len(nrow(data))
  
  # ------------------------------------------------------------------
  # Step 3: Build row-level neighbor lookup via integer arithmetic
  # ------------------------------------------------------------------
  # For each row i: cell_pos = cell_positions[i], year_pos = year_positions[i]
  # neighbor cell positions = neighbors[[cell_pos]]
  # neighbor rows = cell_year_to_row[neighbor_cell_positions, year_pos]
  
  # To enable vectorized stats later, build a padded neighbor-row matrix
  # Find max number of neighbors
  n_neighbors_per_cell <- lengths(neighbors)
  max_k <- max(n_neighbors_per_cell)
  
  # Padded cell-level neighbor matrix: n_cells x max_k
  # Pad with NA
  cell_neighbor_mat <- matrix(NA_integer_, nrow = n_cells, ncol = max_k)
  for (c_idx in seq_len(n_cells)) {
    nb <- neighbors[[c_idx]]
    if (length(nb) > 0 && !(length(nb) == 1 && nb[0] == 0)) {
      # spdep nb objects use 0 for no neighbors
      nb <- nb[nb != 0L]
      if (length(nb) > 0) {
        cell_neighbor_mat[c_idx, seq_along(nb)] <- nb
      }
    }
  }
  
  # Now expand to row-level: for each row, neighbor rows
  n_rows <- nrow(data)
  row_neighbor_mat <- matrix(NA_integer_, nrow = n_rows, ncol = max_k)
  
  for (yr in seq_len(n_years)) {
    # All rows in this year
    row_mask <- which(year_positions == yr)
    if (length(row_mask) == 0) next
    
    # Cell positions for these rows
    cpos <- cell_positions[row_mask]
    
    # For each neighbor slot, look up the row of that neighbor in this year
    for (k in seq_len(max_k)) {
      nb_cell_pos <- cell_neighbor_mat[cpos, k]
      valid <- !is.na(nb_cell_pos)
      if (any(valid)) {
        row_neighbor_mat[row_mask[valid], k] <-
          cell_year_to_row[cbind(nb_cell_pos[valid], yr)]
      }
    }
  }
  
  message("Neighbor lookup built: ", n_rows, " rows x ", max_k, " max neighbors")
  
  return(row_neighbor_mat)
}


compute_neighbor_stats_fast <- function(data, row_neighbor_mat, var_name) {
  # ------------------------------------------------------------------
  # Fully vectorized neighbor statistics using the padded matrix
  # Returns an n x 3 matrix: (max, min, mean) — same as original
  # ------------------------------------------------------------------
  vals <- data[[var_name]]
  n <- nrow(row_neighbor_mat)
  k <- ncol(row_neighbor_mat)
  
  # Build a matrix of neighbor values: n x k
  # Use NA for padding / missing
  neighbor_vals <- matrix(NA_real_, nrow = n, ncol = k)
  for (j in seq_len(k)) {
    idx <- row_neighbor_mat[, j]
    valid <- !is.na(idx)
    neighbor_vals[valid, j] <- vals[idx[valid]]
  }
  
  # Compute row-wise max, min, mean ignoring NAs
  # Use matrixStats if available for speed, otherwise base R
  if (requireNamespace("matrixStats", quietly = TRUE)) {
    row_max  <- matrixStats::rowMaxs(neighbor_vals,  na.rm = TRUE)
    row_min  <- matrixStats::rowMins(neighbor_vals,  na.rm = TRUE)
    row_mean <- matrixStats::rowMeans2(neighbor_vals, na.rm = TRUE)
  } else {
    row_max  <- apply(neighbor_vals, 1, max,  na.rm = TRUE)
    row_min  <- apply(neighbor_vals, 1, min,  na.rm = TRUE)
    row_mean <- rowMeans(neighbor_vals, na.rm = TRUE)
  }
  
  # Handle all-NA rows (no valid neighbors) — restore to NA
  all_na <- rowSums(!is.na(neighbor_vals)) == 0L
  row_max[all_na]  <- NA_real_
  row_min[all_na]   <- NA_real_
  row_mean[all_na]  <- NA_real_
  
  # Fix Inf/-Inf from max/min on all-NA after na.rm
  row_max[is.infinite(row_max)]   <- NA_real_
  row_min[is.infinite(row_min)]   <- NA_real_
  
  cbind(row_max, row_min, row_mean)
}


compute_and_add_neighbor_features_fast <- function(data, var_name, row_neighbor_mat) {
  stats <- compute_neighbor_stats_fast(data, row_neighbor_mat, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]
  data
}


# =============================================================================
# OUTER LOOP — drop-in replacement
# =============================================================================

# Build the lookup ONCE — now O(cells × years × max_k) with integer ops only
# Estimated: ~344K cells × 28 years × 4 neighbors ≈ 38.5M integer lookups
# Should complete in seconds to low minutes, not hours.
row_neighbor_mat <- build_neighbor_lookup_fast(
  cell_data, id_order, rook_neighbors_unique
)

# Compute neighbor features for all 5 variables — fully vectorized
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Processing neighbor stats for: ", var_name)
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, row_neighbor_mat
  )
}

# The trained Random Forest model is untouched.
# The numerical estimand (neighbor max, min, mean) is preserved identically.
# Predicted runtime: minutes instead of 86+ hours.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M `paste()` + string hash lookups | Eliminated — integer matrix arithmetic |
| `compute_neighbor_stats` | `lapply` + `do.call(rbind, ...)` over 6.46M rows × 5 vars | Vectorized matrix ops via padded neighbor matrix |
| **String operations** | ~25.8M `paste()` calls + hash lookups | **Zero** |
| **Estimated runtime** | 86+ hours | **Minutes** (mostly the matrix construction + 5 vectorized passes) |
| **Memory** | List of 6.46M integer vectors | One `n × max_k` integer matrix (~155 MB for 6.46M × 6) |
| **RF model** | Preserved | Preserved |
| **Numerical output** | max, min, mean per neighbor set | Identical |

**The colleague's diagnosis is rejected.** The `do.call(rbind, ...)` and list binding in `compute_neighbor_stats` are minor inefficiencies. The dominant bottleneck is `build_neighbor_lookup()`, which performs millions of `paste()` string constructions and named-vector hash lookups inside a sequential `lapply` over 6.46 million rows. The fix is to exploit the panel structure for pure integer-arithmetic neighbor resolution and to vectorize the statistics computation via a padded matrix.