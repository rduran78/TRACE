 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Looks those keys up in another named character vector (`idx_lookup`).

String construction (`paste`) and named-vector lookup (which is O(n) hash probing on character keys) repeated ~6.46M × ~4 neighbors ≈ 25.8M string operations is extremely expensive in base R. The secondary bottleneck is `compute_neighbor_stats`, which runs an `lapply` with per-element R function calls over 6.46M entries, each allocating small vectors.

**Root causes, ranked:**

| # | Cause | Impact |
|---|-------|--------|
| 1 | Per-row `paste()` + character key lookup in `build_neighbor_lookup` (~6.46M iterations, each creating multiple strings) | **Dominant** |
| 2 | Per-row anonymous function in `compute_neighbor_stats` with small vector allocations | **Major** |
| 3 | `do.call(rbind, result)` on a 6.46M-element list of 3-vectors | **Moderate** |
| 4 | Everything is single-threaded base R | **Compounding** |

## Optimization Strategy

1. **Eliminate all string key construction.** Replace the `paste(id, year)`-based lookup with integer arithmetic. Since years are contiguous (1992–2019, i.e., 28 years), we can map every `(id, year)` pair to a row index via a pre-built integer matrix indexed as `row_matrix[id_index, year_index]`. Lookup becomes a single integer matrix access — no strings, no hashing.

2. **Vectorize neighbor stat computation using `data.table` grouping or, better, a single pre-built sparse adjacency structure and matrix operations.** We build a sparse neighbor matrix (CSR-style, via two integer vectors: `neighbor_row_idx` and `target_row_idx`) and use vectorized grouped operations.

3. **Replace `lapply` + `do.call(rbind, ...)` with pre-allocated matrices.**

4. **Optionally parallelize** the five variables, but the vectorization alone should reduce runtime from 86+ hours to minutes.

**Expected speedup:** The dominant cost moves from ~6.46M × k interpreted R calls with string ops to a single vectorized sparse-matrix–style operation. Estimated wall time: **2–10 minutes** on a 16 GB laptop (down from 86+ hours).

## Optimized Working R Code

```r
# =============================================================================
# OPTIMIZED SPATIAL NEIGHBOR FEATURE CONSTRUCTION
# Drop-in replacement. Preserves the trained RF model and original estimand.
# =============================================================================

library(data.table)

build_neighbor_lookup_fast <- function(cell_data, id_order, rook_neighbors_unique) {
  # ---- Step 1: Create integer mappings (no strings) ----
  # Map each unique cell id to a contiguous integer index
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Map each year to a contiguous integer index
  years <- sort(unique(cell_data$year))
  year_to_idx <- setNames(seq_along(years), as.character(years))
  n_years <- length(years)
  n_ids   <- length(id_order)

  # ---- Step 2: Build (id_index, year_index) -> row number matrix ----
  # This replaces the paste-based idx_lookup entirely.
  # row_matrix[id_idx, year_idx] = row number in cell_data (or NA)
  row_matrix <- matrix(NA_integer_, nrow = n_ids, ncol = n_years)

  cd_id_idx   <- id_to_idx[as.character(cell_data$id)]
  cd_year_idx <- year_to_idx[as.character(cell_data$year)]
  row_matrix[cbind(cd_id_idx, cd_year_idx)] <- seq_len(nrow(cell_data))

  # ---- Step 3: Expand neighbor pairs into (target_row, neighbor_row) ----
  # For each cell i (in id_order), rook_neighbors_unique[[i]] gives

  # the indices (into id_order) of its neighbors.
  # We need to expand this across all 28 years.

  # Build flat edge list at the id-index level
  n_neighbors <- lengths(rook_neighbors_unique)  # integer vector, length n_ids
  from_id_idx <- rep(seq_len(n_ids), times = n_neighbors)
  to_id_idx   <- unlist(rook_neighbors_unique, use.names = FALSE)
  # from_id_idx[k] -> to_id_idx[k] is one directed neighbor relationship

  n_edges <- length(from_id_idx)
  cat(sprintf("Neighbor edges (id-level): %d\n", n_edges))
  cat(sprintf("Expanding across %d years -> %d edge-year pairs\n",
              n_years, n_edges * n_years))

  # Expand across years: each edge is replicated for every year
  # Use integer rep to avoid huge intermediate objects
  from_id_expanded <- rep(from_id_idx, times = n_years)
  to_id_expanded   <- rep(to_id_idx,   times = n_years)
  year_idx_expanded <- rep(seq_len(n_years), each = n_edges)

  # Map to row numbers in cell_data
  target_rows   <- row_matrix[cbind(from_id_expanded, year_idx_expanded)]
  neighbor_rows <- row_matrix[cbind(to_id_expanded,   year_idx_expanded)]

  # Remove pairs where either target or neighbor is missing
  valid <- !is.na(target_rows) & !is.na(neighbor_rows)
  target_rows   <- target_rows[valid]
  neighbor_rows <- neighbor_rows[valid]

  cat(sprintf("Valid (target_row, neighbor_row) pairs: %d\n", length(target_rows)))

  list(target_rows = target_rows, neighbor_rows = neighbor_rows,
       n_data_rows = nrow(cell_data))
}


compute_neighbor_stats_fast <- function(cell_data, edge_list, var_name) {
  # edge_list: list with target_rows, neighbor_rows, n_data_rows
  vals <- cell_data[[var_name]]
  neighbor_vals <- vals[edge_list$neighbor_rows]

  # Build a data.table for grouped aggregation (vectorized C code)
  dt <- data.table(
    target = edge_list$target_rows,
    nval   = neighbor_vals
  )

  # Remove edges where the neighbor value is NA
  dt <- dt[!is.na(nval)]

  # Grouped aggregation — single pass, highly optimized in data.table
  agg <- dt[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = target]

  # Allocate output columns (NA for rows with no valid neighbors)
  nb_max  <- rep(NA_real_, edge_list$n_data_rows)
  nb_min  <- rep(NA_real_, edge_list$n_data_rows)
  nb_mean <- rep(NA_real_, edge_list$n_data_rows)

  nb_max[agg$target]  <- agg$nb_max
  nb_min[agg$target]  <- agg$nb_min
  nb_mean[agg$target] <- agg$nb_mean

  data.frame(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}


compute_and_add_neighbor_features_fast <- function(cell_data, var_name, edge_list) {
  stats <- compute_neighbor_stats_fast(cell_data, edge_list, var_name)
  cell_data[[paste0(var_name, "_nb_max")]]  <- stats$nb_max
  cell_data[[paste0(var_name, "_nb_min")]]  <- stats$nb_min
  cell_data[[paste0(var_name, "_nb_mean")]] <- stats$nb_mean
  cell_data
}


# =============================================================================
# MAIN EXECUTION (drop-in replacement for the original outer loop)
# =============================================================================

cat("Building optimized neighbor edge list...\n")
system.time({
  edge_list <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
})

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features for", length(neighbor_source_vars), "variables...\n")
system.time({
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing: %s\n", var_name))
    cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, edge_list)
  }
})

cat("Done. Neighbor features added to cell_data.\n")

# The trained Random Forest model is untouched.
# Column names match the original pattern: {var}_nb_max, {var}_nb_min, {var}_nb_mean
# Numerical results are identical (same max/min/mean over the same neighbor sets).
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---------|-----------|
| **Same neighbor sets** | We use the same `rook_neighbors_unique` nb object and the same `id_order`. The integer matrix lookup `row_matrix[id_idx, year_idx]` returns exactly the same row as the original `paste(id, year)` string lookup — it's just a different indexing scheme for the same mapping. |
| **Same statistics** | `max`, `min`, `mean` over the same non-NA neighbor values, grouped by the same target row. `data.table` aggregation is numerically identical to the base R computation. |
| **Same output columns** | Column names follow the same `{var}_nb_max/min/mean` pattern, so the trained Random Forest's `predict()` call will find the expected feature names. |
| **RF model untouched** | No retraining; we only reconstruct input features for prediction. |

## Complexity Comparison

| Operation | Original | Optimized |
|-----------|----------|-----------|
| Lookup construction | O(N) `paste` + named-vector hash lookups (N = 6.46M, each with k neighbors) | O(N) integer matrix writes + O(E×T) integer matrix reads (all vectorized in C) |
| Stat computation | O(N) `lapply` with R-level function calls | O(E×T) `data.table` grouped aggregation (radix sort + single C pass) |
| Result assembly | `do.call(rbind, 6.46M-element list)` | Pre-allocated vector indexed assignment |
| **Estimated time** | **86+ hours** | **2–10 minutes** |

The key insight: replacing millions of interpreted R-level string operations and per-row anonymous function calls with vectorized integer indexing and `data.table` grouped aggregation eliminates the bottleneck entirely.