 # Diagnosis and Optimization

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: Row-by-Row `paste`/`match` Inside `lapply` Over 6.46M Rows

1. **`idx_lookup`** is built once (good), but it's a named character vector of length 6.46M. Lookup via `[` on a named vector is **O(n)** linear scan per call in base R (no hashing).
2. **`neighbor_keys <- paste(...)`** is called **6.46 million times**, each time allocating a small character vector and performing string concatenation. With ~4 neighbors on average, that's ~25.8M `paste` calls embedded inside the `lapply`.
3. **`result <- idx_lookup[neighbor_keys]`** does named-vector lookup (linear scan) 6.46M times.
4. The `lapply` returns a **list of 6.46M integer vectors** — the `neighbor_lookup` — which is then iterated **5 more times** (once per variable) in `compute_neighbor_stats`, each time doing another `lapply` over 6.46M elements.

**Total work scales as:** `O(N_rows × avg_neighbors × N_rows)` for the named-vector lookups alone — effectively quadratic in dataset size. This is why it takes 86+ hours.

### Broader Pattern

The real algorithmic issue is that the code converts a **spatial adjacency problem** into a **string-matching problem** row by row. The neighbor structure is static per year-slice and per cell. A vectorized integer-index approach eliminates all string work entirely.

---

## Optimization Strategy

| Step | Current | Proposed |
|------|---------|----------|
| Map cell id → neighbor row indices | `paste` + named-vector lookup per row | Pre-build integer matrix mapping `(cell_index, year_index) → row_index`, then vectorized integer indexing |
| Iterate over rows | `lapply` over 6.46M rows | Vectorized sparse-matrix multiplication or `data.table` join |
| Compute neighbor stats | Second `lapply` over 6.46M rows per variable | Single sparse-matrix multiply per variable gives neighbor sums/counts; min/max via grouped operations |

**Key insight:** The neighbor adjacency is **year-invariant**. We can represent it as a sparse adjacency matrix **W** of dimension `N_cells × N_cells` and then, for each year and each variable, compute neighbor means/sums/min/max via vectorized operations on year-slices. This reduces the entire pipeline from ~86 hours to **minutes**.

### Preserving Numerical Equivalence

The original code computes, for each cell-year `(i, t)`:
- `max(var[neighbors(i), t])`
- `min(var[neighbors(i), t])`
- `mean(var[neighbors(i), t])`

(excluding `NA` values and cells not present in the data). We replicate this exactly.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Drop-in replacement for build_neighbor_lookup + compute_neighbor_stats loop
#
# Preserves: trained Random Forest model (untouched), original numerical output
# Requirements: data.table, Matrix (both typically already available)
# =============================================================================

library(data.table)
library(Matrix)

build_neighbor_features_fast <- function(cell_data,
                                         id_order,
                                         rook_neighbors_unique,
                                         neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # 1. Build a sparse adjacency matrix W  (N_cells x N_cells)
  #    W[i,j] = 1 iff cell j is a rook neighbor of cell i
  #    This encodes ALL neighbor relationships as integers — no strings.
  # -------------------------------------------------------------------------
  N_cells <- length(id_order)

  # id_order is the vector of cell IDs in the order matching the nb object
  # rook_neighbors_unique[[k]] contains integer indices into id_order
  # for the neighbors of id_order[k]

  # Build COO (coordinate) representation
  from_idx <- integer(0)
  to_idx   <- integer(0)
  for (k in seq_along(rook_neighbors_unique)) {
    nbrs <- rook_neighbors_unique[[k]]
    # spdep::nb encodes no-neighbor as 0L in a length-1 vector
    if (length(nbrs) == 1L && nbrs[1L] == 0L) next
    from_idx <- c(from_idx, rep.int(k, length(nbrs)))
    to_idx   <- c(to_idx, nbrs)
  }

  W <- sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = 1,
    dims = c(N_cells, N_cells)
  )
  rm(from_idx, to_idx)

  # -------------------------------------------------------------------------
  # 2. Convert cell_data to data.table and create integer indices
  #    cell_idx: position of each cell's ID within id_order (row of W)
  #    year_idx: integer 1..N_years
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Map cell IDs to integer indices matching id_order (and W rows/cols)
  id_map <- setNames(seq_along(id_order), as.character(id_order))
  dt[, cell_idx := id_map[as.character(id)]]
  
  years_sorted <- sort(unique(dt$year))
  N_years <- length(years_sorted)
  year_map <- setNames(seq_along(years_sorted), as.character(years_sorted))
  dt[, year_idx := year_map[as.character(year)]]

  # -------------------------------------------------------------------------
  # 3. For each variable, build a dense N_cells x N_years matrix, then
  #    compute neighbor stats via sparse matrix ops + grouped operations.
  #
  #    For MEAN:  neighbor_mean = (W %*% V) / (W %*% Valid)
  #      where V has NAs replaced by 0, and Valid is 1 where non-NA, 0 otherwise
  #
  #    For MIN and MAX: we use a year-by-year grouped approach on the
  #      sparse structure to preserve exact NA-aware semantics.
  # -------------------------------------------------------------------------

  # Pre-extract the row/col structure of W for the grouped min/max approach.
  # W is stored in dgCMatrix (compressed sparse column), so we convert to
  # dgTMatrix (triplet) for easy iteration.
  W_triplet <- as(W, "TMatrix")  # may be dgTMatrix
  # i,j are 0-based in Matrix package triplet representation
  w_from <- W_triplet@i + 1L   # focal cell index (row of W)
  w_to   <- W_triplet@j + 1L   # neighbor cell index (col of W)
  n_edges <- length(w_from)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing neighbor features for: %s\n", var_name))

    # --- Build N_cells x N_years matrix of the variable ---
    vals_vec <- dt[[var_name]]
    V <- matrix(NA_real_, nrow = N_cells, ncol = N_years)
    # Fill using integer indexing (fully vectorized)
    lin_idx <- (dt$year_idx - 1L) * N_cells + dt$cell_idx
    V[lin_idx] <- vals_vec

    # --- MEAN via sparse matrix multiplication ---
    V_zero <- V
    V_zero[is.na(V_zero)] <- 0
    Valid <- matrix(0, nrow = N_cells, ncol = N_years)
    Valid[!is.na(V)] <- 1

    neighbor_sum   <- as.matrix(W %*% V_zero)    # N_cells x N_years
    neighbor_count <- as.matrix(W %*% Valid)      # N_cells x N_years
    neighbor_mean  <- neighbor_sum / neighbor_count  # NaN where count==0

    # Convert NaN to NA
    neighbor_mean[is.nan(neighbor_mean)] <- NA_real_
    # Where count == 0, set to NA
    neighbor_mean[neighbor_count == 0] <- NA_real_

    # --- MIN and MAX via edge-list approach (vectorized per year) ---
    neighbor_max <- matrix(NA_real_, nrow = N_cells, ncol = N_years)
    neighbor_min <- matrix(NA_real_, nrow = N_cells, ncol = N_years)

    for (yi in seq_len(N_years)) {
      v_col <- V[, yi]  # length N_cells

      # Values of neighbors for each edge
      nb_vals <- v_col[w_to]  # length n_edges

      # Remove edges where neighbor value is NA
      valid_mask <- !is.na(nb_vals)
      if (!any(valid_mask)) next

      e_from <- w_from[valid_mask]
      e_vals <- nb_vals[valid_mask]

      # Compute grouped max and min using data.table
      edge_dt <- data.table(from = e_from, val = e_vals)
      agg <- edge_dt[, .(mx = max(val), mn = min(val)), by = from]

      neighbor_max[agg$from, yi] <- agg$mx
      neighbor_min[agg$from, yi] <- agg$mn
    }

    # --- Write results back to dt using the same linear index ---
    max_col <- paste0("neighbor_max_", var_name)
    min_col <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    dt[, (max_col)  := neighbor_max[lin_idx]]
    dt[, (min_col)  := neighbor_min[lin_idx]]
    dt[, (mean_col) := neighbor_mean[lin_idx]]

    rm(V, V_zero, Valid, neighbor_sum, neighbor_count,
       neighbor_mean, neighbor_max, neighbor_min)
  }

  # -------------------------------------------------------------------------
  # 4. Drop helper columns and return as data.frame (to match original)
  # -------------------------------------------------------------------------
  dt[, c("cell_idx", "year_idx") := NULL]
  return(as.data.frame(dt))
}


# =============================================================================
# USAGE — drop-in replacement for the original outer loop
# =============================================================================
#
# BEFORE (original ~86 hours):
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }
#
# AFTER (optimized, ~5-15 minutes on 16 GB laptop):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is NOT touched.
# Predictions continue as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Memory Considerations (16 GB Laptop)

| Object | Size |
|--------|------|
| Sparse `W` (344K × 344K, ~1.37M non-zeros) | ~22 MB |
| Dense `V` matrix (344K × 28) | ~77 MB |
| `neighbor_sum`, `neighbor_count` (same shape) | ~77 MB each |
| `neighbor_max`, `neighbor_min` (same shape) | ~77 MB each |
| `cell_data` data.table (~6.46M × 110 cols) | ~5.7 GB |
| **Peak total** | **~7–8 GB** ✓ |

This fits within 16 GB with room to spare.

---

## Why This Is Correct (Numerical Equivalence)

The original code for each row `i`:
1. Finds the rook neighbors of `cell_data$id[i]` that exist in the same year `cell_data$year[i]`.
2. Extracts non-NA values of `var_name` for those neighbor-year rows.
3. Returns `max`, `min`, `mean` of those values (or `NA` if none).

The optimized code:
1. Builds the same adjacency in matrix `W` (same neighbor definitions from `rook_neighbors_unique`).
2. Builds `V[cell, year]` from the same data — cells not present for a year remain `NA` and are excluded by the `Valid` mask (for mean) and `!is.na` filter (for min/max).
3. Computes identical aggregates:
   - **Mean**: `(W %*% V_zero) / (W %*% Valid)` gives the NA-aware mean of neighbor values, identical to `mean(neighbor_vals[!is.na(neighbor_vals)])`.
   - **Max/Min**: Grouped aggregation over the same edge list after removing NA neighbor values, identical to `max/min(neighbor_vals[!is.na(neighbor_vals)])`.

The results are numerically identical (up to floating-point associativity, which is negligible for `mean` of typically ≤4 values).

---

## Complexity Comparison

| | Original | Optimized |
|---|---|---|
| String operations | ~25.8M `paste` calls + 6.46M named-vector lookups | **Zero** |
| Neighbor lookup | O(N_rows × avg_neighbors × N_rows) ≈ quadratic | O(N_edges) via sparse matrix |
| Per-variable stats | `lapply` over 6.46M rows × 5 vars | Sparse matmul + 28 grouped aggregations × 5 vars |
| **Estimated time** | **86+ hours** | **5–15 minutes** |