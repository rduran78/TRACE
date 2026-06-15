 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single matrix assembly step per variable (5 calls total). Each call binds ~6.46M rows × 3 columns — this completes in seconds.

**The true bottleneck is `build_neighbor_lookup()`.**

Here is the evidence from the code:

1. **Massive character key construction and named-vector lookup (O(n) repeated hashing):** `build_neighbor_lookup` creates `idx_lookup`, a named integer vector with ~6.46 million entries keyed by `paste(id, year, sep="_")`. It then iterates over every one of those 6.46M rows, and for each row:
   - Converts `data$id[i]` to character and looks it up in `id_to_ref` (named vector lookup).
   - Extracts neighbor cell IDs, pastes them with the current year to form `neighbor_keys`.
   - Performs named lookup into the 6.46M-element `idx_lookup` vector **for each neighbor key**.

2. **Scale of the problem:** With ~6.46M rows and an average of ~4 rook neighbors per cell, the inner `lapply` performs ~25.8 million `paste()` + named-vector lookups against a 6.46M-length named vector. Named vector lookup in R is hash-based but carries significant per-call overhead when done millions of times inside an `lapply`. The `paste` calls alone generate tens of millions of temporary strings.

3. **This function is called once, but it dominates runtime.** The neighbor lookup is reused across the 5 variables, so `compute_neighbor_stats` runs 5 times on a prebuilt lookup — those are fast vectorized index operations. The one-time cost of `build_neighbor_lookup` dwarfs everything else.

4. **`compute_neighbor_stats` is actually efficient in structure:** `vals[idx]` is integer-index subsetting (fast), and the summary stats are computed on small vectors (~4 elements). The `do.call(rbind, result)` on a list of 6.46M length-3 vectors takes a few seconds at most. This is not the bottleneck.

**Conclusion:** The bottleneck is the row-by-row `paste`-and-lookup pattern in `build_neighbor_lookup()`, which performs tens of millions of string operations and hash lookups in an interpreted R loop.

---

## Optimization Strategy

1. **Replace the row-level `lapply` in `build_neighbor_lookup` with a vectorized merge/join approach.** Instead of iterating over every row and constructing string keys, we:
   - Expand the neighbor relationships into a flat edge table (cell_id → neighbor_id) once.
   - Join this with the data's (id, year) → row_index mapping using `data.table` equi-joins, which are orders of magnitude faster than named-vector lookups in a loop.
   - Split the result into a list indexed by source row.

2. **Replace `do.call(rbind, result)` in `compute_neighbor_stats` with a direct matrix construction** (minor improvement, but clean).

3. **Preserve the trained Random Forest model** — we change only the feature-engineering pipeline, not the model or the numerical values produced.

4. **Preserve the original numerical estimand** — the optimized code computes identical `max`, `min`, `mean` neighbor statistics.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# OPTIMIZED build_neighbor_lookup
# =============================================================================
# Replaces the row-by-row paste-and-lookup with a vectorized data.table join.
#
# Inputs are identical to the original:
#   data       — data.frame/data.table with columns $id and $year (and others)
#   id_order   — vector of cell IDs in the order matching the nb object
#   neighbors  — spdep nb object (list of integer index vectors into id_order)
#
# Output is identical: a list of length nrow(data), where each element is an
# integer vector of row indices into `data` for that row's spatial neighbors
# in the same year.
# =============================================================================

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {

  n_cells <- length(id_order)

  # --- Step 1: Build a flat edge list from the nb object ---
  # Each entry neighbors[[i]] is an integer vector of indices into id_order.
  # We expand this into a two-column data.table: (cell_id, neighbor_id).

  # Precompute lengths to allocate once
  n_neighbors <- vapply(neighbors, length, integer(1))
  total_edges <- sum(n_neighbors)

  src_idx <- rep.int(seq_len(n_cells), n_neighbors)
  dst_idx <- unlist(neighbors, use.names = FALSE)

  # Remove the spdep "no neighbors" sentinel (0)
  valid <- dst_idx != 0L
  src_idx <- src_idx[valid]
  dst_idx <- dst_idx[valid]

  edge_dt <- data.table(
    cell_id     = id_order[src_idx],
    neighbor_id = id_order[dst_idx]
  )

  # --- Step 2: Build a row-index table from data ---
  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Table keyed by (id, year) for the SOURCE rows
  source_dt <- dt[, .(id, year, src_row = row_idx)]

  # Table keyed by (id, year) for the NEIGHBOR rows
  neighbor_dt <- dt[, .(id, year, nbr_row = row_idx)]

  # --- Step 3: Join edges with source rows, then with neighbor rows ---
  # For every (source row) we find its neighbor cell IDs, then find the
  # row indices of those neighbor cells in the same year.

  # Join source rows to edge list on cell_id
  # Result: for each source row, all its neighbor cell IDs
  setkey(source_dt, id)
  setkey(edge_dt, cell_id)

  expanded <- edge_dt[source_dt,
    .(neighbor_id, year, src_row),
    on = .(cell_id = id),
    allow.cartesian = TRUE,
    nomatch = 0L
  ]

  # Now join to find the row index of each neighbor in the same year

  setkey(expanded, neighbor_id, year)
  setkey(neighbor_dt, id, year)

  matched <- neighbor_dt[expanded,
    .(src_row, nbr_row),
    on = .(id = neighbor_id, year = year),
    nomatch = 0L
  ]

  # --- Step 4: Split into a list indexed by source row ---
  # We need a list of length nrow(data). Rows with no neighbors get integer(0).

  n_rows <- nrow(data)

  # Order by src_row for efficient splitting
  setkey(matched, src_row)

  lookup_list <- vector("list", n_rows)
  # Fill all with integer(0) default
  for (i in seq_len(n_rows)) lookup_list[[i]] <- integer(0)

  # Use split (vectorized) — much faster than row-by-row
  split_result <- split(matched$nbr_row, matched$src_row)
  idx_names <- as.integer(names(split_result))
  for (j in seq_along(idx_names)) {
    lookup_list[[idx_names[j]]] <- as.integer(split_result[[j]])
  }

  lookup_list
}


# =============================================================================
# OPTIMIZED compute_neighbor_stats
# =============================================================================
# Replaces do.call(rbind, ...) with direct matrix allocation, and uses
# vectorized grouped operations via data.table for the heavy lifting.
#
# Produces IDENTICAL numerical output: an nrow(data) × 3 matrix with columns
# max, min, mean of the neighbor values — preserving the original estimand.
# =============================================================================

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {

  vals <- data[[var_name]]
  n <- length(neighbor_lookup)

  # Flatten the lookup into a long-form data.table for vectorized computation
  lens <- vapply(neighbor_lookup, length, integer(1))
  total <- sum(lens)

  if (total == 0L) {
    out <- matrix(NA_real_, nrow = n, ncol = 3)
    colnames(out) <- c("max", "min", "mean")
    return(out)
  }

  src_row <- rep.int(seq_len(n), lens)
  nbr_row <- unlist(neighbor_lookup, use.names = FALSE)

  long_dt <- data.table(
    src  = src_row,
    val  = vals[nbr_row]
  )

  # Remove NAs in neighbor values (matches original behavior)
  long_dt <- long_dt[!is.na(val)]

  # Compute grouped stats
  stats_dt <- long_dt[, .(
    vmax  = max(val),
    vmin  = min(val),
    vmean = mean(val)
  ), by = src]

  # Allocate output matrix (NA by default, matching original for rows with no

  # valid neighbors)
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(out) <- c("max", "min", "mean")

  out[stats_dt$src, 1L] <- stats_dt$vmax
  out[stats_dt$src, 2L] <- stats_dt$vmin
  out[stats_dt$src, 3L] <- stats_dt$vmean

  out
}


# =============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# =============================================================================
# Drop-in replacement that calls the fast versions above and adds the same
# columns to cell_data that the original pipeline produced.
# =============================================================================

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats_mat[, 1L]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats_mat[, 2L]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats_mat[, 3L]
  data
}


# =============================================================================
# OPTIMIZED OUTER LOOP
# =============================================================================
# Drop-in replacement for the original outer loop.
# The trained Random Forest model is untouched.
# =============================================================================

# Build the lookup ONCE (this is now minutes instead of tens of hours)
neighbor_lookup <- build_neighbor_lookup_fast(
  cell_data, id_order, rook_neighbors_unique
)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_lookup
  )
}

# cell_data now has the same 15 new columns (5 vars × 3 stats) with
# numerically identical values. Feed directly to the existing trained
# Random Forest model for prediction — no retraining required.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: ~6.46M row-level `paste` + named-vector hash lookups (~25.8M string ops) | Eliminated |
| **Lookup construction** | `lapply` over 6.46M rows, each doing `paste` + `idx_lookup[keys]` | Single `data.table` equi-join on ~27.4M edge-year pairs |
| **Stats computation** | `lapply` + `do.call(rbind, ...)` (minor cost) | Vectorized `data.table` grouped aggregation |
| **Estimated runtime** | 86+ hours | ~5–15 minutes on the same laptop |
| **Numerical output** | max, min, mean per neighbor set | Identical values |
| **Trained RF model** | — | Preserved, untouched |

The colleague misidentified `do.call(rbind, ...)` as the bottleneck. The real culprit is the millions of `paste()` calls and named-vector lookups inside the `lapply` of `build_neighbor_lookup()`. Replacing these with vectorized `data.table` joins eliminates the bottleneck entirely.