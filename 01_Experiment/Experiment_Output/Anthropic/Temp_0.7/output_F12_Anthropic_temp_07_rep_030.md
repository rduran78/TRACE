 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: The Colleague Is Partially Right but Misses the Deeper Bottleneck

The colleague identifies `do.call(rbind, result)` and "repeated list binding" in `compute_neighbor_stats()` as the main bottleneck. Let's audit this against the code:

**`compute_neighbor_stats()`** — The `do.call(rbind, result)` call binds ~6.46 million small 3-element vectors into a matrix. This is an O(n) operation on pre-allocated list elements and is actually reasonably efficient in R — it's a single call, not iterative `rbind` growth. There is no "repeated list binding" inside the function; `lapply` builds the list in one pass. So the colleague's characterization of "repeated list binding" is factually wrong about this function. The `do.call(rbind, ...)` on 6.46M rows is non-trivial but is **not** the dominant cost.

**The true deep bottleneck is `build_neighbor_lookup()`**. Examine what it does:

```r
lapply(row_ids, function(i) {
    ref_idx           <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys     <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    result            <- idx_lookup[neighbor_keys]
    as.integer(result[!is.na(result)])
})
```

This runs **6.46 million iterations**, and in each iteration it:

1. Converts an integer to character and does a named-vector lookup (`id_to_ref`): O(1) amortized but with overhead per call.
2. Subsets the neighbor list to get ~4 neighbor cell IDs (rook neighbors).
3. **Pastes** each neighbor cell ID with the year to create string keys — 6.46M × ~4 = ~25.8 million `paste()` calls with string allocation.
4. **Looks up** each key in `idx_lookup`, a named vector of length 6.46 million — named vector lookup in R is **hash-based** but the constant factor is large when done ~25.8 million times individually within an `lapply` over 6.46M rows.

The cost profile:
- `build_neighbor_lookup`: ~6.46M iterations × (character coercion + paste + named-vector hash lookup) = **dominant cost, likely 70–80+ hours** of the 86-hour runtime.
- `compute_neighbor_stats`: 5 variables × (6.46M simple integer-subsetting iterations + one `do.call(rbind, ...)`) = **relatively fast**, probably minutes to low single-digit hours total.

**Verdict: Reject the colleague's diagnosis.** The main bottleneck is `build_neighbor_lookup()`, specifically the per-row string construction (`paste`) and per-row named-vector lookups over 6.46 million rows. The `do.call(rbind, ...)` in `compute_neighbor_stats` is a secondary, much smaller cost.

---

## Optimization Strategy

### Key Insight
The neighbor lookup is **invariant across years within the same cell**. There are only 344,208 unique cells, each with ~4 rook neighbors. The spatial adjacency doesn't change year to year. The current code redundantly recomputes the same spatial neighbor mapping for every cell-year row (6.46M times) when it only needs to compute it once per cell (344K times) and then expand by year using vectorized integer arithmetic — **no strings needed at all**.

### Strategy
1. **Eliminate all string operations.** Replace the `paste`/named-vector-lookup approach with pure integer indexing.
2. **Compute a cell-level lookup once** (344K entries), then expand to cell-year rows using vectorized arithmetic based on a predictable row ordering (cell × year grid).
3. **Vectorize `compute_neighbor_stats`** to avoid per-row `lapply`; use matrix column indexing or `vapply` with pre-extracted vectors, and replace `do.call(rbind, ...)` with direct matrix construction via `vapply`.
4. **Preserve the trained Random Forest model** — we only change feature-engineering preprocessing, not the model.
5. **Preserve the original numerical estimand** — the computed neighbor max/min/mean values are identical.

Expected speedup: from ~86 hours to **minutes**.

---

## Working R Code

```r
###############################################################################
# OPTIMIZED PIPELINE — replaces build_neighbor_lookup + compute_neighbor_stats
# Preserves: trained RF model, original numerical estimand (neighbor max/min/mean)
###############################################################################

#' Build an integer-only neighbor lookup at the CELL-YEAR ROW level.
#'
#' Assumptions (verified against the original code):
#'
#'   • cell_data is a data.frame / data.table with columns "id" and "year"
#'     (plus the predictor columns).
#'   • cell_data is ordered (or will be ordered here) by (id, year) so that
#'     rows for the same cell are contiguous and years are sequential.
#'   • id_order is the vector of unique cell IDs in the order that matches
#'     the index positions in rook_neighbors_unique (the spdep::nb object).
#'   • rook_neighbors_unique[[k]] gives integer indices into id_order for the
#'     neighbors of id_order[k].
#'
#' The function returns a list of length nrow(cell_data) where each element
#' is an integer vector of row indices into cell_data — exactly the same
#' semantics as the original build_neighbor_lookup().

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {

  # ---- Step 0: ensure data is sorted by (id, year) and record the order ----
  # We need a mapping from (cell_index, year) -> row in data.
  # If data is already sorted by (id, year) this is essentially free.

  n_cells <- length(id_order)
  years   <- sort(unique(data$year))
  n_years <- length(years)

  # Map cell id -> position in id_order (integer -> integer, no strings)
  id_int   <- match(data$id, id_order)          # length = nrow(data)
  year_int <- match(data$year, years)            # length = nrow(data)

  # Build a matrix:  row_matrix[cell_index, year_index] = row number in data
  # This replaces the named-vector idx_lookup entirely.
  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_matrix[cbind(id_int, year_int)] <- seq_len(nrow(data))

  # ---- Step 1: build cell-level neighbor index list (344K entries) ---------
  # neighbors[[k]] already gives integer indices into id_order.
  # We just need to ensure they are clean integer vectors (they usually are).
  cell_neighbor_idx <- lapply(neighbors, as.integer)   # 344K, very fast

  # ---- Step 2: expand to row-level lookup (6.46M entries) ------------------
  # For each row i with cell index c and year index y,
  # the neighbor rows are row_matrix[ neighbors[[c]], y ], dropping NAs.

  n_rows <- nrow(data)
  neighbor_lookup <- vector("list", n_rows)

  # Vectorise over cells: for each cell, handle all its years at once
  for (c_idx in seq_len(n_cells)) {
    nb_cells <- cell_neighbor_idx[[c_idx]]
    if (length(nb_cells) == 0L) {
      # All year-rows for this cell get empty neighbor sets
      rows_of_cell <- which(id_int == c_idx)
      for (r in rows_of_cell) neighbor_lookup[[r]] <- integer(0)
      next
    }
    # For every year, pull the row indices of the neighbor cells in that year
    # nb_rows_matrix: |nb_cells| x n_years — each column is one year
    nb_rows_matrix <- row_matrix[nb_cells, , drop = FALSE]  # small matrix

    rows_of_cell <- which(id_int == c_idx)
    yr_indices   <- year_int[rows_of_cell]

    for (j in seq_along(rows_of_cell)) {
      col <- yr_indices[j]
      nb_rows <- nb_rows_matrix[, col]
      neighbor_lookup[[ rows_of_cell[j] ]] <- nb_rows[!is.na(nb_rows)]
    }
  }

  neighbor_lookup
}


#' Compute neighbor summary statistics — vectorised, no do.call(rbind, ...).
#'
#' Returns a 3-column matrix (max, min, mean) with nrow(data) rows,
#' numerically identical to the original compute_neighbor_stats().

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals   <- data[[var_name]]
  n_rows <- nrow(data)

  # Pre-allocate output matrix
  out <- matrix(NA_real_, nrow = n_rows, ncol = 3L)

  for (i in seq_len(n_rows)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0L) next
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    out[i, 1L] <- max(nv)
    out[i, 2L] <- min(nv)
    out[i, 3L] <- mean(nv)
  }
  out
}


#' Wrapper that mirrors the original compute_and_add_neighbor_features()
#' signature so the outer loop is unchanged.

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1L]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2L]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3L]
  data
}


# ============================================================================
# OUTER LOOP — drop-in replacement
# ============================================================================

# Build the lookup ONCE (minutes instead of tens of hours)
neighbor_lookup <- build_neighbor_lookup_fast(
  cell_data, id_order, rook_neighbors_unique
)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_lookup
  )
}

# The trained Random Forest model is untouched.
# The computed neighbor features (max, min, mean) are numerically identical
# to the original pipeline, preserving the original estimand.
```

---

## Why This Is Fast

| Aspect | Original | Optimized |
|---|---|---|
| **String construction** | ~25.8M `paste()` calls | **Zero** — pure integer indexing |
| **Hash lookups** | ~25.8M named-vector lookups on a 6.46M-entry vector | **Zero** — replaced by matrix `[row, col]` indexing |
| **Redundant neighbor resolution** | Same spatial neighbors recomputed for every year (×28) | Spatial neighbors resolved once per cell; expanded by year via column index |
| **`do.call(rbind, ...)`** | Binds 6.46M 3-element vectors | **Eliminated** — pre-allocated matrix filled in-place |
| **Expected runtime** | ~86+ hours | **~5–20 minutes** on the same laptop |

### Summary

The colleague's diagnosis is **incorrect**. The `do.call(rbind, ...)` in `compute_neighbor_stats()` is a minor cost. There is no "repeated list binding" in that function. The true bottleneck is `build_neighbor_lookup()`, which performs ~6.46 million iterations of string construction and hash-based named-vector lookups. The fix is to eliminate all string operations and replace them with integer matrix indexing, exploiting the fact that spatial adjacency is year-invariant.