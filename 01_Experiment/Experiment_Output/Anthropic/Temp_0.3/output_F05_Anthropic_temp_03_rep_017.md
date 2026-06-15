 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` — **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` — fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — **allocates a new character vector per row** (~4 neighbor keys × 6.46M rows ≈ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` — **named-vector lookup on a 6.46M-element character vector, repeated 6.46M times**.

The named-vector `idx_lookup` has 6.46M entries. Each lookup into it is O(n) in the worst case for partial matching, or O(1) amortized with hashing — but R's `[.character` on named vectors is **not a hash table**; it uses linear search or internal hashing that still has significant per-call overhead. Doing this 6.46M times is catastrophic.

### The Deeper Structural Insight

The neighbor relationships are **spatial** (cell-to-cell) and **time-invariant**. The year dimension is only used to find "the same neighbor in the same year." Since the panel is balanced (every cell appears in every year), the neighbor lookup can be reformulated as a **purely positional/integer operation** that never touches strings at all.

**Key insight:** If the data is sorted by `(id, year)` or `(year, id)` in a known order, then for any row `i` belonging to cell `c` in year `t`, its neighbor rows can be computed by **integer arithmetic** on row indices — no string keys, no hash lookups, no `paste`.

Furthermore, `compute_neighbor_stats` is called 5 times, each time iterating over the full 6.46M-element `neighbor_lookup` list. This list-of-integer-vectors structure forces R into slow per-element `lapply` iteration. A **vectorized matrix-based approach** can replace this entirely.

---

## Optimization Strategy

1. **Eliminate all string operations.** Sort data by `(id, year)`. With `N_cells = 344,208` and `N_years = 28`, row index for cell `c` (0-indexed among cells) in year `t` (0-indexed among years) is `c * N_years + t + 1`. Neighbor row indices are pure integer arithmetic.

2. **Replace the row-level `lapply` in `build_neighbor_lookup`** with a single vectorized construction of a sparse neighbor matrix or a flat integer-index structure.

3. **Replace per-variable `lapply` in `compute_neighbor_stats`** with sparse matrix multiplication. If `W` is the row-adjacency matrix (6.46M × 6.46M sparse), then `W %*% x` gives neighbor sums, `W %*% ones` gives neighbor counts, and neighbor means = sums / counts. For max and min, use grouped operations on a long-form edge list.

4. **Compute all 5 variables' stats in one pass** over the edge list for max/min, and via sparse matrix multiply for mean.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Drop-in replacement preserving the original numerical estimand.
# =============================================================================

library(Matrix)   # for sparse matrices
library(data.table)

build_and_apply_neighbor_features <- function(cell_data,
                                               id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # STEP 0: Convert to data.table for fast manipulation; record original order
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, orig_row_idx__ := .I]

  N_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  N_years <- length(years)
  stopifnot(nrow(dt) == N_cells * N_years)  # balanced panel check

  # -------------------------------------------------------------------------
  # STEP 1: Create integer mappings (no strings anywhere)
  # -------------------------------------------------------------------------
  # Map cell id -> 1-based cell index (in id_order ordering)
  id_to_cidx <- setNames(seq_along(id_order), as.character(id_order))

  # Map year -> 1-based year index
  year_to_yidx <- setNames(seq_along(years), as.character(years))

  # Sort data by (cell_index, year) so row position is deterministic
  dt[, cell_idx__ := id_to_cidx[as.character(id)]]
  dt[, year_idx__ := year_to_yidx[as.character(year)]]
  setorder(dt, cell_idx__, year_idx__)

  # Now row index for cell c (1-based) in year t (1-based) is:
  #   row = (c - 1) * N_years + t
  # Verify:
  dt[, expected_row__ := (cell_idx__ - 1L) * N_years + year_idx__]
  stopifnot(all(dt$expected_row__ == seq_len(nrow(dt))))
  dt[, expected_row__ := NULL]

  # -------------------------------------------------------------------------
  # STEP 2: Build cell-level directed edge list from rook_neighbors_unique
  # -------------------------------------------------------------------------
  # rook_neighbors_unique is an nb object: list of length N_cells,
  # where element i contains integer indices of neighbors of cell i
  # (in id_order indexing, matching our cell_idx__).

  # Build edge list: from_cell -> to_cell (1-based cell indices)
  n_neighbors <- vapply(rook_neighbors_unique, length, integer(1))
  from_cell   <- rep(seq_len(N_cells), times = n_neighbors)
  to_cell     <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove the nb "0" sentinel for cells with no neighbors (if any)
  valid <- to_cell > 0L
  from_cell <- from_cell[valid]
  to_cell   <- to_cell[valid]

  N_edges <- length(from_cell)
  cat(sprintf("Cell-level directed edges: %d\n", N_edges))

  # -------------------------------------------------------------------------
  # STEP 3: Expand cell-level edges to row-level edges (across all years)
  #
  # For each year t, edge (c1 -> c2) at cell level becomes
  #   row_from = (c1-1)*N_years + t  ->  row_to = (c2-1)*N_years + t
  #
  # Total row-level edges = N_edges * N_years
  # ~1.37M * 28 ≈ 38.5M edges — manageable in memory as integer vectors.
  # -------------------------------------------------------------------------

  # Pre-compute cell base offsets: (cell_idx - 1) * N_years
  from_base <- (from_cell - 1L) * N_years
  to_base   <- (to_cell   - 1L) * N_years

  # Expand across years
  year_offsets <- seq_len(N_years)  # 1..28

  # Use outer-sum approach: each column is one year
  # row_from[e, t] = from_base[e] + t
  # Flatten in column-major order (all edges for year 1, then year 2, ...)
  row_from <- rep(from_base, times = N_years) +
              rep(year_offsets, each = N_edges)
  row_to   <- rep(to_base,   times = N_years) +
              rep(year_offsets, each = N_edges)

  N_row_edges <- length(row_from)
  N_rows      <- nrow(dt)
  cat(sprintf("Row-level directed edges: %d (rows: %d)\n", N_row_edges, N_rows))

  # -------------------------------------------------------------------------
  # STEP 4: Build sparse adjacency matrix W (N_rows x N_rows)
  #         W[i,j] = 1 means j is a neighbor of i.
  #         So W %*% x gives sum of neighbor values for each row.
  # -------------------------------------------------------------------------
  W <- sparseMatrix(
    i    = row_from,
    j    = row_to,
    x    = 1,
    dims = c(N_rows, N_rows)
  )

  # Neighbor count per row (for computing means)
  neighbor_count <- as.numeric(W %*% rep(1, N_rows))

  # -------------------------------------------------------------------------
  # STEP 5: For each variable, compute max, min, mean of neighbors
  # -------------------------------------------------------------------------
  # Mean: use sparse matrix multiply.
  # Max, Min: use data.table grouped operations on the edge list.

  # Pre-build the edge data.table for grouped max/min
  # We only need row_from and row_to; we'll join variable values on row_to.
  edge_dt <- data.table(from = row_from, to = row_to)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing neighbor stats for: %s\n", var_name))

    vals <- dt[[var_name]]

    # --- MEAN via sparse matrix multiply ---
    # Handle NAs: replace with 0 for sum, and track non-NA count
    not_na   <- as.numeric(!is.na(vals))
    vals_0   <- ifelse(is.na(vals), 0, vals)

    neighbor_sum    <- as.numeric(W %*% vals_0)
    neighbor_nonna  <- as.numeric(W %*% not_na)
    neighbor_mean   <- ifelse(neighbor_nonna > 0,
                              neighbor_sum / neighbor_nonna,
                              NA_real_)
    # Rows with no neighbors at all -> NA
    neighbor_mean[neighbor_count == 0] <- NA_real_

    # --- MAX and MIN via grouped edge-list operations ---
    edge_dt[, val := vals[to]]

    # Remove edges where neighbor value is NA
    valid_edges <- edge_dt[!is.na(val)]

    if (nrow(valid_edges) > 0) {
      agg <- valid_edges[, .(nmax = max(val), nmin = min(val)), by = from]

      # Initialize with NA
      neighbor_max <- rep(NA_real_, N_rows)
      neighbor_min <- rep(NA_real_, N_rows)

      neighbor_max[agg$from] <- agg$nmax
      neighbor_min[agg$from] <- agg$nmin
    } else {
      neighbor_max <- rep(NA_real_, N_rows)
      neighbor_min <- rep(NA_real_, N_rows)
    }

    # --- Assign columns (same naming convention as original) ---
    max_col  <- paste0("n_max_",  var_name)
    min_col  <- paste0("n_min_",  var_name)
    mean_col <- paste0("n_mean_", var_name)

    dt[, (max_col)  := neighbor_max]
    dt[, (min_col)  := neighbor_min]
    dt[, (mean_col) := neighbor_mean]
  }

  # Clean up the temporary edge val column
  edge_dt[, val := NULL]

  # -------------------------------------------------------------------------
  # STEP 6: Restore original row order and return as data.frame
  # -------------------------------------------------------------------------
  setorder(dt, orig_row_idx__)
  dt[, c("orig_row_idx__", "cell_idx__", "year_idx__") := NULL]

  return(as.data.frame(dt))
}


# =============================================================================
# USAGE — drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_and_apply_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names and numerical values are identical to the original.
```

---

## Why This Works and Performance Estimate

| Aspect | Original | Optimized |
|---|---|---|
| **String constructions** | ~51.6M (`paste` calls inside 6.46M-iteration `lapply`) | **0** |
| **Named-vector lookups on 6.46M-element vector** | ~25.8M | **0** |
| **R-level loop iterations** | 6.46M (in `build_neighbor_lookup`) + 5 × 6.46M (in `compute_neighbor_stats`) | **0** (fully vectorized) |
| **Core operation for mean** | R `lapply` + per-element subsetting | **Sparse matrix multiply** (`Matrix::crossprod`-level C code) |
| **Core operation for max/min** | R `lapply` + per-element subsetting | **`data.table` grouped aggregation** (radix-sort + single pass, in C) |
| **Memory for edge list** | 6.46M-element list of integer vectors (high overhead) | 2 integer vectors of ~38.5M each (~308 MB) + sparse matrix |
| **Estimated wall time** | 86+ hours | **~5–15 minutes** |

### Why numerical equivalence is preserved

- The sparse matrix `W` encodes exactly the same directed neighbor relationships as the original `build_neighbor_lookup`.
- The NA-handling logic (exclude NAs before computing max/min/mean) is replicated exactly: for mean, we sum non-NA values and divide by the non-NA count; for max/min, we filter out NA edges before grouping.
- The column names follow the same `n_max_`, `n_min_`, `n_mean_` convention, so the trained Random Forest model's feature references remain valid.
- The original row order is restored via `orig_row_idx__` before returning.