 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` — **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` — fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — **allocates a new character vector per row** (~4 neighbor keys × 6.46M rows ≈ 25.8M paste operations).
4. Looks up `idx_lookup[neighbor_keys]` — **named-vector substring matching on a 6.46M-element named vector**, which is O(N) per probe in the worst case (hash collisions on 6.46M keys).

The total cost is roughly **6.46M × (paste + hash-probe on a 6.46M-key table)**, which is why it takes 86+ hours.

### Why It's a Broader Architectural Issue

The string-keying strategy conflates two independent dimensions (cell ID and year) into a single flat lookup. But the neighbor structure is **time-invariant** — cell A's neighbors don't change from year to year. The lookup should be factored into:

1. **A spatial step** (done once): for each cell, find its neighbor cell indices.
2. **A temporal broadcast** (vectorized): for each year, map cell indices to row indices via simple arithmetic.

This eliminates all string operations entirely.

## Optimization Strategy

**Key insight**: If the data is sorted by `(id, year)` — or we can create a mapping — then for a cell at position `k` in the cell-order (1-indexed among the 344,208 cells) observed across 28 years, its rows are at predictable positions. We can convert the spatial neighbor list into a **row-index neighbor list** using pure integer arithmetic.

**Steps:**

1. Sort data by `(id, year)` (or build an integer index).
2. Create a cell-to-offset map: cell `k` starts at row `(k-1)*T + 1` where `T` = number of years.
3. For each cell, its neighbor rows in year `t` are simply `(neighbor_offset - 1) * T + t`.
4. Vectorize `compute_neighbor_stats` using `data.table` or matrix operations.

**Complexity reduction**: From O(N × M) string operations (N=6.46M rows, M=avg neighbors) to O(C × M) integer operations (C=344K cells), broadcast across years with vectorized arithmetic. This is roughly **a 28× reduction** in the inner loop, plus eliminating all string allocation — realistically **100–1000× faster overall**.

## Working R Code

```r
library(data.table)

# =============================================================================
# STEP 0: Ensure data is a data.table sorted by (id, year)
# =============================================================================
cell_dt <- as.data.table(cell_data)

# Create a canonical cell ordering and year ordering
unique_ids   <- sort(unique(cell_dt$id))
unique_years <- sort(unique(cell_dt$year))
n_cells <- length(unique_ids)  # 344,208
n_years <- length(unique_years)  # 28

# Map each id to an integer index 1..n_cells
id_to_cidx <- setNames(seq_along(unique_ids), as.character(unique_ids))

# Map each year to an integer index 1..n_years
year_to_tidx <- setNames(seq_along(unique_years), as.character(unique_years))

# Add integer indices
cell_dt[, cidx := id_to_cidx[as.character(id)]]
cell_dt[, tidx := year_to_tidx[as.character(year)]]

# Sort by (cidx, tidx) so that row for cell c, year t is at position (c-1)*T + t
setorder(cell_dt, cidx, tidx)

# Verify the layout is dense and complete
stopifnot(nrow(cell_dt) == n_cells * n_years)
stopifnot(all(cell_dt$cidx == rep(seq_len(n_cells), each = n_years)))
stopifnot(all(cell_dt$tidx == rep(seq_len(n_years), times = n_cells)))

# =============================================================================
# STEP 1: Build spatial neighbor list in terms of cidx (integer, done ONCE)
# =============================================================================
# id_order is the original ordering used to build rook_neighbors_unique.
# rook_neighbors_unique[[k]] gives neighbor positions in id_order for the k-th
# element of id_order.

# Map id_order positions to cidx
id_order_to_cidx <- id_to_cidx[as.character(id_order)]

# Build neighbor list in cidx space
# For each cidx, which cidx values are its neighbors?
# First, map each id_order index to cidx, then translate neighbor references.

n_id_order <- length(id_order)
# neighbors_cidx will be a list of length n_cells, indexed by cidx
neighbors_cidx <- vector("list", n_cells)

for (k in seq_len(n_id_order)) {
  c_idx <- id_order_to_cidx[k]
  nb_in_id_order <- rook_neighbors_unique[[k]]
  if (length(nb_in_id_order) == 0L) {
    neighbors_cidx[[c_idx]] <- integer(0)
  } else {
    neighbors_cidx[[c_idx]] <- as.integer(id_order_to_cidx[nb_in_id_order])
  }
}

# =============================================================================
# STEP 2: Build row-index neighbor list using integer arithmetic
# =============================================================================
# Row for (cidx=c, tidx=t) is at position: (c - 1) * n_years + t
# For a given cell c with neighbors nb_1, nb_2, ..., nb_m,
# in year t the neighbor rows are: (nb_j - 1) * n_years + t
#
# We store this as a list of length n_cells*n_years.
# But even building 6.46M list elements is expensive in a loop.
#
# Better approach: for each variable, compute stats using VECTORIZED operations.
# =============================================================================

# =============================================================================
# STEP 3: Vectorized neighbor-stat computation (no per-row loop at all)
# =============================================================================

compute_neighbor_features_fast <- function(dt, var_name, neighbors_cidx,
                                           n_cells, n_years) {
  # Extract the variable as a matrix: rows = years (1..T), cols = cells (1..C)
  # Since dt is sorted by (cidx, tidx), the vector dt[[var_name]] is laid out as:
  #   cell1_year1, cell1_year2, ..., cell1_yearT, cell2_year1, ..., cell2_yearT, ...
  # Reshape to matrix: n_years rows x n_cells cols
  val_mat <- matrix(dt[[var_name]], nrow = n_years, ncol = n_cells)
  # val_mat[t, c] = value for cell c in year-index t

  # For each cell, gather neighbor columns and compute stats across neighbors

  # We'll compute three vectors of length n_cells * n_years:
  #   neighbor_max, neighbor_min, neighbor_mean

  nb_max  <- rep(NA_real_, n_cells * n_years)
  nb_min  <- rep(NA_real_, n_cells * n_years)
  nb_mean <- rep(NA_real_, n_cells * n_years)

  # Loop over cells (344K iterations — fast, no string ops)
  for (c_idx in seq_len(n_cells)) {
    nb <- neighbors_cidx[[c_idx]]
    if (length(nb) == 0L) next

    # nb_mat: n_years x length(nb) — all neighbor values across all years
    nb_mat <- val_mat[, nb, drop = FALSE]

    # Compute row-wise (i.e., per-year) stats
    # Using matrixStats for speed if available, otherwise base R
    row_start <- (c_idx - 1L) * n_years + 1L
    row_end   <- c_idx * n_years
    rows_idx  <- row_start:row_end

    if (ncol(nb_mat) == 1L) {
      nb_max[rows_idx]  <- nb_mat[, 1]
      nb_min[rows_idx]  <- nb_mat[, 1]
      nb_mean[rows_idx] <- nb_mat[, 1]
    } else {
      # Base R row-wise operations
      # For rows with all NA, these return appropriate values
      nb_max[rows_idx]  <- apply(nb_mat, 1, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else max(x)
      })
      nb_min[rows_idx]  <- apply(nb_mat, 1, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else min(x)
      })
      nb_mean[rows_idx] <- apply(nb_mat, 1, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else mean(x)
      })
    }
  }

  # Assign back to data.table
  dt[, paste0(var_name, "_neighbor_max")  := nb_max]
  dt[, paste0(var_name, "_neighbor_min")  := nb_min]
  dt[, paste0(var_name, "_neighbor_mean") := nb_mean]

  return(dt)
}

# =============================================================================
# STEP 3b: Even faster version using matrixStats (recommended)
# =============================================================================

compute_neighbor_features_fastest <- function(dt, var_name, neighbors_cidx,
                                              n_cells, n_years) {
  requireNamespace("matrixStats", quietly = TRUE)

  val_mat <- matrix(dt[[var_name]], nrow = n_years, ncol = n_cells)

  nb_max  <- rep(NA_real_, n_cells * n_years)
  nb_min  <- rep(NA_real_, n_cells * n_years)
  nb_mean <- rep(NA_real_, n_cells * n_years)

  for (c_idx in seq_len(n_cells)) {
    nb <- neighbors_cidx[[c_idx]]
    if (length(nb) == 0L) next

    nb_mat <- val_mat[, nb, drop = FALSE]
    rows_idx <- ((c_idx - 1L) * n_years + 1L):(c_idx * n_years)

    if (ncol(nb_mat) == 1L) {
      nb_max[rows_idx]  <- nb_mat[, 1]
      nb_min[rows_idx]  <- nb_mat[, 1]
      nb_mean[rows_idx] <- nb_mat[, 1]
    } else {
      # matrixStats handles NA via na.rm = TRUE and returns NA for all-NA rows
      nb_max[rows_idx]  <- matrixStats::rowMaxs(nb_mat, na.rm = TRUE)
      nb_min[rows_idx]  <- matrixStats::rowMins(nb_mat, na.rm = TRUE)
      nb_mean[rows_idx] <- matrixStats::rowMeans2(nb_mat, na.rm = TRUE)

      # matrixStats returns -Inf/Inf for all-NA rows with na.rm=TRUE; fix:
      all_na <- matrixStats::rowAlls(is.na(nb_mat))
      if (any(all_na)) {
        fix_idx <- rows_idx[all_na]
        nb_max[fix_idx]  <- NA_real_
        nb_min[fix_idx]  <- NA_real_
        nb_mean[fix_idx] <- NA_real_
      }
    }
  }

  dt[, paste0(var_name, "_neighbor_max")  := nb_max]
  dt[, paste0(var_name, "_neighbor_min")  := nb_min]
  dt[, paste0(var_name, "_neighbor_mean") := nb_mean]

  return(dt)
}

# =============================================================================
# STEP 4: Run for all neighbor source variables
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_dt <- compute_neighbor_features_fastest(
    cell_dt, var_name, neighbors_cidx, n_cells, n_years
  )
}

# =============================================================================
# STEP 5: Restore original row order if needed for the trained RF model
# =============================================================================
# If the original cell_data had a specific row order, restore it:
# Option A: If you saved original row indices
# cell_dt[, orig_order := ...]; setorder(cell_dt, orig_order)

# Option B: Restore to (id, year) order matching original
# setorder(cell_dt, id, year)  # or whatever the original order was

# Convert back to data.frame if the RF model expects one
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is untouched — only the input features
# are recomputed with identical numerical values.
```

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| **Inner loop iterations** | 6.46M (row-level) | 344K (cell-level) |
| **String `paste()` calls** | ~6.46M × 4 ≈ 25.8M | **0** |
| **Named-vector hash probes on 6.46M keys** | ~25.8M | **0** |
| **Stat computation** | R-level `lapply` per row | `matrixStats` C-level rowwise ops on small matrices |
| **Estimated time** | 86+ hours | **~5–15 minutes** |
| **Memory** | 6.46M-element named character vector (~500MB) | One `n_years × n_cells` numeric matrix per variable (~69MB) |
| **Numerical output** | max, min, mean per row | **Identical** max, min, mean per row |

### Why This Preserves the Estimand

The optimized code computes exactly the same quantity: for each cell-year row, it gathers the values of the same variable from the same rook neighbors in the same year, then takes `max`, `min`, and `mean` with the same NA-handling logic. The matrix reshaping is just a reindexing — no values are altered, aggregated differently, or approximated. The trained Random Forest model receives identical feature columns and requires no retraining.