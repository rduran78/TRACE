 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup (one entry per cell-year row ≈ 6.46 million entries), even though the **neighbor topology is static across all 28 years**. The function pastes cell IDs and years together into string keys, performs string-based hash lookups for every neighbor of every row, and repeats all of this identically for every year a cell appears. This means:

1. **Redundant topology resolution:** The neighbor graph has ~344K cells and ~1.37M directed neighbor pairs. But because the lookup is built at the cell-year level, the function effectively resolves ~1.37M × 28 ≈ 38.5 million string-key lookups, all to rediscover the same spatial relationships year after year.

2. **String key overhead:** Using `paste(id, year, sep="_")` as hash keys for ~6.46M rows is memory-intensive and slow. Every call to `idx_lookup[neighbor_keys]` is an O(1)-amortized named-vector lookup, but with millions of string keys the constant factor is large.

3. **Row-level `lapply` in `compute_neighbor_stats`:** Iterating over 6.46 million rows in R-level `lapply`, subsetting a numeric vector, and computing `max/min/mean` per row is extremely slow — this is called 5 times (once per variable), yielding ~32.3 million R-level function calls.

4. **No vectorization:** The entire pipeline is scalar R loops over millions of elements, with no use of vectorized or matrix operations.

**Summary:** The static neighbor topology is being re-resolved per year, and all aggregation is done in scalar R loops. This is why the pipeline takes 86+ hours.

---

## Optimization Strategy

### Key Insight: Separate Static Topology from Dynamic Values

The neighbor relationship among cells **never changes** across years. Only the variable values attached to cells change by year. Therefore:

1. **Build the neighbor lookup once at the cell level (344K cells), not at the cell-year level (6.46M rows).** Store it as a simple integer-index structure mapping each cell's position (in a canonical cell ordering) to its neighbors' positions.

2. **For each year, extract a numeric vector of values (one per cell in canonical order), then use the static cell-level neighbor index to compute neighbor max/min/mean in a vectorized fashion.**

3. **Use `data.table` for fast split-by-year and column assignment**, and use a **sparse adjacency matrix** (`Matrix` package) to compute neighbor means via matrix-vector multiplication — turning the mean computation into a single sparse matrix multiply per variable per year.

4. **For max and min**, use a compiled C++-level row-wise sparse operation via `Rcpp`, or fall back to a tight vectorized R loop over cells (344K iterations, not 6.46M). Since 344K × 28 = 9.6M and each iteration is trivial, even pure R is fast if the inner loop is vectorized.

### Concrete Plan

- **Step A:** Build a cell-level adjacency list once (344K entries) from `rook_neighbors_unique`. Also build a sparse binary adjacency matrix `W` (344K × 344K) for mean computation.
- **Step B:** Organize data so that for each year, we can extract a vector of length 344K (one value per cell in canonical order) and write back three columns.
- **Step C:** For each variable × year:
  - **Mean:** `W %*% x / degree` (one sparse mat-vec multiply).
  - **Max/Min:** Vectorized over the 344K-length adjacency list using `vapply` (344K iterations, not 6.46M).
- **Step D:** Assign results back into the data.

**Expected speedup:** From ~86 hours to ~5–15 minutes. The bottleneck moves from 6.46M R-level iterations with string hashing to 28 sparse matrix multiplies (each ~1.37M nonzeros) and 28 × 344K vectorized operations per variable.

---

## Working R Code

```r
library(data.table)
library(Matrix)

#' Redesigned neighbor feature computation.
#' Separates static topology from dynamic (year-varying) values.
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order        integer vector of cell IDs in the canonical order matching rook_neighbors_unique
#' @param rook_neighbors_unique  spdep nb object (list of integer index vectors)
#' @param neighbor_source_vars   character vector of variable names
#' @return cell_data with neighbor feature columns added

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # ---------------------------------------------------------------
  # STEP A: Build static cell-level topology (done ONCE)
  # ---------------------------------------------------------------
  n_cells <- length(id_order)

  # Cell-level adjacency list: list of integer vectors (indices into id_order)
  # rook_neighbors_unique is already in this form from spdep
  cell_adj <- rook_neighbors_unique  # cell_adj[[i]] = integer vector of neighbor indices for cell i

  # Remove any 0-length entries' NA markers that spdep uses for islands
  cell_adj <- lapply(cell_adj, function(nb) {
    nb <- nb[nb != 0L]  # spdep uses 0 for no-neighbor sentinel in some versions
    nb
  })

  # Build sparse adjacency matrix W (n_cells x n_cells) for fast mean computation
  # Each row i has 1s in columns cell_adj[[i]]
  from_idx <- rep(seq_len(n_cells), lengths(cell_adj))
  to_idx   <- unlist(cell_adj, use.names = FALSE)

  W <- sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = 1,
    dims = c(n_cells, n_cells)
  )

  # Degree vector (number of neighbors per cell) for computing mean = (W %*% x) / degree

  degree <- diff(W@p)  # for dgCMatrix, this gives the number of nonzeros per row

  # Actually for row-compressed we need rowSums:
  degree <- as.numeric(Matrix::rowSums(W))
  degree[degree == 0] <- NA_real_  # cells with no neighbors -> NA

  # ---------------------------------------------------------------
  # STEP B: Organize data as data.table, keyed by (id, year)
  # ---------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Create a mapping from cell id -> canonical index (position in id_order)
  id_to_canon <- setNames(seq_len(n_cells), as.character(id_order))

  # Add canonical index column
  cell_data[, canon_idx := id_to_canon[as.character(id)]]

  # Get sorted unique years

  years <- sort(unique(cell_data$year))

  # Pre-allocate output columns with NA
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)
  }

  # Key for fast subsetting by year
  setkey(cell_data, year, canon_idx)

  # ---------------------------------------------------------------
  # STEP C: For each year x variable, compute neighbor stats
  # ---------------------------------------------------------------
  for (yr in years) {

    # Get the row indices in cell_data for this year
    yr_rows <- cell_data[.(yr), which = TRUE]

    # Get the canonical indices for these rows (should be 1:n_cells if panel is balanced)
    yr_canon <- cell_data$canon_idx[yr_rows]

    for (var_name in neighbor_source_vars) {

      max_col  <- paste0("neighbor_max_", var_name)
      min_col  <- paste0("neighbor_min_", var_name)
      mean_col <- paste0("neighbor_mean_", var_name)

      # Build a full-length vector (length n_cells) of this variable for this year
      # Initialize with NA (handles unbalanced panels / missing cells)
      x_full <- rep(NA_real_, n_cells)
      x_full[yr_canon] <- cell_data[[var_name]][yr_rows]

      # --- MEAN via sparse matrix multiplication ---
      # neighbor_sum = W %*% x_full
      # We need to handle NAs: replace NA with 0 for the multiply, track valid counts
      x_for_mult <- x_full
      x_valid    <- as.numeric(!is.na(x_full))
      x_for_mult[is.na(x_for_mult)] <- 0

      neighbor_sum   <- as.numeric(W %*% x_for_mult)
      neighbor_count <- as.numeric(W %*% x_valid)
      neighbor_count[neighbor_count == 0] <- NA_real_

      neighbor_mean <- neighbor_sum / neighbor_count

      # --- MAX and MIN via vectorized vapply over cell adjacency list ---
      neighbor_max <- rep(NA_real_, n_cells)
      neighbor_min <- rep(NA_real_, n_cells)

      # Only iterate over cells that have neighbors AND are present this year
      cells_with_neighbors <- which(lengths(cell_adj) > 0)

      # Vectorized approach: extract all neighbor values at once, then split
      # This avoids per-cell R function call overhead for large n
      adj_lengths <- lengths(cell_adj[cells_with_neighbors])
      all_neighbor_idx <- unlist(cell_adj[cells_with_neighbors], use.names = FALSE)
      all_neighbor_vals <- x_full[all_neighbor_idx]

      # Create a grouping factor
      group <- rep(cells_with_neighbors, adj_lengths)

      # Use data.table for fast grouped max/min
      dt_tmp <- data.table(
        cell  = group,
        val   = all_neighbor_vals
      )

      # Remove NA values before aggregation
      dt_tmp <- dt_tmp[!is.na(val)]

      if (nrow(dt_tmp) > 0) {
        agg <- dt_tmp[, .(nmax = max(val), nmin = min(val)), by = cell]
        neighbor_max[agg$cell] <- agg$nmax
        neighbor_min[agg$cell] <- agg$nmin
      }

      # --- Write results back to cell_data for this year's rows ---
      set(cell_data, i = yr_rows, j = max_col,  value = neighbor_max[yr_canon])
      set(cell_data, i = yr_rows, j = min_col,  value = neighbor_min[yr_canon])
      set(cell_data, i = yr_rows, j = mean_col, value = neighbor_mean[yr_canon])
    }

    # Optional: progress reporting
    message(sprintf("Year %d complete.", yr))
  }

  # Clean up helper column
  cell_data[, canon_idx := NULL]

  return(cell_data)
}

# ---------------------------------------------------------------
# OUTER CALL (replaces the original outer loop)
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# ---------------------------------------------------------------
# Prediction with the pre-trained Random Forest (UNCHANGED)
# ---------------------------------------------------------------
# The trained RF model is loaded from disk and used as-is.
# The column names produced above (neighbor_max_*, neighbor_min_*, neighbor_mean_*)
# match the original pipeline's output, preserving the original numerical estimand.
#
# Example (unchanged from original pipeline):
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same numerical results** | The neighbor max, min, and mean are computed from exactly the same neighbor sets and the same variable values. The sparse matrix `W` encodes the identical rook adjacency. The NA-handling logic (exclude NAs, return NA if no valid neighbors) is preserved. |
| **Same column names** | Output columns are named `neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>` — matching the original `compute_and_add_neighbor_features` output. |
| **Trained RF model untouched** | No retraining occurs. The model object is loaded and called with `predict()` on the enriched `cell_data` exactly as before. |

## Performance Estimate

| Operation | Original | Redesigned |
|---|---|---|
| Topology resolution | 38.5M string lookups | 1 sparse matrix construction (~1.37M entries) |
| Mean computation | 6.46M R-level loops × 5 vars | 28 × 5 = 140 sparse mat-vec multiplies |
| Max/Min computation | 6.46M R-level loops × 5 vars | 28 × 5 = 140 `data.table` grouped aggregations over ~1.37M rows each |
| **Estimated total time** | **86+ hours** | **~5–15 minutes** |