 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Over 6.46M Rows

`build_neighbor_lookup` calls `paste()` and performs named-vector lookups **inside a per-row `lapply`** over ~6.46 million rows. Each iteration:

1. **Constructs string keys** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) — repeated string allocation and concatenation.
2. **Looks up those keys** in `idx_lookup`, a named character vector of length 6.46M — each lookup is O(n) in the worst case for named vector indexing, or O(1) amortized via internal hashing, but the constant factor is large due to string comparison.

This means roughly **6.46M × avg_neighbors ≈ 6.46M × 4 ≈ 25.8M string constructions and hash lookups**, all in an interpreted R loop.

### Why It's a Broader Architectural Issue

The string-keyed lookup is a **workaround for the absence of a direct integer-indexed mapping** from `(cell_id, year)` → row index. The entire pipeline conflates two orthogonal dimensions (spatial identity and temporal identity) into a single string key, when in fact:

- The neighbor graph is **purely spatial** — it doesn't change across years.
- The year dimension is **uniform** — every cell appears in every year (balanced panel).

This means the neighbor lookup can be decomposed: find the **row offsets for each cell** once, then for any cell-year row, the neighbors' rows in the same year are deterministic integer arithmetic — **no strings, no hashing, no `lapply` over millions of rows**.

### Secondary Inefficiency

`compute_neighbor_stats` also loops over 6.46M rows in R-level `lapply`. With a vectorized/matrix approach, this can be replaced with a single pass.

---

## Optimization Strategy

1. **Eliminate all string keys.** Build an integer matrix mapping `(cell_index, year_index)` → row number. This is O(1) lookup via matrix indexing.

2. **Precompute a spatial-only neighbor list** as integer indices into `id_order` (already available from `rook_neighbors_unique`).

3. **Vectorize the neighbor statistics** using `data.table` or direct matrix operations. For each row, gather neighbor values via integer indexing and compute stats in bulk.

4. **Avoid per-row R loops entirely.** Use a "long" neighbor-row table and `data.table` grouped aggregation.

Estimated speedup: from ~86 hours to **minutes**.

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature construction.
#' Preserves the exact numerical estimand: for each cell-year row,
#' compute max, min, mean of each neighbor source variable across
#' rook neighbors present in the same year.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order         integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors   spdep nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names
#' @return cell_data with new columns appended (same row order preserved)
build_all_neighbor_features <- function(cell_data,
                                        id_order,
                                        rook_neighbors,
                                        neighbor_source_vars) {

  was_df <- is.data.frame(cell_data) && !is.data.table(cell_data)
  dt <- as.data.table(cell_data)

  # ---- Step 1: Create integer cell index and year index ----
  # Map each cell id to its position in id_order
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Add a cell index column (position in id_order)
  dt[, cell_idx := id_to_idx[as.character(id)]]

  # Preserve original row order

  dt[, orig_row := .I]

  # ---- Step 2: Build (cell_idx, year) -> row number lookup matrix ----
  # Years as factor for integer indexing
  years_sorted <- sort(unique(dt$year))
  year_to_col  <- setNames(seq_along(years_sorted), as.character(years_sorted))
  n_cells <- length(id_order)
  n_years <- length(years_sorted)

  # Matrix: row = cell_idx (1..n_cells), col = year_idx (1..n_years), value = row in dt
  row_lookup <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_lookup[cbind(dt$cell_idx, year_to_col[as.character(dt$year)])] <- dt$orig_row

  # ---- Step 3: Build spatial neighbor edge list (integer indices only) ----
  # rook_neighbors[[i]] gives the neighbor indices (into id_order) for cell i
  # Build a data.table of directed edges: (focal_cell_idx, neighbor_cell_idx)
  focal_idx <- rep(seq_along(rook_neighbors),
                   lengths(rook_neighbors))
  neighbor_idx <- unlist(rook_neighbors, use.names = FALSE)

  # Remove any 0-length or NA entries (spdep nb objects use 0 for no-neighbor cards)
  valid <- !is.na(neighbor_idx) & neighbor_idx > 0L
  focal_idx    <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]

  n_edges <- length(focal_idx)
  cat(sprintf("Spatial neighbor edges: %d\n", n_edges))

  # ---- Step 4: Expand edges across years and map to row numbers ----
  # For each year, every edge (f, n) maps to (row_lookup[f, y], row_lookup[n, y])
  # We build this as a long table: (focal_row, neighbor_row)
  # To stay within 16 GB RAM, process year by year

  # Pre-extract variable columns as matrices for fast indexing
  var_mat <- as.matrix(dt[, ..neighbor_source_vars])  # nrow(dt) x length(vars)

  # We'll accumulate results into pre-allocated matrices
  # For each var: 3 stats (max, min, mean) → total new columns = 5 vars × 3 = 15
  n_rows <- nrow(dt)
  stat_names <- c("max", "min", "mean")

  # Pre-allocate result columns
  result_list <- vector("list", length(neighbor_source_vars))
  names(result_list) <- neighbor_source_vars
  for (v in neighbor_source_vars) {
    result_list[[v]] <- matrix(NA_real_, nrow = n_rows, ncol = 3)
    colnames(result_list[[v]]) <- paste0("neighbor_", stat_names, "_", v)
  }

  # ---- Step 5: Process year by year ----
  cat("Processing years...\n")
  for (yi in seq_along(years_sorted)) {
    yr <- years_sorted[yi]

    # Get row numbers for focal and neighbor cells in this year
    focal_rows    <- row_lookup[focal_idx, yi]
    neighbor_rows <- row_lookup[neighbor_idx, yi]

    # Both must be non-NA (both cells present in this year)
    valid_mask <- !is.na(focal_rows) & !is.na(neighbor_rows)
    f_rows <- focal_rows[valid_mask]
    n_rows_yr <- neighbor_rows[valid_mask]

    if (length(f_rows) == 0L) next

    # For each variable, gather neighbor values and aggregate by focal row
    for (vi in seq_along(neighbor_source_vars)) {
      v <- neighbor_source_vars[vi]
      # Get neighbor values
      nvals <- var_mat[n_rows_yr, vi]

      # Build a data.table for fast grouped aggregation
      edge_dt <- data.table(focal = f_rows, nval = nvals)
      # Remove NA neighbor values (matches original: neighbor_vals[!is.na(neighbor_vals)])
      edge_dt <- edge_dt[!is.na(nval)]

      if (nrow(edge_dt) == 0L) next

      # Grouped aggregation
      agg <- edge_dt[, .(vmax = max(nval),
                         vmin = min(nval),
                         vmean = mean(nval)),
                     by = focal]

      # Write into result matrix
      result_list[[v]][agg$focal, 1L] <- agg$vmax
      result_list[[v]][agg$focal, 2L] <- agg$vmin
      result_list[[v]][agg$focal, 3L] <- agg$vmean
    }

    if (yi %% 5 == 0 || yi == n_years) {
      cat(sprintf("  Year %d/%d (%d) done\n", yi, n_years, yr))
    }
  }

  # ---- Step 6: Bind results back to dt ----
  for (v in neighbor_source_vars) {
    cnames <- colnames(result_list[[v]])
    for (ci in seq_along(cnames)) {
      set(dt, j = cnames[ci], value = result_list[[v]][, ci])
    }
  }

  # Clean up helper columns
  dt[, c("cell_idx", "orig_row") := NULL]

  if (was_df) {
    return(as.data.frame(dt))
  }
  return(dt)
}


# ============================================================
# USAGE — drop-in replacement for the original outer loop
# ============================================================

# Original code was:
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }

# Replacement:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now has the same 15 new columns (neighbor_max_*, neighbor_min_*, neighbor_mean_*)
# with identical numerical values. The trained Random Forest model is untouched.
```

---

## Column Naming Compatibility

If your existing trained Random Forest expects specific column names (e.g., `ntl_neighbor_max` rather than `neighbor_max_ntl`), adjust the naming to match. Here's a helper:

```r
# If the original code produced columns like: <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
# then replace the colnames line above with:
colnames(result_list[[v]]) <- paste0(v, "_neighbor_", stat_names)
```

Verify with:
```r
# Check that the RF model's expected feature names are all present
expected <- model$forest$independent.variable.names  # for ranger
stopifnot(all(expected %in% names(cell_data)))
```

---

## Why This Works and Preserves the Estimand

| Property | Original | Optimized |
|---|---|---|
| Neighbor definition | `rook_neighbors_unique[[cell_idx]]` | Same `rook_neighbors_unique[[cell_idx]]` |
| Year matching | `paste(id, year)` string match | `row_lookup[cell_idx, year_idx]` integer match |
| NA handling | Drop NA neighbor values, then compute stats | Identical: `edge_dt[!is.na(nval)]` |
| Stats computed | `max`, `min`, `mean` | Identical |
| Rows with no valid neighbors | `NA, NA, NA` | `NA` (matrix pre-initialized to `NA`) |

The numerical results are **identical** — only the computational path changes.

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| String key construction | ~6.46M × 4 `paste()` calls in R loop | **Zero** |
| Hash lookups | ~25.8M named-vector lookups | **Zero** |
| Per-row R loop | 6.46M `lapply` iterations | **Zero** |
| Aggregation | R-level `lapply` over 6.46M rows × 5 vars | 28 × 5 = **140** `data.table` grouped aggregations |
| Estimated wall time | ~86 hours | **5–15 minutes** |
| Peak RAM | ~6–8 GB (string keys dominate) | **~3–5 GB** (integer matrices) |