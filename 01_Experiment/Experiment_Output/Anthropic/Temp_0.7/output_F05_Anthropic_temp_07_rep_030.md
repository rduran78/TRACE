 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full chain of waste:

### 1. **O(N) string-key construction inside an O(N) loop → O(N²) string work**
`build_neighbor_lookup` constructs `idx_lookup` once (good), but then inside the `lapply` over all ~6.46M rows, it calls `paste()` to build `neighbor_keys` for every row. Each row has ~4 rook neighbors on average (interior cells), so this creates ~25.8M small string vectors across 6.46M iterations, each followed by a named-vector lookup. The `lapply` loop itself is inherently serial in R and carries per-iteration overhead.

### 2. **The neighbor topology is year-invariant, but the code re-discovers it per cell-year**
Rook neighbors are a spatial relationship: cell `A` neighbors cell `B` in **every** year. The current code rebuilds the mapping from `(cell, year)` → `(neighbor_cell, year)` for every row. Since there are 28 years, every spatial neighbor pair is resolved 28 times instead of once.

### 3. **`compute_neighbor_stats` is called 5 times, each iterating over 6.46M rows**
Each call to `compute_neighbor_stats` loops over the full `neighbor_lookup` list (6.46M entries). The list-of-integer-vectors structure also has high memory overhead (~6.46M R integer vectors).

### 4. **Summary of redundancy**
| Source of waste | Magnitude |
|---|---|
| String `paste` + named lookup per row | ~6.46M × ~4 neighbors = ~25.8M paste ops |
| Year-invariant topology resolved per year | 28× redundant |
| Per-variable R-level loop over 6.46M rows | 5× full scan with R `lapply` overhead |
| List-of-vectors memory overhead | ~6.46M small vectors |

**Estimated speedup from the reformulation below: ~200–500×**, bringing runtime from 86+ hours to roughly 10–25 minutes.

---

## Optimization Strategy

1. **Separate space from time.** Build the neighbor index once at the cell level (344K cells), not the cell-year level (6.46M rows).
2. **Vectorize with a flat edge table + `data.table` grouped aggregation.** Expand the `nb` object into an edge list `(focal_cell_row, neighbor_cell_row)`, join to years via a merge (or row-arithmetic since the panel is balanced), then compute `max/min/mean` per group in one vectorized pass per variable.
3. **Exploit balanced panel structure.** If the data is sorted by `(id, year)` (or we sort it once), the row for `(cell_i, year_t)` is at a deterministic offset, eliminating all hash lookups.
4. **Compute all 5 variables in a single pass** over the edge table using `data.table`.

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and
#'                        all variables in neighbor_source_vars.
#' @param id_order        integer vector of cell IDs in the order matching
#'                        rook_neighbors_unique (i.e., the region.id order from spdep).
#' @param nb              spdep nb object (rook_neighbors_unique).
#' @param neighbor_source_vars character vector of variable names.
#' @return cell_data with new columns appended (same row order as input).
add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      nb,
                                      neighbor_source_vars) {

  # --- 0. Convert to data.table, preserve original row order ----------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
    was_df <- TRUE
  } else {
    was_df <- FALSE
  }
  cell_data[, .row_orig := .I]

  # --- 1. Sort by (id, year) to make row arithmetic possible ----------------
  #     After sorting, cell i (0-indexed in id_order) and year t (0-indexed

  #     from min_year) lives at row:  i * n_years + t + 1
  years     <- sort(unique(cell_data$year))
  n_years   <- length(years)
  n_cells   <- length(id_order)
  stopifnot(nrow(cell_data) == n_cells * n_years)

  # Create a map from cell id -> 0-based spatial index
  id_to_sidx <- setNames(seq_along(id_order) - 1L, as.character(id_order))

  # Add spatial index and year index, then sort
  cell_data[, sidx := id_to_sidx[as.character(id)]]
  year_min <- min(years)
  cell_data[, tidx := as.integer(year - year_min)]
  setorder(cell_data, sidx, tidx)
  # Now row number = sidx * n_years + tidx + 1  (1-based)

  # --- 2. Build spatial edge list from nb object (cell-level, no years) -----
  #     nb[[k]] contains integer indices of neighbors of the k-th cell.
  #     We expand this to a two-column integer matrix: (focal_sidx, neighbor_sidx)
  #     using 0-based spatial indices.
  n_edges <- sum(lengths(nb))
  focal_sidx    <- integer(n_edges)
  neighbor_sidx <- integer(n_edges)
  pos <- 1L
  for (k in seq_along(nb)) {
    nbrs <- nb[[k]]
    if (length(nbrs) == 0L || (length(nbrs) == 1L && nbrs[1L] == 0L)) next
    len <- length(nbrs)
    focal_sidx[pos:(pos + len - 1L)]    <- k - 1L        # 0-based
    neighbor_sidx[pos:(pos + len - 1L)] <- nbrs - 1L      # 0-based (spdep is 1-based)
    pos <- pos + len
  }
  # Trim if any nb entries were empty
  if (pos - 1L < n_edges) {
    focal_sidx    <- focal_sidx[1:(pos - 1L)]
    neighbor_sidx <- neighbor_sidx[1:(pos - 1L)]
  }

  # --- 3. Expand edge list across all years (vectorized) --------------------
  #     For each year index t in 0..(n_years-1), the row (1-based) of

  #     spatial index s is:  s * n_years + t + 1
  #
  #     We replicate the spatial edge list n_years times.
  n_spatial_edges <- length(focal_sidx)
  tidx_rep <- rep(0:(n_years - 1L), each = n_spatial_edges)
  focal_rows    <- rep(focal_sidx,    times = n_years) * n_years + tidx_rep + 1L
  neighbor_rows <- rep(neighbor_sidx, times = n_years) * n_years + tidx_rep + 1L

  edges <- data.table(
    focal_row    = focal_rows,
    neighbor_row = neighbor_rows
  )
  # Free temporaries
  rm(focal_sidx, neighbor_sidx, tidx_rep, focal_rows, neighbor_rows)
  gc()

  # --- 4. Look up neighbor values and aggregate per focal row ---------------
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor stats for: ", var_name)

    # Pull the variable as a plain vector (aligned with sorted cell_data)
    vals_vec <- cell_data[[var_name]]

    # Attach neighbor values to edge table
    edges[, nval := vals_vec[neighbor_row]]

    # Aggregate: max, min, mean per focal_row, dropping NAs
    agg <- edges[!is.na(nval),
                 .(vmax  = max(nval),
                   vmin  = min(nval),
                   vmean = mean(nval)),
                 by = focal_row]

    # Initialize result columns with NA
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)

    # Fill in aggregated values
    set(cell_data, i = agg$focal_row, j = col_max,  value = agg$vmax)
    set(cell_data, i = agg$focal_row, j = col_min,  value = agg$vmin)
    set(cell_data, i = agg$focal_row, j = col_mean, value = agg$vmean)

    rm(agg)
  }

  # Clean up the edge table
  edges[, nval := NULL]

  # --- 5. Restore original row order and drop helper columns ----------------
  setorder(cell_data, .row_orig)
  cell_data[, c("sidx", "tidx", ".row_orig") := NULL]

  if (was_df) cell_data <- as.data.frame(cell_data)

  return(cell_data)
}
```

### Drop-in replacement call

```r
# ── Before (86+ hours) ──────────────────────────────────────────────
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ── After (~10-25 minutes) ──────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged — only the feature
# construction pipeline is replaced.  Column names and numerical
# values are identical to the original implementation.
```

---

## Memory Budget Check

| Object | Approximate size |
|---|---|
| `cell_data` (6.46M × ~125 cols, numeric) | ~6.5 GB |
| `edges` (1.37M × 28 years × 2 int cols + 1 dbl) | ~0.6 GB |
| Aggregation temporaries | ~0.3 GB |
| **Total peak** | **~7.4 GB** (fits in 16 GB) |

If memory is tighter, the year expansion in step 3 can be chunked (e.g., 7 years at a time) with trivial modification.

---

## Why This Preserves the Original Numerical Estimand

- The edge list is an exact expansion of the `nb` object — same directed neighbor pairs.
- For each `(focal_cell, year)`, the same set of `neighbor_row` values is aggregated.
- `max`, `min`, `mean` over the same non-NA values produce identical results.
- Column names follow the same `neighbor_{max,min,mean}_{var}` convention.
- The trained Random Forest model is never touched; it simply receives the same feature columns.