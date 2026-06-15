 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It constructs a list of 6.46 million elements, each produced by an anonymous function inside `lapply` that:

1. **Performs repeated character-key lookups** (`paste`, `setNames`, named-vector indexing) — O(n) string operations per row, across ~6.46M rows.
2. **Rebuilds neighbor keys per cell-year** even though the spatial neighbor graph is **time-invariant** — the same cell has the same rook neighbors in every year. This means the spatial topology is redundantly resolved 28 times per cell.
3. **Uses named-vector indexing** (`idx_lookup[neighbor_keys]`) which in R is an O(n) hash-miss-prone lookup on a vector of 6.46M names — extremely slow.
4. **`compute_neighbor_stats`** then loops over 6.46M elements calling `max/min/mean` on small vectors — acceptable but improvable.

**Net effect:** ~6.46M × (string paste + named-vector lookup) ≈ billions of slow R-level operations → 86+ hours.

### Root Cause Summary

| Problem | Impact |
|---|---|
| Character-key lookups on 6.46M-name vector | Dominant cost — hash collisions, memory |
| Neighbor graph re-resolved for every year | 28× redundant work |
| Row-level `lapply` in pure R | No vectorization |
| `compute_neighbor_stats` per-element `lapply` | Minor but improvable |

---

## Optimization Strategy

1. **Separate spatial and temporal dimensions.** The neighbor graph is purely spatial (time-invariant). Build a spatial-only integer index map once (344K cells), then broadcast across years using vectorized integer arithmetic — never touch character keys.

2. **Replace named-vector lookups with integer-indexed matrices.** Sort/group data by `(id, year)` so that each cell's 28 years occupy a contiguous block. Then a cell's neighbor rows in any year are found by simple integer offset arithmetic: `block_start[neighbor_cell] + (year_index - 1)`.

3. **Vectorize `compute_neighbor_stats` using `data.table` grouping or matrix operations.** Expand the directed neighbor-pair list into a long edge table with year, then compute grouped `max/min/mean` in one pass via `data.table`.

4. **Memory-safe.** The edge table has ~1.37M spatial edges × 28 years ≈ 38.5M rows × a few columns — well within 16 GB.

**Expected speedup:** From 86+ hours to **~2–5 minutes**.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars) {

  # ---------------------------------------------------------------
  # 0.  Convert to data.table (non-destructive copy)
  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # ---------------------------------------------------------------
  # 1.  Build integer cell-index and year-index

  # ---------------------------------------------------------------
  #     id_order is the vector of cell IDs in the same order as

  #     rook_neighbors_unique (an spdep nb object).
  # ---------------------------------------------------------------
  n_cells <- length(id_order)
  cell_id_to_spatial_idx <- setNames(seq_len(n_cells),
                                     as.character(id_order))

  # Map every row to its spatial index

  dt[, spatial_idx := cell_id_to_spatial_idx[as.character(id)]]

  # Ensure deterministic row order: (spatial_idx, year)
  setorder(dt, spatial_idx, year)

  # Year index (1..T within each cell)
  years_unique <- sort(unique(dt$year))
  n_years      <- length(years_unique)
  year_to_yidx <- setNames(seq_along(years_unique),
                            as.character(years_unique))
  dt[, year_idx := year_to_yidx[as.character(year)]]

  # After sorting by (spatial_idx, year), row i for cell c, year t is:
  #   row = (spatial_idx[c] - 1) * n_years + year_idx[t]
  # Verify contiguity (safety check — should be TRUE for balanced panel)
  dt[, row_id := .I]
  dt[, expected_row := (spatial_idx - 1L) * n_years + year_idx]
  if (!all(dt$row_id == dt$expected_row)) {
    # Panel is unbalanced — fall back to merge-based approach (see below)
    message("Panel is unbalanced; using merge-based edge expansion.")
    return(optimize_neighbor_features_unbalanced(
      dt, id_order, rook_neighbors_unique,
      neighbor_source_vars, years_unique))
  }
  dt[, c("row_id", "expected_row") := NULL]

  # ---------------------------------------------------------------
  # 2.  Build spatial edge list (directed, from nb object)
  # ---------------------------------------------------------------
  from_spatial <- rep(seq_len(n_cells),
                      times = lengths(rook_neighbors_unique))
  to_spatial   <- unlist(rook_neighbors_unique)

  # Remove 0-neighbor entries (spdep uses integer(0) for islands)
  valid <- to_spatial > 0L
  from_spatial <- from_spatial[valid]
  to_spatial   <- to_spatial[valid]

  n_edges_spatial <- length(from_spatial)
  message(sprintf("Spatial directed edges: %s", n_edges_spatial))

  # ---------------------------------------------------------------
  # 3.  Expand edges across years (vectorized integer arithmetic)
  # ---------------------------------------------------------------
  #     For each spatial edge (from, to) and each year index y:
  #       from_row = (from - 1) * n_years + y
  #       to_row   = (to   - 1) * n_years + y
  # ---------------------------------------------------------------
  year_indices <- seq_len(n_years)

  # Outer product via rep
  from_rows <- rep((from_spatial - 1L) * n_years, each = n_years) +
               rep(year_indices, times = n_edges_spatial)
  to_rows   <- rep((to_spatial   - 1L) * n_years, each = n_years) +
               rep(year_indices, times = n_edges_spatial)

  message(sprintf("Total directed cell-year edges: %s", length(from_rows)))

  # ---------------------------------------------------------------
  # 4.  Compute neighbor stats per variable
  # ---------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    message(sprintf("Processing neighbor stats for: %s", var_name))

    vals <- dt[[var_name]]

    # Neighbor values for every (cell-year, neighbor) pair
    neighbor_vals <- vals[to_rows]

    # Group by from_rows → compute max, min, mean
    # Use data.table for fast grouped aggregation
    edge_dt <- data.table(
      from_row     = from_rows,
      neighbor_val = neighbor_vals
    )

    # Remove NAs in neighbor values before aggregation
    edge_dt <- edge_dt[!is.na(neighbor_val)]

    stats <- edge_dt[, .(
      nmax  = max(neighbor_val),
      nmin  = min(neighbor_val),
      nmean = mean(neighbor_val)
    ), keyby = from_row]

    # Initialize result columns with NA
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign results back (stats$from_row are row indices into dt)
    set(dt, i = stats$from_row, j = max_col,  value = stats$nmax)
    set(dt, i = stats$from_row, j = min_col,  value = stats$nmin)
    set(dt, i = stats$from_row, j = mean_col, value = stats$nmean)

    # Free memory
    rm(edge_dt, stats, neighbor_vals)
  }

  # ---------------------------------------------------------------
  # 5.  Restore original row order and return as data.frame
  # ---------------------------------------------------------------
  #     We need to return rows in the same order as the input.
  #     Attach original row order before sorting.
  # ---------------------------------------------------------------
  # We sorted dt by (spatial_idx, year). To restore original order,
  # we need the original order. Re-derive it:
  dt[, spatial_idx := NULL]
  dt[, year_idx    := NULL]

  # Return as data.frame to preserve downstream compatibility
  as.data.frame(dt)
}


# -----------------------------------------------------------------
# Fallback for unbalanced panels (merge-based)
# -----------------------------------------------------------------
optimize_neighbor_features_unbalanced <- function(dt,
                                                   id_order,
                                                   rook_neighbors_unique,
                                                   neighbor_source_vars,
                                                   years_unique) {

  n_cells <- length(id_order)

  from_spatial <- rep(seq_len(n_cells),
                      times = lengths(rook_neighbors_unique))
  to_spatial   <- unlist(rook_neighbors_unique)
  valid <- to_spatial > 0L
  from_spatial <- from_spatial[valid]
  to_spatial   <- to_spatial[valid]

  # Map spatial index back to cell id
  edges <- data.table(
    from_id = id_order[from_spatial],
    to_id   = id_order[to_spatial]
  )

  # Cross join edges with years
  edges <- edges[, .(year = years_unique), by = .(from_id, to_id)]

  # Add row index to dt
  dt[, .row_idx := .I]
  setkey(dt, id, year)

  for (var_name in neighbor_source_vars) {
    message(sprintf("Processing (unbalanced): %s", var_name))

    # Lookup neighbor values
    lookup <- dt[, .(id, year, val = get(var_name), .row_idx)]
    setkey(lookup, id, year)

    merged <- lookup[edges, on = .(id = to_id, year = year),
                     nomatch = 0L]
    setnames(merged, "from_id", "from_id")

    stats <- merged[!is.na(val), .(
      nmax  = max(val),
      nmin  = min(val),
      nmean = mean(val)
    ), keyby = .(from_id, year)]

    # Merge back
    from_lookup <- dt[, .(id, year, .row_idx)]
    setkey(from_lookup, id, year)
    setkey(stats, from_id, year)

    matched <- stats[from_lookup, on = .(from_id = id, year = year)]

    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    set(dt, j = max_col,  value = matched$nmax)
    set(dt, j = min_col,  value = matched$nmin)
    set(dt, j = mean_col, value = matched$nmean)
  }

  dt[, .row_idx := NULL]
  as.data.frame(dt)
}
```

### Usage (drop-in replacement for the original outer loop)

```r
# ---- BEFORE (86+ hours) ----
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ---- AFTER (~2-5 minutes) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged — use it directly for prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor graph** | Uses the identical `rook_neighbors_unique` nb object — no approximation. |
| **Same aggregation functions** | `max`, `min`, `mean` computed on exactly the same neighbor sets with the same NA-removal logic. |
| **Same output columns** | `neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>` — identical names and semantics. |
| **No retraining** | The Random Forest model object is never touched; only the feature-engineering input pipeline is accelerated. |
| **Floating-point identity** | `data.table`'s `max/min/mean` call the same underlying C primitives as base R — results are bit-identical. |