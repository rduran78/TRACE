 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized concatenation step per variable — called only 5 times total. That is not where 86+ hours is spent.

**The true bottleneck is `build_neighbor_lookup()`.**

Specifically, inside the `lapply` over all ~6.46 million rows:

```r
lapply(row_ids, function(i) {
  ref_idx           <- id_to_ref[as.character(data$id[i])]
  neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
  neighbor_keys     <- paste(neighbor_cell_ids, data$year[i], sep = "_")
  result            <- idx_lookup[neighbor_keys]
  as.integer(result[!is.na(result)])
})
```

For **every single row** (6.46M iterations), this function:

1. Converts `data$id[i]` to character and does a named-vector lookup (`as.character` + named indexing) — **O(1) amortized but with per-call overhead**.
2. Extracts the neighbor cell IDs from the `nb` object.
3. Calls `paste()` to build string keys for every neighbor of that row.
4. Does named-vector lookup (`idx_lookup[neighbor_keys]`) against a 6.46M-element named character vector for each neighbor key.

With ~1,373,394 directed neighbor relationships spread across 344,208 cells, the average cell has ~4 rook neighbors. Over 28 years, each cell-year row triggers ~4 `paste()` calls and ~4 lookups into a 6.46M-length named vector. That is **~25.8 million string constructions and named-vector lookups**, all inside a sequential R `lapply` with per-element R-interpreter overhead. Named vector lookup in R on a vector of length 6.46M uses hashing, but the repeated per-element R-level overhead across 6.46M iterations dominates runtime catastrophically.

`compute_neighbor_stats()` by contrast is a simple numeric `lapply` — index into a numeric vector, compute `max/min/mean` — and `do.call(rbind, ...)` on the result. This is comparatively cheap.

**Conclusion:** The bottleneck is the row-level string-key construction and lookup in `build_neighbor_lookup()`, not the `rbind` or list binding in `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Eliminate per-row string pasting and named-vector lookup entirely.** Instead, exploit the panel structure: every cell appears once per year, so if we sort/index by `(id, year)`, we can compute the neighbor lookup using pure integer arithmetic.

2. **Vectorize `build_neighbor_lookup()`:** Pre-build a mapping from cell ID → set of row indices (one per year), then for each cell, its neighbors' row indices in a given year are directly retrievable by integer indexing. We build the full lookup using a single pass over cells (344K) rather than rows (6.46M), and expand across years with vectorized operations.

3. **Vectorize `compute_neighbor_stats()`:** Replace the `lapply` + `do.call(rbind, ...)` with a sparse-matrix or grouped vectorized approach using `data.table` for the aggregation step, avoiding per-row R function calls entirely.

4. **Preserve the trained Random Forest model** — we only change feature engineering, producing numerically identical columns.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# OPTIMIZED build_neighbor_lookup
# ===========================================================================
# Instead of returning a list-of-integer-vectors (one per row), we return
# a data.table of (row_idx, neighbor_row_idx) pairs — an edge list
# representation that enables fully vectorized aggregation.
#
# This replaces both build_neighbor_lookup() AND compute_neighbor_stats().
# ===========================================================================

build_neighbor_edge_list <- function(data, id_order, neighbors) {
  # Convert data to data.table if not already; work on a copy to avoid
  # mutating the original during construction.
  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # --- Step 1: Map cell id -> position in id_order (ref_idx) ---------------
  id_map <- data.table(
    cell_id = as.integer(id_order),
    ref_idx = seq_along(id_order)
  )

  # --- Step 2: Build the neighbor edge list at the CELL level ---------------
  # neighbors is an nb object: a list of integer vectors (indices into id_order)
  # We expand it into a two-column data.table: (ref_idx, neighbor_ref_idx)
  n_neighbors <- lengths(neighbors)
  cell_edge <- data.table(
    ref_idx          = rep(seq_along(neighbors), times = n_neighbors),
    neighbor_ref_idx = unlist(neighbors, use.names = FALSE)
  )

  # Map ref_idx -> cell_id
  cell_edge[, cell_id          := id_order[ref_idx]]
  cell_edge[, neighbor_cell_id := id_order[neighbor_ref_idx]]

  # Drop helper columns
  cell_edge[, c("ref_idx", "neighbor_ref_idx") := NULL]

  # --- Step 3: Expand to row-level by joining on year -----------------------
  # For each cell-year row, its neighbors are the neighbor_cell_id rows
  # in the SAME year.

  # Build a lean lookup: (cell_id, year) -> row_idx
  row_lookup <- dt[, .(cell_id = as.integer(id), year, row_idx)]

  # Join: for every (cell_id, neighbor_cell_id) pair, cross with every year
  # that cell_id appears in, then find the neighbor's row in that same year.

  # First, attach the focal row's year and row_idx
  # cell_edge has (cell_id, neighbor_cell_id)
  # row_lookup has (cell_id, year, row_idx)
  setkey(cell_edge, cell_id)
  setkey(row_lookup, cell_id)

  # This join gives one record per (focal_row, neighbor_cell, year)
  edge_expanded <- cell_edge[row_lookup,
    on = "cell_id",
    .(focal_row_idx   = i.row_idx,
      focal_year      = i.year,
      neighbor_cell_id = x.neighbor_cell_id),
    allow.cartesian = TRUE,
    nomatch = NULL
  ]

  # Now find the neighbor's row_idx in the same year
  setnames(row_lookup, c("cell_id", "year", "row_idx"),
                       c("neighbor_cell_id", "focal_year", "neighbor_row_idx"))
  setkey(edge_expanded, neighbor_cell_id, focal_year)
  setkey(row_lookup, neighbor_cell_id, focal_year)

  edge_final <- row_lookup[edge_expanded,
    on = c("neighbor_cell_id", "focal_year"),
    .(focal_row_idx    = i.focal_row_idx,
      neighbor_row_idx = x.neighbor_row_idx),
    nomatch = NA_integer_
  ]

  # Drop edges where the neighbor row was not found (boundary / missing year)
  edge_final <- edge_final[!is.na(neighbor_row_idx)]

  edge_final
}

# ===========================================================================
# OPTIMIZED compute_neighbor_stats (fully vectorized via data.table)
# ===========================================================================

compute_neighbor_stats_vec <- function(data, edge_list, var_name) {
  # edge_list: data.table with columns (focal_row_idx, neighbor_row_idx)
  vals <- data[[var_name]]
  n    <- if (is.data.frame(data)) nrow(data) else length(vals)

  # Attach neighbor values
  el <- copy(edge_list)
  el[, neighbor_val := vals[neighbor_row_idx]]

  # Remove NA neighbor values
  el <- el[!is.na(neighbor_val)]

  # Aggregate per focal row
  agg <- el[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = focal_row_idx]

  # Initialize result columns with NA
  res_max  <- rep(NA_real_, n)
  res_min  <- rep(NA_real_, n)
  res_mean <- rep(NA_real_, n)

  res_max[agg$focal_row_idx]  <- agg$nb_max
  res_min[agg$focal_row_idx]  <- agg$nb_min
  res_mean[agg$focal_row_idx] <- agg$nb_mean

  list(nb_max = res_max, nb_min = res_min, nb_mean = res_mean)
}

# ===========================================================================
# OPTIMIZED compute_and_add_neighbor_features
# ===========================================================================

compute_and_add_neighbor_features_vec <- function(data, var_name, edge_list) {
  stats <- compute_neighbor_stats_vec(data, edge_list, var_name)
  data[[paste0(var_name, "_nb_max")]]  <- stats$nb_max
  data[[paste0(var_name, "_nb_min")]]  <- stats$nb_min
  data[[paste0(var_name, "_nb_mean")]] <- stats$nb_mean
  data
}

# ===========================================================================
# OPTIMIZED OUTER PIPELINE
# ===========================================================================

# Build the edge list ONCE (replaces build_neighbor_lookup)
message("Building neighbor edge list...")
edge_list <- build_neighbor_edge_list(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge list: %s rows", format(nrow(edge_list), big.mark = ",")))

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  cell_data <- compute_and_add_neighbor_features_vec(
    cell_data, var_name, edge_list
  )
}

message("Done. Proceed to prediction with the existing trained Random Forest model.")
# The trained RF model is untouched; use predict(rf_model, cell_data) as before.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup()`: 6.46M R-level iterations with `paste()` + named-vector lookup on a 6.46M-element vector | Eliminated entirely |
| **Lookup construction** | `lapply` over 6.46M rows, string key construction | Single vectorized `data.table` join over 344K cells × 28 years |
| **Neighbor stats** | `lapply` over 6.46M rows + `do.call(rbind, ...)` | Vectorized `data.table` grouped aggregation on edge list |
| **Expected runtime** | ~86+ hours | Minutes (edge list build) + seconds per variable (aggregation) |
| **Numerical equivalence** | — | Identical `max`, `min`, `mean` per row per variable |
| **Trained RF model** | Preserved | Preserved (untouched) |

The colleague's diagnosis was wrong: `do.call(rbind, ...)` is a minor cost. The catastrophic bottleneck is the per-row string construction and named-vector lookup in `build_neighbor_lookup()`, which is eliminated by the vectorized `data.table` join approach above.