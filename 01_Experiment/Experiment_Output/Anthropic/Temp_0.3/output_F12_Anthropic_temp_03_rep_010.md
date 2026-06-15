 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized bind operation per variable (only 5 calls total). Each inner function returns a fixed-length-3 numeric vector, so there is no "repeated list binding" inside `compute_neighbor_stats()` — it's a straightforward `lapply` returning pre-allocated length-3 vectors.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** Inside the `lapply`, for every single row `i`, the code:
   - Calls `as.character(data$id[i])` — 6.46M character coercions.
   - Looks up `id_to_ref[as.character(...)]` — 6.46M named-vector hash lookups against a 344K-element named vector.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` — for each row, pastes ~4 neighbor IDs (rook neighbors average ~4) with the year, creating ~25.8M paste operations total.
   - Looks up `idx_lookup[neighbor_keys]` — 25.8M named-vector lookups against a **6.46M-element** named vector. Named vector lookup in R is O(n) per probe in the worst case and uses hashing that becomes very slow at this scale.

2. **The `lapply` over 6.46M rows is inherently serial in R**, and each iteration does multiple allocations, string operations, and hash lookups. This is the dominant cost — likely 85+ of the 86+ hours.

3. `compute_neighbor_stats()` by contrast does only numeric subsetting (`vals[idx]`) and three simple aggregates per row. Even with 6.46M iterations × 5 variables, this is comparatively fast because it avoids string operations entirely.

**Root cause summary:** The pipeline builds 6.46 million per-row neighbor index lists using expensive per-row string concatenation and named-vector lookups against a 6.46M-key lookup table. This is an O(N × k × lookup_cost) operation where N = 6.46M, k ≈ 4, and lookup_cost is high due to R's named-vector hashing at scale.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()` entirely** — eliminate the per-row `lapply`. Instead, expand the neighbor relationships into a flat edge table (cell-year-row → neighbor-cell-year-row) using `data.table` joins, which use radix-based indexing rather than string hashing.

2. **Vectorize `compute_neighbor_stats()`** — use the flat edge table with `data.table` grouped aggregation (`[, .(max, min, mean), by = row_id]`) to compute all neighbor stats in one pass per variable, eliminating both the `lapply` and `do.call(rbind, ...)`.

3. **Preserve the trained Random Forest model** — we only change the feature-engineering pipeline, producing numerically identical columns. The RF model object is untouched.

4. **Preserve the original numerical estimand** — the optimized code computes the same `max`, `min`, and `mean` of non-NA neighbor values, yielding identical results.

Expected speedup: from 86+ hours to **minutes** (the dominant cost becomes a few `data.table` equi-joins on integer keys over ~26M edge rows).

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED build_neighbor_lookup (returns a data.table edge list, not a list)
# ==============================================================================
build_neighbor_edge_table <- function(data, id_order, neighbors) {
  # data must have columns: id, year (and be ordered as the original data.frame)
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  dt <- as.data.table(data[, c("id", "year")])
  dt[, row_idx := .I]  # preserve original row position

  # --- Step 1: Build a flat neighbor table at the cell level ---------------
  #   For each cell index j in id_order, expand its rook neighbors.
  nb_from <- rep(seq_along(neighbors), lengths(neighbors))
  nb_to   <- unlist(neighbors, use.names = FALSE)

  # Map cell indices back to cell IDs
  cell_edges <- data.table(
    from_id = id_order[nb_from],
    to_id   = id_order[nb_to]
  )
  # ~1.37M rows (directed rook-neighbor relationships)

  # --- Step 2: Cross with years to get row-level edges ---------------------
  #   For every (from_id, year) row in the data, find the row index of each
  #   (to_id, year) neighbor row.

  # Create keyed lookup: (id, year) -> row_idx
  setkey(dt, id, year)

  # Expand cell_edges × years via join on the "from" side to get the focal row_idx
  #   and on the "to" side to get the neighbor row_idx.

  # Join from side: attach focal row index
  edges <- cell_edges[dt, on = .(from_id = id), allow.cartesian = TRUE,
                      nomatch = NULL,
                      .(focal_row = i.row_idx,
                        to_id     = x.to_id,
                        year      = i.year)]

  # Join to side: attach neighbor row index
  edges <- dt[edges, on = .(id = to_id, year = year), nomatch = NA,
              .(focal_row    = i.focal_row,
                neighbor_row = x.row_idx)]

  # Drop edges where the neighbor cell-year doesn't exist in the data
  edges <- edges[!is.na(neighbor_row)]

  # Key for fast grouped operations later
  setkey(edges, focal_row)

  return(edges)
}

# ==============================================================================
# OPTIMIZED compute_neighbor_stats (vectorized via data.table grouping)
# ==============================================================================
compute_neighbor_stats_dt <- function(data_dt, edges, var_name) {
  # data_dt: data.table with a row_idx column (1:nrow)
  # edges:   data.table with (focal_row, neighbor_row)
  # var_name: character, column name to aggregate

  vals <- data_dt[[var_name]]

  # Attach neighbor values
  edge_vals <- edges[, .(focal_row, nval = vals[neighbor_row])]

  # Drop NAs in the variable (matches original: neighbor_vals[!is.na(...)])
  edge_vals <- edge_vals[!is.na(nval)]

  # Grouped aggregation
  stats <- edge_vals[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row]

  # Allocate full-length result (NA for rows with no valid neighbors)
  n <- nrow(data_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  out_max[stats$focal_row]  <- stats$nb_max
  out_min[stats$focal_row]  <- stats$nb_min
  out_mean[stats$focal_row] <- stats$nb_mean

  list(nb_max = out_max, nb_min = out_min, nb_mean = out_mean)
}

# ==============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# ==============================================================================
compute_and_add_neighbor_features_dt <- function(data_dt, var_name, edges) {
  stats <- compute_neighbor_stats_dt(data_dt, edges, var_name)

  # Column names must match what the trained RF model expects.
  # Typical naming convention: <var>_nb_max, <var>_nb_min, <var>_nb_mean
  # Adjust suffixes below if the original code used different names.
  set(data_dt, j = paste0(var_name, "_nb_max"),  value = stats$nb_max)
  set(data_dt, j = paste0(var_name, "_nb_min"),  value = stats$nb_min)
  set(data_dt, j = paste0(var_name, "_nb_mean"), value = stats$nb_mean)

  invisible(data_dt)
}

# ==============================================================================
# MAIN PIPELINE (drop-in replacement for the outer loop)
# ==============================================================================

# Convert to data.table (in-place if already); add row index
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]

# Build the edge table ONCE (replaces build_neighbor_lookup)
# This is the expensive step, but now vectorized: ~minutes, not days.
edges <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  compute_and_add_neighbor_features_dt(cell_dt, var_name, edges)
}

# Convert back to data.frame if downstream code / the RF predict() expects one
cell_data <- as.data.frame(cell_dt)
cell_data$row_idx <- NULL

# The trained Random Forest model is untouched.
# Predictions proceed exactly as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Colleague's Diagnosis | Correct Diagnosis |
|---|---|---|
| Claimed bottleneck | `do.call(rbind, ...)` and list binding in `compute_neighbor_stats()` | `build_neighbor_lookup()`: per-row `lapply` with string pasting and named-vector lookups over 6.46M rows against a 6.46M-key hash |
| Actual cost of claimed bottleneck | Minor (~5 calls, each binding 6.46M × 3 matrix) | N/A |
| True dominant cost | — | ~25.8M string-paste + named-vector hash probes inside a serial `lapply` |
| Fix | — | Replace both functions with vectorized `data.table` joins and grouped aggregation |
| Expected runtime | — | Minutes instead of 86+ hours |
| RF model impact | — | None; numerically identical features produced |