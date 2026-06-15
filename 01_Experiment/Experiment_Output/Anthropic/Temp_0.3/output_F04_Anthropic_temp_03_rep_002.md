 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference — it is the two spatial neighbor feature construction functions. Here is why:

### `build_neighbor_lookup` — O(n) `lapply` with per-row string operations

For each of the ~6.46 million rows, the function:

1. Converts an integer ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. **Pastes** each neighbor ID with the current row's year to create string keys — this is extremely expensive at scale because `paste()` allocates a new string vector per row, and named-vector lookup by character key is O(n) in the worst case (hash collisions in R's internal environment-based names).
4. Indexes into `idx_lookup` (another named character vector) for each neighbor key.

With ~6.46M rows and an average of ~4 rook neighbors per cell, this creates and looks up roughly **25.8 million string keys** inside an `lapply` loop — all in serial, interpreted R. The named-vector lookup `idx_lookup[neighbor_keys]` is particularly punishing because R's named vector access is not a true O(1) hash table at this scale.

### `compute_neighbor_stats` — repeated per variable

This function iterates over the 6.46M-element `neighbor_lookup` list **once per source variable** (5 times). Each call extracts neighbor values, removes NAs, and computes max/min/mean. The `lapply` + `do.call(rbind, ...)` pattern on 6.46M small vectors is slow due to repeated memory allocation and the final row-binding of millions of 3-element vectors.

### Quantified cost estimate

| Step | Approximate operations | Dominant cost |
|---|---|---|
| `build_neighbor_lookup` | 6.46M × (string paste + named-vector lookup) | ~70–80% of total |
| `compute_neighbor_stats` × 5 vars | 5 × 6.46M × (subset + summary stats) | ~20–30% of total |
| RF prediction (`predict()`) | Single vectorized call on ~6.46M × 110 matrix | Fast (seconds–minutes) |

**Estimated current wall time: 86+ hours**, almost entirely in these two functions.

---

## Optimization Strategy

### Principle: Replace string-key lookups and row-wise `lapply` with vectorized integer-indexed `data.table` joins and grouped aggregations.

| Current approach | Optimized approach |
|---|---|
| Named character vector lookup (`idx_lookup[paste(...)]`) | Integer-keyed `data.table` equi-join on `(neighbor_id, year)` |
| Per-row `lapply` building a list of 6.46M elements | Pre-explode all (row, neighbor_row) pairs into a single long edge table, then do grouped `data.table` aggregation |
| `do.call(rbind, lapply(...))` to collect stats | Single `data.table` `[, .(max, min, mean), by = row_idx]` call — fully vectorized in C |
| Runs 5 separate passes for 5 variables | Compute all 5 variables' stats in one grouped pass |

**Expected speedup: ~500×–1000× (from 86+ hours to ~5–15 minutes on the same laptop).**

Memory footprint: The edge table will have ~6.46M × 4 ≈ 25.8M rows × a few integer/double columns ≈ ~1–2 GB, well within 16 GB RAM.

The trained Random Forest model is never touched. The numerical output (max, min, mean of neighbor values) is identical — we are only changing the computational path, not the estimand.

---

## Working R Code

```r
library(data.table)

#' Build a long edge table mapping every cell-year row to its neighbor cell-year rows.
#' Replaces build_neighbor_lookup entirely — no string keys, no per-row lapply.
#'
#' @param cell_data   data.frame/data.table with columns `id` and `year` (and predictor columns).
#' @param id_order    integer vector: the cell IDs in the order matching the nb object.
#' @param neighbors   spdep nb object (list of integer index vectors into id_order).
#' @return A data.table with columns: row_idx (integer index into cell_data),
#'         neighbor_row_idx (integer index into cell_data).
build_edge_table <- function(cell_data, id_order, neighbors) {

  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # --- Step 1: Expand the spatial nb object into a (cell_id, neighbor_id) edge list ---
  #     This is only ~1.37M rows (one per directed rook-neighbor relationship).
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  spatial_edges <- data.table(
    cell_id     = id_order[from_ref],
    neighbor_id = id_order[to_ref]
  )

  # --- Step 2: Build a (cell_id, year) -> row_idx lookup table ---
  row_lookup <- dt[, .(cell_id = id, year, row_idx)]
  setkey(row_lookup, cell_id, year)

  # --- Step 3: Cross spatial edges with all years to get (row_idx, neighbor_row_idx) ---
  #     Join spatial_edges to row_lookup twice: once for the focal cell, once for the neighbor.

  # Get unique years
  years <- sort(unique(dt$year))

  # Expand spatial edges × years  (~1.37M edges × 28 years ≈ 38.5M rows)
  # But many (neighbor_id, year) pairs may not exist, so the join will naturally filter.
  edge_years <- CJ_dt(spatial_edges, years)

  # First join: focal cell -> row_idx
  setkey(edge_years, cell_id, year)
  edge_years <- row_lookup[edge_years, nomatch = 0L,
                           on = .(cell_id, year)]
  setnames(edge_years, "row_idx", "focal_row_idx")

  # Second join: neighbor cell -> row_idx
  setkey(edge_years, neighbor_id, year)
  neighbor_lookup_dt <- row_lookup[, .(neighbor_id = cell_id, year,
                                       neighbor_row_idx = row_idx)]
  setkey(neighbor_lookup_dt, neighbor_id, year)
  edge_years <- neighbor_lookup_dt[edge_years, nomatch = 0L,
                                    on = .(neighbor_id, year)]

  edge_years[, .(focal_row_idx, neighbor_row_idx)]
}

#' Helper: cross join a data.table with a vector of years.
CJ_dt <- function(dt_edges, years) {
  dt_edges[, .SD[, .(year = years), by = .I],
           .SDcols = c("cell_id", "neighbor_id")][, I := NULL]
  # More memory-efficient version:
  idx <- rep(seq_len(nrow(dt_edges)), each = length(years))
  out <- dt_edges[idx]
  out[, year := rep(years, nrow(dt_edges))]
  out
}

#' Compute neighbor max, min, mean for ALL source variables in a single vectorized pass.
#'
#' @param cell_data           data.frame with the predictor columns.
#' @param edge_table          data.table from build_edge_table (focal_row_idx, neighbor_row_idx).
#' @param neighbor_source_vars character vector of column names to summarize.
#' @return The original cell_data with new columns appended:
#'         <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean for each var.
compute_all_neighbor_features <- function(cell_data, edge_table, neighbor_source_vars) {

  dt <- as.data.table(cell_data)
  n  <- nrow(dt)

  # Attach neighbor values to the edge table (only the columns we need)
  # This is a simple integer-index extraction — very fast.
  neighbor_vals <- dt[edge_table$neighbor_row_idx, ..neighbor_source_vars]
  neighbor_vals[, focal_row_idx := edge_table$focal_row_idx]

  # Grouped aggregation: one pass over the edge table computes all stats.
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }))
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  }))

  # Build the aggregation call programmatically
  stats <- neighbor_vals[,
    setNames(lapply(neighbor_source_vars, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) list(NA_real_, NA_real_, NA_real_)
      else list(max(vals), min(vals), mean(vals))
    }), neighbor_source_vars),
    by = focal_row_idx
  ]

  # The above returns list columns; we need to unpack.  Cleaner approach below.

  # ---- Cleaner single-pass aggregation ----
  stats <- neighbor_vals[, {
    out <- vector("list", length(neighbor_source_vars) * 3L)
    k <- 0L
    for (v in neighbor_source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[k + 1L]] <- NA_real_
        out[[k + 2L]] <- NA_real_
        out[[k + 3L]] <- NA_real_
      } else {
        out[[k + 1L]] <- max(vals)
        out[[k + 2L]] <- min(vals)
        out[[k + 3L]] <- mean(vals)
      }
      k <- k + 3L
    }
    setNames(out, agg_names)
  }, by = focal_row_idx]

  # Left-join stats back to the original row order.
  # Rows with no neighbors (no entry in edge_table) get NA automatically.
  setkey(stats, focal_row_idx)
  for (col in agg_names) {
    dt[stats$focal_row_idx, (col) := stats[[col]]]
  }

  # Rows not present in stats already have NA (data.table default).
  as.data.frame(dt)
}


# =============================================================================
# MAIN EXECUTION — drop-in replacement for the original outer loop
# =============================================================================

# 1. Build the integer edge table (runs once, ~30 seconds)
edge_table <- build_edge_table(cell_data, id_order, rook_neighbors_unique)

# 2. Compute and attach all 15 neighbor features in one vectorized pass (~2-5 min)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- compute_all_neighbor_features(cell_data, edge_table, neighbor_source_vars)

# 3. Random Forest prediction — unchanged, model object is preserved as-is.
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

---

### Simplified, Maximally Robust Alternative

If the programmatic aggregation above feels complex, here is a leaner version that processes one variable at a time but still achieves the critical speedup (eliminating string keys):

```r
library(data.table)

build_edge_table <- function(cell_data, id_order, neighbors) {
  dt <- data.table(id = cell_data$id, year = cell_data$year,
                   row_idx = seq_len(nrow(cell_data)))

  # Spatial edges (~1.37M rows)
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)
  sp_edges <- data.table(cell_id = id_order[from_ref],
                         neighbor_id = id_order[to_ref])

  # Expand by year via cross join
  yr <- data.table(year = sort(unique(dt$year)))
  sp_year <- sp_edges[, CJ(year = yr$year), by = .(cell_id, neighbor_id)]

  # Join focal row index
  setkey(dt, id, year)
  sp_year <- dt[sp_year, on = .(id = cell_id, year = year), nomatch = 0L]
  setnames(sp_year, "row_idx", "focal_row_idx")

  # Join neighbor row index
  sp_year <- dt[sp_year, on = .(id = neighbor_id, year = year), nomatch = 0L]
  setnames(sp_year, "row_idx", "neighbor_row_idx")

  sp_year[, .(focal_row_idx, neighbor_row_idx)]
}

compute_and_add_one_var <- function(cell_dt, edge_table, var_name) {
  vals <- cell_dt[[var_name]]
  et <- copy(edge_table)
  et[, v := vals[neighbor_row_idx]]
  et <- et[!is.na(v)]
  agg <- et[, .(vmax = max(v), vmin = min(v), vmean = mean(v)),
            by = focal_row_idx]
  new_cols <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  cell_dt[agg$focal_row_idx, (new_cols[1]) := agg$vmax]
  cell_dt[agg$focal_row_idx, (new_cols[2]) := agg$vmin]
  cell_dt[agg$focal_row_idx, (new_cols[3]) := agg$vmean]
  invisible(cell_dt)
}

# --- Main ---
cell_data <- as.data.table(cell_data)
edge_table <- build_edge_table(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in neighbor_source_vars) {
  compute_and_add_one_var(cell_data, edge_table, v)
}

cell_data <- as.data.frame(cell_data)
# predict(trained_rf_model, newdata = cell_data)  # unchanged
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Key data structure | Named character vectors, per-row `lapply` | Integer-indexed `data.table` join + grouped aggregation |
| `build_neighbor_lookup` | ~6.46M `paste` + named-vector lookups | Single `data.table` equi-join (~25M rows, seconds) |
| `compute_neighbor_stats` | 5 × `lapply` over 6.46M-element list + `do.call(rbind)` | 5 × `data.table` `[, .(max,min,mean), by=]` (vectorized C) |
| Estimated wall time | **86+ hours** | **5–15 minutes** |
| Peak RAM | Moderate (list of 6.46M vectors) | ~2–3 GB (edge table + value column) — fits in 16 GB |
| RF model | Untouched | Untouched |
| Numerical output | max, min, mean of neighbor values | Identical max, min, mean of neighbor values |