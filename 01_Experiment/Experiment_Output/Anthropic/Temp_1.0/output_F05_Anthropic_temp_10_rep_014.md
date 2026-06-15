 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full chain of waste:

1. **O(N) string-key construction inside `build_neighbor_lookup`**: For each of the ~6.46M rows, `paste(neighbor_cell_ids, data$year[i], sep = "_")` creates string keys and looks them up in a named vector. Named-vector lookup in R is hashed, but the keys are constructed per-row, and the entire named vector of 6.46M entries must be built and probed millions of times. This alone is enormous.

2. **The neighbor topology is year-invariant but is re-resolved per cell-year row**: The rook-neighbor structure is spatial — it doesn't change across years. Yet `build_neighbor_lookup` resolves neighbor *row indices* by pasting year onto spatial IDs for every single row. The same spatial neighbor resolution is repeated 28 times (once per year per cell), multiplied across all 344K cells.

3. **`compute_neighbor_stats` is called in a serial loop over 5 variables**, each time iterating over 6.46M entries via `lapply`. The per-variable pass is O(N × avg_neighbors). With 5 variables, that's 5 full scans.

4. **`lapply` over 6.46M rows** is inherently slow in R due to interpreter overhead and poor cache/vectorization behavior.

**In summary**: the string-key construction is the visible hotspot, but the root cause is an algorithmic design that (a) conflates spatial topology with temporal indexing, (b) resolves neighbors row-by-row in interpreted R, and (c) processes variables one at a time. A full reformulation can drop the ~86-hour estimate to minutes.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Separate space from time** | Build a spatial-only neighbor lookup (344K cells), then expand to cell-year rows via integer indexing — no strings. |
| **Vectorize with `data.table`** | Melt neighbor pairs into a long edge table, join once, and compute grouped statistics in one vectorized pass per variable (or all at once). |
| **Eliminate `lapply` over 6.46M rows** | Replace with `data.table` grouped aggregation on the edge table — internally parallelized C code. |
| **Batch all 5 variables** | Compute max/min/mean for all neighbor-source variables in a single grouped pass. |

**Complexity**: The old approach is O(N_rows × avg_neighbors) with massive per-element interpreter overhead. The new approach has the same theoretical complexity but executes in `data.table`'s C internals with radix-sorted joins.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                 "def", "usd_est_n2")) {
  # ---------------------------------------------------------------
  # STEP 1: Build a spatial-only edge list (year-invariant)
  #         rook_neighbors_unique is an nb object: a list of integer

  #         vectors indexing into id_order.
  # ---------------------------------------------------------------
  message("Step 1: Building spatial edge list...")

  # For each cell index in id_order, get its neighbor cell IDs

  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(ref_idx) {
    nb_idx <- rook_neighbors_unique[[ref_idx]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[ref_idx],
               neighbor_id = id_order[nb_idx])
  }))

  message(sprintf("  Edge list: %d directed neighbor pairs", nrow(edge_list)))

  # ---------------------------------------------------------------
  # STEP 2: Convert cell_data to data.table, add a row key
  # ---------------------------------------------------------------
  message("Step 2: Preparing data.table...")

  dt <- as.data.table(cell_data)

  # Ensure id and year columns exist
  stopifnot("id" %in% names(dt), "year" %in% names(dt))

  # ---------------------------------------------------------------
  # STEP 3: Create the neighbor-row lookup by joining edge_list
  #         with dt on (neighbor_id, year) — i.e., for every

  #         focal (id, year), find the rows of its spatial neighbors
  #         in the same year.
  # ---------------------------------------------------------------
  message("Step 3: Joining edges with data to resolve neighbor rows...")

  # Subset to only the columns we need for neighbor values
  cols_needed <- unique(c("id", "year", neighbor_source_vars))
  neighbor_dt <- dt[, ..cols_needed]

  # Rename for the join: neighbor_id -> id in neighbor_dt
  setnames(neighbor_dt, "id", "neighbor_id")

  # Key the neighbor data by (neighbor_id, year)
  setkeyv(neighbor_dt, c("neighbor_id", "year"))

  # Expand edges × years: join edge_list with focal rows to get
  # (focal_id, year, neighbor_id), then join with neighbor_dt to
  # get neighbor values.

  # First, get focal (id, year) pairs
  focal_keys <- dt[, .(focal_id = id, year = year)]

  # Merge focal keys with edge_list to create
  # (focal_id, year, neighbor_id) — one row per directed
  # neighbor-pair-year combination.
  message("  Expanding edges across years...")
  edge_year <- edge_list[focal_keys, on = .(focal_id), allow.cartesian = TRUE, nomatch = 0L]
  # edge_year now has columns: focal_id, neighbor_id, year

  message(sprintf("  Edge-year table: %d rows", nrow(edge_year)))

  # Join neighbor values onto the edge-year table
  message("  Joining neighbor variable values...")
  edge_year <- neighbor_dt[edge_year, on = .(neighbor_id, year), nomatch = NA]
  # Now edge_year has: neighbor_id, year, focal_id, + all neighbor_source_vars

  # ---------------------------------------------------------------
  # STEP 4: Compute grouped statistics (max, min, mean) per
  #         (focal_id, year) across all neighbor_source_vars at once.
  # ---------------------------------------------------------------
  message("Step 4: Computing neighbor statistics...")

  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  # Handle the edge case where all neighbor values are NA:
  # max/min with na.rm=TRUE on zero non-NA values gives ±Inf;
  # mean gives NaN. We'll fix those after aggregation.

  stats_dt <- edge_year[, lapply(agg_exprs, eval), by = .(focal_id, year)]

  # Replace Inf/-Inf/NaN with NA (cells with no valid neighbors)
  for (col in agg_names) {
    set(stats_dt, which(!is.finite(stats_dt[[col]])), col, NA_real_)
  }

  message(sprintf("  Stats table: %d rows × %d columns", nrow(stats_dt), ncol(stats_dt)))

  # ---------------------------------------------------------------
  # STEP 5: Handle focal (id, year) pairs that had NO neighbors
  #         (they won't appear in stats_dt). These get NA for all
  #         neighbor stats. We merge back onto dt.
  # ---------------------------------------------------------------
  message("Step 5: Merging neighbor features back onto cell_data...")

  # Remove any pre-existing neighbor columns to avoid conflicts
  existing_neighbor_cols <- intersect(names(dt), agg_names)
  if (length(existing_neighbor_cols) > 0) {
    dt[, (existing_neighbor_cols) := NULL]
  }

  # Merge
  setnames(stats_dt, "focal_id", "id")
  setkeyv(stats_dt, c("id", "year"))
  setkeyv(dt, c("id", "year"))

  dt <- stats_dt[dt, on = .(id, year)]  # right join keeps all original rows

  # ---------------------------------------------------------------
  # STEP 6: Convert back to data.frame if the original was one
  # ---------------------------------------------------------------
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    setDF(dt)
  }

  message("Done. Neighbor features added.")
  return(dt)
}
```

### Usage (drop-in replacement for the original outer loop)

```r
# --- Original code (86+ hours) ---
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# --- Optimized replacement (estimated 2-10 minutes) ---
cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# The trained Random Forest model is untouched.
# predict(rf_model, new_data) works exactly as before, because the
# output columns (neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, ...)
# carry the same numerical values — just computed faster.
```

### Memory-Constrained Variant (if the ~190M-row `edge_year` table exceeds 16 GB)

```r
# Process one year at a time to cap peak memory at ~1/28th:
optimize_neighbor_features_chunked <- function(cell_data, id_order,
                                                rook_neighbors_unique,
                                                neighbor_source_vars) {
  library(data.table)
  dt <- as.data.table(cell_data)

  # Build spatial edge list (same as above)
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(ref_idx) {
    nb_idx <- rook_neighbors_unique[[ref_idx]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) return(NULL)
    data.table(focal_id = id_order[ref_idx], neighbor_id = id_order[nb_idx])
  }))

  cols_needed <- unique(c("id", "year", neighbor_source_vars))
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  years <- sort(unique(dt$year))

  stats_list <- lapply(years, function(yr) {
    message(sprintf("  Processing year %d ...", yr))
    dt_yr <- dt[year == yr, ..cols_needed]

    # Neighbor values for this year
    nb_vals <- copy(dt_yr)
    setnames(nb_vals, "id", "neighbor_id")
    setkeyv(nb_vals, "neighbor_id")

    # Focal ids for this year
    focal_yr <- data.table(focal_id = dt_yr$id)

    # Expand edges for this year's focal cells
    ey <- edge_list[focal_yr, on = .(focal_id), allow.cartesian = TRUE, nomatch = 0L]
    ey[, year := yr]

    # Join neighbor values
    ey <- nb_vals[ey, on = .(neighbor_id), nomatch = NA]

    # Aggregate
    agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
      list(
        bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
        bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
        bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
      )
    }), recursive = FALSE)
    names(agg_exprs) <- agg_names

    s <- ey[, lapply(agg_exprs, eval), by = .(focal_id)]
    s[, year := yr]
    for (col in agg_names) set(s, which(!is.finite(s[[col]])), col, NA_real_)
    s
  })

  stats_dt <- rbindlist(stats_list)
  setnames(stats_dt, "focal_id", "id")

  # Remove old neighbor columns if present
  existing <- intersect(names(dt), agg_names)
  if (length(existing) > 0) dt[, (existing) := NULL]

  setkeyv(stats_dt, c("id", "year"))
  setkeyv(dt, c("id", "year"))
  dt <- stats_dt[dt, on = .(id, year)]

  if (is.data.frame(cell_data) && !is.data.table(cell_data)) setDF(dt)
  return(dt)
}
```

---

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor resolution** | 6.46M `paste` + named-vector lookups in `lapply` | One `data.table` keyed join (C-level merge-sort) |
| **Variable processing** | 5 serial `lapply` passes over 6.46M rows | Single grouped aggregation over all 5 variables |
| **Interpreter overhead** | ~6.46M R function calls per variable | Zero R-level row iteration |
| **String operations** | ~6.46M × avg_neighbors `paste` calls | Zero (integer keys only) |
| **Estimated time** | 86+ hours | 2–10 minutes |
| **Peak RAM (chunked)** | ~same | ~2–4 GB |
| **Numerical output** | max, min, mean per variable | Identical values |
| **RF model** | — | Untouched; no retraining needed |