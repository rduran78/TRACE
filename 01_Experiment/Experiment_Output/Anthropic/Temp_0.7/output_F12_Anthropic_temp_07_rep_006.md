 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of ~6.46 million small 3-element vectors is not free, it is a single vectorized C-level operation that completes in seconds. The `lapply` inside `compute_neighbor_stats()` does simple indexing and three summary functions — also relatively fast.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** For each of the ~6.46M rows, the function calls `as.character(data$id[i])`, performs a named-vector lookup (`id_to_ref[...]`), retrieves neighbor cell IDs, then calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` to build keys, and finally does another named-vector lookup (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is O(n) hash probing per call, and the `idx_lookup` vector has ~6.46M entries. Doing this 6.46M times (each time with ~4 neighbors on average, given ~1.37M directed relationships / 344K cells ≈ 4) means roughly **25+ million string constructions and hash lookups**.

2. **This runs inside a sequential `lapply` with no vectorization.** The entire loop is single-threaded, row-by-row R-level iteration — the worst-case performance pattern in R.

3. **The lookup is built once but costs hours.** The `compute_neighbor_stats()` function is called 5 times (once per variable) but each call just indexes a numeric vector and computes max/min/mean — trivial by comparison. The lookup construction dominates total runtime.

4. **Redundant work structure.** The function converts IDs to strings, pastes them with years, and looks them up — all operations that could be replaced by direct integer indexing via merge/join logic.

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`** — eliminate the per-row `lapply` entirely. Use a data.table join to map (neighbor_id, year) → row_index in one vectorized pass.
2. **Pre-expand the neighbor pairs** into a two-column table of (source_row, neighbor_row) and use direct integer indexing for all stats.
3. **Vectorize `compute_neighbor_stats()`** — use data.table's grouped aggregation (`max`, `min`, `mean` by source row) instead of per-row `lapply`.
4. **Preserve the trained Random Forest model** — we only change how features are computed, not the model or the numerical values produced. The same arithmetic (max, min, mean of neighbor values) is applied, yielding identical results.

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 1. Build neighbor edge list (vectorized, replaces build_neighbor_lookup)
# ──────────────────────────────────────────────────────────────────────
build_neighbor_edges <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id', 'year', and a .ROW_IDX column
  # id_order: vector of cell IDs in the order used by the nb object
  # neighbors: spdep nb object (list of integer index vectors)

  # Step A: Expand the nb object into a two-column data.table of
  #         (source_cell_id, neighbor_cell_id)
  n_cells <- length(id_order)
  src_indices <- rep(seq_len(n_cells), lengths(neighbors))
  nbr_indices <- unlist(neighbors, use.names = FALSE)

  # Remove any zero-length / empty-neighbor entries (already handled by rep/unlist)
  # Convert positional indices back to actual cell IDs
  edge_dt <- data.table(
    src_cell_id = id_order[src_indices],
    nbr_cell_id = id_order[nbr_indices]
  )

  # Step B: For every (src_cell_id, year) row in the data, find the
  #         corresponding (nbr_cell_id, year) row via keyed join.

  # Add row index to data
  data_dt[, .ROW_IDX := .I]

  # Create keyed version for source rows: maps (id, year) -> source row index
  src_key <- data_dt[, .(src_cell_id = id, year, src_row = .ROW_IDX)]

  # Create keyed version for neighbor rows: maps (id, year) -> neighbor row index
  nbr_key <- data_dt[, .(nbr_cell_id = id, year, nbr_row = .ROW_IDX)]

  # Join edges with source rows: for each (src_cell_id, year), attach edge info
  # This cross of edges × years gives us all (src_row, nbr_cell_id, year) combos
  edge_with_src <- merge(
    src_key, edge_dt,
    by = "src_cell_id",
    allow.cartesian = TRUE  # each cell has multiple neighbors
  )
  # edge_with_src now has: src_cell_id, year, src_row, nbr_cell_id

  # Join with neighbor rows to get nbr_row
  setkey(edge_with_src, nbr_cell_id, year)
  setkey(nbr_key, nbr_cell_id, year)

  result <- nbr_key[edge_with_src, nomatch = 0L]
  # result has: nbr_cell_id, year, nbr_row, src_cell_id, src_row

  result[, .(src_row, nbr_row)]
}

# ──────────────────────────────────────────────────────────────────────
# 2. Compute neighbor stats for one variable (vectorized via data.table)
# ──────────────────────────────────────────────────────────────────────
compute_neighbor_stats_fast <- function(data_dt, edge_dt, var_name) {
  # edge_dt has columns: src_row, nbr_row
  # Extract neighbor values via direct integer indexing (vectorized)
  vals <- data_dt[[var_name]]
  work <- edge_dt[, .(src_row, nbr_val = vals[nbr_row])]

  # Drop NAs in neighbor values
  work <- work[!is.na(nbr_val)]

  # Grouped aggregation — single pass, highly optimized in data.table
  stats <- work[, .(
    nb_max  = max(nbr_val),
    nb_min  = min(nbr_val),
    nb_mean = mean(nbr_val)
  ), by = src_row]

  # Allocate output columns (NA for rows with no valid neighbors)
  n <- nrow(data_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  out_max[stats$src_row]  <- stats$nb_max
  out_min[stats$src_row]  <- stats$nb_min
  out_mean[stats$src_row] <- stats$nb_mean

  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")

  data_dt[, (col_max)  := out_max]
  data_dt[, (col_min)  := out_min]
  data_dt[, (col_mean) := out_mean]

  invisible(data_dt)
}

# ──────────────────────────────────────────────────────────────────────
# 3. Full pipeline (drop-in replacement)
# ──────────────────────────────────────────────────────────────────────
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  # Convert to data.table (in-place if already one, otherwise copy)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  cat("Building vectorized neighbor edge list...\n")
  t0 <- proc.time()
  edge_dt <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)
  cat(sprintf("  Edge list built: %d edges in %.1f seconds.\n",
              nrow(edge_dt), (proc.time() - t0)[3]))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
    t1 <- proc.time()
    compute_neighbor_stats_fast(cell_data, edge_dt, var_name)
    cat(sprintf("  Done in %.1f seconds.\n", (proc.time() - t1)[3]))
  }

  cat("Pipeline complete.\n")
  return(cell_data)
}

# ──────────────────────────────────────────────────────────────────────
# 4. Usage (preserves the trained RF model — no retraining needed)
# ──────────────────────────────────────────────────────────────────────
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# # The trained Random Forest model is unchanged. Use it directly:
# # predictions <- predict(rf_model, newdata = cell_data)
```

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| **Alleged bottleneck** | `do.call(rbind, ...)` and list binding in `compute_neighbor_stats()` | These are minor costs (~seconds) |
| **True bottleneck** | — | `build_neighbor_lookup()`: 6.46M sequential R-level iterations each doing string pasting and named-vector hash lookups |
| **Root cause** | — | Row-by-row `lapply` with `paste()` + named-vector indexing is O(n × k) in slow interpreted R; no vectorization |
| **Fix** | — | Replace with a single vectorized `data.table` merge to build an integer edge list `(src_row, nbr_row)`, then use grouped aggregation for stats |
| **Expected speedup** | — | From ~86+ hours to **minutes** (edge build via merge + 5 grouped aggregations) |
| **Numerical equivalence** | — | Identical `max`, `min`, `mean` computed on the same neighbor sets — preserves the original estimand |
| **RF model** | — | Untouched; no retraining required |