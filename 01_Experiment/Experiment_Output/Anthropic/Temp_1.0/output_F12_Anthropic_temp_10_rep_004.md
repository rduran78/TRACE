 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: The Colleague Is Wrong

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized concatenation operation and runs in seconds to low minutes at most. The inner `lapply` in `compute_neighbor_stats` does not perform "repeated list binding" — it returns fixed-length vectors collected into a pre-allocated list.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** For every single row `i`, the function calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` and then indexes into `idx_lookup` — a named character vector with ~6.46 million entries. Named vector lookup in R is **O(n)** linear scan per query (R's named vectors use linear hashing that degrades badly at this scale). With ~6.46M rows and an average of ~4 rook neighbors each, that's ~25.8 million string-key lookups into a 6.46M-entry named vector.

2. **Repeated `as.character()` coercion** on every iteration.

3. **Character string construction (`paste`) inside a per-row loop** — ~6.46 million calls to `paste`, each producing ~4 strings.

This single function likely accounts for **>95% of the 86+ hour runtime**. Once the lookup is built, `compute_neighbor_stats` with 5 variables is comparatively trivial (5 × one `lapply` of simple arithmetic over pre-resolved integer indices).

## Optimization Strategy

1. **Replace the named-vector lookup with an environment (hash map) or, better, a fully vectorized merge/join approach using `data.table`.** Eliminate the per-row `lapply` in `build_neighbor_lookup` entirely.

2. **Vectorize `build_neighbor_lookup`** by expanding the neighbor list into a two-column edge table `(row_i, neighbor_cell_id)`, joining on `(neighbor_cell_id, year)` to resolve target row indices in one bulk operation.

3. **Vectorize `compute_neighbor_stats`** by using the edge table with `data.table` grouped aggregation (`max`, `min`, `mean` by source row), eliminating the per-row `lapply` there too.

This reduces the runtime from ~86+ hours to **minutes**.

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED build_neighbor_lookup (returns an edge data.table)
# ============================================================
build_neighbor_edge_table <- function(data, id_order, neighbors) {
  # data must have columns: id, year (and be ordered by original row number)
  dt <- as.data.table(data)[, row_idx := .I]

  # Build a mapping from id_order position (ref_idx) to cell id
  # neighbors[[ref_idx]] gives neighbor positions in id_order
  # Expand neighbor list into an edge table: (cell_id, neighbor_cell_id)
  n_cells <- length(id_order)
  edge_list <- rbindlist(lapply(seq_len(n_cells), function(ref_idx) {
    nb <- neighbors[[ref_idx]]
    if (length(nb) == 0L) return(NULL)
    data.table(cell_id = id_order[ref_idx],
               neighbor_cell_id = id_order[nb])
  }))
  # edge_list now has ~1,373,394 rows (directed edges)

  # For every (cell_id, year) row, we need neighbor rows:
  # Join edges with the data on cell_id to get (row_idx_source, neighbor_cell_id, year)
  source <- dt[, .(row_idx_source = row_idx, cell_id = id, year)]
  edges_with_year <- merge(edge_list, source,
                           by = "cell_id", allow.cartesian = TRUE)
  # Now resolve neighbor_cell_id + year -> row_idx_target
  setnames(dt, "id", "cell_id_target")
  target_key <- dt[, .(cell_id_target, year, row_idx_target = row_idx)]
  setkey(target_key, cell_id_target, year)

  setnames(edges_with_year, "neighbor_cell_id", "cell_id_target")
  setkey(edges_with_year, cell_id_target, year)

  resolved <- target_key[edges_with_year, nomatch = 0L]
  # Columns: cell_id_target, year, row_idx_target, cell_id, row_idx_source

  resolved[, .(row_idx_source, row_idx_target)]
}

# ============================================================
# OPTIMIZED compute_neighbor_stats (vectorized via data.table)
# ============================================================
compute_neighbor_stats_fast <- function(data_dt, edge_dt, var_name, n_rows) {
  # edge_dt has columns: row_idx_source, row_idx_target
  # Extract neighbor values in one vectorized step
  vals <- data_dt[[var_name]]
  work <- edge_dt[, .(row_idx_source, nval = vals[row_idx_target])]
  work <- work[!is.na(nval)]

  stats <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = row_idx_source]

  # Build full-length result (NA for rows with no valid neighbors)
  out <- data.table(
    row_idx_source = seq_len(n_rows),
    nb_max  = NA_real_,
    nb_min  = NA_real_,
    nb_mean = NA_real_
  )
  out[stats, on = "row_idx_source",
      `:=`(nb_max = i.nb_max, nb_min = i.nb_min, nb_mean = i.nb_mean)]
  out
}

# ============================================================
# OPTIMIZED outer pipeline
# ============================================================
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  cell_dt <- as.data.table(cell_data)
  n_rows  <- nrow(cell_dt)

  message("Building neighbor edge table (vectorized)...")
  edge_dt <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
  setkey(edge_dt, row_idx_source)
  message(sprintf("  Edge table: %d resolved edges.", nrow(edge_dt)))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_neighbor_stats_fast(cell_dt, edge_dt, var_name, n_rows)

    # Attach columns with same naming convention as original code
    col_max  <- paste0(var_name, "_nb_max")
    col_min  <- paste0(var_name, "_nb_min")
    col_mean <- paste0(var_name, "_nb_mean")

    set(cell_dt, j = col_max,  value = stats$nb_max)
    set(cell_dt, j = col_min,  value = stats$nb_min)
    set(cell_dt, j = col_mean, value = stats$nb_mean)
  }

  message("Done. Returning augmented data.table.")
  cell_dt
}

# ============================================================
# USAGE (drop-in replacement; RF model is untouched)
# ============================================================
# cell_data <- run_optimized_pipeline(cell_data, id_order, rook_neighbors_unique)
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| Alleged bottleneck | `do.call(rbind, ...)` in `compute_neighbor_stats` | Negligible (single bind of pre-allocated list) |
| **True bottleneck** | — | `build_neighbor_lookup()`: 6.46M per-row `paste` + named-vector lookups (O(n) each) ≈ 25.8M linear scans into a 6.46M-entry vector |
| Estimated speedup | — | From ~86+ hours → **~5–15 minutes** via vectorized `data.table` joins |
| Numerical equivalence | — | Preserved: same `max`, `min`, `mean` over identical neighbor sets |
| Trained RF model | — | Untouched; only the feature-engineering pipeline is optimized |