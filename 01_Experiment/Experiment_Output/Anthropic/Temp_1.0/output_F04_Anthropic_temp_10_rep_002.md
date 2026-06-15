 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Looks up the cell's reference index via character conversion and named-vector lookup.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Pastes neighbor IDs with the current row's year to form string keys.
4. Matches those keys against a named vector (`idx_lookup`) of ~6.46M entries.

**String operations (`paste`, named-vector character matching) on 6.46M rows × ~4 neighbors each ≈ 25.8M string constructions and hash lookups.** Named vector lookup in R is O(n) in the worst case per query due to hashing overhead at scale. The `lapply` is also not vectorized — it's a pure R loop with per-element allocations.

`compute_neighbor_stats` then loops over the 6.46M-element lookup list again for each of 5 variables (32.3M list iterations), calling `max`/`min`/`mean` individually.

**Estimated cost:** ~6.46M iterations × (string paste + hash lookup) × overhead ≈ 86+ hours on a laptop.

## Optimization Strategy

1. **Replace string-key lookups with integer-arithmetic indexing.** Since `year` is contiguous (1992–2019, 28 values), encode each `(cell_id, year)` pair as a unique integer and use `match()` or direct array indexing instead of named character vectors.

2. **Vectorize `build_neighbor_lookup` using `data.table` joins** — expand all neighbor relationships into a flat edge table, join on `(neighbor_id, year)` to get row indices, then split by source row. This replaces 6.46M R-level iterations with a single vectorized join.

3. **Vectorize `compute_neighbor_stats` using `data.table` grouped aggregation** — instead of `lapply` over a list, perform grouped `max`/`min`/`mean` in one pass per variable.

4. **Compute all 5 variables' neighbor stats in a single grouped pass** to avoid 5 separate iterations.

## Optimized Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build neighbor lookup as a flat edge table (vectorized)
# ============================================================
build_neighbor_edges <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns id, year, and a row index
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors)

  # Map each cell ID to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Expand nb object into a flat edge list: (source_ref, neighbor_ref)
  # This is ~1.37M directed edges
  src_refs <- rep(seq_along(neighbors), lengths(neighbors))
  nbr_refs <- unlist(neighbors, use.names = FALSE)

  edge_dt <- data.table(
    src_cell_id = id_order[src_refs],
    nbr_cell_id = id_order[nbr_refs]
  )

  # Create a row-index lookup: (id, year) -> row_index in data_dt
  data_dt[, row_idx := .I]

  # Cross-join edges with all years present in the data
  years <- sort(unique(data_dt$year))

  # Expand edges × years: each spatial edge exists for every year
  # ~1.37M edges × 28 years ≈ 38.5M rows — fits in 16 GB RAM
  edge_year <- edge_dt[, CJ(edge_row = seq_len(.N), year = years)]
  edge_year[, `:=`(
    src_cell_id = edge_dt$src_cell_id[edge_row],
    nbr_cell_id = edge_dt$nbr_cell_id[edge_row]
  )]
  edge_year[, edge_row := NULL]

  # Join to get source row index
  setkey(data_dt, id, year)
  edge_year <- merge(
    edge_year,
    data_dt[, .(id, year, src_row_idx = row_idx)],
    by.x = c("src_cell_id", "year"),
    by.y = c("id", "year"),
    all.x = FALSE
  )

 # Join to get neighbor row index
  edge_year <- merge(
    edge_year,
    data_dt[, .(id, year, nbr_row_idx = row_idx)],
    by.x = c("nbr_cell_id", "year"),
    by.y = c("id", "year"),
    all.x = FALSE
  )

  return(edge_year)
}

# ============================================================
# STEP 2: Compute all neighbor stats in one vectorized pass
# ============================================================
compute_all_neighbor_stats <- function(data_dt, edge_year, var_names) {
  # Attach neighbor variable values to the edge table
  # We pull only needed columns to keep memory in check

  nbr_vals <- data_dt[edge_year$nbr_row_idx, ..var_names]
  work <- data.table(
    src_row_idx = edge_year$src_row_idx
  )
  work <- cbind(work, nbr_vals)

  # Grouped aggregation: max, min, mean per source row, for all vars at once
  agg_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(var_names, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  stats <- work[, lapply(agg_exprs, eval, envir = .SD), by = src_row_idx]

  # Replace -Inf/Inf (from max/min of empty after na.rm) with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  return(stats)
}

# ============================================================
# STEP 3: Main execution — drop-in replacement for outer loop
# ============================================================
optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {

  cell_dt <- as.data.table(cell_data)
  cell_dt[, row_idx := .I]

  message("Building vectorized neighbor edge table...")
  t0 <- Sys.time()

  # --- memory-efficient edge expansion (no CJ of full edge table) ---
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  src_refs  <- rep(seq_along(rook_neighbors_unique),
                   lengths(rook_neighbors_unique))
  nbr_refs  <- unlist(rook_neighbors_unique, use.names = FALSE)

  edge_dt <- data.table(
    src_cell_id = id_order[src_refs],
    nbr_cell_id = id_order[nbr_refs]
  )

  # Key the data for fast binary-search joins
  setkey(cell_dt, id, year)

  # Instead of expanding all edges × all years at once (38.5M rows),
  # we join edges against existing (id, year) pairs directly.

  # For each edge, the valid years are those where BOTH src and nbr exist.
  # Since this is a balanced panel (344,208 cells × 28 years), every cell
  # appears in every year. We can expand safely.

  years <- sort(unique(cell_dt$year))
  n_edges <- nrow(edge_dt)
  n_years <- length(years)

  # Expand: repeat each edge for each year
  edge_year <- data.table(
    src_cell_id = rep(edge_dt$src_cell_id, each = n_years),
    nbr_cell_id = rep(edge_dt$nbr_cell_id, each = n_years),
    year        = rep(years, times = n_edges)
  )

  message(sprintf("  Edge-year table: %s rows (%.1f MB)",
                  format(nrow(edge_year), big.mark = ","),
                  object.size(edge_year) / 1e6))

  # Join to get src and nbr row indices
  src_lookup <- cell_dt[, .(src_cell_id = id, year, src_row_idx = row_idx)]
  setkey(src_lookup, src_cell_id, year)
  edge_year <- src_lookup[edge_year, on = .(src_cell_id, year), nomatch = 0L]

  nbr_lookup <- cell_dt[, .(nbr_cell_id = id, year, nbr_row_idx = row_idx)]
  setkey(nbr_lookup, nbr_cell_id, year)
  edge_year <- nbr_lookup[edge_year, on = .(nbr_cell_id, year), nomatch = 0L]

  message(sprintf("  Edge table built in %.1f seconds.", difftime(Sys.time(), t0, units = "secs")))

  # --- Compute neighbor statistics for all variables at once ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  message("Computing neighbor statistics (vectorized)...")
  t1 <- Sys.time()

  # Pull neighbor values for all vars at once
  nbr_vals <- cell_dt[edge_year$nbr_row_idx, ..neighbor_source_vars]
  nbr_vals[, src_row_idx := edge_year$src_row_idx]

  # Build aggregation expressions programmatically
  agg_list <- list()
  for (v in neighbor_source_vars) {
    agg_list[[paste0("neighbor_max_", v)]]  <-
      substitute(fifelse(all(is.na(x)), NA_real_, max(x, na.rm = TRUE)), list(x = as.name(v)))
    agg_list[[paste0("neighbor_min_", v)]]  <-
      substitute(fifelse(all(is.na(x)), NA_real_, min(x, na.rm = TRUE)), list(x = as.name(v)))
    agg_list[[paste0("neighbor_mean_", v)]] <-
      substitute(mean(x, na.rm = TRUE), list(x = as.name(v)))
  }

  stats <- nbr_vals[, lapply(agg_list, eval), by = src_row_idx]

  message(sprintf("  Stats computed in %.1f seconds.", difftime(Sys.time(), t1, units = "secs")))

  # --- Merge back into cell_dt by row index ---
  # Rows with no neighbors won't appear in stats; those get NA (correct).
  new_cols <- setdiff(names(stats), "src_row_idx")

  # Remove any pre-existing neighbor columns to avoid conflicts
  for (col in intersect(new_cols, names(cell_dt))) {
    cell_dt[, (col) := NULL]
  }

  cell_dt <- merge(cell_dt, stats, by.x = "row_idx", by.y = "src_row_idx", all.x = TRUE)

  # Restore original row order
  setorder(cell_dt, row_idx)
  cell_dt[, row_idx := NULL]

  # Convert back to data.frame to preserve downstream compatibility
  cell_data <- as.data.frame(cell_dt)

  message("Done. Neighbor features added.")
  return(cell_data)
}

# ============================================================
# Usage (drop-in replacement for the original outer loop):
# ============================================================
# cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique)
#
# Then proceed with predict(trained_rf_model, cell_data) as before.
# The trained Random Forest model is untouched.
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M R-level iterations with `paste`+named-vector lookup | Single vectorized `data.table` keyed join |
| **Stat computation** | 5 variables × 6.46M `lapply` calls (32.3M R iterations) | One grouped `data.table` aggregation pass |
| **String operations** | ~25.8M `paste()` calls + character hash lookups | Zero — integer keys only |
| **Estimated time** | 86+ hours | **~2–10 minutes** on same laptop |
| **Peak RAM** | Moderate (list of 6.46M vectors) | ~4–6 GB (38.5M-row edge table + aggregation) |
| **Numerical output** | max / min / mean per neighbor set | Identical max / min / mean (preserves estimand) |
| **RF model** | — | Untouched; no retraining |

The core insight: **replace 6.46M sequential R-level iterations (each doing string manipulation and hash lookups) with a single flat join-and-group operation on integer keys using `data.table`'s radix-based binary search.** This converts an O(N×k) interpreted-R loop into a compiled-C vectorized operation, yielding roughly a 500–2500× speedup.