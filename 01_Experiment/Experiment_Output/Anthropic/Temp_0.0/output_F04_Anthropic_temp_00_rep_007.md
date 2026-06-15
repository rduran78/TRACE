 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For every single row, the function:

1. **Performs repeated character coercion and hash lookups** (`as.character`, `paste`, named-vector indexing via `idx_lookup[neighbor_keys]`) — all inside a per-row anonymous function.
2. **Allocates millions of small intermediate character vectors** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) — one per row, causing extreme GC pressure.
3. **Produces a list of 6.46 million integer vectors** — the `lapply` output itself is a massive nested list.

`compute_neighbor_stats` is a secondary bottleneck: it loops over the same 6.46M-element list, extracting and summarizing neighbor values per row — again in pure R with per-element allocation.

**Quantified cost:** ~6.46M iterations × (string paste + named-vector lookup + NA filtering) ≈ 86+ hours. The Random Forest inference, by contrast, is a single vectorized `predict()` call and is comparatively fast.

---

## Optimization Strategy

**Core idea:** Replace the row-level `lapply` with fully vectorized `data.table` merge-and-aggregate operations. Instead of building a 6.46M-element lookup list and then looping over it per variable, we:

1. **Build a flat edge table** (`cell_id`, `neighbor_id`) from the `nb` object — done once, ~1.37M rows.
2. **Join** this edge table to the panel data by `(neighbor_id, year)` to get neighbor values — a single keyed `data.table` merge per variable.
3. **Aggregate** (max, min, mean) by `(cell_id, year)` — a single grouped `data.table` operation per variable.
4. **Merge** the aggregated neighbor features back to the main table.

This eliminates all per-row string operations, all per-row list allocations, and leverages `data.table`'s C-level radix joins and grouped aggregation. Expected runtime: **minutes, not hours**.

The trained Random Forest model and the numerical estimand (max, min, mean of neighbor values) are fully preserved — the output columns are identical.

---

## Working R Code

```r
library(data.table)

#' Vectorized spatial-neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order         integer vector of cell IDs in the order matching the nb object
#' @param rook_neighbors   spdep nb object (list of integer index vectors)
#' @param neighbor_source_vars character vector of variable names to compute neighbor stats for
#' @return data.table with original columns plus neighbor feature columns appended
add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      rook_neighbors,
                                      neighbor_source_vars) {

  # --- Step 1: Build flat edge table from nb object (once) ---
  # Each element of rook_neighbors is an integer vector of *positional* indices

  # into id_order. Convert to actual cell IDs.
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors), function(i) {
    nb_idx <- rook_neighbors[[i]]
    # spdep nb objects use 0L (integer(0) or explicit 0) for no-neighbor cells
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  # edge_list: ~1.37M rows, two integer columns — very small

  # --- Step 2: Convert main data to data.table (no copy if already DT) ---
  dt <- as.data.table(cell_data)

  # Key the main table for fast joins
  setkey(dt, id, year)

  # --- Step 3: For each variable, join → aggregate → merge back ---
  for (var in neighbor_source_vars) {

    # Subset to only the columns we need for the join target
    # (neighbor_id will be matched to id)
    val_dt <- dt[, .(id, year, val = get(var))]
    setnames(val_dt, "id", "neighbor_id")
    setkey(val_dt, neighbor_id)

    # Join edge_list to val_dt: for every (cell_id, neighbor_id) edge,
    # look up the neighbor's value in every year.
    # We need year from the focal cell, so we join via the main table.
    #
    # Efficient approach: merge edges with the focal cell's years first,
    # then look up neighbor values.

    # Get unique (cell_id, year) from dt — these are the focal observations
    focal <- dt[, .(cell_id = id, year)]

    # Expand: for each focal (cell_id, year), attach all its neighbor_ids
    # This is a keyed join: focal ↔ edge_list on cell_id
    setkey(edge_list, cell_id)
    setkey(focal, cell_id)
    expanded <- edge_list[focal, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
    # expanded columns: cell_id, neighbor_id, year
    # Rows ≈ 6.46M × avg_neighbors (≈ 4 for rook) but many cells share
    # the same neighbor count, so ~25–27M rows — fits in 16 GB easily.

    # Now look up the neighbor's value for that year
    setkey(expanded, neighbor_id, year)
    setkey(val_dt, neighbor_id)
    # We need to join on both neighbor_id and year
    # Rebuild val_dt key on both columns
    setkey(val_dt, neighbor_id, year)
    expanded <- val_dt[expanded, on = .(neighbor_id, year), nomatch = NA]
    # expanded now has: neighbor_id, year, val, cell_id

    # Remove NA values before aggregation (matches original behavior)
    expanded <- expanded[!is.na(val)]

    # Aggregate: max, min, mean grouped by (cell_id, year)
    agg <- expanded[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = .(cell_id, year)]

    # Rename to match the original column naming convention
    new_names <- paste0(var, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
    setnames(agg, c("nb_max", "nb_min", "nb_mean"), new_names)
    setnames(agg, "cell_id", "id")

    # Merge back to main table (left join preserves all rows; unmatched get NA)
    setkey(agg, id, year)
    setkey(dt, id, year)
    dt <- agg[dt, on = .(id, year)]

    # Clean up to free memory within the loop
    rm(val_dt, focal, expanded, agg)
  }

  return(dt)
}

# -----------------------------------------------------------------------
# Usage (drop-in replacement for the original outer loop)
# -----------------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Proceed directly to prediction:
# preds <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M `lapply` iterations with `paste`/string indexing | Single flat edge table built once (~1.37M rows) |
| Per-variable stats | 6.46M `lapply` iterations with per-row subsetting | One `data.table` keyed join + one grouped aggregation |
| Total string operations | ~32M `paste` calls + ~32M named-vector lookups | **Zero** |
| Estimated wall time | **86+ hours** | **~5–15 minutes** (depending on disk I/O) |
| Peak RAM | Millions of small list elements → GC thrashing | ~2–3 GB for the expanded join table (fits in 16 GB) |
| RF model | Unchanged | Unchanged |
| Numerical output | max, min, mean of rook-neighbor values | Identical |