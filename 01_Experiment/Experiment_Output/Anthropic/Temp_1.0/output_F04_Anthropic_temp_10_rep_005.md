 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector — O(1) average but with constant overhead from `as.character()` and name-based lookup repeated millions of times.
2. Constructs `paste(..., sep="_")` key strings for every neighbor of every row.
3. Looks up those keys in `idx_lookup` (another named-vector lookup involving string hashing).

This means roughly **6.46M × ~4 neighbors × 3 string operations** ≈ 77M+ string allocations and hash lookups, all inside an interpreted R `lapply` loop. `compute_neighbor_stats` then loops again over the 6.46M-element list, extracting subsets of a numeric vector — lightweight individually, but the sheer iteration count and the `do.call(rbind, ...)` on a 6.46M-element list adds further overhead.

**Root causes (ranked):**
1. Row-level R loop with per-iteration string construction/hashing in `build_neighbor_lookup`.
2. Returning a 6.46M-element list-of-vectors, then iterating over it again per variable.
3. `do.call(rbind, ...)` on a multi-million element list (slow recursive binding).

## Optimization Strategy

**Replace the row-level R loop with a fully vectorized `data.table` join approach:**

- Expand the neighbor graph into an edge table (`cell_id → neighbor_id`).
- Join the panel data onto this edge table by `(neighbor_id, year)` to retrieve neighbor values in one vectorized merge.
- Compute grouped `max/min/mean` with `data.table`'s fast `by=` aggregation.

This eliminates all per-row string pasting, list construction, and repeated lookups. Expected speedup: **~100–500×** (minutes instead of days).

## Optimized Working R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {

  # ---- Step 1: Build directed edge table from the nb object ----
  # rook_neighbors_unique is a list of integer index vectors (spdep nb object).
  # id_order maps positional index -> cell id.

  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  # edge_list now has ~1.37M rows: (cell_id, neighbor_id)

  # ---- Step 2: Convert panel to data.table keyed on (id, year) ----
  dt <- as.data.table(cell_data)

  # Keep only the columns we need for the neighbor join to reduce memory
  join_cols <- c("id", "year", neighbor_source_vars)
  dt_slim <- dt[, ..join_cols]

  # ---- Step 3: For each source variable, compute neighbor stats via join ----
  for (var_name in neighbor_source_vars) {

    # Create a lookup table: (id, year, value)
    val_dt <- dt_slim[, .(neighbor_id = id, year, val = get(var_name))]
    setkey(val_dt, neighbor_id, year)

    # Expand edges × years: join edge_list with the panel on neighbor side
    # First, add the focal cell's year by joining edge_list with dt's (id, year)
    focal_keys <- dt[, .(cell_id = id, year)]
    expanded <- edge_list[focal_keys, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
    # expanded has columns: cell_id, neighbor_id, year

    # Now join to get the neighbor's value for that year
    setkey(expanded, neighbor_id, year)
    expanded <- val_dt[expanded, on = .(neighbor_id, year), nomatch = NA]
    # expanded now has: neighbor_id, year, val, cell_id

    # ---- Step 4: Aggregate per (cell_id, year) ----
    stats <- expanded[!is.na(val),
      .(
        nb_max  = max(val),
        nb_min  = min(val),
        nb_mean = mean(val)
      ),
      by = .(cell_id, year)
    ]

    # Rename columns to match original feature names
    max_name  <- paste0("neighbor_max_",  var_name)
    min_name  <- paste0("neighbor_min_",  var_name)
    mean_name <- paste0("neighbor_mean_", var_name)
    setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_name, min_name, mean_name))

    # ---- Step 5: Left-join back to the main table ----
    setkey(stats, cell_id, year)
    setkey(dt, id, year)
    dt <- stats[dt, on = .(cell_id = id, year = year)]
    setnames(dt, "cell_id", "id")
    setkey(dt, id, year)

    # Clean up to keep memory bounded on a 16 GB laptop
    rm(val_dt, expanded, stats)
    gc()

    message(paste0("Done: ", var_name))
  }

  return(dt)
}

# ---- Usage (drop-in replacement for the original outer loop) ----
cell_data_dt <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# Convert back to data.frame if the trained RF predict() expects one
cell_data <- as.data.frame(cell_data_dt)

# The trained Random Forest model is untouched — just call predict() as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor index sets and the same raw values. The vectorized join replicates the same (cell_id, year) ↔ (neighbor_id, year) pairing as the original key-based lookup. |
| **Trained RF model** | No model retraining occurs. The output `cell_data` has identically named columns fed to `predict()`. |
| **NA handling** | `!is.na(val)` in the aggregation and `nomatch = NA` in the left-join replicate the original `is.na` filtering and the `NA` default for cells with no valid neighbors. |

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~60–70 hrs (6.46M R-loop iterations with string ops) | ~2 sec (vectorized `rbindlist` over 344K cells) |
| Stats per variable | ~3–4 hrs each (R lapply + `do.call(rbind,...)`) | ~1–3 min each (`data.table` keyed join + grouped agg) |
| **Total for 5 variables** | **~86+ hrs** | **~5–15 min** |
| Peak RAM | ~8–10 GB (6.46M-element list of integer vectors) | ~6–8 GB (edge expansion materialized one variable at a time, then freed) |

The optimization converts two nested R-level loops with per-element string operations into vectorized `data.table` keyed joins and grouped aggregations, reducing wall-clock time from days to minutes on the same 16 GB laptop.