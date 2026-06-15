 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Performs character coercion and named-vector lookups (`id_to_ref`, `idx_lookup`) — these are hash-table lookups but done **6.46 million times in an interpreted R loop**.
2. Constructs paste-based string keys (`paste(neighbor_cell_ids, year, sep="_")`) for every neighbor of every row — creating and discarding millions of temporary character vectors.
3. Returns variable-length integer vectors, assembled into a list of 6.46M elements.

**`compute_neighbor_stats`** then iterates over that 6.46M-element list again, once per variable (×5 variables = ~32.3M interpreted iterations).

The fundamental problem: **row-level interpreted R loops over millions of rows with per-element string operations and named-vector lookups**. The algorithmic complexity is fine (linear in edges × years), but the constant factor in interpreted R is enormous.

## Optimization Strategy

**Replace the row-level R loop with vectorized operations using `data.table`.**

Key ideas:

1. **Expand the neighbor list into an edge table once** — a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows.
2. **Join on `(neighbor_id, year)`** to get neighbor row indices or values directly — this is a single keyed merge, fully vectorized in C via `data.table`.
3. **Group-by aggregation** `[, .(max, min, mean), by = .(cell_id, year)]` replaces the per-row `lapply` in `compute_neighbor_stats`.
4. Do this once per variable (5 joins + group-bys instead of 32.3M R-level iterations).

This eliminates all interpreted loops, all string-key construction, and all per-row temporary allocations. Expected runtime: **minutes, not days**.

## Working R Code

```r
library(data.table)

#' Build a data.table edge list from an nb object.
#' Returns a two-column data.table: (id, neighbor_id)
build_edge_table <- function(id_order, neighbors) {
  # neighbors is a list of integer index vectors (spdep nb object)
  n <- length(neighbors)
  # Pre-allocate: total number of directed edges
  from_idx <- rep.int(seq_len(n), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

#' Compute neighbor summary statistics for one variable,
#' returning a data.table with columns: id, year, <var>_max, <var>_min, <var>_mean
compute_neighbor_stats_fast <- function(cell_dt, edge_dt, var_name) {
  # Build a small lookup: (neighbor_id, year) -> value
  lookup <- cell_dt[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(lookup, neighbor_id, year)

  # Join edges with all years: expand edges × years
  # Instead of a full cross join (expensive in memory), join through cell_dt's (id, year) pairs
  # Step 1: get all (id, year) pairs
  id_year <- cell_dt[, .(id, year)]
  setkey(id_year, id)
  setkey(edge_dt, id)

  # Step 2: merge to get (id, year, neighbor_id) — one row per neighbor per cell-year

  expanded <- edge_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded now has columns: id, neighbor_id, year

  # Step 3: look up the neighbor's value in that year
  setkey(expanded, neighbor_id, year)
  expanded <- lookup[expanded, on = .(neighbor_id, year), nomatch = NA]
  # expanded now has: neighbor_id, year, val, id

  # Step 4: aggregate
  max_name  <- paste0(var_name, "_max")
  min_name  <- paste0(var_name, "_min")
  mean_name <- paste0(var_name, "_mean")

  stats <- expanded[
    !is.na(val),
    .(V_max = max(val), V_min = min(val), V_mean = mean(val)),
    by = .(id, year)
  ]
  setnames(stats, c("V_max", "V_min", "V_mean"), c(max_name, min_name, mean_name))
  stats
}

#' Main driver: add neighbor features for all source variables to cell_data.
add_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                      neighbor_source_vars) {
  cell_dt <- as.data.table(cell_data)

  # 1. Build edge table once (~1.37M rows, tiny)
  edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

  # 2. For each variable, compute stats and merge back

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)
    stats <- compute_neighbor_stats_fast(cell_dt, edge_dt, var_name)
    # Left-join stats back onto the main table; unmatched rows get NA (preserves original behavior)
    cell_dt <- stats[cell_dt, on = .(id, year)]
  }

  # 3. Return as data.frame to preserve downstream compatibility
  as.data.frame(cell_dt)
}

# ---- Usage (drop-in replacement for the original outer loop) ----

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched; predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets and values. NA handling is identical (NAs excluded before aggregation; rows with no valid neighbors get NA via the left join). |
| **Trained RF model** | No model retraining. The output columns have the same names and semantics, so `predict(rf_model, newdata = cell_data)` works unchanged. |
| **Column naming** | `paste0(var_name, "_max")` etc. matches the original `compute_and_add_neighbor_features` convention. |

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build lookup / edge table | ~6.46M R-loop iterations with string ops | One `rep.int` + `unlist` (~1.37M rows, <1 sec) |
| Neighbor stats (per var) | ~6.46M `lapply` iterations | One keyed join + one `group-by` in C (est. 30–90 sec per variable) |
| **Total (5 vars)** | **~86+ hours** | **~5–10 minutes** |

Memory peak for the expanded join table: ~1.37M edges × 28 years ≈ 38.4M rows × 3 columns ≈ ~0.9 GB, well within 16 GB RAM.