 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, the function:

1. Performs character coercion and named-vector lookups (`id_to_ref`, `idx_lookup`) — these are O(n) hash-style lookups but repeated ~6.46M times with string paste/match overhead.
2. Constructs `paste(..., sep="_")` keys for every neighbor of every row — millions of small string allocations.
3. Returns a ragged list of integer vectors.

**`compute_neighbor_stats`** then iterates over that 6.46M-element list again, extracting and summarizing values per row. The `do.call(rbind, result)` on 6.46M 3-element vectors is also expensive.

Both functions are pure-R row-wise loops over millions of rows with heavy string operations. This is the classic R anti-pattern.

## Optimization Strategy

**Core insight:** The neighbor graph is *static across years*. A cell's spatial neighbors are the same in every year. So we can:

1. **Vectorize the lookup construction** using `data.table` keyed joins instead of per-row `lapply` + string pasting. Build an edge-list of `(cell_id, neighbor_id)` once, then join on `(neighbor_id, year)` to get neighbor row indices for all rows simultaneously.

2. **Vectorize the stats computation** by performing a single grouped aggregation (`max`, `min`, `mean`) over the edge-list joined with variable values — no per-row `lapply` needed.

3. **Avoid ragged list storage entirely.** A `data.table` grouped-by operation replaces both `build_neighbor_lookup` and `compute_neighbor_stats`.

This reduces ~6.46M × k R-level iterations to a handful of vectorized `data.table` joins and group-bys. Expected speedup: **~100–500×**, bringing runtime from 86+ hours to minutes.

## Optimized Working R Code

```r
library(data.table)

#
# Step 1 — One-time: build a flat directed edge list from the nb object.
#          This is done ONCE regardless of how many variables you process.
#
build_edge_dt <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)
  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

#
# Step 2 — Compute neighbor stats for one variable using a single keyed join
#          and grouped aggregation. Returns the original data.table with three
#          new columns appended.
#
compute_neighbor_features_fast <- function(cell_dt, var_name, edge_dt) {
  # --- build a slim lookup: (to_id, year) -> value ---
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setnames(val_dt, "id", "to_id")
  setkey(val_dt, to_id, year)

  # --- expand edges × years: for every (from_id, year) get neighbor values ---
  #     Join edge_dt with cell_dt's (from_id, year) combos, then look up vals.
  #     Memory-efficient: we only need from_id, year from cell_dt.
  from_dt <- cell_dt[, .(from_id = id, year)]
  setkey(edge_dt, from_id)
  setkey(from_dt, from_id)

  # Cartesian-ish join: for each (from_id, year), attach all to_id neighbors

  merged <- edge_dt[from_dt, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # merged now has columns: from_id, to_id, year

  # Attach the neighbor's value for that year
  merged <- val_dt[merged, on = c("to_id", "year"), nomatch = NA]
  # merged now has columns: to_id, year, val, from_id

  # --- grouped aggregation ---
  stats <- merged[!is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    keyby = .(from_id, year)
  ]

  # Name columns to match the original pipeline's convention
  new_names <- paste0(var_name, c("_max", "_min", "_mean"))
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), new_names)

  # --- left-join back onto cell_dt ---
  result <- merge(cell_dt, stats,
    by.x = c("id", "year"),
    by.y = c("from_id", "year"),
    all.x = TRUE, sort = FALSE
  )
  result
}

# ============================================================
# Main pipeline (replaces the entire outer loop)
# ============================================================

# Convert to data.table once (in-place conversion avoids a copy)
setDT(cell_data)

# Build edge list ONCE from the precomputed nb object
edge_dt <- build_edge_dt(id_order, rook_neighbors_unique)

# Feature construction for every source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_data <- compute_neighbor_features_fast(cell_data, var_name, edge_dt)
}

# cell_data now contains <var>_max, <var>_min, <var>_mean for all 5 variables.
# Feed directly into the existing (already trained) Random Forest predict() call.
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed over the identical neighbor sets (same directed rook edges, same year match, same NA exclusion). The estimand is unchanged. |
| **Trained RF model** | No retraining. The code only constructs predictor columns; the model object is untouched. Column names follow the same `<var>_max/min/mean` convention so `predict()` works as before. |
| **Memory (16 GB laptop)** | The largest intermediate object (`merged`) has ~6.46M × avg_neighbors ≈ ~26M rows × 4 columns of simple types ≈ ~0.8 GB. Well within budget. If memory is tight, process one variable at a time (as the loop already does) and `rm(merged)` inside the function. |

## Expected Performance

| Phase | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (6.46M `lapply` + string ops) | ~10 s (`data.table` edge list build) |
| Stats per variable | ~17 h each (6.46M `lapply`) | ~30–90 s each (vectorized join + group-by) |
| **Total (5 variables)** | **86+ hours** | **~5–10 minutes** |

The entire bottleneck is eliminated by replacing row-wise R loops and string key construction with `data.table` vectorized keyed joins and grouped aggregations.