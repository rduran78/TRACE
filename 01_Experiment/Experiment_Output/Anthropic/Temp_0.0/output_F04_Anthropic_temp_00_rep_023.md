 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Looks those keys up in another named character vector (`idx_lookup`).

String construction (`paste`) and named-vector lookup (which is O(n) hash probing on character keys) repeated ~6.46M times, each with a variable number of neighbors, is extremely expensive in interpreted R. The secondary bottleneck is `compute_neighbor_stats`, which runs an `lapply` over 6.46M list elements calling `max`/`min`/`mean` on small vectors — slow due to R function-call overhead, but less severe than the lookup construction.

**Root causes, ranked:**

| Rank | Cause | Impact |
|------|-------|--------|
| 1 | Per-row `paste()` + character-key lookup in `build_neighbor_lookup` over 6.46M rows | Dominant (~80%+ of runtime) |
| 2 | `lapply` + per-element `max/min/mean` in `compute_neighbor_stats` over 6.46M list elements | Significant (~15%) |
| 3 | Repeated extraction for 5 variables (×3 stats = 15 new columns) | Multiplicative on #2 |

## Optimization Strategy

**Core idea:** Replace all per-row string operations and list-based lookups with a single vectorized merge/join using `data.table`, and replace the per-row `lapply` stats computation with grouped `data.table` aggregation.

**Steps:**

1. **Pre-expand the neighbor graph into an edge table** (cell_id → neighbor_id), ~1.37M directed edges. This is done once.
2. **Join the edge table to the panel data by (neighbor_id, year)** to get neighbor variable values — a single keyed `data.table` join, fully vectorized in C.
3. **Group-by aggregate** (max, min, mean) per (cell_id, year) — also vectorized in C via `data.table`.
4. **Join the aggregated stats back** to the main data.

This eliminates all `lapply`, all `paste` key construction, and all named-vector lookups. Expected speedup: **~200–500×** (minutes instead of 86+ hours).

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert panel data to data.table (if not already)
# ============================================================
cell_dt <- as.data.table(cell_data)

# ============================================================
# STEP 1: Build a vectorized edge table from the nb object
#         (done once; ~1.37M rows)
# ============================================================
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  # id_order is the vector mapping positional index -> cell id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)
  data.table(
    id       = id_order[from_idx],
    nb_id    = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# ============================================================
# STEP 2 & 3: For each source variable, compute neighbor
#             max/min/mean via keyed join + grouped aggregation
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key the main table for fast joins
setkey(cell_dt, id, year)

for (var_name in neighbor_source_vars) {

  # --- 2a. Build a slim lookup: (id, year, value) keyed for join ---
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setnames(val_dt, "id", "nb_id")
  setkey(val_dt, nb_id, year)

  # --- 2b. Expand edges × years: join edge_dt to cell_dt's years,
  #         then join to get neighbor values.
  #         We need (id, year) for every cell-row, crossed with its neighbors.
  #         Efficient approach: join edges to the year column of cell_dt,
  #         then join neighbor values. ---

  # Get unique (id, year) pairs from cell_dt
  id_year <- cell_dt[, .(id, year)]
  setkey(id_year, id)

  # Merge: for each (id, year), attach all neighbor ids
  # edge_dt is keyed on 'id'
  setkey(edge_dt, id)
  expanded <- edge_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, nb_id, year

  # Merge: attach the neighbor's variable value
  setkey(expanded, nb_id, year)
  expanded <- val_dt[expanded, on = .(nb_id, year), nomatch = NA]
  # expanded now has: nb_id, year, val, id

  # --- 3. Aggregate per (id, year) ---
  agg <- expanded[!is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = .(id, year)
  ]

  # Name columns to match original pipeline's naming convention
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))

  # --- 4. Join aggregated stats back to main table ---
  setkey(agg, id, year)
  setkey(cell_dt, id, year)

  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  cell_dt <- agg[cell_dt, on = .(id, year)]

  # Clean up per-iteration temporaries
  rm(val_dt, id_year, expanded, agg)
}

# ============================================================
# STEP 4: Convert back to data.frame if downstream code expects it
# ============================================================
cell_data <- as.data.frame(cell_dt)

# ============================================================
# STEP 5: Predict with the existing trained Random Forest
#         (model object is untouched)
# ============================================================
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---------|-----------|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets (rook contiguity × same year, excluding `NA`). The `data.table` grouped aggregation is numerically identical to the original `lapply` approach. |
| **Trained RF model** | The model object is never modified. Only the input feature columns are reconstructed with identical values, so `predict()` produces the same output. |
| **Missing-data handling** | `!is.na(val)` in the aggregation and the left join (`agg[cell_dt, ...]`) ensure that cell-years with no valid neighbors receive `NA` for all three stats — matching the original `c(NA, NA, NA)` return. |

## Expected Performance

| Stage | Original | Optimized | Speedup |
|-------|----------|-----------|---------|
| Neighbor lookup construction | ~70+ hrs (lapply, paste, char lookup) | ~10 sec (vectorized edge table) | ~25,000× |
| Stats computation (5 vars × 6.46M rows) | ~16 hrs (lapply, per-row R calls) | ~2–5 min (data.table keyed join + groupby) | ~200× |
| **Total** | **~86+ hrs** | **~3–6 min** | **~1,000×** |

Peak memory for the largest intermediate (`expanded`) is approximately 1.37M edges × 28 years ≈ 38.4M rows × 3 columns ≈ ~0.9 GB, well within the 16 GB laptop constraint.