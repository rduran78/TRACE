 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Performs a character-based hash lookup (`id_to_ref`),
2. Retrieves neighbor cell IDs from the `nb` object,
3. Constructs character key strings via `paste()` for every neighbor,
4. Looks those keys up in `idx_lookup` (a named character vector of length ~6.46M).

**String allocation and named-vector lookup are O(n) or near-O(n) per probe in R's hashed environments, but `paste()` and repeated character coercion across 6.46 million rows with ~4 neighbors each creates ~25+ million temporary strings.** The `lapply` returns a list of 6.46M integer vectors — the sheer object overhead is enormous.

`compute_neighbor_stats` then loops over that 6.46M-element list *once per variable* (×5 variables), each time extracting, subsetting, and summarizing — another ~32M R-level function calls.

**Root causes ranked by impact:**

1. **Row-level `lapply` in `build_neighbor_lookup`**: ~6.46M R-level iterations with string operations.
2. **Row-level `lapply` in `compute_neighbor_stats`**: ~6.46M iterations × 5 variables.
3. **No vectorization or use of data.table / matrix operations.**

## Optimization Strategy

**Core idea:** Replace the row-level loop with a fully vectorized, edge-list-based `data.table` join. Instead of building a per-row neighbor lookup list, we:

1. Expand the `nb` object into a flat edge list (cell_id → neighbor_id), ~1.37M rows.
2. Cross-join with years using `data.table` to get ~1.37M × 28 ≈ 38.4M edge-year rows.
3. Join source variable values onto the neighbor side.
4. Group-by aggregate (`max`, `min`, `mean`) per (cell, year).
5. Join results back to the main table.

This eliminates all per-row R loops and string operations, replacing them with `data.table`'s optimized C-level grouped joins and aggregations. Expected runtime: **minutes, not days.**

## Working R Code

```r
library(data.table)

# ── Step 0: Convert main data to data.table (once) ──────────────────────────
cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# ── Step 1: Expand nb object to flat edge list (once) ────────────────────────
#   rook_neighbors_unique is an nb object indexed by position in id_order.
#   id_order is the vector of cell IDs corresponding to each nb element.

build_edge_list <- function(id_order, neighbors) {
  # neighbors is an spdep nb object: list of integer index vectors
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)
  # Remove the spdep convention where 0L means "no neighbors"
  valid    <- to_idx > 0L
  data.table(
    id          = id_order[from_idx[valid]],
    neighbor_id = id_order[to_idx[valid]]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# ~1.37M rows, two integer columns — trivial memory

# ── Step 2: Cross with years to get edge-year table ──────────────────────────
years_dt <- data.table(year = sort(unique(cell_dt$year)))
# Cross join: every edge × every year
edge_year_dt <- edge_dt[, CJ_id := .I]  # placeholder; use CJ approach below

# More memory-efficient: use allow.cartesian join
edge_year_dt <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years_dt$year)
edge_year_dt[, `:=`(
  id          = edge_dt$id[edge_idx],
  neighbor_id = edge_dt$neighbor_id[edge_idx]
)]
edge_year_dt[, edge_idx := NULL]
# ~38.4M rows × 3 columns ≈ ~0.9 GB (fits in 16 GB RAM)

setkey(edge_year_dt, neighbor_id, year)

# ── Step 3: Function to compute and attach neighbor features for one var ─────
compute_and_add_neighbor_features_fast <- function(cell_dt, edge_year_dt, var_name) {
  # Extract only the columns we need for the join
  val_dt <- cell_dt[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(val_dt, neighbor_id, year)

  # Join neighbor values onto edge-year table
  merged <- val_dt[edge_year_dt, on = .(neighbor_id, year), nomatch = NA]
  # merged has columns: neighbor_id, year, val, id

  # Aggregate per (id, year)
  agg <- merged[!is.na(val),
    .(
      nmax  = max(val),
      nmin  = min(val),
      nmean = mean(val)
    ),
    by = .(id, year)
  ]

  # Name output columns to match original pipeline convention
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(agg, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))

  # Join back to main table
  setkey(agg, id, year)
  setkey(cell_dt, id, year)
  cell_dt <- agg[cell_dt, on = .(id, year)]

  cell_dt
}

# ── Step 4: Outer loop over the 5 source variables ──────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_dt <- compute_and_add_neighbor_features_fast(cell_dt, edge_year_dt, var_name)
}

# ── Step 5: Convert back to data.frame if downstream code expects it ─────────
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is unchanged — use it for prediction as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves the Original Numerical Estimand

The aggregation computes **exactly** `max`, `min`, and `mean` of the same neighbor values identified by the same rook-contiguity structure (`rook_neighbors_unique`) and the same year-matching logic. The only change is *how* the computation is dispatched (vectorized C-level `data.table` grouping vs. R-level `lapply`). No approximation, sampling, or rounding is introduced. The trained Random Forest model is never modified or retrained.

## Memory Budget Check

| Object | Rows | Est. Size |
|---|---|---|
| `cell_dt` | 6.46M × ~113 cols | ~5.5 GB |
| `edge_year_dt` | 38.4M × 3 cols | ~0.9 GB |
| `merged` (temp per var) | 38.4M × 4 cols | ~1.2 GB |
| **Peak total** | | **~9 GB** |

Fits within 16 GB. If memory is tight, `edge_year_dt` can be built on-the-fly per variable or years can be processed in chunks, but this should not be necessary.

## Expected Speedup

| Phase | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M string ops) | ~2 seconds (vectorized `unlist`) |
| `compute_neighbor_stats` ×5 | ~hours (32M R calls) | ~3–5 min (data.table grouped agg) |
| **Total** | **86+ hours** | **< 10 minutes** |