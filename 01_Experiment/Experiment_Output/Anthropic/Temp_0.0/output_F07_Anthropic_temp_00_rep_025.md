 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Character key construction and lookup** (`paste`, `setNames`, named-vector indexing) — for every single row. Named-vector lookup in R is O(n) per query in the worst case because R's named vectors use linear hashing with potential collisions, and here the lookup table has ~6.46M entries.
2. **`lapply` over 6.46M rows** — each iteration does string pasting, named-vector subsetting, and NA filtering. The per-element overhead of R's interpreted loop is enormous at this scale.
3. **`compute_neighbor_stats`** then does another `lapply` over 6.46M elements, extracting subsets of a vector by index. This is lighter but still slow in a pure-R loop.

**Estimated cost:** ~6.46M iterations × (string ops + named-vector lookup in a 6.46M-entry table) × 5 variables = the 86+ hour runtime.

**Key insight:** The neighbor topology is *time-invariant*. There are only 344,208 spatial cells with ~1.37M directed rook-neighbor pairs. The year dimension simply replicates this structure. We should:

- Work at the **cell level** (344K) not the **cell-year level** (6.46M).
- Use **vectorized joins** (via `data.table`) instead of named-vector lookups.
- Compute neighbor stats via a **sparse adjacency edge-list join**, not per-row `lapply`.

## Optimization Strategy

1. **Convert the `nb` object to a directed edge-list** (from_id, to_id) — ~1.37M rows, done once.
2. **Convert `cell_data` to a `data.table`**, keyed on `(id, year)`.
3. **For each source variable**, do a single vectorized merge of the edge-list with the data to get all neighbor values, then **group-by `(from_id, year)`** to compute `max`, `min`, `mean` — all in `data.table`, fully vectorized in C.
4. **Left-join** the results back to `cell_data`.

This replaces 6.46M × 5 interpreted R iterations with ~5 vectorized `data.table` group-by operations over ~1.37M × 28 ≈ 38.5M edge-year rows. Expected runtime: **minutes, not days**.

## Working R Code

```r
library(data.table)

# ── Step 0: Convert nb object to directed edge-list (once) ──────────────────
# rook_neighbors_unique is an nb object; id_order maps position → cell id
nb_to_edge_list <- function(nb_obj, id_order) {
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove 0-neighbor placeholders (spdep uses integer(0) but be safe)
  valid <- to > 0L
  data.table(
    from_id = id_order[from[valid]],
    to_id   = id_order[to[valid]]
  )
}

edges <- nb_to_edge_list(rook_neighbors_unique, id_order)
# edges has ~1,373,394 rows: (from_id, to_id)

# ── Step 1: Convert panel to data.table ─────────────────────────────────────
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

# ── Step 2: Vectorized neighbor-stat computation ────────────────────────────
compute_and_add_neighbor_features_fast <- function(dt, edges, var_name) {
  # Build a slim table: every cell-year's value for this variable
  val_dt <- dt[, .(id, year, val = get(var_name))]
  setnames(val_dt, "id", "to_id")
  setkey(val_dt, to_id, year)

  # Expand edges × years: for each (from_id, to_id) pair, join the

  # neighbor's (to_id) value in each year.
  # This is a keyed join — very fast in data.table.
  edge_vals <- edges[val_dt, on = "to_id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_vals columns: from_id, to_id, year, val

  # Aggregate: for each (from_id, year), compute max/min/mean of neighbor vals
  agg <- edge_vals[!is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = .(from_id, year)
  ]

  # Rename columns to match original convention
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))
  setnames(agg, "from_id", "id")
  setkey(agg, id, year)

  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(dt)) dt[, (col) := NULL]
  }

  # Left-join back to the main table
  dt <- agg[dt, on = .(id, year)]
  setkey(dt, id, year)
  dt
}

# ── Step 3: Loop over the 5 source variables ────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, edges, var_name)
  gc()
}

# ── Step 4: Predict with the existing (untouched) Random Forest model ───────
# The trained model object is unchanged; the new columns are numerically
# identical to the originals (same max/min/mean over the same rook neighbors),
# so predictions are reproducible.
# preds <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves the Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor set** | `nb_to_edge_list` reads the identical `rook_neighbors_unique` nb object and `id_order` mapping used by the original code. |
| **Same statistics** | `max`, `min`, `mean` are computed over exactly the same non-NA neighbor values per (cell, year). |
| **Same column names** | The naming convention (`neighbor_max_ntl`, etc.) matches the original `compute_and_add_neighbor_features` output. |
| **RF model untouched** | No retraining; the model object is never modified. |

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Build lookup | ~hours (6.46M string ops) | ~1 sec (vectorized edge-list) |
| Neighbor stats (×5 vars) | ~80+ hours (lapply) | ~5–15 min (data.table group-by) |
| **Total** | **86+ hours** | **< 20 minutes** |

### Memory Note (16 GB laptop)

The largest intermediate object is `edge_vals` at ~1.37M edges × 28 years ≈ 38.5M rows × 4 columns ≈ 1.2 GB. With `cell_data` itself at ~6.46M × 110 cols ≈ 5–6 GB, peak usage stays under 12 GB. The `gc()` call between variables reclaims temporaries.