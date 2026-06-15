 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is in **`build_neighbor_lookup`**, which runs an `lapply` over **~6.46 million rows**, performing per-row string pasting and named-vector lookups. Named vector lookups in R via `[` on character keys are O(n) hash probes per call, and doing this 6.46M times with multiple neighbor keys each time is catastrophically slow. The secondary bottleneck is **`compute_neighbor_stats`**, which also loops over 6.46M rows in pure R, extracting and summarizing small vectors one at a time. Together, these two functions dominate the 86+ hour runtime.

**Specific problems:**

1. **`build_neighbor_lookup`**: `paste()` and named-vector indexing inside a per-row `lapply` over 6.46M rows. Each iteration creates temporary character vectors and performs hash lookups. This is the single largest bottleneck.
2. **`compute_neighbor_stats`**: Pure-R `lapply` over 6.46M list elements, each calling `max`, `min`, `mean` on small vectors. The overhead of 6.46M R function calls plus `do.call(rbind, ...)` on a 6.46M-element list is enormous.
3. **Memory**: Storing `neighbor_lookup` as a list of 6.46M integer vectors has high overhead (each list element is a separate R object with header overhead). With 16 GB RAM this is tight.

---

## Optimization Strategy

**Replace row-level R loops with vectorized `data.table` joins and grouped aggregations.**

The key insight: `build_neighbor_lookup` is essentially building a **join table** — for each `(cell, year)`, find all `(neighbor_cell, year)` rows. This is a classic equi-join that `data.table` handles in seconds. Then `compute_neighbor_stats` is a **grouped aggregation** (max, min, mean by group), which `data.table` also handles natively in C.

**Plan:**

1. Build an **edge table** (`data.table`) mapping each `cell_id` to its neighbor `cell_id`s — this is a one-time expansion of the `nb` object (~1.37M rows).
2. For each variable, join the edge table with the panel data on `(neighbor_id, year)` to retrieve neighbor values, then compute grouped `max`, `min`, `mean` by `(cell_id, year)`.
3. Merge the results back into `cell_data`.

This eliminates all per-row R loops, replaces them with vectorized C-level operations, and reduces memory by never materializing a 6.46M-element list.

**Expected speedup:** From 86+ hours to roughly **5–20 minutes** depending on disk I/O and RAM pressure.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Build the edge table from the nb object (one-time, ~1.37M rows)
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: a list of integer index vectors
  # id_order maps positional index -> cell_id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the 0-neighbor sentinel if present (spdep uses 0 for no neighbors)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id       = id_order[from_idx],   # focal cell id
    nb_id    = id_order[to_idx]      # neighbor cell id
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# ──────────────────────────────────────────────────────────────────────
# Step 2: Convert cell_data to data.table (in-place, no copy if already DT)
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns have consistent types
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]
edge_dt[,   id   := as.integer(id)]
edge_dt[,   nb_id := as.integer(nb_id)]

# ──────────────────────────────────────────────────────────────────────
# Step 3: Compute neighbor stats for each variable via join + group-by
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  cat(sprintf("[%s] Computing neighbor features for: %s\n", Sys.time(), var_name))

  # --- 3a. Build a slim lookup: (cell_id, year, value) ---
  # Only the columns we need, to minimize memory during the join.
  val_dt <- cell_data[, .(nb_id = id, year, val = get(var_name))]
  setkey(val_dt, nb_id, year)

  # --- 3b. Expand edges × years via join ---
  # For every (focal_id, nb_id) edge and every year, pull the neighbor's value.
  # Join edge_dt with val_dt on (nb_id, year).
  # First, cross edge_dt with the year column by joining through cell_data's (id, year).
  # More efficient: join edge_dt to cell_data to get (id, year, nb_id),
  # then join to val_dt to get the neighbor value.

  focal_dt <- cell_data[, .(id, year)]
  setkey(focal_dt, id)
  setkey(edge_dt, id)

  # This gives every (focal_id, year, neighbor_id) triple
  expanded <- edge_dt[focal_dt, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded columns: id, nb_id, year

  # Now join to get the neighbor's value for that (nb_id, year)
  setkey(expanded, nb_id, year)
  expanded[val_dt, on = .(nb_id, year), val := i.val]

  # --- 3c. Grouped aggregation ---
  stats <- expanded[!is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    keyby = .(id, year)
  ]

  # Name the new columns to match the original pipeline's naming convention
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  # --- 3d. Merge back into cell_data ---
  # Remove old columns if they exist (idempotent reruns)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- stats[cell_data, on = .(id, year)]

  # Clean up large temporaries

  rm(val_dt, focal_dt, expanded, stats)
  gc()

  cat(sprintf("[%s] Done: %s\n", Sys.time(), var_name))
}

# ──────────────────────────────────────────────────────────────────────
# Step 4: Predict with the existing trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The trained model object is used as-is. No retraining.
# Example (adjust to your actual model object and predict call):
#
#   predictions <- predict(trained_rf_model, newdata = cell_data)
#
# The numerical estimand is preserved because we compute identical
# max, min, mean aggregations over the same neighbor sets.
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor sets** | `build_edge_table` expands the identical `nb` object and `id_order` mapping used by the original `build_neighbor_lookup`. |
| **Same statistics** | `max`, `min`, `mean` are computed on the same non-NA neighbor values per `(id, year)` group. |
| **NA handling** | Rows with no valid neighbors get `NA` for all three stats (the left join in step 3d fills unmatched rows with `NA`). |
| **Trained RF model** | No model code is touched. The feature columns have the same names and identical values, so `predict()` produces the same output. |

---

## Memory Notes for 16 GB RAM

- `expanded` is the largest temporary: ~6.46M rows × ~4 neighbors ≈ 26M rows × 3 columns ≈ ~600 MB. This is processed one variable at a time and freed with `rm(); gc()`.
- `cell_data` itself (~6.46M × 110 cols) is ~5–7 GB. This leaves adequate headroom.
- If memory is still tight, process years in batches (e.g., 7 years at a time) inside the `for` loop, then `rbindlist` the stats. But this is unlikely to be necessary.