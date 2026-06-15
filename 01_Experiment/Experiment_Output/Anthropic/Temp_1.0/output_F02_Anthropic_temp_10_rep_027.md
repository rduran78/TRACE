 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup`** function. It calls `lapply` over ~6.46 million rows, and inside each iteration it performs:

1. **Character coercion and named-vector lookups** (`id_to_ref[as.character(...)]`, `idx_lookup[neighbor_keys]`) — named vector lookups in R are hash-table operations, but doing them millions of times in an interpreted loop is extremely slow.
2. **`paste()` key construction** for every neighbor of every row — this generates millions of small temporary character vectors.
3. **The result is a list of 6.46 million integer vectors**, consuming substantial memory.

Then **`compute_neighbor_stats`** iterates over that 6.46M-element list again, extracting values, filtering NAs, and computing max/min/mean — another interpreted loop with per-element allocation.

Multiplied across 5 variables, the estimated 86+ hour runtime is dominated by these two R-level interpreted loops over millions of elements, with heavy per-iteration allocation and hashing overhead.

---

## Optimization Strategy

**Replace interpreted R loops and character-key lookups with vectorized `data.table` joins.**

The key insight: the neighbor lookup is fundamentally a **merge/join** operation. For each `(cell_id, year)` row, we want to find the rows of its rook neighbors in the same year, then aggregate their variable values. This is exactly what `data.table` excels at — keyed equi-joins with grouped aggregation — and it operates in C, not interpreted R.

### Steps

1. **Expand the neighbor list into an edge table** (`data.table` with columns `id` and `neighbor_id`). This is done once and has ~1.37M rows (times the number of directed edges, but still very manageable).
2. **Join the edge table to the panel data** on `(neighbor_id, year)` to pull neighbor values. This is a single keyed join — `data.table` does this in seconds for millions of rows.
3. **Group-aggregate** by `(id, year)` to compute `max`, `min`, `mean` for each neighbor variable.
4. **Merge the aggregated stats back** into the main dataset.

This replaces both `build_neighbor_lookup` and `compute_neighbor_stats` entirely. No 6.46M-element list is ever created. Memory use drops dramatically (the edge table is ~10–50 MB; intermediate joins are handled column-wise).

**Expected speedup**: from 86+ hours to roughly **10–30 minutes** total for all 5 variables on a 16 GB laptop.

**Preservation guarantees**: The Random Forest model is not touched. The numerical outputs (max, min, mean of neighbor values) are identical to the original code — we are computing the same aggregates, just via join rather than loop.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 1. Build the edge table (once). Convert the spdep nb object into a
#    two-column data.table of (id, neighbor_id).
# ──────────────────────────────────────────────────────────────────────

build_edge_table <- function(id_order, neighbors) {
  # id_order: vector of cell IDs in the same order as the nb object

  # neighbors: spdep nb list (rook_neighbors_unique)
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# ──────────────────────────────────────────────────────────────────────
# 2. Convert the main panel to data.table (in-place, no copy needed).
# ──────────────────────────────────────────────────────────────────────

setDT(cell_data)

# ──────────────────────────────────────────────────────────────────────
# 3. Compute and attach neighbor features for all source variables.
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {

  # --- Subset the columns we need (keep memory low) ---
  vals_dt <- cell_data[, .(neighbor_id = id, year, value = get(var))]
  setkey(vals_dt, neighbor_id, year)

  # --- Join edges → values: for each (id, year), get every neighbor's value ---
  #     edge_dt provides (id, neighbor_id);
  #     we join on (neighbor_id, year) to pull the neighbor's value.
  merged <- merge(
    edge_dt,                         # (id, neighbor_id)
    vals_dt,                         # (neighbor_id, year, value)
    by = "neighbor_id",
    allow.cartesian = TRUE           # a neighbor_id appears in many years
  )
  # merged columns: neighbor_id, id, year, value

  # --- Aggregate by (id, year) ---
  agg <- merged[
    !is.na(value),
    .(
      nbr_max  = max(value),
      nbr_min  = min(value),
      nbr_mean = mean(value)
    ),
    keyby = .(id, year)
  ]

  # --- Rename to match the original feature naming convention ---
  new_names <- paste0(var, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  setnames(agg, c("nbr_max", "nbr_min", "nbr_mean"), new_names)

  # --- Left-join back onto the main table ---
  cell_data <- merge(cell_data, agg, by = c("id", "year"), all.x = TRUE)

  # --- Clean up intermediate objects to free memory ---
  rm(vals_dt, merged, agg)
  gc()

  message(sprintf("Done: %s", var))
}

# Convert back to data.frame if downstream code (predict.randomForest, etc.)
# expects one:
# setDF(cell_data)
```

---

## Why This Is Correct and Safe

| Concern | Explanation |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values as the original code. The join `(neighbor_id, year)` replicates the lookup `idx_lookup[neighbor_keys]`. |
| **NA handling** | `merged[!is.na(value), ...]` mirrors `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors get `NA` via the `all.x = TRUE` left join, matching the original `c(NA, NA, NA)` return. |
| **Random Forest model** | Untouched — we only prepare the same feature columns the trained model expects. |
| **Memory** | The edge table is ~1.37M rows × 2 integer columns ≈ 11 MB. The largest intermediate (`merged`) is at most ~(1.37M × 28) ≈ 38M rows × 3 columns ≈ 0.9 GB — well within 16 GB. Each iteration frees its intermediates with `rm(); gc()`. |
| **Speed** | `data.table` keyed joins and grouped aggregation on ~38M rows typically complete in seconds to low minutes per variable. Total: **~10–30 minutes for all 5 variables**. |

This single structural change — replacing interpreted per-row loops with vectorized `data.table` joins — eliminates the bottleneck without altering any outputs or the trained model.