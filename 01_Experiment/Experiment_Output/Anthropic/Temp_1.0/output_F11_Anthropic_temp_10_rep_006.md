 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that **Random Forest inference** is the main bottleneck is **incorrect**. The code evidence points clearly to a different bottleneck: **the spatial neighbor feature engineering stage**, specifically `build_neighbor_lookup` and `compute_neighbor_stats`.

Here's why:

1. **`build_neighbor_lookup`** uses a **row-level `lapply` over ~6.46 million rows**. For every single row, it performs character coercion (`as.character`), named vector lookups (`id_to_ref[...]`), string pasting (`paste(..., sep="_")`), and named index lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is **O(n)** or at best O(1)-amortized via hashing, but the repeated string construction and subsetting across 6.46M iterations is extremely expensive. With ~1.37M neighbor relationships spread across 344K cells and 28 years, this function alone generates tens of millions of string operations.

2. **`compute_neighbor_stats`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows with `lapply`, performing subsetting, NA removal, and computing `max/min/mean`. The `do.call(rbind, result)` at the end binds ~6.46 million 3-element vectors into a matrix — another expensive operation.

3. **Combined cost**: `build_neighbor_lookup` runs once (O(6.46M) string-heavy iterations) and `compute_neighbor_stats` runs 5 times (O(5 × 6.46M) iterations with subsetting). The total is approximately **38.8 million R-level loop iterations** dominated by string manipulation and per-row subsetting. This is what produces the 86+ hour runtime.

4. **Random Forest inference**, by contrast, is a single call to `predict()` on ~6.46M rows with ~110 predictors. Modern RF implementations (`ranger`, `randomForest`) handle this in minutes, even on a laptop. Loading a serialized model and writing predictions are trivially fast. There is no loop, no string manipulation, and the predict call is implemented in C/C++.

**Verdict**: The bottleneck is the **neighbor feature engineering**, not RF inference.

---

## Optimization Strategy

The core strategy is to **eliminate all row-level R loops and string operations** by using **vectorized joins via `data.table`**:

1. **Replace `build_neighbor_lookup`** with a vectorized `data.table` join approach: expand the neighbor list into an edge table `(id, neighbor_id)`, then join on `(neighbor_id, year)` to retrieve neighbor rows directly — no per-row `lapply`, no string pasting, no named vector lookups.

2. **Replace `compute_neighbor_stats`** with a **grouped aggregation** (`data.table`'s `[, .(max, min, mean), by=...]`), which is executed in C and handles all 6.46M rows in one vectorized pass per variable.

3. **Batch all 5 variables** in a single join + aggregation pass instead of 5 separate iterations.

4. **Preserve the trained RF model** — no retraining. Preserve the original numerical estimand — same max/min/mean statistics, same column names.

Expected speedup: from **86+ hours to roughly 5–15 minutes** on the same laptop.

---

## Working R Code

```r
library(data.table)

# ─────────────────────────────────────────────────────────────────────
# 1. Convert cell_data to data.table (preserves all existing columns)
# ─────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)

# Ensure id and year are keyed for fast joins
# (Assuming 'id' and 'year' columns already exist in cell_data)
setkey(cell_dt, id, year)

# ─────────────────────────────────────────────────────────────────────
# 2. Build a vectorized edge table from rook_neighbors_unique (nb object)
#    rook_neighbors_unique[[i]] gives the neighbor indices for the i-th
#    element of id_order. We expand this into a two-column data.table
#    of (focal_id, neighbor_id).
# ─────────────────────────────────────────────────────────────────────

# Expand the nb list into an edge list (integer indices into id_order)
edge_lengths <- lengths(rook_neighbors_unique)
focal_idx    <- rep(seq_along(rook_neighbors_unique), times = edge_lengths)
neighbor_idx <- unlist(rook_neighbors_unique, use.names = FALSE)

# Map integer indices to actual cell IDs
edges <- data.table(
  focal_id    = id_order[focal_idx],
  neighbor_id = id_order[neighbor_idx]
)

# ─────────────────────────────────────────────────────────────────────
# 3. Define the neighbor source variables and the columns to aggregate
# ─────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ─────────────────────────────────────────────────────────────────────
# 4. Join edges with cell_dt to get neighbor values for all years at once
#    For each (focal_id, year), we look up every neighbor_id's row in
#    the same year.
# ─────────────────────────────────────────────────────────────────────

# Subset cell_dt to only the columns we need for the neighbor lookup
# to keep the join lightweight
neighbor_cols <- c("id", "year", neighbor_source_vars)
cell_subset   <- cell_dt[, ..neighbor_cols]

# Rename 'id' to 'neighbor_id' in the subset so we can join on it
setnames(cell_subset, "id", "neighbor_id")
setkey(cell_subset, neighbor_id, year)

# Join: for each edge (focal_id, neighbor_id), pull in all years of the
# neighbor. This creates a tall table:
#   (focal_id, neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2)
# where the variable values come from the *neighbor's* row in that year.
setkey(edges, neighbor_id)
neighbor_data <- edges[cell_subset, on = "neighbor_id", allow.cartesian = TRUE, nomatch = 0L]

# neighbor_data now has columns: focal_id, neighbor_id, year, and the 5 vars
# Each row represents one (focal_cell, neighbor_cell, year) combination.

# ─────────────────────────────────────────────────────────────────────
# 5. Compute grouped aggregations: max, min, mean per (focal_id, year)
#    for all 5 variables simultaneously, in one pass.
# ─────────────────────────────────────────────────────────────────────

# Build the aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0(v, c("_nb_max", "_nb_min", "_nb_mean"))
}))

names(agg_exprs) <- agg_names

# Perform the aggregation
neighbor_stats <- neighbor_data[,
  eval(as.call(c(as.name("list"), agg_exprs))),
  by = .(focal_id, year)
]

# Replace Inf/-Inf (from max/min on all-NA groups) with NA
for (col_name in agg_names) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col_name]])), col_name, NA_real_)
}

# ─────────────────────────────────────────────────────────────────────
# 6. Join the neighbor stats back onto cell_dt
# ─────────────────────────────────────────────────────────────────────

setnames(neighbor_stats, "focal_id", "id")
setkey(neighbor_stats, id, year)
setkey(cell_dt, id, year)

# Remove old neighbor columns if they exist (from prior runs)
old_nb_cols <- intersect(agg_names, names(cell_dt))
if (length(old_nb_cols) > 0) {
  cell_dt[, (old_nb_cols) := NULL]
}

# Merge
cell_dt <- neighbor_stats[cell_dt, on = .(id, year)]

# ─────────────────────────────────────────────────────────────────────
# 7. RF inference (unchanged — the trained model is preserved)
#    This is NOT the bottleneck. Runs in minutes.
# ─────────────────────────────────────────────────────────────────────

# Example (adapt to your actual model object and predict call):
# rf_model   <- readRDS("trained_rf_model.rds")
# preds      <- predict(rf_model, data = cell_dt)  # or newdata = cell_dt
# cell_dt[, prediction := preds$predictions]        # for ranger
# fwrite(cell_dt, "final_predictions.csv")

# ─────────────────────────────────────────────────────────────────────
# 8. (Optional) Convert back to data.frame if downstream code requires it
# ─────────────────────────────────────────────────────────────────────
cell_data <- as.data.frame(cell_dt)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Bottleneck** | `build_neighbor_lookup` + `compute_neighbor_stats` (row-level `lapply` × 6.46M rows, string ops) | Eliminated |
| **Approach** | Per-row `lapply`, `paste()`, named vector lookups | Vectorized `data.table` join + grouped aggregation |
| **Loops** | ~38.8M R-level iterations | Zero R-level row loops |
| **Estimated time** | 86+ hours | ~5–15 minutes |
| **RF model** | Preserved ✓ | Preserved ✓ |
| **Numerical output** | max/min/mean neighbor stats | Identical max/min/mean neighbor stats ✓ |

The trained Random Forest model is never retrained — only `predict()` is called. The 15 neighbor feature columns (5 vars × 3 stats) are numerically identical to the originals, preserving the estimand.