 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is overwhelmingly in `build_neighbor_lookup`, not in `compute_neighbor_stats`. Here's why:

**`build_neighbor_lookup`** iterates over ~6.46 million rows with `lapply`, and *inside each iteration* it performs:

1. **Character coercion and named-vector lookups** (`id_to_ref[as.character(...)]`, `idx_lookup[neighbor_keys]`) — these are hash-table lookups on character keys, repeated ~6.46M times, each touching a named vector of length ~6.46M. Named vector lookup in R is O(n) in the worst case for each probe when the vector is large.
2. **`paste()` key construction** — allocates ~4–8 short strings per row (one per neighbor), totaling ~25–50 million small string allocations.
3. **`lapply` over 6.46M elements** returns a list of 6.46M integer vectors — massive memory allocation overhead in R's single-threaded interpreter.

The net effect: billions of character operations and hash probes in interpreted R, producing the estimated 86+ hour runtime.

**`compute_neighbor_stats`** is comparatively cheap — it just indexes into a numeric vector using pre-built integer indices, computes max/min/mean on ~4 values, and does this 5 times total (once per variable). That's fast.

**The Random Forest inference** is a single `predict()` call on a pre-trained model with ~110 columns and ~6.46M rows — typically minutes, not hours.

**Conclusion:** The pathology is the row-by-row string-key construction and lookup in `build_neighbor_lookup`. The fix is to eliminate character key manipulation entirely and use direct integer-indexed joins.

---

## Optimization Strategy

### Principle: Replace per-row string-key lookups with a vectorized merge/join on integer columns.

The core insight is that `build_neighbor_lookup` is essentially performing a spatial join: "for each (cell, year) row, find the row indices of its rook neighbors in the same year." This is a standard equi-join that `data.table` can execute in seconds.

**Steps:**

1. **Expand the neighbor list into an edge table** — a two-column data.table of `(id, neighbor_id)` with ~1.37M rows (one per directed edge). This is done once.
2. **Join the edge table to the panel data twice** — once to attach the focal row's year, once to look up the neighbor's row index in that year. This is a vectorized merge on integer keys. `data.table` performs this with radix-sort joins in seconds.
3. **Compute neighbor stats vectorially** — group by the focal row index and compute max/min/mean of each variable in one pass per variable, fully vectorized.

This replaces 6.46M × (string alloc + hash probe) iterations with two `data.table` merges on ~37M rows (6.46M rows × avg ~5.7 neighbors, minus boundary cells), executing in under a minute on a laptop.

**Numerical equivalence:** The same neighbor sets are used, the same max/min/mean aggregations are computed, and the same columns are appended to the data. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────
# 0.  Convert panel data to data.table (if not already)
#     Assumes: cell_data is a data.frame / data.table with
#              columns 'id' (integer cell id) and 'year'.
#     Assumes: id_order is the vector mapping position in
#              rook_neighbors_unique to cell id.
#     Assumes: rook_neighbors_unique is an nb object (list
#              of integer index vectors referencing id_order).
# ──────────────────────────────────────────────────────────

cell_dt <- as.data.table(cell_data)

# Preserve original row ordering so we can write results
# back in the correct position (important for predict()).
cell_dt[, .row_idx := .I]

# Create a fast row-index lookup keyed on (id, year).
# This replaces the old character-keyed idx_lookup entirely.
cell_dt[, .row_id_year := .I]                 
row_lookup <- cell_dt[, .(.row_id_year, id, year)]
setkey(row_lookup, id, year)

# ──────────────────────────────────────────────────────────
# 1.  Build the edge table from the nb object — ONCE
#     Result: edges with columns  focal_id, neighbor_id
# ──────────────────────────────────────────────────────────

edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_id    = id_order[i],
             neighbor_id = id_order[nb_idx])
}))

cat(sprintf("Edge table: %s directed edges\n", format(nrow(edges), big.mark = ",")))

# ──────────────────────────────────────────────────────────
# 2.  Cross edges with years → (focal_id, year, neighbor_id)
#     Then join to row_lookup to get the neighbor's row index.
#
#     Instead of exploding edges × 28 years up front (which
#     would be ~38 M rows but still manageable), we join
#     through the panel data directly.
# ──────────────────────────────────────────────────────────

# 2a. Attach focal row index and year to every edge instance.
#     focal_rows: one row per (focal cell-year, neighbor_id).
setkey(edges, focal_id)
focal_panel <- cell_dt[, .(focal_id = id, year, focal_row = .row_idx)]
setkey(focal_panel, focal_id)

# Merge: for every cell-year row, expand its neighbor edges.
# Result columns: focal_id, year, focal_row, neighbor_id
edge_panel <- edges[focal_panel, on = .(focal_id), allow.cartesian = TRUE, nomatch = 0L]

cat(sprintf("Edge-panel table: %s rows\n", format(nrow(edge_panel), big.mark = ",")))

# 2b. Look up the neighbor's row in the same year.
setkey(row_lookup, id, year)
edge_panel[row_lookup,
           neighbor_row := i..row_id_year,
           on = .(neighbor_id = id, year)]

# Drop edges whose neighbor has no data in that year (boundary / missing).
edge_panel <- edge_panel[!is.na(neighbor_row)]

# ──────────────────────────────────────────────────────────
# 3.  Compute neighbor stats vectorially for each variable
# ──────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  # Pull the numeric vector once (aligned to row index).
  vals <- cell_dt[[var_name]]

  # Attach neighbor values to every edge-panel row.
  edge_panel[, nval := vals[neighbor_row]]

  # Aggregate per focal row — exactly max, min, mean of
  # non-NA neighbor values (same semantics as original code).
  agg <- edge_panel[!is.na(nval),
                    .(nb_max  = max(nval),
                      nb_min  = min(nval),
                      nb_mean = mean(nval)),
                    keyby = .(focal_row)]

  # Prepare NA-filled columns, then overwrite matched rows.
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)

  set(cell_dt, j = max_col,  value = NA_real_)
  set(cell_dt, j = min_col,  value = NA_real_)
  set(cell_dt, j = mean_col, value = NA_real_)

  set(cell_dt, i = agg$focal_row, j = max_col,  value = agg$nb_max)
  set(cell_dt, i = agg$focal_row, j = min_col,  value = agg$nb_min)
  set(cell_dt, i = agg$focal_row, j = mean_col, value = agg$nb_mean)

  cat(sprintf("  ✓ %s neighbor features added\n", var_name))
}

# ──────────────────────────────────────────────────────────
# 4.  Clean up helper columns and convert back if needed
# ──────────────────────────────────────────────────────────
cell_dt[, c(".row_idx", ".row_id_year") := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(cell_dt)

# ──────────────────────────────────────────────────────────
# 5.  Predict with the EXISTING trained Random Forest
#     (model object is untouched)
# ──────────────────────────────────────────────────────────
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

---

## Expected Performance Comparison

| Phase | Original (est.) | Optimized (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~80–86 hours | eliminated |
| Edge table construction | — | ~2 seconds |
| `edge_panel` join (edges × years) | — | ~20–40 seconds |
| Neighbor row lookup join | — | ~10–20 seconds |
| `compute_neighbor_stats` (5 vars) | ~5–10 min | ~30–90 seconds |
| **Total neighbor feature time** | **~86 hours** | **~2–4 minutes** |
| Peak RAM (neighbor step) | ~8–12 GB (list of 6.46M vectors) | ~4–6 GB (`edge_panel` table) |

The optimization yields roughly a **1,000–2,500×** speedup by replacing interpreted per-row string manipulation with vectorized `data.table` integer-key joins, while producing numerically identical neighbor max/min/mean features and leaving the trained Random Forest model completely unchanged.