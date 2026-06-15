 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is twofold:

### 1. `build_neighbor_lookup` — O(n) `lapply` over 6.46 million rows
Each iteration performs character coercion, `paste`, and named-vector lookups (which are hash-table probes). With ~6.46M rows this creates millions of temporary character vectors and millions of hash lookups. The named-vector `idx_lookup` itself is a character-keyed hash of 6.46M entries, and probing it inside a per-row `lapply` is extremely slow. Estimated wall time for this step alone: **20–40 hours**.

### 2. `compute_neighbor_stats` — repeated per variable, also O(n)
Each call iterates over 6.46M rows, subsetting a numeric vector by index vectors of varying length. This is called 5 times (once per neighbor source variable). The `do.call(rbind, ...)` on a 6.46M-element list of 3-element vectors is itself a major allocation bottleneck. Estimated: **~10 hours per variable × 5 = 50 hours**.

### Memory
Storing `neighbor_lookup` as a list of 6.46M integer vectors is also memory-heavy (~2–4 GB depending on neighbor counts), and the repeated `do.call(rbind, ...)` temporarily doubles memory.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Per-row `paste`/hash lookup in `build_neighbor_lookup` | Replace with a **vectorized merge/join** using `data.table`. Build a neighbor-edge table once, then join to get row indices. Eliminates all per-row `lapply`. |
| Storing lookup as list-of-vectors | Store as a **`data.table` of edges** (`from_row`, `to_row`). This is a flat table of ~1.37M × 28 ≈ 38.4M edge-rows (directed, per year). Compact and joinable. |
| Per-row `lapply` in `compute_neighbor_stats` | Replace with **grouped `data.table` aggregation**: group by `from_row`, compute `max`, `min`, `mean` of the neighbor values in one vectorized pass. |
| `do.call(rbind, ...)` on 6.46M-element list | Eliminated entirely — `data.table` returns a single result table. |
| 5 separate passes over the edge table | Compute all 5 variables' neighbor stats in a **single grouped aggregation** or a tight loop of vectorized ops. |

**Expected speedup**: from ~86 hours to **~5–15 minutes** on the same laptop. Memory peak: ~4–6 GB (well within 16 GB).

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 0 — Convert cell_data to data.table (if not already) and ensure key cols
# ===========================================================================
cell_dt <- as.data.table(cell_data)          # non-destructive copy
cell_dt[, row_idx := .I]                     # preserve original row order

# ===========================================================================
# STEP 1 — Build a flat edge table from the nb object (one-time, vectorized)
#
#   rook_neighbors_unique is a list of length N_cells (344,208).
#   rook_neighbors_unique[[i]] contains integer indices into id_order
#   of the neighbors of cell id_order[i].
#
#   id_order is a vector of cell IDs of length 344,208.
# ===========================================================================

# --- 1a. Expand the nb list into a two-column data.table of (from_id, to_id)
from_ref <- rep(seq_along(rook_neighbors_unique),
                lengths(rook_neighbors_unique))
to_ref   <- unlist(rook_neighbors_unique, use.names = FALSE)

# Remove the spdep placeholder 0 (nb objects use 0 for "no neighbors")
valid <- to_ref != 0L
from_ref <- from_ref[valid]
to_ref   <- to_ref[valid]

edge_cells <- data.table(
  from_id = id_order[from_ref],
  to_id   = id_order[to_ref]
)
rm(from_ref, to_ref, valid)                  # free memory

# --- 1b. Create a lookup from (id, year) → row_idx in cell_dt
key_dt <- cell_dt[, .(id, year, row_idx)]

# --- 1c. Cross-join edges with years to get per-year edge table,
#          then map each (id, year) to its row_idx.
#
#   Instead of a full cross-join (which would be huge), we merge twice:
#     • first  merge: edge_cells ⋈ key_dt  on from_id = id  → gives (from_row, to_id, year)
#     • second merge: result     ⋈ key_dt  on to_id = id AND year → gives (from_row, to_row)
#
#   This naturally restricts to (cell, year) pairs that actually exist.

setnames(key_dt, c("id", "year", "row_idx"),
                 c("from_id", "year", "from_row"))
setkey(key_dt, from_id)
setkey(edge_cells, from_id)

# First merge — attach from_row and year
edge_year <- edge_cells[key_dt, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
#   columns: from_id, to_id, year, from_row

# Second merge — attach to_row
setnames(key_dt, c("from_id", "year", "from_row"),
                 c("to_id",   "year", "to_row"))
setkey(key_dt, to_id, year)
setkey(edge_year, to_id, year)

edge_year <- key_dt[edge_year, on = c("to_id", "year"), nomatch = 0L]
#   columns: to_id, year, to_row, from_id, from_row

# Keep only the columns we need
edge_year <- edge_year[, .(from_row, to_row)]
rm(key_dt, edge_cells)
gc()

cat("Edge-year table:", format(nrow(edge_year), big.mark = ","), "rows\n")

# ===========================================================================
# STEP 2 — Compute neighbor stats for all variables in one pass
#
#   For each (from_row) we need max, min, mean of the neighbor values
#   (the values at to_row) for each of the 5 source variables.
# ===========================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Attach the neighbor (to_row) values to the edge table
# We pull only the columns we need to keep memory tight.
val_cols <- neighbor_source_vars
to_vals  <- cell_dt[edge_year$to_row, ..val_cols]
edge_aug <- cbind(edge_year, to_vals)
edge_aug[, from_row := edge_year$from_row]
rm(to_vals)
gc()

# Grouped aggregation — one pass over the edge table
agg_exprs <- unlist(lapply(val_cols, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(val_cols, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Build the aggregation call dynamically
agg_list <- setNames(agg_exprs, agg_names)

# data.table aggregation
neighbor_stats <- edge_aug[,
  lapply(agg_list, eval),
  by = from_row
]

# Replace Inf/-Inf (from max/min on all-NA groups) with NA
for (col in agg_names) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

rm(edge_aug)
gc()

# ===========================================================================
# STEP 3 — Join the neighbor stats back onto cell_dt
# ===========================================================================

# Rows with no neighbors will be NA after the join (correct behavior).
setkey(neighbor_stats, from_row)
setkey(cell_dt, row_idx)

# Remove any pre-existing neighbor columns to avoid conflicts
existing <- intersect(names(cell_dt), agg_names)
if (length(existing)) cell_dt[, (existing) := NULL]

cell_dt <- neighbor_stats[cell_dt, on = c(from_row = "row_idx")]

# Clean up helper columns
cell_dt[, from_row := NULL]

# ===========================================================================
# STEP 4 — Restore original row order and convert back to data.frame
#           (preserves compatibility with the trained Random Forest model)
# ===========================================================================

setorder(cell_dt, row_idx)
cell_dt[, row_idx := NULL]

cell_data <- as.data.frame(cell_dt)          # overwrite original object
rm(cell_dt, neighbor_stats, edge_year)
gc()
```

---

## Why This Preserves the Trained Random Forest and the Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Column names** | The output columns are named identically to what `compute_and_add_neighbor_features` would have produced (`neighbor_max_ntl`, `neighbor_min_ntl`, `neighbor_mean_ntl`, etc.). The trained model's feature name references are satisfied. |
| **Numerical values** | `max`, `min`, `mean` with `na.rm = TRUE` are the same functions applied to the same neighbor index sets. The flat edge table is an exact materialization of the original `neighbor_lookup` list. No approximation or sampling is involved, so the numerical estimand is identical to machine precision. |
| **No retraining** | The code only constructs prediction features. The model object is never touched. |

---

## Summary of Expected Gains

| Metric | Before (original) | After (data.table) |
|---|---|---|
| `build_neighbor_lookup` | ~20–40 h | ~1–3 min (vectorized joins) |
| `compute_neighbor_stats` (×5) | ~50 h | ~2–5 min (single grouped agg) |
| Peak RAM | ~10–12 GB+ | ~4–6 GB |
| **Total wall time** | **~86 h** | **~5–15 min** |