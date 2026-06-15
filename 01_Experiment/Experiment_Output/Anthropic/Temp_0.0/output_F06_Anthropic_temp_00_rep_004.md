 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) list with per-row string operations

`build_neighbor_lookup` iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's rook neighbors by cell ID.
2. Constructs `paste(neighbor_id, year, "_")` keys.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

Named character vector lookup in R is **hash-based but slow at scale** when called millions of times in an `lapply`. The resulting `neighbor_lookup` is a list of ~6.46M integer vectors — enormous in memory and slow to build.

### Bottleneck B: `compute_neighbor_stats` — per-row subsetting in a loop

For each of the 5 variables, `compute_neighbor_stats` loops over 6.46M entries, subsets `vals[idx]`, removes NAs, and computes `max/min/mean`. That's **~32.3 million R-level function calls** (5 vars × 6.46M rows), each involving vector allocation and subsetting.

### Why raster focal/kernel operations are a useful *analogy* but not directly applicable

Focal operations (e.g., `terra::focal`) assume a regular rectangular grid with a fixed kernel. Here, the grid cells have a **rook-neighbor structure stored as an `spdep::nb` object**, which may include irregular boundaries (coastal cells, edge cells with fewer than 4 neighbors). A focal approach would require reshaping data into a 3D raster stack (x × y × time) and carefully handling NA/missing cells. This is feasible but risks introducing subtle numerical differences at boundaries. The better approach is to **vectorize the neighbor computation directly using data.table joins**, which preserves the exact `nb` structure and results.

### Summary

| Component | Current Cost | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~hours | 6.46M string-paste + named-vector lookups |
| `compute_neighbor_stats` | ~hours × 5 vars | 6.46M R-level loops × 5 variables |
| **Total** | **86+ hours** | Interpreted R loops, no vectorization |

---

## 2. Optimization Strategy

**Replace both functions with a single vectorized `data.table` join-and-aggregate approach.**

The key insight: instead of building a per-row lookup list and then looping, we:

1. **Expand the `nb` object into an edge list** (cell_id → neighbor_id), ~1.37M directed edges.
2. **Join** this edge list to the panel data by `(neighbor_id, year)` to get neighbor values — this is a single equi-join, handled in C by `data.table`.
3. **Group-by aggregate** `(cell_id, year)` to compute `max`, `min`, `mean` for all 5 variables simultaneously.
4. **Join** the aggregated stats back to the main data.

This eliminates all R-level loops. Expected runtime: **minutes, not hours**.

### Why this preserves the original numerical estimand

- The rook-neighbor relationships are identical (same `nb` object).
- `max`, `min`, `mean` are computed on exactly the same neighbor sets.
- NA handling is identical (neighbors missing from the panel or with NA values are excluded).
- The trained Random Forest model is never retrained — we only recompute the input features, which are numerically identical.

---

## 3. Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 0: Convert panel data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Build edge list from the spdep nb object (one-time, fast)
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : vector mapping position in nb list → cell id
#
#   Result: a data.table with columns  (id, neighbor_id)
#           representing every directed rook-neighbor pair.
# ──────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order for the
  # neighbors of the cell whose id is id_order[i].
  # spdep nb objects use 0L to denote "no neighbors" for an isolate.
  from_ids <- rep(id_order, times = lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the 0-entries that spdep uses for isolates
  valid    <- to_idx > 0L
  from_ids <- from_ids[valid]
  to_ids   <- id_order[to_idx[valid]]

  data.table(id = from_ids, neighbor_id = to_ids)
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat("Edge list rows:", nrow(edge_dt), "\n")
# Expected: ~1,373,394

# ──────────────────────────────────────────────────────────────────────
# Step 2: Vectorized neighbor-stat computation via join + group-by
#
#   For every (cell, year) we need max, min, mean of each source
#   variable across that cell's rook neighbors in the same year.
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare a slim table of just the columns we need for the neighbor side
# to keep the join memory-efficient.
neighbor_cols <- c("id", "year", neighbor_source_vars)
neighbor_dt   <- cell_data[, ..neighbor_cols]

# Key the neighbor table for fast join
setnames(neighbor_dt, "id", "neighbor_id")
setkeyv(neighbor_dt, c("neighbor_id", "year"))

# Key the edge list
setkeyv(edge_dt, "neighbor_id")

# Join: for each edge (id, neighbor_id) and each year, attach the
# neighbor's variable values.
#
#   edge_dt  has columns: id, neighbor_id
#   We need to join on (neighbor_id, year).
#   Strategy: first cross edge_dt with the years present for each id,
#   but that would be huge.  Instead, join cell_data's (id, year) with
#   edge_dt to get (id, year, neighbor_id), then join neighbor values.

# Slim version of cell_data with just id and year (one row per cell-year)
cell_year <- unique(cell_data[, .(id, year)])
setkeyv(cell_year, "id")
setkeyv(edge_dt, "id")

# Expand: every (id, year) gets its neighbor_ids
# This produces ~1,373,394 * 28 ≈ 38.5M rows but is manageable in 16 GB
# because each row is just three integer/numeric columns.
#
# Actually, each edge applies to ALL 28 years, so:
expanded <- edge_dt[cell_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
# expanded has columns: id, neighbor_id, year
cat("Expanded edge-year rows:", nrow(expanded), "\n")

# Now attach the neighbor variable values
setkeyv(expanded, c("neighbor_id", "year"))
expanded <- neighbor_dt[expanded, on = c("neighbor_id", "year"), nomatch = NA]
# expanded now has: neighbor_id, year, ntl, ec, ..., id

# ──────────────────────────────────────────────────────────────────────
# Step 3: Aggregate by (id, year) — compute max, min, mean per variable
# ──────────────────────────────────────────────────────────────────────

# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

names(agg_exprs) <- agg_names

# Evaluate
stats_dt <- expanded[,
  eval(as.call(c(as.name("list"), agg_exprs))),
  by = .(id, year)
]

# Replace -Inf/Inf (from max/min on all-NA groups) with NA
inf_cols <- grep("_max$|_min$", names(stats_dt), value = TRUE)
for (col in inf_cols) {
  set(stats_dt, which(is.infinite(stats_dt[[col]])), col, NA_real_)
}

cat("Stats rows:", nrow(stats_dt), "\n")
# Expected: ~6,460,000 (one per cell-year)

# ──────────────────────────────────────────────────────────────────────
# Step 4: Join aggregated stats back to cell_data
# ──────────────────────────────────────────────────────────────────────

# Remove any pre-existing neighbor columns to avoid duplication
old_neighbor_cols <- intersect(names(cell_data), agg_names)
if (length(old_neighbor_cols) > 0) {
  cell_data[, (old_neighbor_cols) := NULL]
}

setkeyv(cell_data, c("id", "year"))
setkeyv(stats_dt,  c("id", "year"))

cell_data <- stats_dt[cell_data, on = c("id", "year")]

cat("Done. cell_data now has", ncol(cell_data), "columns and",
    nrow(cell_data), "rows.\n")

# ──────────────────────────────────────────────────────────────────────
# Step 5: Predict with the existing trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# cell_data$prediction <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Memory Budget Check

| Object | Approximate Size |
|---|---|
| `cell_data` (6.46M × 110 cols) | ~5.7 GB |
| `edge_dt` (1.37M × 2 int cols) | ~11 MB |
| `expanded` (38.5M × 7 cols) | ~2.2 GB |
| `stats_dt` (6.46M × 17 cols) | ~0.9 GB |
| **Peak total** | **~9–10 GB** |

This fits within 16 GB. If memory is tight, process variables in batches of 2–3 instead of all 5 simultaneously, then `cbind` the results.

---

## 5. Expected Speedup

| Phase | Before | After |
|---|---|---|
| Neighbor lookup construction | ~30–40 hours | ~2 seconds (vectorized edge list) |
| Neighbor stats (5 vars) | ~45–50 hours | ~2–5 minutes (data.table join + group-by in C) |
| **Total** | **86+ hours** | **< 10 minutes** |

The trained Random Forest model is untouched, and the 15 neighbor-derived features (`5 vars × {max, min, mean}`) are numerically identical to the original implementation.