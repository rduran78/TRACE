 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's reference index.
2. Retrieves that cell's rook neighbors (spatial, ~4 per cell).
3. Pastes together `neighbor_id_year` keys (character concatenation over millions of rows).
4. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) — each lookup is O(n) hash probe on a 6.46M-element named vector, repeated ~4 times per row.

This produces a **list of 6.46 million integer vectors**. The `paste`-and-lookup pattern is executed ~25.8 million times (6.46M rows × ~4 neighbors). This is the dominant cost, likely accounting for 70+ hours of the 86-hour estimate.

### Bottleneck B: `compute_neighbor_stats` — repeated `lapply` over 6.46M rows

For each of 5 variables, another `lapply` iterates over the 6.46M-element lookup list, subsetting a numeric vector and computing `max`, `min`, `mean`. That's 5 × 6.46M = 32.3M small R function calls. This is slow but secondary (~15 hours estimated).

### Why raster focal/kernel operations don't directly apply

Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed kernel applied uniformly. While the data *is* on a grid, the panel structure (cell × year), missing cells, and the need to match neighbors within the same year make a direct focal approach fragile. The correct analogy is a **sparse adjacency join**, best handled via `data.table` joins.

---

## 2. Optimization Strategy

### Replace both functions with a vectorized `data.table` join approach:

1. **Expand the neighbor list into an edge table** (a two-column data.table: `id`, `neighbor_id`) — done once, ~1.37M rows.
2. **Join the edge table to the panel data by `(neighbor_id, year)`** — this is a single keyed `data.table` merge producing ~1.37M × 28 ≈ 38.5M rows (the "long neighbor-values" table).
3. **Group by `(id, year)` and compute `max`, `min`, `mean`** for each variable in one pass — a single `data.table` aggregation.

This eliminates all per-row R function calls, all `paste` key construction, and all named-vector lookups. Expected runtime: **2–5 minutes** on a 16 GB laptop.

### Preserving the estimand

The computation is numerically identical: for each `(cell, year)`, we gather the same rook neighbors' values (excluding `NA`), and compute the same `max`, `min`, `mean`. The trained Random Forest model is never retouched — we simply produce the same predictor columns it expects.

---

## 3. Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 0: Convert panel data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Build a flat edge table from the nb object (one-time, fast)
#
#   rook_neighbors_unique : an nb object (list of integer vectors)
#   id_order              : vector mapping list position -> cell id
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb[[i]] contains the indices (into id_order) of cell i's
  # rook neighbors. An entry of 0L means no neighbors (spdep convention).
  edges <- rbindlist(lapply(seq_along(neighbors_nb), function(i) {
    nb_idx <- neighbors_nb[[i]]
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  edges
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has columns: id, neighbor_id
# ~1.37 M rows (directed pairs)

cat("Edge table rows:", nrow(edge_dt), "\n")

# ──────────────────────────────────────────────────────────────────────
# Step 2: For each variable, join + aggregate in one vectorized pass
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare a minimal lookup table: (id, year, var1, var2, …)
# We key it on (id, year) so the join is O(n log n) or hash-based.
lookup_cols <- c("id", "year", neighbor_source_vars)
neighbor_vals_dt <- cell_data[, ..lookup_cols]
setnames(neighbor_vals_dt, "id", "neighbor_id")   # rename for join
setkeyv(neighbor_vals_dt, c("neighbor_id", "year"))

# Cross edge_dt with all years present in cell_data
# But it's more efficient to join edge_dt to the data directly:
#   edge_dt[neighbor_vals_dt] gives us, for every (neighbor_id, year),
#   the focal cell id that has that neighbor.
# However, we want: for each (id, year), get neighbor values.
# Strategy: join cell_data's year onto edge_dt, then join neighbor values.

# 2a. Get the set of (id, year) pairs from cell_data
id_year <- cell_data[, .(id, year)]

# 2b. Expand: for each (id, year), attach all neighbor_ids
#     Result: ~1.37M * 28 ≈ 38.5M rows (but many cells share years,
#     so we do a keyed join which is fast)
setkeyv(edge_dt, "id")
setkeyv(id_year, "id")

# This join replicates each id's neighbors across all years for that id
expanded <- edge_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
# expanded has columns: id, neighbor_id, year
# ~38.5M rows

cat("Expanded neighbor-year rows:", nrow(expanded), "\n")

# 2c. Attach neighbor variable values by joining on (neighbor_id, year)
setkeyv(expanded, c("neighbor_id", "year"))
expanded <- neighbor_vals_dt[expanded, on = c("neighbor_id", "year"), nomatch = NA]
# Now expanded has: neighbor_id, year, id, ntl, ec, pop_density, def, usd_est_n2
# where the variable columns are the NEIGHBOR's values

# 2d. Aggregate: group by (id, year), compute max/min/mean per variable
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  vsym <- as.name(v)
  list(
    bquote(as.numeric(max(.(vsym), na.rm = TRUE))),
    bquote(as.numeric(min(.(vsym), na.rm = TRUE))),
    bquote(mean(.(vsym), na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0(v, c("_max_neighbor", "_min_neighbor", "_mean_neighbor"))
}))

# Build the aggregation call dynamically
agg_call <- as.call(c(as.name("list"),
  setNames(agg_exprs, agg_names)
))

neighbor_stats <- expanded[, eval(agg_call), by = .(id, year)]

# Replace Inf/-Inf (from max/min of all-NA groups) with NA
for (col_name in agg_names) {
  vals <- neighbor_stats[[col_name]]
  set(neighbor_stats, i = which(is.infinite(vals) | is.nan(vals)),
      j = col_name, value = NA_real_)
}

cat("Aggregated stats rows:", nrow(neighbor_stats), "\n")

# ──────────────────────────────────────────────────────────────────────
# Step 3: Merge neighbor stats back onto cell_data
# ──────────────────────────────────────────────────────────────────────

# Remove old neighbor columns if they exist (idempotency)
old_cols <- intersect(agg_names, names(cell_data))
if (length(old_cols) > 0) cell_data[, (old_cols) := NULL]

setkeyv(cell_data, c("id", "year"))
setkeyv(neighbor_stats, c("id", "year"))
cell_data <- neighbor_stats[cell_data, on = c("id", "year")]

cat("Done. cell_data now has", ncol(cell_data), "columns and",
    nrow(cell_data), "rows.\n")

# ──────────────────────────────────────────────────────────────────────
# Step 4: Predict with the pre-trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The model object (e.g., `rf_model`) expects the same column names
# that were just created (*_max_neighbor, *_min_neighbor, *_mean_neighbor).
# Prediction proceeds exactly as before:
#
#   cell_data$predicted <- predict(rf_model, newdata = cell_data)
#
# The Random Forest is NOT retrained.
```

---

## 4. Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `paste` + named-vector lookups | One `data.table` keyed join (seconds) |
| `compute_neighbor_stats` | 5 × 6.46M R-level `lapply` calls | One grouped aggregation (seconds) |
| Memory pattern | 6.46M-element R list of integer vectors (~2–4 GB) | Flat `data.table` columns (~1.5 GB peak for expanded) |
| Estimated wall time | **86+ hours** | **2–5 minutes** |
| Numerical result | max/min/mean of non-NA rook-neighbor values | **Identical** |
| RF model | Unchanged | **Unchanged** |

The key insight: the original code performs millions of individual R function calls with character key construction; replacing this with a single vectorized **equi-join + grouped aggregation** in `data.table` reduces the work to a handful of highly optimized C-level passes over contiguous memory.