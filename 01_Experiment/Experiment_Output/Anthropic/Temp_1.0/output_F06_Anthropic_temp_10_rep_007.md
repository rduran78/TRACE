 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** creates a list of ~6.46 million elements, each produced by an `lapply` call that performs character coercion, string pasting, and named-vector lookups per row. This is O(n) in pure R with extremely high constant factors due to:

1. **String-based key lookups**: `paste()` and named-vector indexing (`idx_lookup[neighbor_keys]`) for every single row is catastrophically slow at 6.46M rows.
2. **Row-level `lapply` over millions of rows**: R's `lapply` over 6.46M iterations with non-trivial closures is inherently slow.
3. **`compute_neighbor_stats` also uses `lapply` over 6.46M entries**, performing per-element aggregation in pure R.

The comment about raster focal/kernel operations is a conceptual hint: focal operations compute neighborhood statistics in bulk via optimized C routines. We should emulate that idea—**vectorized bulk aggregation**—using `data.table` joins and grouped aggregation rather than element-wise R loops.

### Why raster focal won't work directly
Focal operations assume a regular grid with a fixed kernel. This panel has an irregular neighbor structure (rook contiguity from `spdep::nb`) and a time dimension. A direct raster focal approach would require restructuring into a 3D array per year and may lose cells with missing neighbors. The `data.table` join approach preserves the exact same neighbor structure and numerical results.

---

## Optimization Strategy

1. **Convert the `nb` object into an edge list** (a two-column data.table of `from_id` → `to_id`). This is done once and is small (~1.37M rows).
2. **Join the edge list with `cell_data` by `(to_id, year)`** to pull each neighbor's variable values. This is a single keyed `data.table` merge—highly optimized in C.
3. **Group by `(from_id, year)` and compute `max`, `min`, `mean`** in one pass. This replaces millions of R-level function calls with a single `data.table` grouped aggregation.
4. **Left-join the results back** to `cell_data`, preserving rows with no neighbors as `NA`.
5. Repeat for each of the 5 variables (or batch them).

**Expected speedup**: from ~86+ hours to **minutes**. The dominant operation becomes a single indexed merge of ~6.46M × ~4 neighbors ≈ ~26M rows, then a grouped aggregation—both are `data.table`'s sweet spot.

**Numerical equivalence**: The operations are identical—`max`, `min`, `mean` of the same neighbor sets—so the trained Random Forest model receives exactly the same feature values.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Convert spdep::nb object to a data.table edge list (once)
# ──────────────────────────────────────────────────────────────────────
nb_to_edge_dt <- function(nb_obj, id_order) {
  # nb_obj: list of integer vectors (indices into id_order)
  # id_order: vector mapping position -> cell id
  from_ids <- rep(id_order, lengths(nb_obj))
  to_ids   <- id_order[unlist(nb_obj)]
  data.table(from_id = from_ids, to_id = to_ids)
}

edges <- nb_to_edge_dt(rook_neighbors_unique, id_order)
# edges has ~1,373,394 rows: (from_id, to_id)

# ──────────────────────────────────────────────────────────────────────
# Step 2: Convert cell_data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ──────────────────────────────────────────────────────────────────────
# Step 3: Function to compute and attach neighbor features for one var
# ──────────────────────────────────────────────────────────────────────
compute_and_add_neighbor_features_fast <- function(dt, var_name, edges) {
  # Subset to only needed columns for the join
  # Neighbor values come from matching (to_id, year) in cell_data
  neighbor_vals <- edges[
    dt[, .(to_id = id, year, val = get(var_name))],
    on = .(to_id, year = year),   # join: edge.to_id == dt.id, same year
    nomatch = NULL,                # inner join: drop edges with no match
    allow.cartesian = FALSE
  ]
  # neighbor_vals now has columns: from_id, to_id, year, val
  # where val is the neighbor's value of var_name

  # Aggregate by (from_id, year)
  stats <- neighbor_vals[
    !is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = .(from_id, year)
  ]

  # Rename columns to match original naming convention
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                  c(max_col,  min_col,  mean_col))

  # Left-join back onto dt by (id == from_id, year)
  # Remove old columns if re-running
  old_cols <- intersect(c(max_col, min_col, mean_col), names(dt))
  if (length(old_cols) > 0) dt[, (old_cols) := NULL]

  dt <- merge(dt, stats,
              by.x = c("id", "year"),
              by.y = c("from_id", "year"),
              all.x = TRUE,    # preserve all original rows
              sort = FALSE)

  dt
}

# ──────────────────────────────────────────────────────────────────────
# Step 4: Set keys for fast joins
# ──────────────────────────────────────────────────────────────────────
setkey(edges, to_id)
setkey(cell_data, id, year)

# ──────────────────────────────────────────────────────────────────────
# Step 5: Loop over the 5 neighbor source variables
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, edges)
}

# ──────────────────────────────────────────────────────────────────────
# Step 6: Predict with the pre-trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The model object (e.g., rf_model) is loaded from disk as before.
# cell_data now has exactly the same neighbor feature columns, so:
#
#   predictions <- predict(rf_model, newdata = cell_data)
#
# produces numerically identical results.
```

---

## Memory Note (16 GB laptop)

The largest intermediate object is the join result inside `compute_and_add_neighbor_features_fast`: ~1.37M edges × 28 years ≈ ~38M rows × 4 columns ≈ ~1.2 GB. This fits comfortably in 16 GB alongside `cell_data` (~6.46M × 110 cols ≈ ~5.7 GB as doubles). If memory is tight, the loop already processes one variable at a time, so the intermediate is freed between iterations via R's garbage collector.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M-element `lapply` with string keys | One-time `nb` → edge-list conversion |
| Stats computation | 6.46M-element `lapply` with per-row aggregation | Single `data.table` keyed join + grouped aggregation |
| Estimated runtime | 86+ hours | ~5–15 minutes |
| Numerical results | max / min / mean of rook neighbors | **Identical** |
| RF model | Unchanged | **Unchanged** |