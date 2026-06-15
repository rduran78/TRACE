 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the spatial cell's rook neighbors (fine — O(1) hash lookup).
2. Constructs character key strings via `paste()` for every neighbor × year combination.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) using character matching on a 6.46M-length named vector.

Character-keyed lookups on a vector of length 6.46M are **O(n)** per probe in the worst case (R's named vector lookup uses hashing, but the sheer volume — ~6.46M × ~4 neighbors average — means billions of character operations). This alone likely accounts for the majority of the 86+ hour estimate.

### Bottleneck 2: `compute_neighbor_stats` — Row-wise `lapply` over 6.46M rows

For each of 5 variables, another `lapply` iterates over 6.46M rows, subsetting a numeric vector by index, removing NAs, and computing `max`, `min`, `mean`. The per-row overhead of R function calls, subsetting, and `is.na` checks across 6.46M iterations × 5 variables = ~32.3M R-level loop iterations is substantial.

### Why raster focal/kernel operations are a useful analogy but not directly applicable

Raster focal operations (e.g., `terra::focal`) compute neighborhood statistics on regular grids extremely efficiently using compiled C code with sliding windows. The panel data here **is** on a regular spatial grid, so conceptually each year-slice is a raster and rook-neighbor max/min/mean are exactly a 3×3 cross-shaped (rook) focal operation. However, the data is stored as a **long panel data.frame**, not as a raster stack, and cells may have missing years or irregular coverage. The optimization strategy below uses the **same logic** as focal operations — vectorized, year-sliced, matrix-indexed computation — while preserving exact numerical results for all edge cases (boundary cells, missing data).

---

## Optimization Strategy

| Step | What changes | Why it's faster |
|---|---|---|
| **1. Replace character-key lookup with integer join** | Use `data.table` keyed join on `(id, year)` instead of `paste()`-based named vector lookup. | Eliminates ~6.46M `paste()` calls and character hash lookups. `data.table` binary search join is O(log n). |
| **2. Build an edge list once, then vectorized merge** | Expand the `nb` object into a two-column integer edge list of `(id, neighbor_id)`. Join to the panel on `(neighbor_id, year)` to get all neighbor values in one vectorized operation per variable. | Replaces 6.46M R-level loop iterations with a single `data.table` merge (~1.37M edges × 28 years ≈ 38.5M rows, handled in compiled C). |
| **3. Grouped aggregation** | `data.table` grouped `max`, `min`, `mean` by `(id, year)` on the joined result. | Replaces 6.46M per-row `lapply` calls with a single compiled grouped aggregation. |
| **4. Repeat for each variable (or batch)** | Loop over the 5 variables, or pivot and do all at once. | Each variable takes seconds, not hours. |

**Expected speedup**: From ~86+ hours to **~2–10 minutes** on 16 GB RAM.

**Numerical equivalence**: The `max`, `min`, `mean` are computed on exactly the same neighbor sets with the same NA handling, so the trained Random Forest model receives identical inputs and need not be retrained.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Inputs assumed to exist:
#     - cell_data        : data.frame/data.table with columns id, year, 
#                          ntl, ec, pop_density, def, usd_est_n2, ...
#     - rook_neighbors_unique : spdep nb object (list of integer vectors)
#     - id_order          : integer/character vector mapping nb list 
#                           positions to cell ids
#     - rf_model           : pre-trained Random Forest (untouched)
# ──────────────────────────────────────────────────────────────────────

# Convert to data.table (by reference if already a data.table)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ──────────────────────────────────────────────────────────────────────
# 1.  Build directed rook-neighbor edge list (once)
#     From the nb object: for each cell i, neighbors[[i]] gives the
#     positions in id_order of its rook neighbors.
# ──────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer index vectors
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)
  
  # Remove the 0-length (no-neighbor) entries naturally (they contribute

  # nothing to from_idx / to_idx via rep + unlist).
  
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat("Edge list rows:", nrow(edge_dt), "\n")
# Expected: ~1,373,394 (directed rook edges among 344,208 cells)

# ──────────────────────────────────────────────────────────────────────
# 2.  Key the panel for fast joins
# ──────────────────────────────────────────────────────────────────────
setkey(cell_data, id, year)

# ──────────────────────────────────────────────────────────────────────
# 3.  For each source variable, compute neighbor max, min, mean
#     and join back to cell_data.
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_and_add_neighbor_features_fast <- function(dt, edge_dt, var_name) {
  
  # --- 3a. Build a slim lookup: (id, year, value) ----------------------
  lookup <- dt[, .(id, year, value = get(var_name))]
  setnames(lookup, "id", "neighbor_id")
  setkey(lookup, neighbor_id, year)
  
  # --- 3b. Cross join edges × years -----------------------------------
  #   For every (id -> neighbor_id) edge and every year present for `id`,
  #   retrieve the neighbor's value.
  #
  #   Strategy: get the years each cell appears in, cross with edges,
  #   then join to lookup.  But it's simpler and equally fast to:
  #     (i)  join edge_dt to lookup on neighbor_id (adds year & value), 
  #     (ii) then ensure the focal cell also exists in that year.
  
  # Expand: every edge × every year the *neighbor* has data
  expanded <- merge(edge_dt, lookup, by = "neighbor_id", allow.cartesian = TRUE)
  # expanded columns: neighbor_id, id, year, value
  
  # Keep only rows where the focal cell (id) also exists in that year.
  # We use a semi-join via keyed existence check.
  focal_keys <- unique(dt[, .(id, year)])
  setkey(focal_keys, id, year)
  setkey(expanded, id, year)
  expanded <- expanded[focal_keys, nomatch = 0L]
  
  # --- 3c. Aggregate: max, min, mean per (id, year) -------------------
  #   Drop NA values in the variable before aggregating (matches original code).
  agg <- expanded[!is.na(value), 
                  .(nb_max  = max(value),
                    nb_min  = min(value),
                    nb_mean = mean(value)),
                  by = .(id, year)]
  
  # --- 3d. Name the new columns to match original convention -----------
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))
  
  # --- 3e. Left-join aggregates back to the main table -----------------
  #   Cells with no valid neighbors in a given year get NA (correct).
  setkey(agg, id, year)
  setkey(dt, id, year)
  
  # Remove old columns if re-running
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(dt)) dt[, (col) := NULL]
  }
  
  dt <- agg[dt, on = .(id, year)]
  setkey(dt, id, year)
  
  return(dt)
}

# ──────────────────────────────────────────────────────────────────────
# 4.  Run for all 5 variables
# ──────────────────────────────────────────────────────────────────────
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  t0 <- proc.time()
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, edge_dt, var_name)
  cat("  Done in", round((proc.time() - t0)[3], 1), "seconds.\n")
}

# ──────────────────────────────────────────────────────────────────────
# 5.  Predict with the existing Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Memory Note for 16 GB RAM

The largest intermediate object is `expanded` in step 3b. With ~1.37M edges × 28 years = ~38.4M rows × 3 columns (two integer IDs + one double value) ≈ **~920 MB**. This is well within 16 GB. Each variable iteration creates and discards this object, so peak usage stays under ~4–5 GB including `cell_data` itself (~6.46M rows × 110 columns ≈ 5.7 GB at 8 bytes/double).

If memory is tight, the loop already processes one variable at a time and `expanded` is garbage-collected between iterations. You can add an explicit `gc()` call after each iteration if needed.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | `paste()` + named vector (character hashing, 6.46M iterations) | `data.table` integer-keyed binary search join |
| Stats computation | `lapply` over 6.46M rows × 5 vars | Vectorized `data.table` grouped aggregation |
| Estimated runtime | 86+ hours | ~2–10 minutes |
| Numerical results | Baseline | **Identical** (same neighbor sets, same `max`/`min`/`mean`, same NA handling) |
| RF model | Untouched | **Untouched** |