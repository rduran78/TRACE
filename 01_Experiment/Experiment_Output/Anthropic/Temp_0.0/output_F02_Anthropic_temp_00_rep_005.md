 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and triggers repeated hash lookups on a named vector of 6.46 million entries — an O(n²)-like workload in practice. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again for each of the 5 variables, subsetting numeric vectors by index lists. The combination of these two stages, repeated 5 times, produces the estimated 86+ hour runtime.

**Specific problems:**

1. **String-key lookups at scale:** `idx_lookup` is a named vector of length ~6.46M. Named-vector lookup in R uses hashing, but constructing 6.46M paste keys and looking up variable-length neighbor sets against this hash is extremely slow in an interpreted `lapply` loop.
2. **Per-row `lapply` overhead:** 6.46M iterations of anonymous R functions, each allocating small vectors, causes massive interpreter and GC overhead.
3. **Redundant computation:** The neighbor *structure* is static across years — cell A's rook neighbors are the same cells every year. But the lookup is rebuilt as if the topology could change per row.
4. **Memory pressure:** Storing 6.46M list elements in `neighbor_lookup` (each a variable-length integer vector) is memory-inefficient and cache-unfriendly.

---

## Optimization Strategy

**Key insight:** The spatial neighbor topology is year-invariant. Cell `i`'s neighbors are the same cells regardless of year. We can therefore separate the spatial structure from the temporal indexing and use vectorized, column-oriented operations.

**Strategy (3 pillars):**

### 1. Vectorized Neighbor Lookup via `data.table` Joins (eliminate `lapply` entirely)

Instead of building a per-row list, we:
- Expand the `rook_neighbors_unique` nb object into an edge list `(focal_id, neighbor_id)`.
- Join this edge list to the panel data by `(neighbor_id, year)` to retrieve neighbor values.
- Group by `(focal_id, year)` and compute `max`, `min`, `mean` in one vectorized pass.

This replaces both `build_neighbor_lookup` and `compute_neighbor_stats` with a single `data.table` merge + grouped aggregation — no R-level loops at all.

### 2. Process One Variable at a Time (control peak memory)

With 6.46M rows and ~1.37M directed edges, the expanded join table is ~6.46M × (avg ~4 neighbors) ≈ 26M rows, but only needs 3 columns at a time (`focal_id`, `year`, `value`). At ~26M rows × 3 columns × 8 bytes ≈ 0.6 GB per variable, this fits comfortably in 16 GB alongside the original data (~5.7 GB for 6.46M × 110 cols).

### 3. Preserve the Trained Model and Numerical Estimand

We only change *how* the features are computed, not *what* is computed. The `max`, `min`, `mean` aggregations over the identical neighbor sets produce bit-identical results. The Random Forest model is never touched.

**Expected speedup:** Each variable's join + aggregation should take ~30–90 seconds on a modern laptop. Total for 5 variables: **~3–8 minutes** (vs. 86+ hours).

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# Step 0: Convert the spdep nb object to a data.table edge list (once)
# ──────────────────────────────────────────────────────────────────────
# id_order is the vector of cell IDs aligned with rook_neighbors_unique
# (i.e., id_order[k] is the cell ID for the k-th element of the nb list)

build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer vectors (indices into id_order)
  n <- length(neighbors)
  # Pre-allocate by counting total edges
  lens <- vapply(neighbors, length, integer(1))
  total <- sum(lens)
  
  focal_id    <- integer(total)
  neighbor_id <- integer(total)
  
  pos <- 1L
  for (k in seq_len(n)) {
    nb <- neighbors[[k]]
    m  <- lens[k]
    if (m == 0L) next
    idx <- pos:(pos + m - 1L)
    focal_id[idx]    <- id_order[k]
    neighbor_id[idx] <- id_order[nb]
    pos <- pos + m
  }
  
  data.table(focal_id = focal_id, neighbor_id = neighbor_id)
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# ~1.37M rows, two integer columns — trivial memory

# ──────────────────────────────────────────────────────────────────────
# Step 1: Convert panel data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure keyed for fast joins
setkey(cell_data, id, year)

# ──────────────────────────────────────────────────────────────────────
# Step 2: Compute neighbor features for each variable
# ──────────────────────────────────────────────────────────────────────
compute_and_add_neighbor_features_fast <- function(cell_data, var_name, edge_dt) {
  # Extract only the columns we need for the join (minimise memory)
  # Columns: neighbor_id (to join on), year, and the variable value
  val_dt <- cell_data[, .(id, year, value = get(var_name))]
  setnames(val_dt, "id", "neighbor_id")
  setkey(val_dt, neighbor_id, year)
  
  # Join edge list with values: for each (focal_id, neighbor_id) pair,
  # look up the neighbor's value in every year
  # Result: one row per (focal_id, neighbor_id, year) with the neighbor's value
  joined <- edge_dt[val_dt, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NULL]
  # joined has columns: focal_id, neighbor_id, year, value
  
  # Aggregate by (focal_id, year)
  agg <- joined[!is.na(value),
                 .(nb_max  = max(value),
                   nb_min  = min(value),
                   nb_mean = mean(value)),
                 by = .(focal_id, year)]
  
  # Build target column names (must match what the trained RF model expects)
  col_max  <- paste0(var_name, "_nb_max")
  col_min  <- paste0(var_name, "_nb_min")
  col_mean <- paste0(var_name, "_nb_mean")
  
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(col_max,  col_min,  col_mean))
  setnames(agg, "focal_id", "id")
  setkey(agg, id, year)
  
  # Merge back into cell_data (left join to preserve all rows, NAs where no neighbors)
  # Remove columns if they already exist (idempotent re-runs)
  for (col in c(col_max, col_min, col_mean)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  cell_data <- merge(cell_data, agg, by = c("id", "year"), all.x = TRUE)
  setkey(cell_data, id, year)
  
  # Clean up intermediate objects to free memory
  rm(val_dt, joined, agg)
  gc()
  
  return(cell_data)
}

# ──────────────────────────────────────────────────────────────────────
# Step 3: Outer loop — identical interface, vastly faster
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, edge_dt)
  message("  Done. Columns: ", ncol(cell_data))
}

# ──────────────────────────────────────────────────────────────────────
# Step 4: Predict with the existing trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# predictions <- predict(trained_rf_model, newdata = cell_data)
# (model object and predict call are untouched)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-iteration `lapply` with string hashing | One-time edge-list build (simple `for` over 344K cells) |
| **Stat computation** | 6.46M-iteration `lapply` per variable × 5 | `data.table` equi-join + grouped aggregation per variable × 5 |
| **Estimated runtime** | 86+ hours | ~3–8 minutes |
| **Peak memory** | Uncontrolled (millions of list elements + named vectors) | ~2–3 GB transient per variable, freed between iterations |
| **Numerical output** | max / min / mean of rook-neighbor values | Identical max / min / mean of identical neighbor sets |
| **Trained RF model** | — | Untouched; same feature names, same values |