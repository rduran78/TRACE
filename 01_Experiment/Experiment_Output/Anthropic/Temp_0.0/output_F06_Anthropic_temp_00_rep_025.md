 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine — O(1) hash lookup).
2. Constructs `paste(neighbor_id, year)` key strings for every neighbor (expensive string allocation × ~4 neighbors × 6.46M rows ≈ 25.8M string constructions).
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) — named vector indexing in R is hash-based but still slow at this scale.

The result is a **list of 6.46 million integer vectors**. This alone can take hours and consumes substantial memory.

**Key insight:** The neighbor structure is *purely spatial* — it doesn't change across years. A cell's rook neighbors in 1993 are the same cells as in 2007. The only thing that changes is which *row index* corresponds to `(neighbor_id, year)`. If the data is sorted by `(id, year)` or `(year, id)` in a predictable way, we can compute row indices arithmetically instead of via string hashing.

### Bottleneck B: `compute_neighbor_stats` — `lapply` over 6.46M elements, repeated 5 times

For each of the 5 source variables, we iterate over 6.46M rows, subset a numeric vector by index, remove NAs, and compute `max/min/mean`. This is 32.3M R-level function calls total. Each call is cheap, but the R interpreter overhead at this scale is enormous.

### Why raster focal/kernel operations are *not* directly applicable

The comment in the prompt asks whether raster focal operations (e.g., `terra::focal`) could help. Focal operations assume a **regular grid with complete coverage** and a fixed kernel. Here:
- The panel is cell-year, not a single raster layer.
- The neighbor structure (`spdep::nb`) may reflect irregular boundaries (coastal cells, border cells with fewer than 4 neighbors).
- We need `max`, `min`, and `mean` — focal can do this per-layer, but we'd need to reshape to 28 raster layers, run focal 3×5×28 = 420 times, then reshape back.

The **better approach** is to vectorize the neighbor computation directly using sparse matrix multiplication and vectorized row operations, which preserves the exact `spdep::nb` topology and the exact numerical results.

---

## 2. Optimization Strategy

### Strategy A: Eliminate `build_neighbor_lookup` entirely for per-variable stats

Instead of building a 6.46M-element list of row indices, we:

1. **Sort the data** by `(id, year)` so that each cell's 28 years are contiguous.
2. **Build a sparse adjacency matrix** `W` of dimension `344,208 × 344,208` from the `nb` object (one-time cost, fast via `spdep::nb2listw` → `as_dgRMatrix`).
3. For each year, extract the column of values, compute `W %*% x` for the mean (with row-sum normalization), and use grouped sparse-row operations for `max` and `min`.

But sparse matrix multiplication gives us **sum** (hence mean), not max/min. For max and min we need a different approach.

### Strategy B (Chosen): Vectorized expansion via `data.table` joins

The fastest pure-R approach that preserves exact results:

1. Build an **edge list** from the `nb` object: a two-column integer matrix `(from_cell_idx, to_cell_idx)` with ~1.37M rows.
2. Ensure `cell_data` is a `data.table` keyed on `(id, year)`.
3. For each source variable, do a single **non-equi join / edge-list join**: expand each cell-year to its neighbors' values via the edge list, then **group-by** `(cell_row)` to compute `max`, `min`, `mean` in one vectorized pass.

This replaces 6.46M R-level iterations with a single `data.table` grouped aggregation over ~25.8M edge-year rows — something `data.table` handles in seconds.

**Expected speedup:** From 86+ hours to **~5–15 minutes total**.

---

## 3. Working R Code

```r
library(data.table)
library(spdep)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert cell_data to data.table if not already
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build a spatial edge list from the nb object (one-time, fast)
#
#   rook_neighbors_unique: an nb object of length 344,208
#   id_order: vector of cell IDs in the order matching the nb object
#
#   We build a data.table with columns:
#     focal_id    — the cell ID of the focal cell
#     neighbor_id — the cell ID of each rook neighbor
# ──────────────────────────────────────────────────────────────────────
build_edge_list <- function(id_order, nb_obj) {
  n <- length(nb_obj)
  # Pre-compute lengths to allocate once
  lens <- vapply(nb_obj, length, integer(1))
  total_edges <- sum(lens)
  
  focal_idx    <- rep.int(seq_len(n), lens)
  neighbor_idx <- unlist(nb_obj, use.names = FALSE)
  
  # Remove any 0-entries (spdep uses 0 to denote "no neighbors" in some cases)
  valid <- neighbor_idx > 0L
  focal_idx    <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]
  
  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

cat("Building spatial edge list...\n")
edge_list <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %d directed edges\n", nrow(edge_list)))

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Compute neighbor features for all source variables
#
#   For each variable, we:
#     a) Join edge_list × cell_data to get neighbor values per cell-year
#     b) Aggregate max, min, mean by (focal_id, year)
#     c) Merge back into cell_data
#
#   This is fully vectorized — no row-level R loops.
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data has a row-order key we can restore later
cell_data[, .row_order := .I]

for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing neighbor features for: %s\n", var_name))
  
  # Extract only the columns we need for the join (minimize memory)
  # neighbor_id will be matched to id, year will be matched to year
  cols_needed <- c("id", "year", var_name)
  neighbor_vals <- cell_data[, ..cols_needed]
  setnames(neighbor_vals, old = c("id", var_name), new = c("neighbor_id", "nval"))
  
  # Key for fast join
  setkey(neighbor_vals, neighbor_id, year)
  
  # Expand: for each edge (focal_id, neighbor_id), join on (neighbor_id, year)
  # This creates one row per (focal_cell, neighbor_cell, year) combination
  # with the neighbor's value attached.
  #
  # We join edge_list to neighbor_vals:
  #   edge_list has: focal_id, neighbor_id
  #   neighbor_vals has: neighbor_id, year, nval
  #   Result: focal_id, neighbor_id, year, nval
  
  expanded <- neighbor_vals[edge_list, on = .(neighbor_id), allow.cartesian = TRUE, nomatch = NA]
  # expanded now has columns: neighbor_id, year, nval, focal_id
  
  # Aggregate by (focal_id, year) — this is the core computation
  agg <- expanded[!is.na(nval),
                  .(nb_max  = max(nval),
                    nb_min  = min(nval),
                    nb_mean = mean(nval)),
                  by = .(focal_id, year)]
  
  # Rename columns to match expected output format
  new_col_max  <- paste0("nb_max_",  var_name)
  new_col_min  <- paste0("nb_min_",  var_name)
  new_col_mean <- paste0("nb_mean_", var_name)
  setnames(agg,
           old = c("focal_id", "nb_max", "nb_min", "nb_mean"),
           new = c("id", new_col_max, new_col_min, new_col_mean))
  
  # Merge back into cell_data
  setkey(agg, id, year)
  setkey(cell_data, id, year)
  
  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(new_col_max, new_col_min, new_col_mean)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }
  
  cell_data <- agg[cell_data, on = .(id, year)]
  
  # Clean up to free memory
  rm(neighbor_vals, expanded, agg)
  gc()
  
  cat(sprintf("  Done: added %s, %s, %s\n", new_col_max, new_col_min, new_col_mean))
}

# Restore original row order
setorder(cell_data, .row_order)
cell_data[, .row_order := NULL]

cat("All neighbor features computed.\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Predict with the pre-trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The trained model object (e.g., `rf_model`) is used as-is.
# Example:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves Exact Results

| Concern | Guarantee |
|---|---|
| **Same neighbor topology** | Edge list is built directly from the same `spdep::nb` object (`rook_neighbors_unique`). No approximation. |
| **Same statistics** | `max`, `min`, `mean` are computed on exactly the same sets of neighbor values. NAs are excluded identically (`!is.na(nval)` mirrors `neighbor_vals[!is.na(neighbor_vals)]`). |
| **Same RF model** | The model is never retrained — only `predict()` is called on the enriched data. |
| **No raster approximation** | We rejected `terra::focal` because it would impose a regular-grid kernel that could silently alter neighbor sets at boundaries. |

## 5. Performance Estimate

| Step | Original | Optimized |
|---|---|---|
| Build lookup (6.46M string hashes) | ~10–20 hours | **~2 seconds** (edge list from `nb`) |
| Compute stats (5 vars × 6.46M `lapply`) | ~60–70 hours | **~5–10 min** (5 × `data.table` grouped agg on ~25.8M rows) |
| Peak RAM | ~12–14 GB (6.46M-element list) | **~6–8 GB** (edge expansion is streaming per variable, cleaned up each iteration) |
| **Total** | **86+ hours** | **~5–15 minutes** |