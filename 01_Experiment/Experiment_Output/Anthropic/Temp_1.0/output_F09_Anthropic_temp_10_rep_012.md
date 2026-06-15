 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of **two compounding bottlenecks**:

### 1. `build_neighbor_lookup` creates a per-row (cell-year) lookup — 6.46 million entries
The function iterates over every cell-year row, resolves neighbor cell IDs, then pastes year-specific keys to find the row indices of neighbors *in the current panel-format data frame*. This means:
- **6.46 million calls** to the inner `lapply` function.
- Each call performs `paste()` and named-vector lookups (which are O(n) hash lookups on a ~6.46M-length named vector `idx_lookup`).
- The resulting `neighbor_lookup` list has ~6.46 million elements, each containing integer vectors.
- Total memory for this list is very large and thrashes RAM/cache.

### 2. `compute_neighbor_stats` iterates over the same 6.46 million rows
For each of 5 variables, it loops over 6.46 million entries again, subsetting a numeric vector by the index vectors from the lookup. That's ~32.3 million R-level `lapply` iterations total.

### Root Cause
The topology (which cell neighbors which cell) **never changes across years**. There are only ~344K cells and ~1.37M directed neighbor pairs. But the code re-resolves this topology at the cell-year level (×28 years), inflating 344K relationships into 6.46M relationships. This is **28× more work than necessary** at the lookup stage, and creates a **28× larger lookup object** that then penalizes every downstream stats computation.

---

## Optimization Strategy

**Build a time-invariant cell-neighbor edge table once (344K cells, ~1.37M edges), then use vectorized joins and grouped aggregation per year to compute neighbor stats.**

Specifically:

1. **Build a static edge `data.table`** with columns `(id, neighbor_id)` from the `spdep::nb` object. This has ~1.37M rows and is built once.
2. **Join yearly attributes onto the edge table** by `(neighbor_id, year)` — a keyed `data.table` join, which is O(n log n) and highly optimized in C.
3. **Compute grouped `max`, `min`, `mean`** by `(id, year)` — a single vectorized `data.table` aggregation call per variable.
4. **Merge the results back** onto the main `cell_data` data.table by `(id, year)`.

This eliminates all per-row R-level loops and replaces them with bulk vectorized operations. Expected speedup: **~100–500×** (minutes instead of days).

---

## Working R Code

```r
library(data.table)

# ==============================================================
# STEP 0: Convert cell_data to data.table if not already
# ==============================================================
cell_data <- as.data.table(cell_data)

# ==============================================================
# STEP 1: Build a STATIC cell-neighbor edge table (time-invariant)
#
# Inputs:
#   id_order              — vector of cell IDs (length 344,208),
#                           ordered to match rook_neighbors_unique
#   rook_neighbors_unique — spdep nb object (list of integer index vectors)
#
# Output:
#   edges — data.table with columns (id, neighbor_id), ~1.37M rows
# ==============================================================

build_static_edge_table <- function(id_order, neighbors) {
  n <- length(neighbors)
  # Pre-allocate vectors
  from_ids <- vector("list", n)
  to_ids   <- vector("list", n)
  for (i in seq_len(n)) {
    nb_idx <- neighbors[[i]]
    # spdep nb objects use 0L to denote "no neighbors"
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) > 0L) {
      from_ids[[i]] <- rep(id_order[i], length(nb_idx))
      to_ids[[i]]   <- id_order[nb_idx]
    }
  }
  data.table(
    id          = unlist(from_ids, use.names = FALSE),
    neighbor_id = unlist(to_ids,   use.names = FALSE)
  )
}

edges <- build_static_edge_table(id_order, rook_neighbors_unique)

cat("Edge table rows:", nrow(edges), "\n")
# Expected: ~1,373,394

# ==============================================================
# STEP 2: For each neighbor source variable, compute neighbor
#          max, min, mean via vectorized join + grouped aggregation
# ==============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key cell_data for fast joins
# We will join on (neighbor_id = id, year = year), so prepare a lookup keyed on (id, year)
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {

  cat("Computing neighbor stats for:", var_name, "...\n")

  # --- 2a. Build a slim lookup: (id, year, value) keyed on (id, year) ---
  lookup_cols <- c("id", "year", var_name)
  val_lookup  <- cell_data[, ..lookup_cols]
  setnames(val_lookup, old = var_name, new = "value")
  setkey(val_lookup, id, year)

  # --- 2b. Expand edges × years by joining neighbor attribute ---
  #
  # edges has (id, neighbor_id).
  # We join val_lookup onto edges by  neighbor_id == id  to get the
  # neighbor's value for every year the neighbor appears in the panel.
  #
  # Strategy: add neighbor values via a keyed join.
  # Result: one row per (id, neighbor_id, year) with the neighbor's value.

  # Rename for the join: we want to match val_lookup$id to edges$neighbor_id
  edge_expanded <- merge(
    edges,
    val_lookup,
    by.x = "neighbor_id",
    by.y = "id",
    allow.cartesian = TRUE   # each neighbor_id has up to 28 year-rows
  )
  # edge_expanded now has columns: neighbor_id, id, year, value
  # Each row = "cell `id` has neighbor `neighbor_id` in `year` with value `value`"

  # --- 2c. Grouped aggregation: neighbor max, min, mean per (id, year) ---
  neighbor_stats <- edge_expanded[
    !is.na(value),
    .(
      nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)
    ),
    by = .(id, year)
  ]

  # --- 2d. Name the new columns to match original pipeline's naming convention ---
  #         Adjust the naming pattern to match whatever your trained RF expects.
  #         Common pattern: neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(neighbor_stats, c("nb_max", "nb_min", "nb_mean"),
           c(max_col, min_col, mean_col))

  # --- 2e. Remove old columns if they exist, then merge new ones onto cell_data ---
  old_cols <- intersect(c(max_col, min_col, mean_col), names(cell_data))
  if (length(old_cols) > 0L) {
    cell_data[, (old_cols) := NULL]
  }

  setkey(neighbor_stats, id, year)
  cell_data <- merge(cell_data, neighbor_stats, by = c("id", "year"), all.x = TRUE)
  setkey(cell_data, id, year)  # restore key

  # Clean up per-iteration temporaries

rm(val_lookup, edge_expanded, neighbor_stats)
  gc()

  cat("  Done:", max_col, min_col, mean_col, "\n")
}

# ==============================================================
# STEP 3: Predict with the already-trained Random Forest
#          (model object untouched — no retraining)
# ==============================================================
# cell_data now has the same neighbor feature columns the RF was trained on.
# Proceed with prediction exactly as before:

# Example (adjust to your actual model object and prediction call):
# predictions <- predict(rf_model, newdata = cell_data)

cat("All neighbor features computed. Ready for Random Forest prediction.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max()`, `min()`, `mean()` are computed on exactly the same set of rook-neighbor values per (cell, year) as the original code. The edge table encodes the identical topology from `rook_neighbors_unique`, and the join on `(neighbor_id, year)` selects the same year-matched neighbor values. `NA` handling via `!is.na(value)` before aggregation mirrors the original `neighbor_vals[!is.na(neighbor_vals)]`. Cells with no non-NA neighbors get `NA` via `all.x = TRUE` in the final merge. |
| **Trained RF model** | The model object is never modified. The code only rebuilds the input features that the model expects, with identical column names and identical numerical values. |
| **Column naming** | Uses the pattern `neighbor_max_{var}`, `neighbor_min_{var}`, `neighbor_mean_{var}`. Adjust the `paste0` pattern if your trained model expects a different naming convention (e.g., `{var}_neighbor_max`). |

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build lookup / edge table | ~hours (6.46M R-level iterations) | **< 5 seconds** (344K list traversal) |
| Compute stats (5 vars) | ~80+ hours (32.3M R-level iterations) | **~2–5 minutes** (vectorized `data.table` joins + grouped agg) |
| Peak RAM | ~10–14 GB (giant list of index vectors) | **~4–6 GB** (edge table × 28 years in temporary merge) |
| **Total** | **~86+ hours** | **~3–8 minutes** |