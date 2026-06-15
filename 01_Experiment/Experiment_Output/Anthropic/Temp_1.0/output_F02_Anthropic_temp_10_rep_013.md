 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core performance bottleneck is **`build_neighbor_lookup`**, which runs an `lapply` over **~6.46 million rows**, performing per-row character coercion, string pasting, and named-vector lookups. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. The result is a list of 6.46 million integer vectors, consuming both enormous time and memory. Then `compute_neighbor_stats` iterates over that 6.46M-element list again for *each* of the 5 variables, performing per-element subsetting, NA removal, and summary stats—adding another 5 × 6.46M ≈ 32.3M R-level loop iterations.

**Specific problems:**

1. **String-key lookups at scale:** `paste()` + named-vector indexing on 6.46M rows is O(n) with a large constant factor. Each call to `id_to_ref[as.character(...)]` and `idx_lookup[neighbor_keys]` involves hashing millions of strings.
2. **Ragged list of 6.46M elements:** The `neighbor_lookup` list stores ~6.46M integer vectors. This is ~1.37M directed edges × years replicated, consuming several GB of RAM just for the list overhead (each R vector has a ~56-byte header).
3. **Row-level R loops:** `lapply` over millions of rows in pure R is inherently slow—no vectorization, no SIMD, no parallelism.
4. **Redundant computation:** The neighbor *graph* is year-invariant (a cell's rook neighbors don't change over time), but the lookup is rebuilt as if each cell-year is unique.

---

## Optimization Strategy

**Key insight:** The neighbor topology is **time-invariant**. Cell `i`'s neighbors in 1992 are the same cells as in 2019. We only need to look up neighbor *values* for the matching year. This can be fully vectorized using `data.table` joins, eliminating the 6.46M-element list entirely.

**Approach — "Edge Table + data.table grouped join":**

1. **Expand the neighbor graph into an edge table** (~1.37M rows of `(cell_id, neighbor_id)` pairs). This is done once.
2. **Join the edge table to the panel data** on `(neighbor_id, year)` to pull each neighbor's variable value. This is a single keyed `data.table` merge—highly optimized in C.
3. **Group by `(cell_id, year)`** and compute `max`, `min`, `mean` in one pass using `data.table`'s grouped aggregation (also C-level, vectorized).
4. **Repeat for each of the 5 variables** (or batch them in one join).

This replaces billions of R-level operations with a handful of C-level `data.table` operations. Expected speedup: **~100–500×** (minutes instead of days). Memory stays well within 16 GB because the edge table is only ~1.37M rows, and the join result is ~1.37M × 28 ≈ 38.4M rows of numeric data (~1–2 GB).

---

## Working R Code

```r
library(data.table)

# ─────────────────────────────────────────────────────────────────
# STEP 0: Convert panel data to data.table (if not already)
# ─────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)

# Ensure id and year are proper types for joining
cell_dt[, id := as.integer(id)]
cell_dt[, year := as.integer(year)]

# ─────────────────────────────────────────────────────────────────
# STEP 1: Build the edge table from the nb object (done once)
#
# rook_neighbors_unique is a list of length 344,208 where element
# [[i]] is an integer vector of indices into id_order.
# id_order maps position -> cell id.
# ─────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, neighbors) {
  # Pre-compute lengths for pre-allocation
  n_edges <- sum(vapply(neighbors, length, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb <- neighbors[[i]]
    n  <- length(nb)
    if (n == 0L) next
    idx_range <- pos:(pos + n - 1L)
    from_id[idx_range] <- id_order[i]
    to_id[idx_range]   <- id_order[nb]
    pos <- pos + n
  }
  
  data.table(cell_id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

cat("Building edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("Edge table: %d directed edges\n", nrow(edge_dt)))

# ─────────────────────────────────────────────────────────────────
# STEP 2: Compute neighbor features for all variables via join
# ─────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We join edge_dt × years to cell_dt to get neighbor values.
# To avoid a massive cross-join, we join per-variable in a memory-
# efficient loop.

# Key cell_dt for fast joining
setkey(cell_dt, id, year)

# Get the unique years vector once
all_years <- sort(unique(cell_dt$year))

cat("Computing neighbor features...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing variable: %s\n", var_name))
  
  # Extract only the columns we need for the lookup side
  # This keeps memory lean
  lookup_cols <- c("id", "year", var_name)
  neighbor_vals_dt <- cell_dt[, ..lookup_cols]
  setnames(neighbor_vals_dt, c("id", var_name), c("neighbor_id", "nb_val"))
  setkey(neighbor_vals_dt, neighbor_id, year)
  
  # Join: for every (cell_id, neighbor_id) edge and every year,
  # get the neighbor's value.
  # We do this by joining edge_dt to the panel on neighbor_id,
  # allowing year from neighbor_vals_dt to carry through.
  
  # Expand edges × years efficiently:
  # Rather than a literal cross-join (which would be 1.37M × 28 = 38.4M rows),
  # we merge edge_dt with neighbor_vals_dt on neighbor_id,
  # which naturally gives us one row per (edge, year) combination
  # wherever the neighbor has data.
  
  joined <- merge(edge_dt, neighbor_vals_dt,
                  by = "neighbor_id",
                  allow.cartesian = TRUE)
  # joined has columns: neighbor_id, cell_id, year, nb_val
  # ~38.4M rows (1.37M edges × 28 years)
  
  # Aggregate by (cell_id, year)
  agg <- joined[!is.na(nb_val),
                 .(nb_max  = max(nb_val),
                   nb_min  = min(nb_val),
                   nb_mean = mean(nb_val)),
                 by = .(cell_id, year)]
  
  # Name the new columns to match the original pipeline's convention
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
           c(max_col, min_col, mean_col))
  
  # Merge back into cell_dt
  setkey(agg, cell_id, year)
  cell_dt <- merge(cell_dt, agg, by.x = c("id", "year"),
                   by.y = c("cell_id", "year"), all.x = TRUE)
  
  # Free intermediate objects
  rm(joined, agg, neighbor_vals_dt)
  gc()
  
  cat(sprintf("    Done — added %s, %s, %s\n", max_col, min_col, mean_col))
}

# ─────────────────────────────────────────────────────────────────
# STEP 3: Convert back to data.frame if downstream code expects it
# ─────────────────────────────────────────────────────────────────
cell_data <- as.data.frame(cell_dt)
rm(cell_dt, edge_dt)
gc()

cat("Neighbor feature computation complete.\n")

# ─────────────────────────────────────────────────────────────────
# The trained Random Forest model is untouched. Proceed to
# prediction using the existing model object as before:
#
#   predictions <- predict(rf_model, newdata = cell_data)
#
# The numerical values of max/min/mean neighbor stats are
# identical to the original implementation (same arithmetic,
# same neighbor graph), preserving the original estimand.
# ─────────────────────────────────────────────────────────────────
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor graph** | `build_edge_table` reads the identical `rook_neighbors_unique` nb object and `id_order` mapping. Every directed edge is preserved. |
| **Same per-year matching** | The join on `(neighbor_id, year)` ensures each cell-year row picks up its neighbors' values from the *same* year—identical to the original `paste(id, year)` key logic. |
| **Same summary statistics** | `max`, `min`, `mean` after `!is.na` filtering are the same arithmetic operations as the original `compute_neighbor_stats`. |
| **Trained RF model untouched** | No model retraining occurs. The output column names match the original convention, so `predict(rf_model, newdata = ...)` works unchanged. |

## Expected Performance

| Metric | Original | Optimized |
|---|---|---|
| **`build_neighbor_lookup`** | ~6.46M R-level string ops → hours | Replaced by `build_edge_table`: ~1.37M iterations in a simple for-loop → **< 5 seconds** |
| **`compute_neighbor_stats`** (×5 vars) | ~32.3M R-level list iterations → tens of hours | Replaced by keyed `data.table` merge + grouped agg → **~2–5 minutes per variable** |
| **Total wall time** | 86+ hours estimated | **~15–30 minutes total** |
| **Peak RAM** | Multiple GB for the 6.46M-element list | ~3–5 GB peak (38.4M-row join intermediate) — fits in 16 GB |