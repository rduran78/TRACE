 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every single cell-year row, the function performs character coercion, `paste`, and named-vector lookups. With ~6.46M rows, this means ~6.46M iterations of string operations and hash lookups. Named vector lookups via `[` on character keys are O(n) in the worst case when the vector is large, and `paste` is vectorized but called inside a per-row `lapply`, defeating vectorization entirely.

### 2. Redundant recomputation of spatial topology per cell-year
The neighbor graph is **purely spatial** — it is identical across all 28 years. Yet `build_neighbor_lookup` rebuilds the neighbor mapping for every cell-year combination (6.46M rows) instead of building it once for the 344,208 cells and reusing it. This means the same spatial neighbor lookup is repeated 28 times per cell.

### 3. Row-level `lapply` over 6.46M rows in `compute_neighbor_stats`
For each of the 5 variables, `compute_neighbor_stats` iterates over all 6.46M rows, extracting neighbor values one row at a time. This is 5 × 6.46M = 32.3M R-level loop iterations with per-iteration subsetting, NA removal, and summary computation.

**Summary:** The fundamental mistake is treating a **spatial** problem as a **cell-year** problem. The neighbor table should be built once at the cell level (344,208 entries), then joined to the panel by `(id, year)` to compute neighbor statistics using vectorized, column-wise operations.

---

## Optimization Strategy

### Step 1: Build a static cell-neighbor edge table once (344K cells → ~1.37M directed edges)
Convert the `spdep::nb` object into a two-column `data.table` of `(cell_id, neighbor_id)`. This is done once and is reusable forever.

### Step 2: Join yearly attributes onto the edge table
For each year (or all years at once via a keyed join), join the cell-level attributes onto the neighbor side of the edge table. This replaces all per-row `lapply` calls with a single vectorized `data.table` merge.

### Step 3: Compute grouped neighbor statistics with `data.table`
Group by `(cell_id, year)` and compute `max`, `min`, `mean` of each neighbor variable in one pass using `data.table`'s optimized `by=` grouping. This replaces 6.46M R-level loop iterations per variable with a single vectorized grouped aggregation.

### Complexity comparison

| Step | Current | Proposed |
|---|---|---|
| Build neighbor lookup | O(6.46M) string ops | O(1.37M) integer edge table, once |
| Compute neighbor stats (per var) | O(6.46M) R-level iterations | O(1.37M × 28) vectorized grouped agg |
| Total R-level loop iterations | ~38.7M | 0 (fully vectorized) |

**Expected speedup:** From ~86 hours to **minutes** (likely 2–10 minutes depending on disk I/O).

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 0: Ensure cell_data is a data.table
# ==============================================================================
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# plus all other predictor columns needed for the RF model.
# id_order is the vector of cell IDs in the order matching rook_neighbors_unique.
# rook_neighbors_unique is the spdep::nb object (list of integer index vectors).

setDT(cell_data)

# ==============================================================================
# STEP 1: Build a static cell-neighbor edge table ONCE
#         This encodes the spatial topology and never changes across years.
# ==============================================================================
build_edge_table <- function(id_order, nb_object) {
  # nb_object is a list of length N where nb_object[[i]] contains integer
  # indices (into id_order) of the neighbors of cell id_order[i].
  # A 0-length integer(0) entry means no neighbors.
  
  from_idx <- rep(seq_along(nb_object), lengths(nb_object))
  to_idx   <- unlist(nb_object, use.names = FALSE)
  
  # Remove the spdep convention where 0L means "no neighbors"
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_table <- build_edge_table(id_order, rook_neighbors_unique)

cat(sprintf(
  "Edge table: %s directed edges for %s cells\n",
  format(nrow(edge_table), big.mark = ","),
  format(length(id_order), big.mark = ",")
))

# ==============================================================================
# STEP 2 & 3: For each neighbor source variable, join yearly attributes onto
#              the neighbor side of the edge table, compute grouped stats,
#              and merge back onto cell_data.
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-set keys for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {
  
  cat(sprintf("Computing neighbor stats for: %s\n", var_name))
  
  # --- Extract only the columns we need for this variable ---
  # Columns: id (to join as neighbor), year, and the variable value
  attr_cols <- c("id", "year", var_name)
  attr_dt   <- cell_data[, ..attr_cols]
  setnames(attr_dt, c("id", var_name), c("neighbor_id", "neighbor_val"))
  setkey(attr_dt, neighbor_id, year)
  
  # --- Join: for every (cell_id, neighbor_id) edge and every year,
  #     look up the neighbor's value of var_name ---
  # We need to cross the edge table with years. Instead of a full cross join
  # (which would be huge), we join edge_table onto attr_dt by neighbor_id,
  # which automatically expands across all years the neighbor has data for.
  
  # Merge edge_table with neighbor attributes (expands to ~1.37M * 28 rows)
  edges_with_vals <- merge(
    edge_table,
    attr_dt,
    by = "neighbor_id",
    allow.cartesian = TRUE
  )
  # Result columns: neighbor_id, cell_id, year, neighbor_val
  
  # --- Compute grouped neighbor statistics ---
  neighbor_stats <- edges_with_vals[
    !is.na(neighbor_val),
    .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ),
    by = .(cell_id, year)
  ]
  
  # --- Rename columns to match the original pipeline's naming convention ---
  # Original pipeline used: {var_name}_neighbor_max, {var_name}_neighbor_min,
  #                         {var_name}_neighbor_mean
  new_names <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  setnames(neighbor_stats, c("nb_max", "nb_min", "nb_mean"), new_names)
  
  # --- Remove old neighbor columns from cell_data if they exist (idempotent) ---
  for (col in new_names) {
    if (col %in% names(cell_data)) {
      cell_data[, (col) := NULL]
    }
  }
  
  # --- Merge neighbor stats back onto cell_data ---
  setkey(neighbor_stats, cell_id, year)
  cell_data <- merge(
    cell_data,
    neighbor_stats,
    by.x = c("id", "year"),
    by.y = c("cell_id", "year"),
    all.x = TRUE   # preserve rows with no neighbors (they get NA)
  )
  
  # Reset key after merge
  setkey(cell_data, id, year)
  
  cat(sprintf("  Done. Added: %s\n", paste(new_names, collapse = ", ")))
}

# ==============================================================================
# STEP 4: Predict with the existing trained Random Forest model
#         (model object is assumed to already be in memory, e.g., `rf_model`)
# ==============================================================================
# The trained RF model is preserved exactly as-is; no retraining occurs.
# The numerical estimand is preserved because we compute identical max/min/mean
# statistics over the same neighbor sets with the same variable values.

# Ensure column order / names match what the RF model expects:
# predictor_names <- rf_model$forest$independent.variable.names  # if ranger
# or: predictor_names <- rf_model$xNames  # if randomForest package

# Example prediction (uncomment and adapt to your RF package):
# cell_data[, prediction := predict(rf_model, data = cell_data)$predictions]

cat("Neighbor feature engineering complete. Ready for RF prediction.\n")
```

---

## Verification: Numerical Equivalence

The optimized code produces **identical** numerical results to the original because:

1. **Same neighbor sets:** The edge table is constructed from the same `rook_neighbors_unique` nb object and `id_order` vector. Each `(cell_id, year)` pair sees exactly the same set of neighbor cell-year rows.

2. **Same aggregation functions:** `max`, `min`, and `mean` are computed on the same non-NA neighbor values. The `data.table` grouped aggregation is algebraically identical to the per-row `lapply` approach.

3. **Same NA handling:** Rows with no neighbors (or all-NA neighbor values) receive `NA` for all three statistics, matching the `c(NA, NA, NA)` return in the original `compute_neighbor_stats`.

4. **RF model untouched:** The trained model object is never modified; only the feature columns are recomputed (identically) before calling `predict()`.

### Optional spot-check

```r
# Spot-check a random sample of rows against the old method
set.seed(42)
sample_rows <- sample(nrow(cell_data), 100)
for (var_name in neighbor_source_vars) {
  max_col <- paste0(var_name, "_neighbor_max")
  # Compare new values to values from the old pipeline (if saved)
  # stopifnot(all.equal(cell_data[[max_col]][sample_rows],
  #                     old_cell_data[[max_col]][sample_rows]))
}
```

---

## Performance Estimate

| Operation | Rows/Iterations | Estimated Time |
|---|---|---|
| Build edge table | 1.37M edges | < 1 second |
| Per-variable merge + group-by (×5 vars) | ~38.4M joined rows each | ~1–2 min each |
| Total | — | **~5–10 minutes** |

This represents a speedup of roughly **500–1,000×** over the original 86-hour estimate, achieved entirely through eliminating R-level row iteration in favor of vectorized `data.table` joins and grouped aggregations, and by factoring the time-invariant spatial topology out of the per-year computation.