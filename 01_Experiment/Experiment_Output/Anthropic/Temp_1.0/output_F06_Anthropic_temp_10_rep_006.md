 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the cell's spatial neighbor IDs via `id_to_ref` and `neighbors`.
2. Pastes neighbor IDs with the current year to create keys (`paste(..., sep="_")`).
3. Looks up those keys in `idx_lookup` (a named vector of 6.46M elements).

This is repeated 6.46 million times. The `paste` calls and named-vector lookups are extremely slow in base R at this scale. Worse, the spatial neighbor structure is **identical across all 28 years** — only the year suffix changes — yet it's recomputed from scratch for every row.

### Bottleneck B: `compute_neighbor_stats` — O(n) `lapply` with per-row subsetting

For each of the 5 variables, the function loops over 6.46M rows, subsets the variable vector by neighbor indices, removes NAs, and computes `max`, `min`, `mean`. This produces ~32.3M individual R function calls (5 vars × 6.46M rows), each with overhead from subsetting, `is.na`, and summary statistics.

### Why raster focal/kernel operations are tempting but wrong

Raster focal operations (e.g., `terra::focal`) assume a regular grid with uniform kernel geometry. This panel has an **irregular spdep::nb neighbor structure** (rook contiguity on potentially irregular polygons, with edge/boundary cells having fewer neighbors). A focal approach would either silently change the neighbor definitions or require padding/masking that negates the speed gain. We must **preserve the exact nb object** to preserve the numerical estimand.

### Summary of waste

| Source | Estimated share of 86h |
|---|---|
| 6.46M `paste` + named-vector lookups in `build_neighbor_lookup` | ~40% |
| 6.46M × 5 per-row `lapply` calls in `compute_neighbor_stats` | ~55% |
| Memory pressure / GC | ~5% |

---

## 2. Optimization Strategy

### Strategy A: Vectorize the neighbor lookup via sparse matrix multiplication

Instead of row-by-row subsetting, represent the neighbor structure as a **sparse adjacency matrix W** (344,208 × 344,208). Then for each year-slice, computing `neighbor_mean` is just a sparse matrix-vector product `W %*% x / row_degree`. `neighbor_max` and `neighbor_min` can be computed similarly using grouped operations.

However, sparse matrix multiplication only gives us the **sum** (and thus mean given known row-degrees). Max and min require a different approach.

### Strategy B: Expand to a long edge-list and use `data.table` grouped aggregation

Convert the nb object to an edge data.frame `(from_id, to_id)`. Join with the panel on `(to_id, year)` to get neighbor values. Then group by `(from_id, year)` and compute `max`, `min`, `mean` in one pass.

This is the **optimal approach** because:
- `data.table` grouped aggregation is vectorized in C.
- The join is a single hash-join, not 6.46M individual lookups.
- All three statistics are computed in one grouped call.
- Memory is manageable: the edge-list × 28 years ≈ 1.37M × 28 ≈ 38.4M rows of (from, to, year, value) — about 1.2 GB per variable, well within 16 GB.
- **The numerical results are identical** because the same neighbor pairs and the same aggregation functions are used.

### Strategy C: Hybrid — sparse matrix for mean, data.table for max/min

Use `Matrix::sparseMatrix` for the mean (fastest possible: one sparse mat-vec per year-variable), and `data.table` for max/min. This is faster but adds complexity.

**I recommend Strategy B (pure `data.table`)** for clarity and maintainability, with expected runtime of **2–5 minutes** instead of 86+ hours.

---

## 3. Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 0: Convert spdep::nb object to a data.table edge list (one-time cost)
# ==============================================================================
# rook_neighbors_unique is an nb object indexed by position in id_order.
# id_order is the vector of cell IDs in the same order as the nb object.

nb_to_edge_dt <- function(nb_obj, id_order) {
  # Pre-allocate by computing total number of edges
  n_edges <- sum(vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos <- 1L
  
  for (i in seq_along(nb_obj)) {
    nbs <- nb_obj[[i]]
    # spdep uses 0 to denote "no neighbors"
    if (length(nbs) == 1L && nbs[1] == 0L) next
    n <- length(nbs)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nbs]
    pos <- pos + n
  }
  
  data.table(from_id = from_id, to_id = to_id)
}

edges <- nb_to_edge_dt(rook_neighbors_unique, id_order)
# edges has columns: from_id, to_id
# Each row means: "to_id is a rook neighbor of from_id"

cat(sprintf("Edge list: %d directed neighbor pairs\n", nrow(edges)))

# ==============================================================================
# STEP 1: Convert cell_data to data.table if not already
# ==============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ==============================================================================
# STEP 2: Compute neighbor features for all source variables
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a minimal lookup table: only id, year, and the source variables
lookup_cols <- c("id", "year", neighbor_source_vars)
lookup_dt <- cell_data[, ..lookup_cols]

# Set key for fast join
setkey(lookup_dt, id, year)

# For each variable, join edges with neighbor values, aggregate, and merge back
for (var_name in neighbor_source_vars) {
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))
  
  # Build a small table with just id, year, and the variable of interest
  val_dt <- lookup_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)
  
  # Join edges with the *neighbor's* values:
  # For each (from_id, to_id) pair, get all years of to_id
  # We need: for each (from_id, year), the values of all to_id neighbors in that year
  
  # Expand edges by year via join on to_id
  # edge_vals: from_id | to_id | year | val (val is the neighbor's value)
  edge_vals <- merge(
    edges,
    val_dt,
    by.x = "to_id",
    by.y = "id",
    allow.cartesian = TRUE  # each edge × 28 years
  )
  # edge_vals columns: to_id, from_id, year, val
  
  # Aggregate: for each (from_id, year), compute max, min, mean of neighbor vals
  # Remove NAs before aggregation
  edge_vals <- edge_vals[!is.na(val)]
  
  agg <- edge_vals[,
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = .(from_id, year)
  ]
  
  # Rename columns to match expected output format
  max_col  <- paste0("max_neighbor_", var_name)
  min_col  <- paste0("min_neighbor_", var_name)
  mean_col <- paste0("mean_neighbor_", var_name)
  
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  setnames(agg, "from_id", "id")
  
  # Merge back into cell_data
  # First remove these columns if they already exist (idempotent re-runs)
  existing <- intersect(c(max_col, min_col, mean_col), names(cell_data))
  if (length(existing) > 0) {
    cell_data[, (existing) := NULL]
  }
  
  cell_data <- merge(cell_data, agg, by = c("id", "year"), all.x = TRUE)
  
  cat(sprintf("  Done. Added: %s, %s, %s\n", max_col, min_col, mean_col))
}

# ==============================================================================
# STEP 3: Predict with existing Random Forest model (unchanged)
# ==============================================================================
# The trained RF model object is assumed to be loaded (e.g., `rf_model`).
# Prediction proceeds exactly as before — the feature columns have identical
# names and identical numerical values.

# Example (uncomment and adapt to your pipeline):
# cell_data$predicted <- predict(rf_model, newdata = cell_data)

cat("Neighbor feature engineering complete.\n")
```

---

## 4. Why This Preserves the Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor pairs** | `nb_to_edge_dt` reads the identical `rook_neighbors_unique` nb object and `id_order`. Every `(from_id, to_id)` pair is preserved. |
| **Same aggregation functions** | `max`, `min`, `mean` are applied to the same sets of non-NA neighbor values per `(from_id, year)` group. |
| **NA handling** | `edge_vals[!is.na(val)]` before aggregation reproduces the `neighbor_vals[!is.na(neighbor_vals)]` filter. Cells with zero valid neighbors get `NA` via the `all.x = TRUE` merge. |
| **No raster resampling** | No focal window or kernel is used. The exact irregular nb topology is preserved. |
| **RF model untouched** | The trained model is loaded and used as-is for prediction. No retraining occurs. |

## 5. Expected Performance

| Step | Estimated Time | Estimated Peak RAM |
|---|---|---|
| `nb_to_edge_dt` | < 1 second | ~20 MB |
| Per-variable `merge` + `aggregate` | ~20–40 seconds | ~2–3 GB |
| All 5 variables total | **2–4 minutes** | ~4 GB peak |
| **Speedup vs. 86 hours** | **~1,500×** | — |

The dominant cost is the `merge` (cartesian join of 1.37M edges × 28 years = 38.4M rows) followed by the grouped aggregation, both of which `data.table` executes in optimized C.