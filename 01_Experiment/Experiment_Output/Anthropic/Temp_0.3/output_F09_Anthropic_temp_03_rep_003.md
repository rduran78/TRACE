 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(N²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs **character key strings** (`paste(id, year)`) and performs **named-vector lookups** (`idx_lookup[neighbor_keys]`) — these are hash-table lookups but done 6.46 million times, each with ~4 neighbor keys on average.
- The result is a **list of 6.46 million integer vectors**, consuming significant memory and time.

The fundamental problem: **the neighbor topology is purely spatial and does not change across years**, yet the lookup is rebuilt per cell-year row, fusing spatial structure with temporal indexing in a single expensive pass.

### 2. `compute_neighbor_stats` iterates over the same 6.46M-element list
For each of the 5 variables, another `lapply` over 6.46 million elements computes `max`, `min`, `mean` from R-level subsetting. That's **~32.3 million R-level function calls** (5 vars × 6.46M rows), each involving vector subsetting and three aggregation functions.

### Memory pressure
The `neighbor_lookup` list of 6.46M integer vectors, plus intermediate copies, can easily push past 16 GB on a laptop, causing swapping and further slowdown.

---

## Optimization Strategy

**Core insight:** The neighbor graph is *time-invariant*. Build a **spatial adjacency table once** (344K cells × ~4 neighbors ≈ 1.37M directed edges), then **join yearly attributes onto it** using vectorized `data.table` operations. This replaces all `lapply` loops with vectorized grouped aggregations.

| Step | What | Complexity |
|------|------|------------|
| 1 | Build a `data.table` of directed edges: `(cell_id, neighbor_id)` — ~1.37M rows, built once. | O(E) where E ≈ 1.37M |
| 2 | Cross-join edges with years → ~38.4M edge-year rows (1.37M × 28). Or better: join cell-year attributes onto edges. | Vectorized join |
| 3 | For each variable, join neighbor's value onto the edge table, then `group_by(cell_id, year)` to compute `max`, `min`, `mean`. | Vectorized grouped agg |
| 4 | Join the resulting stats back onto the main cell-year table. | Keyed join |

**Expected speedup:** From ~86 hours to **minutes** (typically 5–15 min on a 16 GB laptop), because:
- No R-level loops over 6.46M rows.
- `data.table` grouped aggregations are C-optimized.
- Memory footprint is controlled (edge table is ~1.37M rows, not 6.46M lists).

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert cell_data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
# cell_data must have columns: id, year, and all predictor columns.
# rook_neighbors_unique is the spdep::nb object (list of integer vectors).
# id_order is the vector mapping nb-list index → cell id.

cell_dt <- as.data.table(cell_data)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build the spatial adjacency edge table ONCE
#         This is time-invariant — ~1.37M directed edges.
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, nb_list) {
  # nb_list[[i]] contains integer indices into id_order for neighbors of cell i
  # Expand into a two-column edge table
  from_idx <- rep(seq_along(nb_list), lengths(nb_list))
  to_idx   <- unlist(nb_list, use.names = FALSE)

  # Remove the spdep "no-neighbor" sentinel (0)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edge_dt), "\n")
# Expected: ~1,373,394

# ──────────────────────────────────────────────────────────────────────
# STEP 2: For each neighbor-source variable, compute neighbor stats
#         using vectorized data.table joins + grouped aggregation,
#         then join results back onto cell_dt.
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key the main table for fast joins
setkey(cell_dt, id, year)

for (var in neighbor_source_vars) {

  cat("Computing neighbor stats for:", var, "\n")

  # --- 2a. Extract only the columns we need for the neighbor values ---
  # This is a slim table: (id, year, value) — 6.46M rows
  val_dt <- cell_dt[, .(neighbor_id = id, year, nbr_val = get(var))]
  setkey(val_dt, neighbor_id, year)

  # --- 2b. Join neighbor values onto the edge table × year ---
  # We need each edge to carry the year dimension.
  # Strategy: join edge_dt with val_dt on (neighbor_id, year).
  # First, create edge-year table by joining edges onto the set of
  # (cell_id, year) combinations that exist in the data.

  # Get the unique (cell_id, year) pairs
  cy <- cell_dt[, .(cell_id = id, year)]

  # Merge edges: for each (cell_id, year), get all neighbor_ids
  # This produces ~38.4M rows (6.46M × avg ~6? or 1.37M edges × 28 years)
  edge_year <- edge_dt[cy, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_year now has columns: cell_id, neighbor_id, year

  # Join the neighbor's value for that year
  edge_year[val_dt, nbr_val := i.nbr_val, on = .(neighbor_id, year)]

  # --- 2c. Grouped aggregation: max, min, mean per (cell_id, year) ---
  stats <- edge_year[
    !is.na(nbr_val),
    .(
      nbr_max  = max(nbr_val),
      nbr_min  = min(nbr_val),
      nbr_mean = mean(nbr_val)
    ),
    by = .(cell_id, year)
  ]

  # Rename columns to match the original pipeline's naming convention
  max_col  <- paste0(var, "_neighbor_max")
  min_col  <- paste0(var, "_neighbor_min")
  mean_col <- paste0(var, "_neighbor_mean")
  setnames(stats, c("nbr_max", "nbr_min", "nbr_mean"),
                  c(max_col,   min_col,   mean_col))

  # --- 2d. Join stats back onto the main table ---
  # Use cell_id = id for the join
  setnames(stats, "cell_id", "id")
  setkey(stats, id, year)

  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  cell_dt <- stats[cell_dt, on = .(id, year)]

  # Clean up intermediate objects to stay within 16 GB

  rm(val_dt, cy, edge_year, stats)
  gc()

  cat("  Done:", var, "\n")
}

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Predict with the existing trained Random Forest model
#         (model object is NOT retrained — preserved exactly as-is)
# ──────────────────────────────────────────────────────────────────────
# Ensure column order / names match what the RF model expects.
# The trained model object is assumed to be called `rf_model`.

# Extract the predictor names the model was trained on:
rf_predictors <- if (inherits(rf_model, "ranger")) {
  rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  rownames(rf_model$importance)
} else {
  stop("Unsupported RF model class. Adjust predictor extraction.")
}

# Verify all required predictors are present
missing_cols <- setdiff(rf_predictors, names(cell_dt))
if (length(missing_cols) > 0) {
  stop("Missing predictor columns: ", paste(missing_cols, collapse = ", "))
}

# Predict
pred_input <- cell_dt[, ..rf_predictors]

cell_dt[, prediction := predict(rf_model, data = pred_input)$predictions]
# For randomForest class, use: predict(rf_model, newdata = pred_input)

cat("Prediction complete. Rows:", nrow(cell_dt), "\n")

# ──────────────────────────────────────────────────────────────────────
# STEP 4 (optional): Convert back to data.frame if downstream code expects it
# ──────────────────────────────────────────────────────────────────────
# cell_data <- as.data.frame(cell_dt)
```

---

## Memory-Optimized Variant (if 16 GB is tight)

The `edge_year` table (~38.4M rows) is the peak memory object. If RAM is constrained, process **one year at a time**:

```r
# Memory-safe variant: process year-by-year within each variable
for (var in neighbor_source_vars) {

  max_col  <- paste0(var, "_neighbor_max")
  min_col  <- paste0(var, "_neighbor_min")
  mean_col <- paste0(var, "_neighbor_mean")

  # Pre-allocate columns with NA
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  years <- sort(unique(cell_dt$year))

  for (yr in years) {
    # Subset to this year
    yr_dt <- cell_dt[year == yr, .(id, val = get(var))]
    setkey(yr_dt, id)

    # Join neighbor values via edge table
    edges_with_vals <- edge_dt[yr_dt, on = .(cell_id = id), nomatch = 0L]
    # edges_with_vals: cell_id, neighbor_id, val (cell's own val — not needed)
    edges_with_vals[, val := NULL]

    # Now get the neighbor's value
    setnames(yr_dt, "id", "neighbor_id")
    edges_with_vals[yr_dt, nbr_val := i.val, on = "neighbor_id"]

    # Aggregate
    stats_yr <- edges_with_vals[
      !is.na(nbr_val),
      .(nbr_max = max(nbr_val), nbr_min = min(nbr_val), nbr_mean = mean(nbr_val)),
      by = cell_id
    ]

    # Write back into cell_dt
    idx <- cell_dt[year == yr, which = TRUE]
    m   <- match(cell_dt$id[idx], stats_yr$cell_id)

    set(cell_dt, i = idx, j = max_col,  value = stats_yr$nbr_max[m])
    set(cell_dt, i = idx, j = min_col,  value = stats_yr$nbr_min[m])
    set(cell_dt, i = idx, j = mean_col, value = stats_yr$nbr_mean[m])

    rm(yr_dt, edges_with_vals, stats_yr)
  }

  gc()
  cat("Done:", var, "\n")
}
```

This variant processes only ~1.37M edge rows per year-variable combination (peak ~50 MB), keeping total RAM well under 16 GB.

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Neighbor lookup | 6.46M-element R list built with `lapply` + character hashing | 1.37M-row `data.table` built once with vectorized indexing |
| Stat computation | `lapply` over 6.46M elements × 5 vars | `data.table` grouped `max`/`min`/`mean` (C-level) |
| Estimated runtime | ~86+ hours | ~5–15 minutes |
| Peak RAM | >16 GB (list overhead) | ~4–8 GB (edge-year join) or ~2 GB (year-by-year variant) |
| RF model | Preserved, not retrained | Preserved, not retrained |
| Numerical results | Identical `max`, `min`, `mean` | Identical — same aggregation, same edges |