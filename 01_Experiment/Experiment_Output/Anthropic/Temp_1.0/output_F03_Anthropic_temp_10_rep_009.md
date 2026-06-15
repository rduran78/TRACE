 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### Primary Bottlenecks

**A. `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows with per-row string operations**

- Each of the ~6.46 million rows performs: a character lookup in `id_to_ref`, subsetting `neighbors`, pasting neighbor IDs with the year, and named-vector lookups in `idx_lookup`.
- `paste()` and named-vector character matching inside a per-row `lapply` is extremely slow at this scale. Named character vector lookup is O(n) amortized in base R (hash-based, but constant factor is large when called millions of times with allocation overhead).
- The output is a **list of 6.46M integer vectors** — enormous memory footprint and allocation pressure.

**B. `compute_neighbor_stats` — repeated per-row subsetting of a numeric vector**

- Called 5 times (once per variable). Each call iterates over 6.46M list elements, subsets `vals[idx]`, removes NAs, and computes max/min/mean.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is a classic R anti-pattern — it creates millions of tiny row-objects and binds them one by one.

**C. Memory pressure and object copying**

- `cell_data` is modified in a loop (`cell_data <- compute_and_add_neighbor_features(...)`) — each assignment likely triggers a full copy of the ~6.46M × 110+ column data.frame (COW semantics in R, but modification forces a copy).
- Storing the neighbor lookup as a 6.46M-element list of integer vectors consumes substantial RAM.

**D. Random Forest inference**

- Predicting 6.46M rows × 110 features with `predict.randomForest` or `predict.ranger` is I/O and compute intensive. If using the `randomForest` package, prediction is done in R-level loops and is dramatically slower than `ranger`.
- If the model is loaded from disk on every run, deserialization of a large RF object adds time.
- If prediction is done row-by-row or in small batches rather than as a single vectorized call, overhead is massive.

### Estimated time breakdown (86+ hours)

| Stage | Estimated share |
|---|---|
| `build_neighbor_lookup` | ~25–35% |
| `compute_neighbor_stats` (×5) | ~25–35% |
| Data.frame copying in loop | ~10–15% |
| RF prediction | ~15–25% |

---

## 2. OPTIMIZATION STRATEGY

### Strategy A: Vectorize neighbor lookup via `data.table` join (eliminate per-row `lapply`)

Replace the row-by-row `paste`/lookup approach with a single **equi-join**. Pre-expand the neighbor relationships into an edge table `(id, year) → (neighbor_id, year)`, then join against the data to get row indices. Group by source row to collect neighbor indices.

### Strategy B: Vectorized neighbor stats via `data.table` grouped aggregation

Instead of building a list of neighbor indices and then looping, join the edge table directly to the variable columns and compute `max`, `min`, `mean` in a single grouped `data.table` operation — per variable, zero R-level loops.

### Strategy C: Use `data.table` set-by-reference to avoid copies

Use `:=` to add columns in place — no full-table copy per iteration.

### Strategy D: Ensure single-call, vectorized RF prediction with `ranger` or `predict.randomForest`

If the model is a `ranger` object, call `predict()` once on the full matrix. If `randomForest`, consider converting. Load the model once and cache it.

---

## 3. WORKING R CODE

```r
library(data.table)

# ===========================================================================
# STEP 0: Convert cell_data to data.table (by reference, no copy)
# ===========================================================================
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns exist and are of the right type
stopifnot(all(c("id", "year") %in% names(cell_data)))

# Create a row index column (used later to assign results back)
cell_data[, .row_idx := .I]

# ===========================================================================
# STEP 1: Build vectorized edge table from spdep nb object
#
#   rook_neighbors_unique is a list of length = number of unique cell IDs.
#   id_order[i] is the cell ID of the i-th element.
#   rook_neighbors_unique[[i]] gives integer indices (into id_order) of
#   neighbors of cell id_order[i].
#
#   We expand this into a two-column data.table: (id, neighbor_id)
# ===========================================================================
message("Building edge table from nb object...")

edge_list <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    # spdep nb encodes "no neighbors" as a single 0L
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[nb])
  }),
  use.names = TRUE
)

# Expected: ~1.37M rows (directed edges)
message(sprintf("Edge table: %s rows", format(nrow(edge_list), big.mark = ",")))

# ===========================================================================
# STEP 2: Compute neighbor features for all variables — fully vectorized
#
#   For each source variable, we:
#     1. Join edge_list × year combinations to cell_data to get neighbor values.
#     2. Aggregate (max, min, mean) grouped by (id, year).
#     3. Assign back to cell_data by reference.
#
#   This replaces build_neighbor_lookup + compute_neighbor_stats entirely.
# ===========================================================================

# Build a mapping from (id, year) → row index for fast joins
setkey(cell_data, id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We only need to expand edges × years once.
# Unique years in the data
years <- unique(cell_data$year)

message("Expanding edge × year combinations...")

# Cross-join edges with years: for every edge (id, neighbor_id), the neighbor
# value is looked up in the SAME year. So we need (id, year, neighbor_id).
# But not every (id, year) pair or (neighbor_id, year) pair necessarily exists
# in cell_data. The join handles this naturally (non-matches become NA).

# Create the expanded edge-year table:
#   Rather than a full cross-join (which would be 1.37M × 28 = 38.4M rows),
#   we join edges onto the actual (id, year) pairs present in cell_data.
#   This is more memory-efficient and only keeps rows that exist.

# Slim lookup: which (id, year) pairs exist?
id_year_keys <- cell_data[, .(id, year, .row_idx)]

# Join: for each (id, year) row, attach all neighbor_ids
# This is an equi-join on `id`
setkey(edge_list, id)
setkey(id_year_keys, id)

message("Joining edges to (id, year) pairs...")
# For each row in id_year_keys, find all matching edges
edge_year <- edge_list[id_year_keys, on = "id", allow.cartesian = TRUE, nomatch = NULL]
# edge_year now has columns: id, neighbor_id, year, .row_idx
# .row_idx is the source row in cell_data

# Now join to get the neighbor's data values.
# We need to look up (neighbor_id, year) in cell_data.
# Prepare a slim lookup table for neighbor values.
# We'll do this per variable to limit peak memory.

message(sprintf("Computing neighbor stats for %d variables...", length(neighbor_source_vars)))

# Key edge_year for the neighbor join
setnames(edge_year, "neighbor_id", "nb_id")

for (var_name in neighbor_source_vars) {
  message(sprintf("  Processing variable: %s", var_name))
  
  # Build a slim lookup: (id, year, value)
  nb_vals <- cell_data[, .(nb_id = id, year, nb_val = get(var_name))]
  setkey(nb_vals, nb_id, year)
  
  # Join neighbor values onto edge_year
  setkey(edge_year, nb_id, year)
  edge_with_val <- nb_vals[edge_year, on = c("nb_id", "year"), nomatch = NA]
  # Columns: nb_id, year, nb_val, id, .row_idx
  
  # Aggregate by source row
  agg <- edge_with_val[
    !is.na(nb_val),
    .(
      nb_max  = max(nb_val),
      nb_min  = min(nb_val),
      nb_mean = mean(nb_val)
    ),
    by = .row_idx
  ]
  
  # Prepare NA-filled columns, then fill in computed values
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  # Set columns by reference — no copy of the whole table
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
  
  cell_data[agg$.row_idx, (max_col)  := agg$nb_max]
  cell_data[agg$.row_idx, (min_col)  := agg$nb_min]
  cell_data[agg$.row_idx, (mean_col) := agg$nb_mean]
  
  # Free intermediate objects

rm(nb_vals, edge_with_val, agg)
}

# Clean up helper column
cell_data[, .row_idx := NULL]

message("Neighbor features complete.")

# ===========================================================================
# STEP 3: Random Forest Prediction — single vectorized call
#
#   Key optimisations:
#     1. Load the model ONCE from disk (if not already in memory).
#     2. Build the prediction matrix ONCE.
#     3. Call predict() a single time on the full dataset.
#     4. If the model is a `randomForest` object, consider converting to
#        ranger for ~5-10× faster prediction (not always possible).
#     5. Use num.threads for ranger.
# ===========================================================================

message("Preparing prediction matrix...")

# Identify the predictor columns expected by the model.
# This works for both randomForest and ranger objects.
if (inherits(trained_model, "ranger")) {
  predictor_names <- trained_model$forest$independent.variable.names
} else if (inherits(trained_model, "randomForest")) {
  # randomForest stores the variable names used in training
  predictor_names <- rownames(trained_model$importance)
} else {
  stop("Unsupported model class: ", class(trained_model)[1])
}

# Subset to predictors, ensuring correct column order
pred_data <- cell_data[, ..predictor_names]

message(sprintf("Predicting %s rows × %d features...",
                format(nrow(pred_data), big.mark = ","), ncol(pred_data)))

# --- Prediction ---
if (inherits(trained_model, "ranger")) {
  # ranger: fast C++ prediction, use all available threads
  predictions <- predict(
    trained_model,
    data = pred_data,
    num.threads = parallel::detectCores(logical = FALSE)
  )$predictions
  
} else if (inherits(trained_model, "randomForest")) {
  # randomForest predict is slower but still vectorized.
  # Convert to matrix for faster internal handling.
  pred_matrix <- as.matrix(pred_data)
  predictions <- predict(trained_model, newdata = pred_matrix)
}

# Attach predictions back to cell_data by reference
cell_data[, predicted_gdp := predictions]

message("Prediction complete.")

# ===========================================================================
# STEP 4 (Optional): Memory cleanup
# ===========================================================================
rm(edge_list, edge_year, pred_data)
if (exists("pred_matrix")) rm(pred_matrix)
gc()

message("Pipeline complete.")
```

---

## 4. EXPECTED PERFORMANCE GAINS

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| **Neighbor lookup** | ~6.46M `lapply` iterations with `paste`/character match → hours | Single `data.table` equi-join (C-level hash join) | **~100–500×** |
| **Neighbor stats** (×5 vars) | ~6.46M `lapply` per var + `do.call(rbind, ...)` | Grouped `data.table` aggregation (GForce-optimized `max`/`min`/`mean`) | **~50–200×** |
| **Data.frame copying** (5 loop iters) | Full copy of 6.46M × 110 columns per iteration | `:=` set by reference — zero copies | **5× less memory, ~10× faster** |
| **RF prediction** | Depends on current approach | Single vectorized `predict()` call; `ranger` multi-threaded | **1–10×** (already vectorized → 1×; if batched before → 10×) |

### Estimated total runtime

| Stage | Estimated time |
|---|---|
| Edge table construction | ~5–15 seconds |
| Edge × year join | ~30–60 seconds |
| Neighbor stats (5 vars) | ~2–5 minutes |
| RF prediction (ranger, multi-threaded) | ~5–20 minutes |
| **Total** | **~10–30 minutes** |

This represents a reduction from **86+ hours to under 30 minutes** — approximately a **170–500× overall speedup** — while preserving the trained Random Forest model and producing numerically identical predictions.