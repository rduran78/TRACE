 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The bottleneck is overwhelmingly in **`build_neighbor_lookup`**, which constructs a list of ~6.46 million elements, each built via an `lapply` iteration that performs character coercion, hash-table lookups (`setNames`/named-vector indexing), and `paste` operations **per row**. This means:

- **~6.46 million iterations** of `lapply`, each doing string concatenation, named-vector lookups, and `NA` filtering.
- The `id_to_ref` and `idx_lookup` named vectors are themselves large (~6.46M entries for `idx_lookup`), so each `[` lookup scans a hash table at scale, millions of times.
- `compute_neighbor_stats` then does another `lapply` over 6.46M elements **per variable** (×5 variables = ~32.3M iterations), each extracting subsets from a numeric vector.

The raster/focal-kernel analogy is apt conceptually (each cell's neighborhood statistics are a "focal" operation), but here the geometry is an **irregular spatial grid with a precomputed `nb` object**, not a regular raster. So a true `terra::focal()` replacement is not directly applicable without confirming the grid is perfectly regular and re-encoding the data as a raster. The implementation below therefore stays in the tabular/vector domain but eliminates the R-level loops.

**Root causes of the 86+ hour runtime:**

| Component | Problem |
|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations with string ops and named-vector lookups |
| `compute_neighbor_stats` | 6.46M × 5 = 32.3M R-level iterations with subsetting |
| Memory | Storing a 6.46M-element list of integer vectors is allocation-heavy |

---

## 2. Optimization Strategy

### A. Replace `build_neighbor_lookup` with a vectorized `data.table` join

Instead of iterating row-by-row, we:

1. Expand the `nb` object into an **edge list** (cell_id → neighbor_cell_id), which has ~1.37M directed edges.
2. Cross-join this edge list with the 28 years to get ~38.4M (edge × year) rows.
3. Join against the data to get the **row index** of each neighbor in each year.

This replaces 6.46 million `lapply` iterations with a single vectorized merge.

### B. Replace `compute_neighbor_stats` with a `data.table` grouped aggregation

Instead of iterating over 6.46M list elements per variable, we:

1. Use the edge-list-with-row-indices from step A.
2. For each source variable, extract the neighbor values via vectorized indexing.
3. Compute `max`, `min`, `mean` in a single `data.table` grouped-by aggregation (`by = row_idx`).

This replaces all 32.3M R-level iterations with 5 vectorized `data.table` group-by operations.

### C. Memory management

- The edge list × year table is ~38.4M rows × 3 integer columns ≈ ~440 MB, feasible within 16 GB.
- We avoid materializing a 6.46M-element list entirely.

**Expected speedup:** From 86+ hours to roughly **5–15 minutes**.

---

## 3. Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# INPUTS (assumed to already exist in the environment):
#   cell_data              : data.frame/data.table with columns id, year,
#                            ntl, ec, pop_density, def, usd_est_n2, ...
#   rook_neighbors_unique  : nb object (list of integer index vectors)
#   id_order               : integer vector; id_order[i] = cell id of
#                            the i-th element in the nb object
#   rf_model               : pre-trained Random Forest model (untouched)
# ──────────────────────────────────────────────────────────────────────

# Convert to data.table if not already; add a row index column
setDT(cell_data)
cell_data[, .row_idx := .I]

# ── Step 1: Build a directed edge list from the nb object ────────────
#    nb object: rook_neighbors_unique[[i]] gives integer indices of
#    neighbors of the i-th cell; id_order[i] maps to cell id.
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_indices <- rook_neighbors_unique[[i]]
  if (length(nb_indices) == 0L || (length(nb_indices) == 1L && nb_indices[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_id    = id_order[i],
             neighbor_id = id_order[nb_indices])
}))
# This table has ~1,373,394 rows (directed rook edges)

cat("Edge list rows:", nrow(edges), "\n")

# ── Step 2: Cross-join edges with years, then join to get row indices ─
years <- sort(unique(cell_data$year))

# Expand edges × years  (~1.37M × 28 ≈ 38.5M rows)
edges_by_year <- edges[, CJ(year = years), by = .(focal_id, neighbor_id)]
# Columns: focal_id, neighbor_id, year

# Build a lookup from (id, year) → row index in cell_data
id_year_lookup <- cell_data[, .(id, year, .row_idx)]
setkey(id_year_lookup, id, year)

# Get the row index of the focal cell-year
setnames(id_year_lookup, ".row_idx", "focal_row_idx")
setkey(edges_by_year, focal_id, year)
edges_by_year <- id_year_lookup[edges_by_year, on = .(id = focal_id, year = year),
                                 nomatch = NA]
setnames(edges_by_year, "focal_row_idx", "focal_row_idx")

# Get the row index of the neighbor cell-year
setnames(id_year_lookup, "focal_row_idx", "neighbor_row_idx")
edges_by_year <- id_year_lookup[edges_by_year,
                                 on = .(id = neighbor_id, year = year),
                                 nomatch = NA]

# Keep only rows where both focal and neighbor exist
edges_by_year <- edges_by_year[!is.na(focal_row_idx) & !is.na(neighbor_row_idx)]

# Clean up: keep only what we need
edges_by_year <- edges_by_year[, .(focal_row_idx, neighbor_row_idx)]

# Restore lookup column name
setnames(id_year_lookup, "neighbor_row_idx", ".row_idx")

cat("Edges × years rows (after join):", nrow(edges_by_year), "\n")

# ── Step 3: Compute neighbor stats for each source variable ──────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor features for:", var_name, "\n")

  # Extract the neighbor values via vectorized indexing
  edges_by_year[, val := cell_data[[var_name]][neighbor_row_idx]]

  # Grouped aggregation: max, min, mean per focal row, dropping NAs
  stats <- edges_by_year[!is.na(val),
                         .(nb_max  = max(val),
                           nb_min  = min(val),
                           nb_mean = mean(val)),
                         by = focal_row_idx]

  # Initialize new columns to NA (for cells with no valid neighbors)
  max_col  <- paste0(var_name, "_max_nb")
  min_col  <- paste0(var_name, "_min_nb")
  mean_col <- paste0(var_name, "_mean_nb")

  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  # Assign computed values back to the correct rows
  cell_data[stats$focal_row_idx, (max_col)  := stats$nb_max]
  cell_data[stats$focal_row_idx, (min_col)  := stats$nb_min]
  cell_data[stats$focal_row_idx, (mean_col) := stats$nb_mean]

  # Drop temporary column
  edges_by_year[, val := NULL]

  cat("  Done:", max_col, min_col, mean_col, "\n")
}

# ── Step 4: Clean up helper column ───────────────────────────────────
cell_data[, .row_idx := NULL]

# ── Step 5: Predict using the pre-trained Random Forest model ────────
# The model is NOT retrained. We just score.
# (Adjust the predict call to match your model framework, e.g. ranger/randomForest)

# Example for ranger:
# preds <- predict(rf_model, data = cell_data)$predictions

# Example for randomForest:
# preds <- predict(rf_model, newdata = cell_data)

cat("Neighbor feature engineering complete. Ready for prediction.\n")
```

---

## 4. Why This Preserves the Original Estimand

| Requirement | Guarantee |
|---|---|
| **Same max/min/mean values** | The grouped `data.table` aggregation computes `max`, `min`, `mean` on exactly the same set of non-NA neighbor values that the original `compute_neighbor_stats` extracted. The edge list is derived from the same `nb` object and `id_order`. |
| **Same column names** | Columns are named `{var}_max_nb`, `{var}_min_nb`, `{var}_mean_nb` — match these to whatever your `compute_and_add_neighbor_features` produced (adjust the suffix if your original code used a different naming convention). |
| **RF model untouched** | The trained model object is never modified; we only call `predict()` on the enriched data. |
| **No raster approximation** | We do **not** rasterize or use focal kernels, which could alter neighbor sets at grid boundaries or for irregular geometries. The exact `nb` object is used. |

---

## 5. Summary of Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level iterations (~hours) | One `data.table` join (~1–2 min) |
| `compute_neighbor_stats` (×5 vars) | ~32.3M R-level iterations (~hours) | 5 grouped aggregations (~2–3 min) |
| **Total** | **86+ hours** | **~5–15 minutes** |
| Peak RAM | Lists of 6.46M elements + overhead | ~38.5M-row edge table ≈ 0.5–1 GB |