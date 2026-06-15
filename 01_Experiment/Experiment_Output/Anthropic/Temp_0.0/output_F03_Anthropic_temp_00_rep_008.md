 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys
- Named-vector lookup on `idx_lookup` (a named character vector of length 6.46M — each lookup is O(n) hash probe on a very large vector)
- NA filtering

With 6.46M iterations, each doing string concatenation and multiple named-vector lookups against a 6.46M-element vector, this is extremely slow — likely **hours** on its own.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, subsetting a numeric vector and computing `max/min/mean`. The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also slow (repeated memory allocation). This is called 5 times (once per neighbor source variable), so ~32.3M iterations total.

**`do.call(rbind, ...)` on millions of small vectors** is a classic R anti-pattern — it creates millions of tiny row matrices and binds them one by one.

### 1.2 Prediction Workflow Bottlenecks (Inferred)

- **Model loading:** If the serialized Random Forest is large (110 predictors, many trees), `readRDS` can take significant time and memory.
- **Single-call vs. loop prediction:** If `predict()` is called row-by-row or in small batches instead of once on the full data.frame, overhead is enormous.
- **Object copying:** R's copy-on-modify semantics mean that repeatedly doing `cell_data$new_col <- ...` inside a loop triggers full-dataframe copies each time. With ~6.46M rows × 110+ columns, each copy is hundreds of MB.
- **Memory pressure:** 6.46M rows × 110 columns × 8 bytes ≈ 5.7 GB for the numeric matrix alone. Combined with the RF model, neighbor lookup lists, and intermediate objects, 16 GB RAM is tight, causing garbage collection thrashing.

### 1.3 Root Cause Summary

| Component | Problem | Estimated Impact |
|---|---|---|
| `build_neighbor_lookup` | Per-row string ops + named-vector lookup on 6.46M keys | ~hours |
| `compute_neighbor_stats` | Per-row lapply + `do.call(rbind, ...)` × 5 vars | ~hours |
| Column assignment in loop | Copy-on-modify of full data.frame × 15 new columns | ~tens of minutes, GB of RAM churn |
| Prediction (likely) | Possible row-level or batch predict loop; large model load | ~hours if looped |
| Memory | 16 GB RAM saturated → GC thrashing | Multiplier on all above |

---

## 2. OPTIMIZATION STRATEGY

### Principle: Vectorize everything, eliminate per-row R loops, use `data.table` for zero-copy column addition, and call `predict()` once.

| Step | Action | Speedup Factor |
|---|---|---|
| **A.** Replace `build_neighbor_lookup` | Build a `data.table` join between (id, year) and neighbor-id, yielding a two-column integer matrix of (row_index, neighbor_row_index). No per-row loop. | ~100–500× |
| **B.** Replace `compute_neighbor_stats` | Group-by aggregation on the edge table using `data.table`: group by `row_idx`, compute max/min/mean of neighbor values. Fully vectorized. | ~50–200× |
| **C.** Use `data.table` for `cell_data` | Eliminates copy-on-modify. Column addition by reference (`:=`). | ~10× for column ops |
| **D.** Single `predict()` call | Ensure prediction is one call: `predict(model, newdata = cell_data)`. | Critical |
| **E.** Memory management | Convert to matrix for predict if ranger; `gc()` after large intermediates; remove lookup objects when done. | Prevents thrashing |

**Expected total runtime: ~5–20 minutes** (down from 86+ hours).

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Preserves: trained RF model (no retraining), original numerical estimand
# =============================================================================

library(data.table)

# ---- A. Optimized neighbor lookup: returns a data.table edge list -----------
#
# Instead of a list of length nrow(data), we build a two-column data.table:
#   from_row  : integer row index in cell_data
#   to_row    : integer row index of the neighbor in cell_data
#
# This is fully vectorized and avoids all per-row string operations.

build_neighbor_edges <- function(dt, id_order, neighbors) {
  # dt must be a data.table with columns 'id' and 'year'
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  # Step 1: Expand the nb object into a two-column data.table of
  #         (focal_cell_id, neighbor_cell_id)
  n_cells <- length(id_order)
  from_lengths <- vapply(neighbors, length, integer(1))  # fast C-level
  edge_from_idx <- rep(seq_len(n_cells), times = from_lengths)
  edge_to_idx   <- unlist(neighbors, use.names = FALSE)

  edges_cell <- data.table(
    focal_id    = id_order[edge_from_idx],
    neighbor_id = id_order[edge_to_idx]
  )
  rm(edge_from_idx, edge_to_idx, from_lengths)

  # Step 2: Map (id, year) -> row index in dt
  dt[, row_idx := .I]

  # Step 3: Cross-join edges with all years present for each focal cell.
  #         We need (focal_id, year) -> row_idx  AND  (neighbor_id, year) -> row_idx
  id_year_map <- dt[, .(id, year, row_idx)]
  setkey(id_year_map, id, year)

  # Join focal side: get focal row_idx for every (focal_id, year) combination
  # First, get all (focal_id, year) pairs by joining edges_cell with the years
  # present for each focal_id.
  focal_years <- dt[, .(year), by = .(focal_id = id)]
  setkey(focal_years, focal_id)
  setkey(edges_cell, focal_id)

  # Merge: for each edge, replicate across all years of the focal cell
  edge_year <- edges_cell[focal_years, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_year now has columns: focal_id, neighbor_id, year
  rm(edges_cell, focal_years)
  gc()

  # Join to get focal row index
  edge_year[id_year_map, on = .(focal_id = id, year = year), from_row := i.row_idx]


  # Join to get neighbor row index
  edge_year[id_year_map, on = .(neighbor_id = id, year = year), to_row := i.row_idx]

  # Drop edges where either side is missing
  edge_year <- edge_year[!is.na(from_row) & !is.na(to_row),
                         .(from_row, to_row)]

  rm(id_year_map)
  gc()

  return(edge_year)
}


# ---- B. Optimized neighbor stats: vectorized group-by aggregation -----------

compute_neighbor_stats_vec <- function(dt, edge_dt, var_name, nrow_dt) {
  # dt       : data.table with the variable column
  # edge_dt  : data.table with columns from_row, to_row
  # var_name : character, name of the variable
  # nrow_dt  : total number of rows in dt
  #
  # Returns a data.table with columns:
  #   <var_name>_neighbor_max, <var_name>_neighbor_min, <var_name>_neighbor_mean

  vals <- dt[[var_name]]

  # Attach neighbor values to edge table (by reference-safe copy of needed cols)
  work <- edge_dt[, .(from_row, to_row)]
  work[, nval := vals[to_row]]

  # Remove edges where neighbor value is NA

  work <- work[!is.na(nval)]

  # Aggregate by focal row
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = from_row]

  # Build full-length result (NA for rows with no valid neighbors)
  result <- data.table(
    nb_max  = rep(NA_real_, nrow_dt),
    nb_min  = rep(NA_real_, nrow_dt),
    nb_mean = rep(NA_real_, nrow_dt)
  )
  result[agg$from_row, `:=`(
    nb_max  = agg$nb_max,
    nb_min  = agg$nb_min,
    nb_mean = agg$nb_mean
  )]

  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  setnames(result, c("nb_max", "nb_min", "nb_mean"),
                   c(col_max,  col_min,  col_mean))

  return(result)
}


# ---- C. Full pipeline ------------------------------------------------------

run_optimized_pipeline <- function(cell_data_input,
                                   id_order,
                                   rook_neighbors_unique,
                                   rf_model_path,
                                   neighbor_source_vars = c("ntl", "ec",
                                                            "pop_density",
                                                            "def",
                                                            "usd_est_n2")) {

  # --- 0. Convert to data.table (no copy if already data.table) ---
  if (!is.data.table(cell_data_input)) {
    cell_data <- as.data.table(cell_data_input)
  } else {
    cell_data <- copy(cell_data_input)
  }

  cat("Rows:", nrow(cell_data), " Cols:", ncol(cell_data), "\n")
  nrow_cd <- nrow(cell_data)

  # --- 1. Build vectorized neighbor edge list ---
  cat("Building neighbor edge list...\n")
  t0 <- proc.time()
  edge_dt <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)
  cat("  Edge list:", nrow(edge_dt), "edges. Time:",
      round((proc.time() - t0)[3], 1), "s\n")

  # Remove temporary row_idx if added
  if ("row_idx" %in% names(cell_data)) {
    cell_data[, row_idx := NULL]
  }

  # --- 2. Compute neighbor features (vectorized) ---
  cat("Computing neighbor features...\n")
  t0 <- proc.time()
  for (var_name in neighbor_source_vars) {
    cat("  Variable:", var_name, "...")
    stats_dt <- compute_neighbor_stats_vec(cell_data, edge_dt, var_name, nrow_cd)

    # Add columns by reference — no copy of cell_data
    new_cols <- names(stats_dt)
    for (col in new_cols) {
      set(cell_data, j = col, value = stats_dt[[col]])
    }
    rm(stats_dt)
    cat(" done\n")
  }
  cat("  Neighbor features time:", round((proc.time() - t0)[3], 1), "s\n")

  # Free edge table
  rm(edge_dt)
  gc()

  # --- 3. Load trained Random Forest model ---
  cat("Loading RF model...\n")
  t0 <- proc.time()
  rf_model <- readRDS(rf_model_path)
  cat("  Model load time:", round((proc.time() - t0)[3], 1), "s\n")

  # --- 4. Predict — single vectorized call ---
  cat("Running prediction on", nrow_cd, "rows...\n")
  t0 <- proc.time()

  # Identify the predictor columns the model expects
  # Works for randomForest, ranger, and most RF implementations
  if (inherits(rf_model, "ranger")) {
    # ranger: predict expects a data.frame or data.table
    pred <- predict(rf_model, data = cell_data)$predictions
  } else if (inherits(rf_model, "randomForest")) {
    # randomForest: predict expects newdata as data.frame
    pred <- predict(rf_model, newdata = cell_data)
  } else {
    # Generic fallback
    pred <- predict(rf_model, newdata = cell_data)
  }

  cat("  Prediction time:", round((proc.time() - t0)[3], 1), "s\n")

  # --- 5. Attach predictions ---
  cell_data[, predicted_gdp := pred]

  # Clean up model from memory
  rm(rf_model, pred)
  gc()

  cat("Pipeline complete.\n")
  return(cell_data)
}


# ---- D. Example invocation -------------------------------------------------
#
# result <- run_optimized_pipeline(
#   cell_data_input      = cell_data,
#   id_order             = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   rf_model_path        = "path/to/trained_rf_model.rds",
#   neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
```

---

## 4. WHY THIS PRESERVES CORRECTNESS

| Concern | Guarantee |
|---|---|
| **Trained RF model** | Loaded via `readRDS`; never modified or retrained. `predict()` is called exactly once on the full dataset. |
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets as the original code. The edge-list join replicates the same (focal_id, year) ↔ (neighbor_id, year) mapping. NA handling is identical: edges with NA neighbor values are excluded before aggregation; rows with no valid neighbors get NA. |
| **Column names** | The naming convention `<var>_neighbor_max/min/mean` matches the pattern from `compute_and_add_neighbor_features`. Adjust the suffix pattern if your original wrapper uses different names. |

---

## 5. EXPECTED PERFORMANCE

| Stage | Original (est.) | Optimized (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~10–30 hours | ~1–3 minutes |
| `compute_neighbor_stats` × 5 | ~30–50 hours | ~2–5 minutes |
| Column assignment (×15 cols) | ~2–5 hours (copy-on-modify) | ~seconds (by-reference) |
| RF prediction | Unknown (depends on implementation) | ~2–10 minutes (single call) |
| **Total** | **86+ hours** | **~5–20 minutes** |

The key transformation is replacing **6.46 million R-level loop iterations** with **vectorized `data.table` joins and group-by aggregations** that execute in compiled C code internally.