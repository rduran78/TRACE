 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys
- Named-vector lookup on `idx_lookup` (a named character vector of length 6.46M — each lookup is O(n) hash probe on a very large vector)
- NA filtering and `as.integer` coercion

**Cost:** ~6.46M iterations × multiple string allocations and hash lookups = extremely slow. This alone could take hours.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, each extracting a variable-length subset of a numeric vector, removing NAs, and computing three summary statistics. The `do.call(rbind, result)` at the end binds 6.46M small vectors into a matrix — this is a known R anti-pattern that is very slow and memory-hungry.

**Outer loop:** This is called 5 times (once per neighbor source variable), so `compute_neighbor_stats` runs 5 × 6.46M = ~32.3M iterations total.

### 1.2 Prediction / Inference Bottlenecks

Although the code shown is feature preparation, the user states the main problem is the overall prediction pipeline. Likely issues:

- **Model loading:** If the Random Forest model is loaded from disk repeatedly (e.g., per chunk or per year), deserialization of a large RF object is expensive.
- **Prediction in a loop:** If `predict()` is called row-by-row or in small batches instead of in one vectorized call, overhead dominates.
- **Object copying:** R's copy-on-modify semantics mean that repeatedly adding columns to a large data.frame (`cell_data$new_col <- ...`) triggers full copies of the data.frame each time.
- **Memory pressure:** With 6.46M rows × 110 columns of doubles ≈ 5.4 GB, plus the RF model, plus intermediate objects, 16 GB RAM is tight. Swapping to disk would catastrophically slow everything.

### 1.3 Root-Cause Summary

| Component | Problem | Severity |
|---|---|---|
| `build_neighbor_lookup` | Per-row string ops + hash lookups on 6.46M-length named vector | **Critical** |
| `compute_neighbor_stats` | Per-row `lapply` + `do.call(rbind, ...)` on 6.46M lists | **Critical** |
| Column addition in loop | Copy-on-modify of large data.frame 15+ times (5 vars × 3 stats) | **High** |
| `predict()` call pattern | Possibly called in loop or on subsets instead of once | **High** |
| Model loading | Possibly redundant deserialization | **Medium** |
| Memory | ~5–10 GB working set on 16 GB machine | **Medium** |

---

## 2. OPTIMIZATION STRATEGY

### Strategy A: Vectorize neighbor lookup using `data.table` joins

Replace the per-row `lapply` in `build_neighbor_lookup` with a single merge/join operation. Pre-expand the neighbor list into an edge-list data.table `(id, year, neighbor_id)`, then join to get row indices. This turns 6.46M R-level iterations into a single vectorized join.

### Strategy B: Vectorize neighbor stats using `data.table` grouped aggregation

Instead of iterating per row, join the neighbor edge-list to the data, then compute `max`, `min`, `mean` grouped by the focal row index — all in one `data.table` operation per variable.

### Strategy C: Use `data.table` throughout to avoid copy-on-modify

Use `:=` (assignment by reference) to add new columns without copying the entire table.

### Strategy D: Single vectorized `predict()` call

Load the model once, build the full feature matrix, call `predict()` once on all 6.46M rows.

### Strategy E: Memory management

- Use `data.table` (column-store, no row names, no copy-on-modify).
- Remove intermediate objects and call `gc()` at key points.
- Optionally predict in large chunks (e.g., 1M rows) if the model's `predict` method builds a dense matrix internally.

**Expected speedup:** From 86+ hours to roughly 10–30 minutes for feature preparation, plus prediction time (depends on RF size).

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Requirements: data.table, ranger (or randomForest — adapt predict call)
# Preserves: trained RF model (no retraining), original numerical estimand
# =============================================================================

library(data.table)

# ---- 0. Load pre-trained model ONCE ----------------------------------------
# Adjust path and loading method to your serialization format.
# If using ranger:
#   rf_model <- readRDS("path/to/trained_rf_model.rds")
# If using randomForest:
#   rf_model <- readRDS("path/to/trained_rf_model.rds")
# Do NOT reload inside any loop.

rf_model <- readRDS("path/to/trained_rf_model.rds")


# ---- 1. Convert cell_data to data.table (by reference if possible) ----------
# Assumes cell_data is a data.frame with columns: id, year, ntl, ec,
# pop_density, def, usd_est_n2, and ~110 predictor columns.

if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in place — no copy
}

# Create a row-index column for the focal cell
cell_data[, .row_idx := .I]


# ---- 2. Build vectorized neighbor edge list ---------------------------------
# rook_neighbors_unique: an nb object (list of integer vectors).
#   rook_neighbors_unique[[k]] gives the neighbor indices (into id_order) of
#   the k-th element of id_order.
# id_order: vector of cell IDs in the order matching the nb object.

build_neighbor_edgelist <- function(id_order, neighbors) {
  # Expand the nb list into a two-column data.table of (focal_id, neighbor_id)
  n <- length(neighbors)
  # Pre-allocate: count total edges
  lens <- lengths(neighbors)
  total_edges <- sum(lens)

  focal_idx    <- rep.int(seq_len(n), lens)
  neighbor_idx <- unlist(neighbors, use.names = FALSE)

  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

cat("Building neighbor edge list...\n")
edge_dt <- build_neighbor_edgelist(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %s rows\n", format(nrow(edge_dt), big.mark = ",")))


# ---- 3. Expand edges across years and join to row indices -------------------
# We need (focal_id, year, neighbor_id) -> neighbor's row index in cell_data.

# Unique years
years_vec <- sort(unique(cell_data$year))

# Cross-join edges with years (all edges exist in every year)
# This creates ~1.37M edges × 28 years ≈ 38.5M rows — fits in memory as
# a 3-column integer table (~900 MB).
cat("Expanding edges across years...\n")
edge_year_dt <- edge_dt[, .(year = years_vec), by = .(focal_id, neighbor_id)]

# Build a lookup from (id, year) -> row index in cell_data
id_year_idx <- cell_data[, .(id, year, .row_idx)]
setkey(id_year_idx, id, year)

# Join to get focal row index
setnames(id_year_idx, ".row_idx", "focal_row")
edge_year_dt <- id_year_idx[edge_year_dt, on = .(id = focal_id, year = year), nomatch = 0L]
setnames(id_year_idx, "focal_row", ".row_idx")

# Join to get neighbor row index
setnames(id_year_idx, c("id", "year", "neighbor_row"))
edge_year_dt <- id_year_idx[edge_year_dt, on = .(id = neighbor_id, year = year), nomatch = 0L]
setnames(id_year_idx, c("id", "year", ".row_idx"))  # restore names

# Now edge_year_dt has columns: focal_row, neighbor_id, id (=focal_id),
# year, neighbor_row.  We only need focal_row and neighbor_row.
edge_year_dt <- edge_year_dt[, .(focal_row, neighbor_row)]

# Clean up
rm(id_year_idx, edge_dt)
gc()

cat(sprintf("  Expanded edge table: %s rows\n",
            format(nrow(edge_year_dt), big.mark = ",")))


# ---- 4. Compute neighbor features (vectorized) -----------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))

  # Extract the neighbor values via the edge table
  # edge_year_dt$neighbor_row indexes into cell_data
  edge_year_dt[, val := cell_data[[var_name]][neighbor_row]]

  # Compute grouped stats: max, min, mean per focal_row, excluding NAs
  stats_dt <- edge_year_dt[!is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = focal_row
  ]

  # Assign to cell_data by reference (no copy)
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)

  # Initialize with NA
  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)

  # Fill in computed values
  set(cell_data, i = stats_dt$focal_row, j = max_col,  value = stats_dt$nb_max)
  set(cell_data, i = stats_dt$focal_row, j = min_col,  value = stats_dt$nb_min)
  set(cell_data, i = stats_dt$focal_row, j = mean_col, value = stats_dt$nb_mean)

  # Drop temporary column from edge table
  edge_year_dt[, val := NULL]

  rm(stats_dt)
}

# Clean up edge table — no longer needed
rm(edge_year_dt)
gc()

cat("Neighbor features complete.\n")


# ---- 5. Prepare prediction matrix ------------------------------------------
# Identify the predictor columns expected by the model.
# For ranger: rf_model$forest$independent.variable.names
# For randomForest: colnames(rf_model$forest$xlevels) or stored separately

# Adapt this to your model type:
if (inherits(rf_model, "ranger")) {
  pred_vars <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores the variable names used in training
  pred_vars <- rownames(importance(rf_model))
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all required columns exist
missing_vars <- setdiff(pred_vars, names(cell_data))
if (length(missing_vars) > 0) {
  stop("Missing predictor columns: ", paste(missing_vars, collapse = ", "))
}

cat(sprintf("Prediction matrix: %s rows x %s predictors\n",
            format(nrow(cell_data), big.mark = ","), length(pred_vars)))


# ---- 6. Predict (single vectorized call, or chunked for memory) ------------
cat("Running Random Forest prediction...\n")

# Option A: Single call (fastest if memory allows)
# For ranger:
#   preds <- predict(rf_model, data = cell_data[, ..pred_vars])$predictions
# For randomForest:
#   preds <- predict(rf_model, newdata = cell_data[, ..pred_vars])

# Option B: Chunked prediction (safer for 16 GB RAM)
chunk_size <- 500000L  # 500K rows per chunk — adjust based on available RAM
n_rows     <- nrow(cell_data)
n_chunks   <- ceiling(n_rows / chunk_size)

preds <- numeric(n_rows)

for (ch in seq_len(n_chunks)) {
  i_start <- (ch - 1L) * chunk_size + 1L
  i_end   <- min(ch * chunk_size, n_rows)
  idx     <- i_start:i_end

  chunk_data <- cell_data[idx, ..pred_vars]

  if (inherits(rf_model, "ranger")) {
    preds[idx] <- predict(rf_model, data = chunk_data)$predictions
  } else {
    preds[idx] <- predict(rf_model, newdata = chunk_data)
  }

  if (ch %% 5 == 0 || ch == n_chunks) {
    cat(sprintf("  Chunk %d/%d complete (rows %s-%s)\n",
                ch, n_chunks,
                format(i_start, big.mark = ","),
                format(i_end, big.mark = ",")))
  }

  rm(chunk_data)
}

# Assign predictions back to cell_data (by reference)
cell_data[, predicted_gdp := preds]
rm(preds)
gc()

cat("Prediction complete.\n")


# ---- 7. Clean up helper column --------------------------------------------
cell_data[, .row_idx := NULL]


# ---- 8. Summary ------------------------------------------------------------
cat(sprintf(
  "\nPipeline finished.\n  Rows:        %s\n  Predictors:  %d\n  Predictions: %s non-NA\n",
  format(nrow(cell_data), big.mark = ","),
  length(pred_vars),
  format(sum(!is.na(cell_data$predicted_gdp)), big.mark = ",")
))
```

---

## 4. EXPECTED PERFORMANCE COMPARISON

| Stage | Original (estimated) | Optimized (estimated) |
|---|---|---|
| `build_neighbor_lookup` | ~20–40 hours (6.46M `lapply` with string ops) | ~1–3 minutes (vectorized edge-list expansion + `data.table` keyed join) |
| `compute_neighbor_stats` (×5 vars) | ~40–50 hours (32.3M `lapply` + `do.call(rbind)`) | ~3–8 minutes (5 × grouped `data.table` aggregation) |
| Column assignment (×15 cols) | ~5–10 hours (copy-on-modify of 6.46M-row data.frame) | ~seconds (`:=` / `set()` by reference) |
| RF prediction (6.46M rows) | Unknown (possibly looped) | ~5–20 minutes (chunked vectorized `predict()`) |
| **Total** | **86+ hours** | **~10–30 minutes** |

---

## 5. KEY PRINCIPLES PRESERVED

1. **Trained RF model is never retrained** — only `predict()` is called.
2. **Original numerical estimand is preserved** — the same `max`, `min`, `mean` neighbor statistics are computed from the same neighbor structure; the same predictor columns are fed to the same model; predictions are numerically identical.
3. **No approximations or sampling** — every cell-year row is processed.
4. **Memory-safe for 16 GB** — chunked prediction, `data.table` by-reference operations, and explicit `gc()` calls keep the working set manageable.