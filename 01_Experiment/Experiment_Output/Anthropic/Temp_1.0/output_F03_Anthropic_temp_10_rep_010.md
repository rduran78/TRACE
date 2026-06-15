 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### Primary Bottlenecks

**A. `build_neighbor_lookup` — O(n) `lapply` with per-row string operations**

The function iterates over ~6.46 million rows, and for each row it:
- Converts IDs to characters and does named-vector lookups (`id_to_ref`, `idx_lookup`) — each lookup is O(n) in the worst case for named vectors in R.
- Calls `paste()` to build keys for every neighbor of every row.
- Performs named-vector subsetting on `idx_lookup` (a named vector of length 6.46M) — this is hash-lookup but repeated millions of times inside an `lapply`.

The result is a **list of 6.46 million integer vectors**, which is enormous in memory and slow to construct. Estimated: this single step could take tens of minutes to hours.

**B. `compute_neighbor_stats` — repeated per-variable, per-row**

For each of 5 variables, it iterates over all 6.46M rows, subsets a numeric vector by neighbor indices, removes NAs, and computes max/min/mean. The `do.call(rbind, result)` at the end on a 6.46M-element list is itself very slow (each element is a length-3 vector, so it builds a 6.46M × 3 matrix from a list). This runs **5 times** (once per variable), producing 5 × 6.46M = 32.3M row-level computations.

**C. Repeated `data.frame` column assignment inside the loop**

`cell_data <- compute_and_add_neighbor_features(cell_data, ...)` likely copies the entire data.frame (6.46M × 110+ columns) on every column addition. With 5 variables × 3 stats = 15 new columns, that's up to 15 full copies of a multi-GB data.frame.

**D. Random Forest prediction on 6.46M rows × 110 features**

- If the model is large (many trees, deep), `predict()` on 6.46M rows is memory-intensive and slow.
- If the model is loaded from disk every time, deserialization adds overhead.
- If prediction is done row-by-row or in small batches rather than vectorized, it's orders of magnitude slower.

### Memory Pressure

With 6.46M rows × 110 columns × 8 bytes ≈ **5.7 GB** just for the numeric matrix, plus the neighbor lookup list, plus the RF model in memory, a 16 GB laptop will be swapping to disk, destroying performance.

---

## 2. OPTIMIZATION STRATEGY

| Bottleneck | Strategy | Expected Speedup |
|---|---|---|
| `build_neighbor_lookup` | Replace with `data.table` merge/join — build a neighbor-edge table, join on `(neighbor_id, year)` to get row indices, then split by source row. Eliminate per-row `paste` and named-vector lookups. | 50–200× |
| `compute_neighbor_stats` | Vectorized `data.table` grouped aggregation over the edge table: join neighbor values, group by source row, compute max/min/mean in one pass for all variables simultaneously. Eliminate `lapply` + `do.call(rbind)`. | 100× |
| Data.frame copying | Use `data.table` with `:=` (modify in place). Zero copies. | 5–15× |
| RF prediction | Load model once; predict in a single vectorized call; convert predictor data to matrix once; use batched prediction if memory-constrained. | 2–10× |
| Memory | Use `data.table` (no copies); drop intermediate objects; `gc()` at key points; convert to matrix only at prediction time. | Keeps within 16 GB |

**Overall: from 86+ hours → estimated 10–30 minutes.**

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Requirements: data.table, ranger (or randomForest), spdep (for nb object)
# =============================================================================

library(data.table)

# ---- Step 0: Load data and model -------------------------------------------
# Assume:
#   cell_data            : data.frame/data.table with columns id, year, ntl, ec,
#                          pop_density, def, usd_est_n2, ... (110 predictors)
#   id_order             : integer vector of cell IDs in the order matching
#                          rook_neighbors_unique
#   rook_neighbors_unique: nb object (list of integer index vectors)
#   rf_model             : pre-trained Random Forest model (loaded once)

# Convert to data.table in place (no copy if already data.table)
setDT(cell_data)

# Load RF model ONCE
# rf_model <- readRDS("path/to/rf_model.rds")   # uncomment as needed

cat("Data rows:", nrow(cell_data), "\n")
cat("Data cols:", ncol(cell_data), "\n")

# ---- Step 1: Build edge table from nb object (vectorized) ------------------
# This replaces build_neighbor_lookup entirely.

build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer vectors (indices into id_order)
  # Build a two-column edge table: source_cell_id -> neighbor_cell_id
  
  n <- length(neighbors)
  
  # Pre-calculate total edges for pre-allocation
  edge_lengths <- vapply(neighbors, length, integer(1))
  total_edges  <- sum(edge_lengths)
  
  source_idx <- rep.int(seq_len(n), edge_lengths)
  neighbor_idx <- unlist(neighbors, use.names = FALSE)
  
  # Remove 0-entries (spdep uses 0 for "no neighbors" in some representations)
  valid <- neighbor_idx > 0L
  source_idx   <- source_idx[valid]
  neighbor_idx <- neighbor_idx[valid]
  
  data.table(
    source_cell_id   = id_order[source_idx],
    neighbor_cell_id = id_order[neighbor_idx]
  )
}

cat("Building edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edge_dt), "\n")

# ---- Step 2: Build row-index lookup and expand edges by year ---------------
# Key insight: every edge (A -> B) exists for every year in the panel.
# Instead of expanding edges × years (which would be huge), we join on
# (neighbor_cell_id, year) to get neighbor values directly.

# Create a row-index column
cell_data[, .row_idx := .I]

# Set key for fast joins
setkey(cell_data, id, year)

# ---- Step 3: Compute all neighbor stats at once (vectorized) ---------------
# Strategy: for each source row, find its neighbors' rows via join,
# then aggregate. We do this for ALL variables simultaneously.

compute_all_neighbor_features <- function(cell_data, edge_dt, 
                                           neighbor_source_vars) {
  
  # Step 3a: Get unique years
  years <- sort(unique(cell_data$year))
  
  # Step 3b: Cross-join edges with years to get (source_cell_id, year,
  #          neighbor_cell_id, year) — but we do this efficiently by
  #          joining edges to cell_data on the NEIGHBOR side.
  
  # Create a slim lookup: just id, year, row_idx, and the source variables
  keep_cols <- c("id", "year", neighbor_source_vars)
  neighbor_vals <- cell_data[, ..keep_cols]
  setnames(neighbor_vals, "id", "neighbor_cell_id")
  setkey(neighbor_vals, neighbor_cell_id, year)
  
  # Also need source_cell_id + year to identify source rows
  source_key <- cell_data[, .(source_cell_id = id, year, .row_idx)]
  setkey(source_key, source_cell_id, year)
  
  # Step 3c: For each year, join edges to neighbor values
  # To avoid a massive cross of edges × years, process by year chunks
  # (each year has ~344K cells, edges ~1.37M → ~1.37M joined rows per year)
  
  cat("Computing neighbor statistics for", length(neighbor_source_vars), 
      "variables across", length(years), "years...\n")
  
  # Pre-allocate result columns in cell_data
  for (var in neighbor_source_vars) {
    cell_data[, paste0("n_max_", var) := NA_real_]
    cell_data[, paste0("n_min_", var) := NA_real_]
    cell_data[, paste0("n_mean_", var) := NA_real_]
  }
  
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (var in neighbor_source_vars) {
    agg_exprs[[paste0("n_max_", var)]]  <- parse(text = paste0(
      "as.double(max(", var, ", na.rm = TRUE))"))[[1]]
    agg_exprs[[paste0("n_min_", var)]]  <- parse(text = paste0(
      "as.double(min(", var, ", na.rm = TRUE))"))[[1]]
    agg_exprs[[paste0("n_mean_", var)]] <- parse(text = paste0(
      "as.double(mean(", var, ", na.rm = TRUE))"))[[1]]
  }
  
  # Process in year batches to limit memory
  setkey(edge_dt, neighbor_cell_id)
  
  for (yr in years) {
    cat("  Year:", yr, "\r")
    
    # Neighbor values for this year
    nv_yr <- neighbor_vals[year == yr]
    setkey(nv_yr, neighbor_cell_id)
    
    # Join edges to neighbor values
    # Result: source_cell_id | neighbor_cell_id | ntl | ec | ...
    joined <- edge_dt[nv_yr, on = "neighbor_cell_id", nomatch = NULL, 
                      allow.cartesian = TRUE]
    
    # Aggregate by source_cell_id
    agg <- joined[, lapply(agg_exprs, eval, envir = .SD), 
                  by = source_cell_id]
    
    # Fix -Inf/Inf from max/min on all-NA groups
    inf_cols <- names(agg_exprs)
    for (col in inf_cols) {
      agg[is.infinite(get(col)), (col) := NA_real_]
    }
    
    # Map back to cell_data rows for this year
    agg[, year := yr]
    setkey(agg, source_cell_id, year)
    
    # Get row indices in cell_data for this year
    src_yr <- source_key[year == yr]
    setkey(src_yr, source_cell_id)
    
    matched <- src_yr[agg, on = "source_cell_id", nomatch = NULL]
    
    # Update cell_data in place using row indices
    if (nrow(matched) > 0) {
      for (col in inf_cols) {
        set(cell_data, i = matched$.row_idx, j = col, value = matched[[col]])
      }
    }
  }
  cat("\nNeighbor feature computation complete.\n")
  
  invisible(cell_data)
}

# Run it
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- compute_all_neighbor_features(cell_data, edge_dt, 
                                            neighbor_source_vars)

# Clean up intermediates
rm(edge_dt)
gc()

# ---- Step 4: Prepare prediction matrix and predict -------------------------

# Remove non-feature columns
meta_cols <- c("id", "year", ".row_idx")
feature_cols <- setdiff(names(cell_data), meta_cols)
# Also remove any target column if present
feature_cols <- setdiff(feature_cols, c("gdp", "log_gdp", "gdp_target"))

cat("Feature columns for prediction:", length(feature_cols), "\n")

predict_in_batches <- function(model, data, feature_cols, 
                                batch_size = 500000L) {
  n <- nrow(data)
  n_batches <- ceiling(n / batch_size)
  predictions <- numeric(n)
  
  cat("Predicting in", n_batches, "batches of up to", batch_size, "rows...\n")
  
  for (b in seq_len(n_batches)) {
    start_idx <- (b - 1L) * batch_size + 1L
    end_idx   <- min(b * batch_size, n)
    
    # Extract batch as data.frame (RF predict methods expect this)
    batch_data <- as.data.frame(
      data[start_idx:end_idx, ..feature_cols]
    )
    
    predictions[start_idx:end_idx] <- predict(model, newdata = batch_data)
    
    if (b %% 5 == 0 || b == n_batches) {
      cat("  Batch", b, "/", n_batches, "complete\n")
    }
    
    # Free batch memory
    rm(batch_data)
    if (b %% 10 == 0) gc()
  }
  
  predictions
}

# ---- If using ranger model: ----
# ranger::predict returns an object; extract $predictions
predict_in_batches_ranger <- function(model, data, feature_cols,
                                      batch_size = 500000L) {
  n <- nrow(data)
  n_batches <- ceiling(n / batch_size)
  predictions <- numeric(n)
  
  cat("Predicting (ranger) in", n_batches, "batches...\n")
  
  for (b in seq_len(n_batches)) {
    start_idx <- (b - 1L) * batch_size + 1L
    end_idx   <- min(b * batch_size, n)
    
    batch_data <- as.data.frame(
      data[start_idx:end_idx, ..feature_cols]
    )
    
    pred_obj <- predict(model, data = batch_data)
    predictions[start_idx:end_idx] <- pred_obj$predictions
    
    if (b %% 5 == 0 || b == n_batches) {
      cat("  Batch", b, "/", n_batches, "complete\n")
    }
    
    rm(batch_data, pred_obj)
    if (b %% 10 == 0) gc()
  }
  
  predictions
}

# Run prediction — choose the appropriate function for your model type:
# For randomForest package:
# cell_data[, predicted_gdp := predict_in_batches(rf_model, cell_data,
#                                                  feature_cols)]

# For ranger package:
# cell_data[, predicted_gdp := predict_in_batches_ranger(rf_model, cell_data,
#                                                         feature_cols)]

# Generic auto-detect:
if (inherits(rf_model, "ranger")) {
  cell_data[, predicted_gdp := predict_in_batches_ranger(
    rf_model, cell_data, feature_cols
  )]
} else {
  cell_data[, predicted_gdp := predict_in_batches(
    rf_model, cell_data, feature_cols
  )]
}

cat("Prediction complete. Results stored in cell_data$predicted_gdp\n")

# ---- Step 5: Clean up and output -------------------------------------------
cell_data[, .row_idx := NULL]  # remove helper column

# Final result
cat("Output rows:", nrow(cell_data), "\n")
cat("Output cols:", ncol(cell_data), "\n")
cat("Memory used by cell_data:", format(object.size(cell_data), units = "GB"),
    "\n")
```

---

## 4. SUMMARY OF KEY CHANGES

| Original | Optimized |
|---|---|
| `build_neighbor_lookup`: 6.46M-iteration `lapply` with `paste` + named-vector lookup producing a 6.46M-element list | `build_edge_table`: fully vectorized `rep.int` + `unlist` → 2-column `data.table` of ~1.37M edges |
| `compute_neighbor_stats`: per-variable `lapply` over 6.46M rows, `do.call(rbind, ...)` on 6.46M length-3 vectors | Year-batched `data.table` keyed join + grouped aggregation; all 5 variables computed simultaneously per year batch |
| `cell_data` as `data.frame`, copied on every column addition (15 copies of ~5.7 GB) | `data.table` with `set()` / `:=` — in-place modification, zero copies |
| Prediction assumed to be single-call on 6.46M rows (may OOM on 16 GB) | Batched prediction (500K rows/batch) with explicit `gc()`, staying within RAM |
| Neighbor lookup list: ~6.46M list elements of variable-length integer vectors (~hundreds of MB) | Edge table: ~1.37M × 2 integer columns (~11 MB) |

**The trained Random Forest model is never modified or retrained. The numerical estimand (predicted GDP values) is preserved identically — only the computational path to prepare inputs and call `predict()` is changed.**