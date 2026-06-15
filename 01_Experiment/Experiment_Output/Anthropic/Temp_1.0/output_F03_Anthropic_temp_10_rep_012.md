 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first major bottleneck:

1. **Row-level `lapply` over 6.46M rows.** Each iteration performs character coercion (`as.character`), multiple named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), string pasting (`paste(..., sep="_")`), and NA filtering. Named vector lookup in R is O(n) hash-probe per call, but doing it millions of times with freshly-pasted keys is extremely expensive due to repeated memory allocation for temporary character vectors.
2. **String key construction inside the loop.** `paste(neighbor_cell_ids, data$year[i], sep = "_")` creates a new character vector for every single row. Over 6.46M rows, this alone generates billions of short-lived string objects, hammering the garbage collector.
3. **The output is a list of 6.46M integer vectors.** This consumes substantial memory and is slow to construct element-by-element.

**`compute_neighbor_stats`** is the second major bottleneck:

1. **Another `lapply` over 6.46M elements**, each extracting a subset of a numeric vector, removing NAs, and computing three summary statistics. This is pure interpreted R with no vectorisation.
2. **`do.call(rbind, result)` on a 6.46M-element list of length-3 vectors** is notoriously slow — it creates a temporary list-of-rows then binds them, causing massive copying.
3. **This runs 5 times** (once per neighbor source variable), so the inefficiency multiplies.

**Outer loop — `compute_and_add_neighbor_features`:**

1. Each call likely modifies `cell_data` (a 6.46M × 110+ column data.frame), triggering R's copy-on-modify semantics. With a ~5.7 GB frame, even one unnecessary copy can exhaust 16 GB RAM and force swapping.

### B. Random Forest Inference Bottlenecks (Primary Focus)

Although the code shown is feature preparation, the user states the main problem is the **prediction workflow**:

1. **Model object size.** A Random Forest trained on 6.46M rows × 110 features with, say, 500 trees can easily be 2–8 GB in memory. On a 16 GB machine, having both the model and the data in memory simultaneously may cause swapping.
2. **Single `predict()` call on 6.46M rows.** `ranger::predict` and `randomForest::predict` both allocate an internal matrix of `n_rows × n_trees` and reduce. For 6.46M × 500, that's a ~24 GB intermediate (double precision) — far beyond available RAM.
3. **If using `randomForest` (not `ranger`)**, prediction is done in interpreted R loops over trees and is dramatically slower.
4. **No chunking.** Predicting all 6.46M rows at once maximises peak memory; chunked prediction keeps memory bounded.
5. **Repeated model loading.** If the serialized model is re-read from disk on every invocation, deserialization of a multi-GB RDS is slow.

### C. Summary of Root Causes

| Component | Issue | Estimated Impact |
|---|---|---|
| `build_neighbor_lookup` | 6.46M iterations of string ops + named-vector lookup | ~hours |
| `compute_neighbor_stats` | 6.46M `lapply` + `do.call(rbind, ...)` × 5 vars | ~hours |
| Data.frame copy-on-modify | Full-frame copy on each feature addition × 5 vars | ~minutes + RAM pressure |
| RF prediction (peak memory) | Full-matrix intermediate exceeds 16 GB RAM | Swapping → hours |
| RF prediction (no chunking) | No way to bound memory | Compounds above |
| Possible `randomForest` pkg | Interpreted-R tree traversal | 10–50× slower than `ranger` |

**Estimated time breakdown of 86+ hours:** ~30–40 h in neighbor lookup construction, ~20–30 h in neighbor stats, ~10–20 h in prediction (much of it swapping), remainder in data manipulation overhead.

---

## 2. OPTIMIZATION STRATEGY

### A. Feature Preparation

1. **Replace string-keyed lookups with integer-arithmetic indexing.** Encode `(id, year)` → row index as a matrix or integer hash: `row_index = (id_position - 1) * n_years + (year - min_year + 1)`. This eliminates all `paste()` and named-vector lookups.
2. **Precompute a flat neighbor-index matrix** using `data.table` joins instead of row-by-row `lapply`. Build an edge list `(row_i, neighbor_row_j)` once, then compute grouped summaries with `data.table`.
3. **Vectorised neighbor stats** via `data.table` grouped aggregation on the edge list: one operation computes max/min/mean for all 6.46M rows simultaneously, fully in C.
4. **Use `data.table` for the main dataset** to avoid copy-on-modify (`:=` modifies in place).

### B. Random Forest Inference

1. **Ensure the model is `ranger`, not `randomForest`.** If the existing model is `randomForest`, wrap prediction but still chunk. If it's `ranger`, prediction is already C++-backed and much faster.
2. **Chunked prediction.** Split the 6.46M rows into chunks of ~500K, predict each chunk, concatenate results. Peak memory drops from ~24 GB to ~2 GB.
3. **Load the model once** and keep it in memory; don't reload per chunk.
4. **Trim the model object** before prediction: remove training-only slots (e.g., `inbag.counts`, `predictions` on training data) to save RAM.
5. **Garbage collect between chunks** to keep memory bounded.

### C. Expected Speedup

| Component | Before | After | Factor |
|---|---|---|---|
| Neighbor lookup | ~30–40 h | ~2–5 min | ~500× |
| Neighbor stats (×5) | ~20–30 h | ~2–5 min | ~400× |
| Data manipulation | ~hours | seconds | ~100× |
| RF prediction | ~10–20 h (swapping) | ~10–30 min | ~30–60× |
| **Total** | **86+ h** | **~20–45 min** | **~100–250×** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Dependencies: data.table, ranger (or randomForest)
# Preserves: trained RF model, original numerical estimand
# =============================================================================

library(data.table)

# -------------------------------------------------------------------------
# STEP 0: Load model once, trim unnecessary slots to save RAM
# -------------------------------------------------------------------------
load_and_trim_model <- function(model_path) {
  message("Loading trained model from: ", model_path)
  model <- readRDS(model_path)
  
  # Trim training-only slots to reduce memory footprint
  # Works for both ranger and randomForest objects
  if (inherits(model, "ranger")) {
    model$predictions      <- NULL
    model$inbag.counts     <- NULL
    # Keep: num.trees, forest, variable.importance, etc.
  } else if (inherits(model, "randomForest")) {
    model$predicted   <- NULL
    model$oob.times   <- NULL
    model$votes       <- NULL
    # Keep: forest, ntree, mtry, etc.
  }
  
  gc()
  message("Model loaded. Object size: ",
          format(object.size(model), units = "GB"))
  model
}

# -------------------------------------------------------------------------
# STEP 1: Build integer-indexed neighbor edge list (vectorised)
# -------------------------------------------------------------------------
build_neighbor_edgelist <- function(cell_data_dt, id_order, rook_neighbors) {
  # cell_data_dt must be a data.table with columns: id, year
  # id_order: vector of cell IDs in the order matching rook_neighbors
  # rook_neighbors: spdep nb object (list of integer neighbor indices)
  
  message("Building neighbor edge list...")
  
  n_years  <- cell_data_dt[, uniqueN(year)]
  min_year <- cell_data_dt[, min(year)]
  
  # Map each id to its position in id_order (1-based)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Build edge list from the nb object: (source_pos, target_pos) in id_order space
  # This is done once and is fast because rook_neighbors is already integer-indexed
  n_cells <- length(id_order)
  
  source_pos <- rep(seq_len(n_cells),
                    times = lengths(rook_neighbors))
  target_pos <- unlist(rook_neighbors, use.names = FALSE)
  
  edges <- data.table(source_pos = source_pos, target_pos = target_pos)
  
  # Now expand over all years: for each year, each (source, target) pair

# corresponds to row indices in cell_data_dt.
  # 
  # We need a mapping: (id_pos, year) -> row index in cell_data_dt.
  # Strategy: add id_pos to cell_data_dt, then create the mapping.
  
  # Add id_pos to cell_data_dt
  cell_data_dt[, id_pos := id_to_pos[as.character(id)]]
  
  # Create mapping table: (id_pos, year) -> row_idx
  pos_year_map <- cell_data_dt[, .(id_pos, year, row_idx = .I)]
  setkey(pos_year_map, id_pos, year)
  
  # Get unique years
  years <- sort(unique(cell_data_dt$year))
  
  # For each year, join edges with row indices
  # Vectorised: cross join edges × years, then map to row indices
  message("  Expanding edges across ", length(years), " years...")
  
  edge_year <- CJ(edge_idx = seq_len(nrow(edges)), year = years)
  edge_year[, source_pos := edges$source_pos[edge_idx]]
  edge_year[, target_pos := edges$target_pos[edge_idx]]
  edge_year[, edge_idx := NULL]
  
  # Join to get source row index (the row whose neighbors we're computing)
  setkey(edge_year, source_pos, year)
  edge_year <- pos_year_map[edge_year,
                            .(source_row = row_idx,
                              target_pos,
                              year = i.year),
                            on = .(id_pos = source_pos, year)]
  
  # Join to get target row index (the neighbor row)
  setkey(edge_year, target_pos, year)
  edge_year <- pos_year_map[edge_year,
                            .(source_row = i.source_row,
                              target_row = row_idx),
                            on = .(id_pos = target_pos, year = year)]
  
  # Remove edges where either source or target row is missing
  edge_year <- edge_year[!is.na(source_row) & !is.na(target_row)]
  
  setkey(edge_year, source_row)
  
  message("  Edge list complete: ", format(nrow(edge_year), big.mark = ","),
          " directed edges across all years.")
  
  edge_year
}

# -------------------------------------------------------------------------
# STEP 2: Compute neighbor stats for one variable (fully vectorised)
# -------------------------------------------------------------------------
compute_neighbor_stats_fast <- function(cell_data_dt, edge_list, var_name) {
  # edge_list: data.table with columns source_row, target_row (keyed on source_row)
  # Returns nothing; modifies cell_data_dt in place via :=
  
  message("  Computing neighbor stats for: ", var_name)
  
  # Extract neighbor values via the edge list
  edge_list[, val := cell_data_dt[[var_name]][target_row]]
  
  # Remove NA values before aggregation
  edge_valid <- edge_list[!is.na(val)]
  
  # Grouped aggregation: max, min, mean per source_row
  stats <- edge_valid[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = source_row]
  
  # Prepare column names matching original output convention
  max_col  <- paste0("max_neighbor_", var_name)
  min_col  <- paste0("min_neighbor_", var_name)
  mean_col <- paste0("mean_neighbor_", var_name)
  
  # Initialize with NA, then fill matched rows (in-place, no copy)
  n <- nrow(cell_data_dt)
  cell_data_dt[, (max_col)  := NA_real_]
  cell_data_dt[, (min_col)  := NA_real_]
  cell_data_dt[, (mean_col) := NA_real_]
  
  cell_data_dt[stats$source_row, (max_col)  := stats$nb_max]
  cell_data_dt[stats$source_row, (min_col)  := stats$nb_min]
  cell_data_dt[stats$source_row, (mean_col) := stats$nb_mean]
  
  # Clean up temp column from edge_list
  edge_list[, val := NULL]
  
  invisible(NULL)
}

# -------------------------------------------------------------------------
# STEP 3: Prepare all neighbor features
# -------------------------------------------------------------------------
prepare_all_neighbor_features <- function(cell_data_dt, id_order,
                                          rook_neighbors,
                                          neighbor_source_vars) {
  # Build edge list once
  edge_list <- build_neighbor_edgelist(cell_data_dt, id_order, rook_neighbors)
  
  # Compute neighbor stats for each variable (in-place modification)
  for (var_name in neighbor_source_vars) {
    compute_neighbor_stats_fast(cell_data_dt, edge_list, var_name)
  }
  
  # Clean up
  rm(edge_list)
  gc()
  
  invisible(NULL)
}

# -------------------------------------------------------------------------
# STEP 4: Chunked Random Forest prediction (memory-bounded)
# -------------------------------------------------------------------------
predict_chunked <- function(model, newdata_dt, chunk_size = 500000L) {
  # newdata_dt: data.table of predictor features
  # Returns: numeric vector of predictions (same length as nrow(newdata_dt))
  # Preserves original numerical estimand (no transformation)
  
  n <- nrow(newdata_dt)
  n_chunks <- ceiling(n / chunk_size)
  predictions <- numeric(n)
  
  is_ranger <- inherits(model, "ranger")
  
  message("Predicting ", format(n, big.mark = ","), " rows in ",
          n_chunks, " chunks of up to ", format(chunk_size, big.mark = ","),
          " rows...")
  
  for (i in seq_len(n_chunks)) {
    start_idx <- (i - 1L) * chunk_size + 1L
    end_idx   <- min(i * chunk_size, n)
    
    chunk <- newdata_dt[start_idx:end_idx]
    
    if (is_ranger) {
      pred <- predict(model, data = chunk)$predictions
    } else {
      # randomForest package
      pred <- predict(model, newdata = chunk)
    }
    
    predictions[start_idx:end_idx] <- pred
    
    if (i %% 5 == 0 || i == n_chunks) {
      message("  Chunk ", i, "/", n_chunks, " complete.")
      gc()
    }
  }
  
  predictions
}

# -------------------------------------------------------------------------
# STEP 5: Full pipeline
# -------------------------------------------------------------------------
run_optimized_pipeline <- function(
    cell_data,            # data.frame or data.table with cell-year panel
    id_order,             # vector of cell IDs matching rook_neighbors order
    rook_neighbors_unique,# spdep nb object (pre-loaded or path to RDS)
    model_path,           # path to saved RF model RDS file
    predictor_cols,       # character vector of ~110 predictor column names
    neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
    chunk_size = 500000L
) {
  total_start <- Sys.time()
  
  # --- Load neighbor object if path is given ---
  if (is.character(rook_neighbors_unique)) {
    message("Loading neighbor object...")
    rook_neighbors_unique <- readRDS(rook_neighbors_unique)
  }
  
  # --- Convert to data.table (in-place if already data.table) ---
  if (!is.data.table(cell_data)) {
    message("Converting to data.table...")
    cell_data <- as.data.table(cell_data)
  }
  
  # --- Step A: Feature preparation ---
  message("\n=== FEATURE PREPARATION ===")
  t0 <- Sys.time()
  
  prepare_all_neighbor_features(
    cell_data_dt         = cell_data,
    id_order             = id_order,
    rook_neighbors       = rook_neighbors_unique,
    neighbor_source_vars = neighbor_source_vars
  )
  
  message("Feature preparation time: ",
          round(difftime(Sys.time(), t0, units = "mins"), 1), " minutes")
  
  # --- Step B: Load and trim model ---
  message("\n=== MODEL LOADING ===")
  model <- load_and_trim_model(model_path)
  gc()
  
  # --- Step C: Prepare prediction matrix ---
  message("\n=== PREDICTION ===")
  
  # Verify all predictor columns exist
  missing_cols <- setdiff(predictor_cols, names(cell_data))
  if (length(missing_cols) > 0) {
    stop("Missing predictor columns: ",
         paste(head(missing_cols, 10), collapse = ", "),
         if (length(missing_cols) > 10) paste0(" ... (", length(missing_cols), " total)"))
  }
  
  # Subset to predictor columns only (avoids passing unnecessary data)
  pred_data <- cell_data[, ..predictor_cols]
  
  # --- Step D: Chunked prediction ---
  t0 <- Sys.time()
  cell_data[, predicted_gdp := predict_chunked(model, pred_data, chunk_size)]
  
  message("Prediction time: ",
          round(difftime(Sys.time(), t0, units = "mins"), 1), " minutes")
  
  # --- Clean up ---
  rm(pred_data, model)
  gc()
  
  message("\n=== PIPELINE COMPLETE ===")
  message("Total time: ",
          round(difftime(Sys.time(), total_start, units = "mins"), 1), " minutes")
  
  cell_data
}

# -------------------------------------------------------------------------
# USAGE EXAMPLE
# -------------------------------------------------------------------------
# result <- run_optimized_pipeline(
#   cell_data             = cell_data,
#   id_order              = id_order,
#   rook_neighbors_unique = "path/to/rook_neighbors_unique.rds",
#   model_path            = "path/to/trained_rf_model.rds",
#   predictor_cols        = predictor_column_names,  # character vector of ~110 names
#   neighbor_source_vars  = c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
#   chunk_size            = 500000L
# )
#
# # Predictions are in: result$predicted_gdp
```

---

## 4. KEY DESIGN DECISIONS EXPLAINED

### Why the edge-list approach replaces `build_neighbor_lookup`

The original code builds a **list of 6.46M integer vectors** by looping row-by-row, pasting strings, and doing named-vector lookups. The optimized version:

- Expands the `spdep::nb` object into a flat edge table `(source_pos, target_pos)` in **one `unlist()` + `rep()` call** — pure vectorised C code.
- Cross-joins with years using `data.table::CJ` and resolves row indices via keyed joins — no string operations at all.
- The resulting `edge_list` data.table has ~38.5M rows (1.37M edges × 28 years) and is ~600 MB, well within RAM budget.

### Why `data.table` grouped aggregation replaces `compute_neighbor_stats`

The original does 6.46M separate `lapply` iterations followed by `do.call(rbind, ...)`. The optimized version:

- Extracts all neighbor values in one vectorised column assignment: `edge_list[, val := cell_data_dt[[var_name]][target_row]]`.
- Computes `max`, `min`, `mean` grouped by `source_row` in a single `data.table` aggregation — executed in C with radix-sort grouping.
- Writes results back via `:=` (in-place, zero-copy).
- **Each variable takes seconds instead of hours.**

### Why chunked prediction

A `ranger` model with 500 trees predicting 6.46M rows allocates an internal `n × num.trees` matrix during prediction. At 8 bytes per double, that's `6.46M × 500 × 8 ≈ 24 GB` — far exceeding 16 GB RAM. Chunking to 500K rows bounds this intermediate to `500K × 500 × 8 ≈ 1.9 GB`, keeping total memory well within budget and **eliminating swap-induced slowdowns** that likely account for a significant portion of the original 86+ hour runtime.

### Preservation guarantees

- **Trained model is never modified** — we only remove non-prediction slots (`predictions`, `inbag.counts`) that are not used during `predict()`.
- **Numerical estimand is preserved exactly** — predictions use the same `predict()` function with identical inputs; no transformations, rounding, or retraining occur. The chunked results are concatenated in order and assigned back to the original row positions.