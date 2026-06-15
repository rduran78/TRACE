 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### Primary Bottlenecks (ranked by severity)

**A. `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows with per-row string operations**

The function creates 6.46 million character-pasted keys (`paste(id, year, sep="_")`) and then does named-vector lookups (which are hash-table lookups in R, but repeated millions of times inside `lapply`). Each iteration also calls `as.character`, `paste`, subsetting a named vector, and `is.na` filtering. This is the single most expensive pre-prediction step. Estimated cost: **hours** on 6.46M rows.

**B. `compute_neighbor_stats` — repeated per-variable `lapply` over 6.46M rows**

Called 5 times (once per neighbor source variable). Each call iterates over all 6.46M rows, subsets a numeric vector by index, removes NAs, and computes `max/min/mean`. The `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself very slow (creates 6.46M tiny vectors then row-binds them).

**C. Object copying in the outer loop**

`cell_data <- compute_and_add_neighbor_features(...)` likely triggers a full copy of the data.frame on every column addition (R's copy-on-modify semantics). With ~110 columns × 6.46M rows, each copy is ~5–7 GB, and it happens 5 times (once per variable, adding 3 columns each time).

**D. Random Forest prediction on 6.46M rows × 110 features**

`predict.randomForest` in the `randomForest` package is implemented in R with a loop over trees. For a large forest (e.g., 500 trees) on 6.46M rows, this is extremely slow. The `ranger` package's `predict` is written in C++ and is 10–50× faster, but model compatibility must be preserved.

**E. Memory pressure**

6.46M × 110 columns of doubles ≈ 5.4 GB. Neighbor lookup list with ~6.46M elements ≈ 1–2 GB. Intermediate copies push total past 16 GB → swapping to disk → catastrophic slowdown.

---

## 2. OPTIMIZATION STRATEGY

| Bottleneck | Solution | Expected Speedup |
|---|---|---|
| A. `build_neighbor_lookup` | Replace string-key lookup with integer arithmetic: encode `(id, year)` → row index via a matrix or `data.table` join. Vectorize entirely. | **50–200×** |
| B. `compute_neighbor_stats` | Pre-build a flat edge-list (row_i → neighbor_row_j), then use `data.table` grouped aggregation or vectorized C++ (Rcpp) to compute stats in one pass per variable. | **20–100×** |
| C. Object copying | Use `data.table` with `:=` (modification by reference, zero copies). | **5–15×** |
| D. RF prediction | If model is `randomForest` class, convert predictions to use `ranger` (if possible) or at minimum predict in chunks to control memory. If stuck with `randomForest::predict`, no easy fix without model retraining. Use matrix input instead of data.frame. | **2–50×** |
| E. Memory | `data.table` + in-place ops + chunked prediction keep peak RAM under ~10 GB. | Avoids swap |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Dependencies: data.table, Rcpp (optional but recommended)
# =============================================================================

library(data.table)

# ---- Step 0: Convert cell_data to data.table (if not already) ---------------
# This is the single most impactful change: all subsequent column additions
# are by reference (zero-copy).

if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ---- Step 1: Optimized neighbor lookup via integer indexing ------------------
# Goal: for every row i in cell_data, find the row indices of its
# rook neighbors in the SAME year.
#
# Strategy:
#   - Create a keyed lookup table: (id, year) -> row_index
#   - Expand the nb object into a flat edge list of (focal_id, neighbor_id)
#   - Join with year to get (focal_id, year, neighbor_id) -> neighbor_row_index
#   - This is fully vectorized via data.table joins.

build_neighbor_edgelist_dt <- function(cell_dt, id_order, neighbors) {
  # cell_dt must have columns: id, year
  # id_order: vector mapping position in nb list -> cell id
  # neighbors: spdep nb object (list of integer vectors of neighbor positions)
  
  # 1. Build flat edge list: focal_position -> neighbor_position
  #    Then map positions to actual cell IDs.
  n_focal <- length(neighbors)
  
  # Pre-compute lengths for pre-allocation
  lens <- lengths(neighbors)
  total_edges <- sum(lens)  # ~1.37M directed edges
  
  focal_pos <- rep(seq_len(n_focal), times = lens)
  neighbor_pos <- unlist(neighbors, use.names = FALSE)
  
  edge_dt <- data.table(
    focal_id    = id_order[focal_pos],
    neighbor_id = id_order[neighbor_pos]
  )
  
  # 2. Build row-index lookup: (id, year) -> row in cell_dt
  cell_dt[, .row_idx := .I]
  
  lookup <- cell_dt[, .(id, year, .row_idx)]
  setkey(lookup, id, year)
  
  # 3. Get all unique years
  years <- unique(cell_dt$year)
  
  # 4. Cross join edges × years, then look up row indices for both focal and neighbor
  #    But edges × 28 years = ~38M rows — manageable.
  #    More memory-efficient: join focal rows to get year, then join neighbor.
  
  # Approach: start from cell_dt rows, attach their neighbor IDs, then look up neighbor rows.
  
  # focal_lookup: for each row in cell_dt, get its neighbor cell IDs
  focal_lookup <- cell_dt[, .(focal_id = id, year, focal_row = .row_idx)]
  setkey(edge_dt, focal_id)
  
  # For each (focal_id, year) row, find all neighbor_ids
  # This is a join: focal_lookup[edge_dt, on = "focal_id", allow.cartesian = TRUE]
  # Result: one row per (focal_row, neighbor_id, year)
  
  setkey(focal_lookup, focal_id)
  expanded <- edge_dt[focal_lookup,
                      on = "focal_id",
                      .(focal_row, neighbor_id, year),
                      allow.cartesian = TRUE,
                      nomatch = NULL]
  
  # Now look up the neighbor's row index for the same year
  setnames(lookup, c("id", "year", ".row_idx"), c("neighbor_id", "year", "neighbor_row"))
  setkey(lookup, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  
  result <- lookup[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # result has columns: neighbor_id, year, neighbor_row, focal_row
  
  # Drop rows where neighbor has no data for that year
  result <- result[!is.na(neighbor_row)]
  
  # Clean up temporary column
  cell_dt[, .row_idx := NULL]
  
  return(result[, .(focal_row, neighbor_row)])
}

cat("Building neighbor edge list (vectorized)...\n")
system.time({
  edge_dt <- build_neighbor_edgelist_dt(cell_data, id_order, rook_neighbors_unique)
})
# Expected: ~30–120 seconds for ~38M expanded edges (vs hours before)


# ---- Step 2: Vectorized neighbor stats via data.table grouping ---------------
# For each focal_row and each variable, compute max, min, mean of neighbor values.

compute_all_neighbor_features_dt <- function(cell_dt, edge_dt, var_names) {
  # edge_dt: data.table with columns focal_row (int), neighbor_row (int)
  # var_names: character vector of variable names
  
  setkey(edge_dt, focal_row)
  
  for (vname in var_names) {
    cat(sprintf("  Computing neighbor stats for: %s\n", vname))
    
    # Attach neighbor values
    edge_dt[, nval := cell_dt[[vname]][neighbor_row]]
    
    # Group by focal_row, compute stats (NA-aware)
    stats <- edge_dt[!is.na(nval),
                     .(
                       v_max  = max(nval),
                       v_min  = min(nval),
                       v_mean = mean(nval)
                     ),
                     by = focal_row]
    
    # Prepare column names
    max_col  <- paste0("neighbor_max_", vname)
    min_col  <- paste0("neighbor_min_", vname)
    mean_col <- paste0("neighbor_mean_", vname)
    
    # Initialize columns with NA (for rows with no valid neighbors)
    set(cell_dt, j = max_col,  value = NA_real_)
    set(cell_dt, j = min_col,  value = NA_real_)
    set(cell_dt, j = mean_col, value = NA_real_)
    
    # Fill in computed values by reference (zero-copy)
    set(cell_dt, i = stats$focal_row, j = max_col,  value = stats$v_max)
    set(cell_dt, i = stats$focal_row, j = min_col,  value = stats$v_min)
    set(cell_dt, i = stats$focal_row, j = mean_col, value = stats$v_mean)
    
    # Clean up temp column
    edge_dt[, nval := NULL]
  }
  
  invisible(cell_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (vectorized, in-place)...\n")
system.time({
  compute_all_neighbor_features_dt(cell_data, edge_dt, neighbor_source_vars)
})
# Expected: ~2–5 minutes total for all 5 variables (vs hours before)

# Free the edge list
rm(edge_dt); gc()


# ---- Step 3: Optimized Random Forest Prediction -----------------------------
# Key optimizations:
#   a) Convert predictor data to a plain matrix (RF predict on matrix is faster
#      than on data.frame — avoids repeated type-checking per tree).
#   b) Predict in chunks to limit peak memory (each chunk's internal
#      allocation is bounded).
#   c) If model is of class "randomForest", use predict.randomForest with
#      matrix input. If "ranger", use predict.ranger (already fast).

predict_rf_optimized <- function(model, cell_dt, feature_names,
                                 chunk_size = 500000L) {
  # Prepare a numeric matrix of predictors
  cat("Preparing prediction matrix...\n")
  
  # Allocate matrix once
  n <- nrow(cell_dt)
  p <- length(feature_names)
  pred_mat <- matrix(NA_real_, nrow = n, ncol = p,
                     dimnames = list(NULL, feature_names))
  
  for (j in seq_along(feature_names)) {
    col_vals <- cell_dt[[feature_names[j]]]
    if (is.numeric(col_vals)) {
      pred_mat[, j] <- col_vals
    } else {
      # For factors/characters, convert to numeric codes
      # (Random Forest expects same types as training data)
      pred_mat[, j] <- as.numeric(as.factor(col_vals))
    }
  }
  
  # Determine model class
  model_class <- class(model)[1]
  
  # Chunked prediction
  n_chunks <- ceiling(n / chunk_size)
  predictions <- numeric(n)
  
  cat(sprintf("Predicting %d rows in %d chunks of %d...\n",
              n, n_chunks, chunk_size))
  
  for (ch in seq_len(n_chunks)) {
    start_i <- (ch - 1L) * chunk_size + 1L
    end_i   <- min(ch * chunk_size, n)
    idx     <- start_i:end_i
    
    chunk_data <- pred_mat[idx, , drop = FALSE]
    
    if (model_class == "ranger") {
      # ranger::predict expects a data.frame or matrix
      pred_obj <- predict(model, data = as.data.frame(chunk_data))
      predictions[idx] <- pred_obj$predictions
      
    } else if (model_class == "randomForest") {
      # randomForest::predict.randomForest accepts newdata as data.frame
      predictions[idx] <- predict(model, newdata = as.data.frame(chunk_data))
      
    } else {
      # Generic fallback
      predictions[idx] <- predict(model, newdata = as.data.frame(chunk_data))
    }
    
    if (ch %% 5 == 0 || ch == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %d–%d)\n",
                  ch, n_chunks, start_i, end_i))
    }
    gc()  # Free intermediate allocations between chunks
  }
  
  rm(pred_mat); gc()
  return(predictions)
}

# Load the pre-trained model
# rf_model <- readRDS("path/to/trained_rf_model.rds")

# Define the feature names used during training (must match exactly)
# feature_names <- names(rf_model$forest$xlevels)
#   — or however your feature names are stored. For ranger:
# feature_names <- rf_model$forest$independent.variable.names

cat("Running optimized Random Forest prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_rf_optimized(
    model         = rf_model,
    cell_dt       = cell_data,
    feature_names = feature_names,
    chunk_size    = 500000L
  )]
})
# Expected: 10–60 minutes depending on forest size (vs many hours before)


# ---- Step 4 (Optional): If model is randomForest, consider one-time --------
# conversion to ranger for dramatically faster predict().
# This does NOT retrain — it's only useful if you can re-save as ranger.
# If you MUST keep the randomForest object, skip this.

# Alternative: use the 'trimTrees' or external C-level predict if available.


# =============================================================================
# SUMMARY OF CHANGES
# =============================================================================
# 
# | Component                  | Before                        | After                              |
# |----------------------------|-------------------------------|-------------------------------------|
# | Neighbor lookup            | lapply + paste + named vector | data.table join (vectorized)        |
# |                            | O(6.46M) string ops           | O(1) join on integer keys           |
# | Neighbor stats             | lapply per variable           | data.table grouped aggregation      |
# |                            | + do.call(rbind, 6.46M list)  | single pass, no list overhead       |
# | Column addition            | data.frame copy-on-modify     | data.table set()/`:=` (zero-copy)  |
# |                            | ~5 GB copy × 15 additions     | 0 bytes copied                      |
# | RF prediction              | Full data.frame, single call  | Matrix input, chunked, gc() calls  |
# | Peak memory                | >16 GB (swap)                 | ~8–10 GB (fits in RAM)             |
# | Estimated total time       | 86+ hours                     | ~15–90 minutes                     |
# =============================================================================
```

---

## 4. KEY EXPLANATIONS

### Why `build_neighbor_lookup` was the worst offender

The original code calls `paste()` and does named-vector hash lookups **inside a per-row `lapply`** over 6.46M rows. R's `lapply` has ~1–3 µs overhead per iteration just for the function call, so 6.46M iterations × ~10 µs of string work each ≈ **18+ hours** for this function alone. The replacement uses `data.table` keyed joins which execute the same logic in vectorized C code in ~1 minute.

### Why `do.call(rbind, ...)` on millions of small vectors is devastating

`do.call(rbind, list_of_6.46M_vectors)` must allocate a 6.46M × 3 matrix and copy each 3-element vector one at a time. The `data.table` grouped aggregation returns the result directly as columns—no intermediate list is ever created.

### Why `:=` / `set()` matters

Adding a column to a `data.frame` with `df$new_col <- values` triggers R's copy-on-modify: the entire data.frame (~5 GB) is duplicated. With 15 new neighbor columns (5 vars × 3 stats), that's 15 copies = **~75 GB of memory churn**, guaranteed to cause swap thrashing on a 16 GB machine. `data.table`'s `set()` modifies in place with zero copies.

### Random Forest prediction

The chunked approach keeps peak memory bounded (only one chunk's worth of internal tree-traversal allocations at a time) and calls `gc()` between chunks to reclaim memory. Converting input to a matrix avoids repeated `is.numeric()` checks inside the predict method. If your model is a `randomForest` object and you have the option to retrain with `ranger` in the future, `ranger::predict` is typically **10–50× faster** due to its C++ implementation.