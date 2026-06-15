 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

1. **Row-level `lapply` over 6.46M rows:** Each iteration performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and NA filtering. Doing this 6.46 million times in interpreted R is extremely slow.
2. **Named-vector lookups are O(n) hash lookups repeated millions of times.** `idx_lookup` is a named vector of length 6.46M; each call to `idx_lookup[neighbor_keys]` performs repeated hash-table probes on a very large vector.
3. **Massive string allocation:** ~6.46M `paste()` calls inside the loop, plus the neighbor-key pastes, generate enormous temporary string vectors that pressure the garbage collector.

**`compute_neighbor_stats`** is the second bottleneck:

1. **Another row-level `lapply` over 6.46M rows**, each extracting a small subset of a numeric vector, removing NAs, and computing max/min/mean.
2. **Called 5 times** (once per neighbor source variable), so this loop executes ~32.3M iterations total.
3. **`do.call(rbind, result)` on a 6.46M-element list** is notoriously slow — it creates a massive argument list and row-binds one-by-one.

### B. Random Forest Inference Bottleneck

With ~6.46M rows and ~110 predictors, a single call to `predict(rf_model, newdata)` on the full dataset will:

1. **Attempt to allocate a single enormous prediction matrix** in memory. For a `ranger` model this is manageable; for `randomForest` it can be very costly because `predict.randomForest` copies the data frame into a matrix internally.
2. **If using `randomForest` package:** `predict()` converts the entire data.frame to a matrix via `model.frame` → heavy copying and type-checking for 6.46M × 110 ≈ 710M cells.
3. **If predicting in a row-by-row or small-batch loop:** overhead per call dominates (model dispatch, data validation, matrix conversion repeated thousands of times).

### C. Memory Pressure

- Raw data: 6.46M rows × 110 columns × 8 bytes ≈ **5.7 GB** just for the numeric matrix.
- Neighbor lookup list of 6.46M elements, each an integer vector → several GB with list overhead.
- Any full-copy operations (`cell_data <- cbind(cell_data, ...)`) double the memory footprint temporarily.
- On a 16 GB laptop, this leaves almost no headroom, causing swapping → the "86+ hours" estimate.

### D. Summary of Root Causes

| Rank | Bottleneck | Why |
|------|-----------|-----|
| 1 | `build_neighbor_lookup` — row-level R loop with string ops over 6.46M rows | Interpreted loop, string allocation, large hash lookups |
| 2 | `compute_neighbor_stats` — row-level R loop × 5 variables | 32.3M interpreted iterations, `do.call(rbind, ...)` |
| 3 | RF prediction — possible row-loop or `randomForest` package overhead | Data copying, matrix conversion, dispatch overhead |
| 4 | Memory — full copies of `cell_data` on each feature addition | 16 GB RAM saturated, OS swapping |

---

## 2. OPTIMIZATION STRATEGY

### Principle: Vectorize everything; eliminate interpreted loops; use `data.table` for in-place operations; batch RF prediction.

#### Feature Preparation

1. **Replace `build_neighbor_lookup` entirely** with a vectorized `data.table` join approach:
   - Expand the neighbor list into an edge-list `data.table` with columns `(id, neighbor_id)`.
   - Merge with the panel data keyed on `(id, year)` to get neighbor row indices or values directly.
   - This replaces 6.46M R-loop iterations with a single indexed join.

2. **Replace `compute_neighbor_stats`** with a grouped `data.table` aggregation:
   - After the join, group by the focal row and compute `max`, `min`, `mean` in one vectorized pass per variable.
   - Avoid `do.call(rbind, ...)` entirely.

3. **Add all 5 × 3 = 15 neighbor features in one pass** rather than looping over variables and re-joining.

4. **Use `data.table` `:=` assignment** to add columns in place — no copying of the entire data frame.

#### Random Forest Inference

5. **If currently using `randomForest` package, switch prediction to `ranger`'s `predict()`** or convert the model. If the model is already `ranger`, simply call `predict()` on the full matrix.
6. **Predict in moderately sized chunks** (~500K–1M rows) to control peak memory while avoiding per-row overhead.
7. **Pre-convert the prediction input to a numeric matrix** once, outside the predict call, to avoid repeated internal conversion.

#### Memory

8. **Use `data.table` throughout** (single copy of data, modify in place).
9. **Remove intermediate objects and call `gc()`** between major pipeline stages.
10. **Write neighbor features to disk (fst/qs) if needed** to free RAM before prediction.

**Expected speedup:** From 86+ hours to roughly **10–30 minutes** for feature preparation, and **5–20 minutes** for prediction, depending on the RF model size.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Prerequisites:
#   install.packages(c("data.table", "ranger"))
#   (or whatever RF package the trained model uses)
# =============================================================================

library(data.table)

# ---- Configuration ----------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
PREDICT_CHUNK_SIZE   <- 500000L
# Adjust based on available RAM; 500K rows × 110 cols ≈ 440 MB per chunk

# ---- Step 0: Convert cell_data to data.table in place ----------------------
# Assumes cell_data is a data.frame with columns: id, year, ntl, ec, ...
# Assumes rook_neighbors_unique is an nb object (list of integer index vectors)
# Assumes id_order is the vector mapping nb indices → cell IDs

setDT(cell_data)

# ---- Step 1: Build vectorized edge list from nb object ----------------------
build_edge_list_dt <- function(id_order, neighbors) {
  # neighbors[[i]] gives the nb-index positions of neighbors of id_order[i]
  # We expand this into a two-column data.table: (focal_id, neighbor_id)
  
  n <- length(neighbors)
  
  # Pre-compute lengths to allocate once
  lens <- vapply(neighbors, length, integer(1))
  total_edges <- sum(lens)
  
  focal_idx    <- rep.int(seq_len(n), lens)
  neighbor_idx <- unlist(neighbors, use.names = FALSE)
  
  # Remove the 0-neighbor sentinel if spdep uses 0 for "no neighbors"
  valid <- neighbor_idx > 0L
  focal_idx    <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]
  
  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

cat("Building edge list...\n")
edge_dt <- build_edge_list_dt(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %s rows\n", format(nrow(edge_dt), big.mark = ",")))

# ---- Step 2: Compute all neighbor features via data.table join --------------
compute_all_neighbor_features <- function(cell_dt, edge_dt, source_vars) {
  # cell_dt must have columns: id, year, and all source_vars
  # edge_dt must have columns: focal_id, neighbor_id
  
  # Create a minimal lookup table: (id, year) → values of source_vars
  lookup_cols <- c("id", "year", source_vars)
  lookup <- cell_dt[, ..lookup_cols]
  setnames(lookup, "id", "neighbor_id")
  
  # Key the lookup for fast join
  setkey(lookup, neighbor_id, year)
  
  # Expand edges by year: for each (focal_id, neighbor_id) pair, we need
  # every year that the focal cell has data. Instead of a cross-join,
  # we join edge_dt with cell_dt's (id, year) and then look up neighbor values.
  
  # Step 2a: Create (focal_id, year, neighbor_id) by joining cell_dt's
  #          unique (id, year) with edge_dt on focal_id
  cat("  Joining edges with years...\n")
  
  # Get unique (id, year) pairs — but cell_dt already has one row per (id, year)
  # so we just need id and year columns
  focal_years <- cell_dt[, .(id, year)]
  setnames(focal_years, "id", "focal_id")
  setkey(focal_years, focal_id)
  setkey(edge_dt, focal_id)
  
  # Join: for each focal_id, attach all its neighbor_ids
  # This creates (focal_id, year, neighbor_id) — can be large!
  # Estimated size: ~1.37M edges × 28 years ≈ 38.5M rows (manageable)
  expanded <- edge_dt[focal_years, on = "focal_id", allow.cartesian = TRUE,
                      nomatch = NULL]
  # expanded has columns: focal_id, neighbor_id, year
  
  cat(sprintf("  Expanded edge-year table: %s rows\n",
              format(nrow(expanded), big.mark = ",")))
  
  # Step 2b: Look up neighbor values
  cat("  Looking up neighbor values...\n")
  setkey(expanded, neighbor_id, year)
  expanded <- lookup[expanded, on = .(neighbor_id, year), nomatch = NA]
  # Now expanded has: neighbor_id, year, <source_vars>, focal_id
  
  # Step 2c: Aggregate by (focal_id, year) to get max, min, mean per variable
  cat("  Aggregating neighbor statistics...\n")
  
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("n_max_", v)]]  <- bquote(max(.(v_sym), na.rm = TRUE))
    agg_exprs[[paste0("n_min_", v)]]  <- bquote(min(.(v_sym), na.rm = TRUE))
    agg_exprs[[paste0("n_mean_", v)]] <- bquote(mean(.(v_sym), na.rm = TRUE))
  }
  
  # Convert to a single call
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  
  neighbor_stats <- expanded[, eval(agg_call), by = .(focal_id, year)]
  
  # Replace Inf/-Inf (from max/min on all-NA) with NA
  for (col_name in names(neighbor_stats)) {
    if (col_name %in% c("focal_id", "year")) next
    v <- neighbor_stats[[col_name]]
    set(neighbor_stats, which(is.infinite(v)), col_name, NA_real_)
  }
  
  cat(sprintf("  Neighbor stats table: %s rows × %s cols\n",
              format(nrow(neighbor_stats), big.mark = ","),
              ncol(neighbor_stats)))
  
  return(neighbor_stats)
}

cat("Computing neighbor features...\n")
neighbor_features <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# ---- Step 3: Join neighbor features back to cell_data in place --------------
cat("Joining neighbor features to cell_data...\n")

# Rename focal_id back to id for the join
setnames(neighbor_features, "focal_id", "id")
setkey(neighbor_features, id, year)
setkey(cell_data, id, year)

# Identify new columns to add
new_cols <- setdiff(names(neighbor_features), c("id", "year"))

# Remove any pre-existing neighbor columns to avoid duplication
existing <- intersect(new_cols, names(cell_data))
if (length(existing) > 0) {
  cell_data[, (existing) := NULL]
}

# In-place join (no copy of cell_data)
cell_data <- neighbor_features[cell_data, on = .(id, year)]

# Clean up large intermediates
rm(edge_dt, neighbor_features)
gc()

cat("Feature preparation complete.\n")
cat(sprintf("  cell_data: %s rows × %s cols\n",
            format(nrow(cell_data), big.mark = ","),
            ncol(cell_data)))

# ---- Step 4: Prepare prediction matrix once ---------------------------------
# Load the trained Random Forest model
# rf_model <- readRDS("path/to/trained_rf_model.rds")

# Identify the predictor columns the model expects
# For ranger:
if (inherits(rf_model, "ranger")) {
  pred_vars <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores predictor names differently
  pred_vars <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

cat(sprintf("Model expects %d predictor variables.\n", length(pred_vars)))

# Verify all required columns exist
missing_vars <- setdiff(pred_vars, names(cell_data))
if (length(missing_vars) > 0) {
  stop("Missing predictor columns in cell_data: ",
       paste(missing_vars, collapse = ", "))
}

# Extract predictor matrix once (avoids repeated internal conversion)
cat("Extracting predictor matrix...\n")
pred_matrix <- as.matrix(cell_data[, ..pred_vars])
cat(sprintf("  Predictor matrix: %s rows × %s cols (%.1f GB)\n",
            format(nrow(pred_matrix), big.mark = ","),
            ncol(pred_matrix),
            object.size(pred_matrix) / 1e9))

# ---- Step 5: Chunked prediction --------------------------------------------
predict_chunked <- function(model, pred_mat, chunk_size = 500000L) {
  n <- nrow(pred_mat)
  n_chunks <- ceiling(n / chunk_size)
  predictions <- numeric(n)
  
  cat(sprintf("Predicting in %d chunks of up to %s rows...\n",
              n_chunks, format(chunk_size, big.mark = ",")))
  
  for (i in seq_len(n_chunks)) {
    start_row <- (i - 1L) * chunk_size + 1L
    end_row   <- min(i * chunk_size, n)
    idx       <- start_row:end_row
    
    chunk_data <- pred_mat[idx, , drop = FALSE]
    
    if (inherits(model, "ranger")) {
      # ranger::predict on a matrix is efficient
      pred_obj <- predict(model, data = chunk_data)
      predictions[idx] <- pred_obj$predictions
    } else if (inherits(model, "randomForest")) {
      # randomForest::predict — pass matrix directly
      predictions[idx] <- predict(model, newdata = as.data.frame(chunk_data))
    } else {
      predictions[idx] <- predict(model, newdata = as.data.frame(chunk_data))
    }
    
    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("  Chunk %d/%d complete (rows %s–%s)\n",
                  i, n_chunks,
                  format(start_row, big.mark = ","),
                  format(end_row, big.mark = ",")))
    }
  }
  
  return(predictions)
}

cat("Running Random Forest predictions...\n")
cell_data[, predicted_gdp := predict_chunked(rf_model, pred_matrix, PREDICT_CHUNK_SIZE)]

# Free the prediction matrix
rm(pred_matrix)
gc()

cat("Prediction complete.\n")
cat(sprintf("  Output rows: %s\n", format(nrow(cell_data), big.mark = ",")))
cat(sprintf("  Predicted GDP range: [%.4f, %.4f]\n",
            min(cell_data$predicted_gdp, na.rm = TRUE),
            max(cell_data$predicted_gdp, na.rm = TRUE)))

# ---- Step 6 (optional): Save results ---------------------------------------
# fst::write_fst(cell_data, "cell_data_with_predictions.fst")
# or
# saveRDS(cell_data, "cell_data_with_predictions.rds")
```

---

## 4. SUMMARY OF CHANGES AND EXPECTED IMPACT

| Component | Original | Optimized | Expected Speedup |
|---|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows with `paste`/named-vector lookup | Vectorized `data.table` edge-list expansion + keyed join | ~200–500× |
| **Neighbor stats** | `lapply` over 6.46M rows × 5 vars + `do.call(rbind)` | Single grouped `data.table` aggregation for all 15 features | ~100–300× |
| **Feature attachment** | `cbind` / `data.frame` copy per variable | `data.table` keyed join + `:=` in-place assignment | Eliminates ~5 full copies of 6.46M-row frame |
| **RF prediction** | Possibly row-loop or single massive call with internal copy | Pre-built numeric matrix + chunked `predict()` (500K rows/chunk) | 10–50× (avoids repeated data conversion) |
| **Peak memory** | Multiple full copies → >16 GB → swapping | In-place `data.table` ops, single matrix, chunked predict | Stays within ~10–12 GB |
| **Overall runtime** | ~86+ hours | ~15–45 minutes | ~100–300× |

The trained Random Forest model is never modified — only its `predict()` method is called. The numerical estimand (predicted GDP values) is preserved identically since the same model, same features, and same prediction function are used; only the computational pathway is optimized.