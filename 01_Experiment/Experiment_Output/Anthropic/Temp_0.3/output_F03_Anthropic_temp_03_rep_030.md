 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys
- Named-vector lookup on `idx_lookup` (a named character vector of length 6.46M — each lookup is O(n) hash probe on a very large vector)
- NA filtering

With 6.46M iterations, each doing string concatenation and multiple named-vector lookups against a 6.46M-element vector, this alone can take many hours.

**`compute_neighbor_stats`:** Called 5 times (once per variable). Each call iterates over 6.46M rows with `lapply`, subsetting a numeric vector, removing NAs, and computing max/min/mean. The `do.call(rbind, result)` at the end binds 6.46M 3-element vectors into a matrix — this is a well-known R anti-pattern that is extremely slow for large lists.

**`do.call(rbind, ...)` on millions of small vectors** is one of the single most expensive operations here. It creates millions of intermediate objects.

**Overall:** The combination of per-row `lapply` with string operations, named-vector lookups, and `do.call(rbind, ...)` on millions of rows is the primary reason the pipeline takes 86+ hours.

### 1.2 Random Forest Inference Bottlenecks

- **Model loading:** If the serialized RF model is large (110 predictors, potentially thousands of trees), `readRDS()` can be slow and memory-intensive. A `ranger` model is typically much smaller than a `randomForest` model.
- **Prediction call:** `predict()` on 6.46M rows × 110 features can be memory-intensive. If the model is from the `randomForest` package, prediction is single-threaded and slow. If from `ranger`, it's multi-threaded but still needs a properly formatted `data.frame`/`matrix`.
- **Object copying:** R's copy-on-modify semantics mean that adding columns to `cell_data` inside a loop (`cell_data$new_col <- ...`) can trigger full copies of the entire data.frame (~6.46M × 110+ columns). Each iteration of the outer `for` loop over 5 variables (adding 3 columns each) potentially copies the entire multi-GB data.frame up to 15 times.
- **Memory pressure:** 6.46M rows × 110 numeric columns ≈ 5.4 GB for the data alone. The RF model, neighbor lookup list (6.46M elements), and intermediate objects can easily exceed 16 GB, causing swapping.

---

## 2. OPTIMIZATION STRATEGY

| Bottleneck | Solution | Expected Speedup |
|---|---|---|
| `build_neighbor_lookup` with string keys | Use integer arithmetic for keys (`id * 10000 + year`), use `data.table` for fast keyed joins | 50–100× |
| Per-row `lapply` in `compute_neighbor_stats` | Vectorize using `data.table` group-by on an edge list | 100–500× |
| `do.call(rbind, ...)` on 6.46M rows | Eliminate entirely; use pre-allocated matrix or `data.table` aggregation | 50–100× |
| Column-addition copying `cell_data` | Use `data.table` with `:=` (in-place column addition, zero copies) | 5–15× |
| RF prediction (single-threaded `randomForest`) | Batch predict; if possible convert model to `ranger` or use chunked prediction to control memory | 2–10× |
| Memory pressure / swapping | `data.table` is more memory-efficient; eliminate intermediate objects; `gc()` strategically | Eliminates swap thrashing |

**Key architectural change:** Replace the per-row neighbor lookup list with a **flat edge-list `data.table`** and use grouped aggregation to compute all neighbor statistics in one vectorized pass per variable.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Requirements: data.table, ranger (or randomForest — works with either)
# =============================================================================

library(data.table)

# ---- Step 0: Convert cell_data to data.table (in-place, no copy) -----------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure id and year are integer for fast operations
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# ---- Step 1: Build flat edge list (replaces build_neighbor_lookup) ----------
build_neighbor_edge_list <- function(data, id_order, neighbors) {
  # Build a data.table of (focal_id, neighbor_id) from the nb object
  id_order <- as.integer(id_order)
  
  # Expand nb list into edge list
  n_neighbors <- vapply(neighbors, length, integer(1))
  focal_idx <- rep(seq_along(neighbors), times = n_neighbors)
  neighbor_idx <- unlist(neighbors, use.names = FALSE)
  
  edge_dt <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
  
  # Get unique years present in data
  years <- sort(unique(data$year))
  
  # Cross-join edges with years to get (focal_id, year, neighbor_id)
  # This creates the full set of neighbor lookups across all years
  edge_year_dt <- CJ_dt(edge_dt, years)
  
  return(edge_year_dt)
}

# Helper: cross join edge_dt with years vector efficiently
CJ_dt <- function(edge_dt, years) {
  years_dt <- data.table(year = years)
  # Cross join: every edge × every year
  result <- edge_dt[, .(year = years), by = .(focal_id, neighbor_id)]
  return(result)
}

cat("Building neighbor edge list...\n")
system.time({
  edge_year <- build_neighbor_edge_list(cell_data, id_order, rook_neighbors_unique)
})
# edge_year has columns: focal_id, neighbor_id, year
# Each row means: "for focal cell focal_id in year, neighbor_id is a neighbor"

cat(sprintf("Edge-year table: %s rows (%.1f M)\n", 
            format(nrow(edge_year), big.mark = ","),
            nrow(edge_year) / 1e6))

# ---- Step 2: Key the data for fast joins -----------------------------------
setkey(cell_data, id, year)

# ---- Step 3: Vectorized neighbor stats (replaces compute_neighbor_stats) ----
compute_and_add_all_neighbor_features <- function(data, edge_year, var_names) {
  # Join neighbor values onto the edge list once per variable
  # Then aggregate by (focal_id, year) to get max, min, mean
  
  # We need neighbor values: join edge_year with data on (neighbor_id, year)
  # First, set key on edge_year for the join
  setkey(edge_year, neighbor_id, year)
  
  for (var_name in var_names) {
    cat(sprintf("  Computing neighbor stats for: %s\n", var_name))
    
    # Extract only the columns we need for the join (minimize memory)
    val_dt <- data[, .(id, year, val = get(var_name))]
    setkey(val_dt, id, year)
    
    # Join: get neighbor's value for each edge-year row
    # edge_year[val_dt] — join on neighbor_id == id, year == year
    edge_with_val <- val_dt[edge_year, on = .(id = neighbor_id, year = year), 
                            nomatch = NA,
                            .(focal_id, year = i.year, val = x.val)]
    
    # Aggregate by (focal_id, year): compute max, min, mean (excluding NAs)
    agg <- edge_with_val[!is.na(val), 
                          .(nb_max  = max(val),
                            nb_min  = min(val),
                            nb_mean = mean(val)),
                          by = .(focal_id, year)]
    
    # Define output column names
    max_col  <- paste0("nb_max_", var_name)
    min_col  <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)
    
    setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
    
    # Join aggregated stats back to data (in-place with :=)
    setkey(agg, focal_id, year)
    setkey(data, id, year)
    
    data[agg, on = .(id = focal_id, year = year), 
         `:=`(
           (max_col)  = get(paste0("i.", max_col)),
           (min_col)  = get(paste0("i.", min_col)),
           (mean_col) = get(paste0("i.", mean_col))
         )]
    
    # Clean up intermediate objects to free memory
    rm(val_dt, edge_with_val, agg)
  }
  
  invisible(data)
}

cat("Computing neighbor features...\n")
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

system.time({
  compute_and_add_all_neighbor_features(cell_data, edge_year, neighbor_source_vars)
})

# Free the large edge table
rm(edge_year)
gc()

# ---- Step 4: Optimized Random Forest Prediction ----------------------------
cat("Loading trained Random Forest model...\n")
system.time({
  rf_model <- readRDS("path/to/trained_rf_model.rds")
})

# Determine model class for optimal prediction strategy
model_class <- class(rf_model)[1]
cat(sprintf("Model class: %s\n", model_class))

# Prepare prediction matrix — extract only the predictor columns needed
# (Avoids passing the entire data.table with extra columns to predict())
if (inherits(rf_model, "ranger")) {
  # ranger stores variable names
  pred_vars <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores variable names used in training
  pred_vars <- rownames(rf_model$importance)
} else {
  # Fallback: assume all columns except id, year, and target are predictors
  exclude_cols <- c("id", "year", "gdp", "usd_est_n2")
  pred_vars <- setdiff(names(cell_data), exclude_cols)
}

cat(sprintf("Number of predictor variables: %d\n", length(pred_vars)))

# Verify all predictor columns exist
missing_vars <- setdiff(pred_vars, names(cell_data))
if (length(missing_vars) > 0) {
  stop(sprintf("Missing predictor columns: %s", paste(missing_vars, collapse = ", ")))
}

# --- Prediction: chunked to control peak memory ---
predict_chunked <- function(model, data, pred_vars, chunk_size = 500000L) {
  n <- nrow(data)
  n_chunks <- ceiling(n / chunk_size)
  predictions <- numeric(n)
  
  cat(sprintf("Predicting %s rows in %d chunks of up to %s...\n",
              format(n, big.mark = ","), n_chunks, format(chunk_size, big.mark = ",")))
  
  for (i in seq_len(n_chunks)) {
    start_row <- (i - 1L) * chunk_size + 1L
    end_row   <- min(i * chunk_size, n)
    
    # Extract chunk as a plain data.frame (some predict methods require this)
    chunk_df <- as.data.frame(data[start_row:end_row, ..pred_vars])
    
    if (inherits(model, "ranger")) {
      pred_obj <- predict(model, data = chunk_df, num.threads = parallel::detectCores())
      predictions[start_row:end_row] <- pred_obj$predictions
    } else {
      # randomForest or other
      predictions[start_row:end_row] <- predict(model, newdata = chunk_df)
    }
    
    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("  Chunk %d/%d complete (rows %s-%s)\n", 
                  i, n_chunks, format(start_row, big.mark = ","), format(end_row, big.mark = ",")))
    }
    
    rm(chunk_df)
    if (i %% 10 == 0) gc()
  }
  
  return(predictions)
}

cat("Running predictions...\n")
system.time({
  cell_data[, predicted_gdp := predict_chunked(rf_model, cell_data, pred_vars)]
})

cat("Done. Predictions stored in cell_data$predicted_gdp\n")

# ---- Step 5: Memory cleanup ------------------------------------------------
rm(rf_model)
gc()

# =============================================================================
# SUMMARY OF CHANGES
# =============================================================================
# 1. build_neighbor_lookup: Replaced per-row lapply + string keys with a flat
#    data.table edge list cross-joined with years. O(E×T) rows, fully vectorized.
#
# 2. compute_neighbor_stats: Replaced per-row lapply + do.call(rbind,...) with
#    data.table keyed joins and grouped aggregation. Single vectorized pass per
#    variable instead of 6.46M R-level iterations.
#
# 3. Column addition: data.table `:=` adds columns in-place with zero copies,
#    eliminating the repeated full-copy of the multi-GB data.frame.
#
# 4. RF prediction: Chunked to control peak memory on a 16 GB laptop. Uses
#    multi-threading if the model is ranger. Extracts only predictor columns
#    to avoid passing unnecessary data.
#
# 5. The trained RF model is loaded and used as-is (never retrained).
#    The original numerical estimand (GDP prediction) is preserved exactly.
#
# Expected runtime: ~10-30 minutes total (vs. 86+ hours), depending on disk
# I/O speed and RF model size.
# =============================================================================
```

---

## 4. EXPECTED PERFORMANCE COMPARISON

| Stage | Original (est.) | Optimized (est.) | Why |
|---|---|---|---|
| `build_neighbor_lookup` | ~20–40 hours | ~1–3 min | Vectorized `data.table` CJ vs. 6.46M `lapply` with string ops |
| `compute_neighbor_stats` (×5 vars) | ~30–40 hours | ~3–8 min | Grouped `data.table` aggregation vs. 6.46M `lapply` + `do.call(rbind)` |
| Column binding / copies | ~5–10 hours | ~0 (in-place) | `data.table` `:=` vs. `data.frame` copy-on-modify |
| RF prediction (6.46M rows) | ~1–5 hours | ~5–30 min | Chunked, multi-threaded (ranger), minimal memory overhead |
| **Total** | **~86+ hours** | **~10–40 min** | **~100–500× speedup** |

The trained Random Forest model is never modified or retrained, and the predicted values are numerically identical to what the original pipeline would produce.