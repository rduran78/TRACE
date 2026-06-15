 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

1. **Row-level `lapply` over 6.46M rows.** Each iteration performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. This is ~6.46 million R-level function calls with repeated allocation.
2. **String-keyed lookups are O(n) hash probes on a 6.46M-element named vector.** The `idx_lookup` named vector has 6.46M entries; each probe is expensive at that scale, and each row probes multiple neighbors.
3. **Redundant recomputation.** Every cell's neighbors are the same across all 28 years, but the lookup is rebuilt from scratch for every cell-year combination rather than exploiting the temporal structure.

**`compute_neighbor_stats`** is the second bottleneck:

1. **Another `lapply` over 6.46M rows**, each extracting a variable-length subset of a numeric vector, removing NAs, and computing three summary statistics. This is called 5 times (once per neighbor source variable), totaling ~32.3M R-level iterations.
2. **`do.call(rbind, result)` on a 6.46M-element list of 3-vectors** is a well-known R anti-pattern that creates enormous intermediate object churn.

**Combined cost estimate:** The two functions together perform roughly **38.8 million interpreted R function calls** with heavy string manipulation and list allocation. On a standard laptop this easily accounts for the 86+ hour estimate.

### B. Random Forest Inference Bottleneck

With ~6.46M rows and ~110 predictors, a single `predict()` call on a `ranger` or `randomForest` model will:

1. **Allocate a prediction matrix** of ~6.46M × 110 ≈ 710M cells (≈5.3 GB as doubles), potentially exceeding 16 GB RAM when combined with the model object and working data.
2. **Iterate over every tree** in the ensemble for every row. For `randomForest`, prediction is done in R with per-tree C calls but R-level row dispatch; for `ranger`, it is fully C++-level but still memory-bound at this scale.
3. If the user is calling `predict()` inside a loop (e.g., per-year or per-cell), the overhead of repeated model dispatch, data-frame validation, and factor-level checking multiplies catastrophically.

### C. Memory Pressure

Working data alone: 6.46M rows × 110 columns × 8 bytes ≈ **5.7 GB**. Add the RF model (often 1–4 GB), neighbor lookup list (~200–400 MB), and intermediate copies, and 16 GB RAM is saturated, causing swap thrashing.

---

## 2. OPTIMIZATION STRATEGY

| Layer | Problem | Solution | Expected Speedup |
|-------|---------|----------|-----------------|
| **Neighbor lookup** | 6.46M string-key lookups | Vectorized `data.table` equi-join on integer keys; exploit year×neighbor structure | ~100–500× |
| **Neighbor stats** | 6.46M `lapply` iterations × 5 vars | Single `data.table` grouped aggregation (`j = .(max, min, mean), by = row_id`) | ~50–200× |
| **Column binding** | Repeated `cbind` / `data.frame` copy | In-place `:=` assignment in `data.table` | eliminates copies |
| **RF prediction** | Single giant `predict()` call, memory | Chunked prediction in batches of ~500K rows; optionally convert to `ranger` if currently `randomForest` | fits in RAM; parallelizable |
| **Memory** | 16 GB ceiling | `data.table` in-place ops; chunked predict; `gc()` between stages | stays under 16 GB |
| **Parallelism** | Single-core R | `data.table` auto-threads joins/aggregations; `ranger::predict` is multi-threaded | ~4× on 4-core laptop |

**Key invariants preserved:**
- The trained RF model object is never modified or retrained.
- The numerical predictions (the estimand) are identical to the original pipeline's output.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Requirements: data.table, ranger (or randomForest — both handled)
# =============================================================================

library(data.table)

# ---- 0. Convert working data to data.table (once) --------------------------

if (!is.data.table(cell_data)) {
  setDT(cell_data)  # in-place conversion, no copy
}

# Ensure key columns are integer for fast joins
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# Add a row index (used later for joining stats back)
cell_data[, .row_id := .I]


# =============================================================================
# STEP 1: BUILD NEIGHBOR EDGE LIST (vectorized, replaces build_neighbor_lookup)
# =============================================================================

build_neighbor_edgelist <- function(id_order, neighbors) {
  # id_order : integer vector of cell IDs in the order matching `neighbors`
  # neighbors: spdep nb object (list of integer index vectors)
  #
  # Returns a data.table with columns: focal_id, neighbor_id
  # This is year-independent; we cross-join with years later.

  n <- length(neighbors)
  # Pre-allocate lengths
  lens <- vapply(neighbors, length, integer(1))
  total <- sum(lens)

  focal_ref    <- rep.int(seq_len(n), lens)
  neighbor_ref <- unlist(neighbors, use.names = FALSE)

  data.table(
    focal_id    = id_order[focal_ref],
    neighbor_id = id_order[neighbor_ref]
  )
}

cat("Building neighbor edge list...\n")
edge_dt <- build_neighbor_edgelist(id_order, rook_neighbors_unique)

# Cross-join edges with all years present in the data
all_years <- sort(unique(cell_data$year))
edge_year_dt <- edge_dt[, .(year = all_years), by = .(focal_id, neighbor_id)]

# This is the full (focal_id, year) -> (neighbor_id, year) mapping.
# Now join to get the row index of each neighbor observation.

# Key the main data for fast join
setkey(cell_data, id, year)

# Map neighbor_id + year -> .row_id of the neighbor row
neighbor_row_map <- cell_data[, .(neighbor_id = id, year, neighbor_row = .row_id)]
setkey(neighbor_row_map, neighbor_id, year)
setkey(edge_year_dt, neighbor_id, year)

edge_year_dt <- neighbor_row_map[edge_year_dt, nomatch = 0L]

# Map focal_id + year -> .row_id of the focal row
focal_row_map <- cell_data[, .(focal_id = id, year, focal_row = .row_id)]
setkey(focal_row_map, focal_id, year)
setkey(edge_year_dt, focal_id, year)

edge_year_dt <- focal_row_map[edge_year_dt, nomatch = 0L]

# Now edge_year_dt has columns: focal_row, neighbor_row (and ids/year)
# Key by focal_row for grouped aggregation
setkey(edge_year_dt, focal_row)

cat(sprintf("Edge-year table: %s rows\n", format(nrow(edge_year_dt), big.mark = ",")))

# Clean up intermediates
rm(neighbor_row_map, focal_row_map, edge_dt)
gc()


# =============================================================================
# STEP 2: COMPUTE NEIGHBOR STATS (vectorized, replaces compute_neighbor_stats)
# =============================================================================

compute_and_add_all_neighbor_features <- function(cell_data, edge_year_dt,
                                                   neighbor_source_vars) {
  # For each variable, compute max/min/mean of neighbor values in one
  # vectorized data.table aggregation, then join back.

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Computing neighbor stats for: %s\n", var_name))

    # Attach the neighbor's value to each edge row
    edge_year_dt[, nval := cell_data[[var_name]][neighbor_row]]

    # Grouped aggregation — fully vectorized C-level in data.table
    stats <- edge_year_dt[
      !is.na(nval),
      .(
        nb_max  = max(nval),
        nb_min  = min(nval),
        nb_mean = mean(nval)
      ),
      by = focal_row
    ]

    # Prepare target column names (match original pipeline naming convention)
    max_col  <- paste0("nb_max_",  var_name)
    min_col  <- paste0("nb_min_",  var_name)
    mean_col <- paste0("nb_mean_", var_name)

    # Initialize with NA, then fill matched rows — in-place, no copy
    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)

    set(cell_data, i = stats$focal_row, j = max_col,  value = stats$nb_max)
    set(cell_data, i = stats$focal_row, j = min_col,  value = stats$nb_min)
    set(cell_data, i = stats$focal_row, j = mean_col, value = stats$nb_mean)

    # Drop temporary column
    edge_year_dt[, nval := NULL]
  }

  invisible(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
compute_and_add_all_neighbor_features(cell_data, edge_year_dt, neighbor_source_vars)

# Free the large edge table
rm(edge_year_dt)
gc()


# =============================================================================
# STEP 3: CHUNKED RANDOM FOREST PREDICTION (memory-safe)
# =============================================================================

chunked_rf_predict <- function(model, newdata, feature_names,
                                chunk_size = 500000L) {
  # Works with both ranger and randomForest model objects.
  # Preserves exact numerical output (no approximation).

  n <- nrow(newdata)
  n_chunks <- ceiling(n / chunk_size)
  preds <- numeric(n)

  is_ranger <- inherits(model, "ranger")

  cat(sprintf("Predicting %s rows in %d chunks of up to %s...\n",
              format(n, big.mark = ","), n_chunks,
              format(chunk_size, big.mark = ",")))

  for (ch in seq_len(n_chunks)) {
    i_start <- (ch - 1L) * chunk_size + 1L
    i_end   <- min(ch * chunk_size, n)
    idx     <- i_start:i_end

    # Extract only the needed columns for this chunk (minimizes memory)
    chunk_df <- as.data.frame(newdata[idx, ..feature_names])

    if (is_ranger) {
      preds[idx] <- predict(model, data = chunk_df)$predictions
    } else {
      # randomForest
      preds[idx] <- predict(model, newdata = chunk_df)
    }

    if (ch %% 5 == 0 || ch == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %s–%s)\n",
                  ch, n_chunks,
                  format(i_start, big.mark = ","),
                  format(i_end, big.mark = ",")))
    }

    rm(chunk_df)
    if (ch %% 10 == 0) gc()
  }

  preds
}

# ---- Identify feature columns (exclude id, year, target, row_id) -----------
# Adjust 'target_var' to whatever your response variable is called.
target_var <- "gdp"  # <-- change if needed
exclude_cols <- c("id", "year", ".row_id", target_var)

feature_names <- setdiff(names(cell_data), exclude_cols)
# Verify count matches expectation
cat(sprintf("Number of predictor features: %d\n", length(feature_names)))

# ---- Load model (assumed already in memory or on disk) ----------------------
# rf_model <- readRDS("path/to/trained_rf_model.rds")  # uncomment if needed

# ---- Run prediction ---------------------------------------------------------
cat("Starting Random Forest prediction...\n")
cell_data[, predicted_gdp := chunked_rf_predict(
  model         = rf_model,
  newdata       = cell_data,
  feature_names = feature_names,
  chunk_size    = 500000L
)]

cat("Pipeline complete.\n")
```

---

## 4. SUMMARY OF EXPECTED IMPROVEMENTS

| Stage | Original | Optimized | Reason |
|---|---|---|---|
| `build_neighbor_lookup` | ~6.46M `lapply` calls with string ops | Single vectorized `data.table` join | Eliminates all string-key hashing; C-level radix join |
| `compute_neighbor_stats` (×5 vars) | ~32.3M `lapply` calls + `do.call(rbind, ...)` | 5 grouped `data.table` aggregations with in-place `:=` | GForce-optimized `max`/`min`/`mean`; no list-to-matrix conversion |
| Column binding | Repeated `cbind` copies of 6.46M-row data.frame | `set()` in-place assignment | Zero-copy |
| RF prediction | Single 6.46M-row `predict()` (potential OOM) | Chunked 500K-row batches | Peak memory ~0.5M × 110 × 8B ≈ 440 MB per chunk instead of 5.7 GB |
| **Total estimated time** | **86+ hours** | **~15–45 minutes** (depending on RF model size/tree count) | ~100–300× overall speedup |

The trained Random Forest model is never modified. The numerical predictions are identical — chunking simply partitions rows; each row's prediction depends only on its own feature values, so the output is bitwise identical to a single `predict()` call.