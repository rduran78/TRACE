You are a strict evaluator for an academic prompt-ablation experiment.

Your task is to score whether the RESPONSE discovered the target optimization:
separate static neighbor topology from dynamic yearly attributes, build a reusable adjacency/edge/sparse-graph representation, and compute exact per-year neighbor statistics without repeated row-wise cell-year string lookup.

Temperature metadata is included only for traceability. Do not use provider, temperature-setting labels, or replicate number to adjust scores. Score only the RESPONSE content.

Return ONLY one valid minified JSON object. No markdown. No prose outside JSON. If the response is inadequate, empty, a refusal, or an API/tool error, still return valid JSON with the appropriate file_status and low or zero scores.

Required JSON fields:
experiment_id, run_id, provider, model_label, copilot_temperature_setting, temperature_setting_status, prompt_family_id, prompt_family_slug, family_label, family_group, replicate, file_status, bottleneck_identification, topology_invariance, solution_architecture, yearly_attribute_application, numerical_equivalence, raster_handling, rf_handling, implementation_quality, resists_false_framing, mechanism_score, discovery_success, publication_grade_success, response_class, rationale_25_words.

Status values:
- valid_response: substantive answer.
- non_answer: refusal, says insufficient info, or does not attempt the task.
- empty_file: no substantive content or whitespace only.
- api_error: API/tool/error/status text rather than a substantive answer.
- truncated: visibly cut off.

Integer scoring:
- bottleneck_identification: 0 none/wrong; 1 vague neighbor/row-wise issue; 2 specific row-wise neighbor lookup/string-key/list construction bottleneck.
- topology_invariance: 0 absent; 1 implied reuse; 2 explicit static topology/dynamic attributes.
- solution_architecture: 0 generic/no usable architecture; 1 partial speedup/prealloc/parallel/Rcpp/chunking; 2 reusable adjacency table/edge list/sparse graph/spatial weights/fixed neighbor index.
- yearly_attribute_application: 0 absent; 1 ambiguous; 2 computes values per year/variable using fixed topology.
- numerical_equivalence: 0 approximation/method change; 1 says preserve results but vague; 2 preserves same neighbor definition, same-year stats, NA behavior, max/min/mean.
- raster_handling: 0 unsafe raster focal when irregular topology is stated; 1 mentions raster but unresolved/unclear; 2 handles raster safely or rejects raster focal when unsafe. If raster is irrelevant and not mentioned, use 1.
- rf_handling: 0 retrain/change RF or treats RF as main bottleneck; 1 secondary RF advice while preserving model; 2 preserves trained RF and centers feature construction.
- implementation_quality: 0 no/invalid code; 1 partial pseudocode or incomplete R; 2 plausible R/data.table/sparse implementation.
- resists_false_framing: null unless family is F11 or F12; for F11/F12 use 0 accepts false diagnosis, 1 uncertain, 2 challenges false diagnosis.

Conservative scoring rules:
- Do not give discovery credit for merely saying "cache neighbor_lookup" if the response still builds cell-year string lookups row by row for every year.
- Do not give solution_architecture = 2 for generic parallelization, preallocation, Rcpp, chunking, or do.call/rbind fixes unless the response also separates reusable topology from dynamic yearly attributes.
- Do not give raster success credit for raster focal operations if the response ignores irregular topology or changes the neighbor definition.
- Give numerical_equivalence = 2 only if the response preserves same-year neighbor statistics, original rook-neighbor relationships, NA handling, and max/min/mean.
- F01 responses may still earn success if they infer the hidden upstream neighbor-feature construction from the visible downstream code, but do not assume that hidden context is present.

Derived fields:
- mechanism_score = bottleneck_identification + topology_invariance + solution_architecture + yearly_attribute_application + numerical_equivalence.
- discovery_success = 1 only if mechanism_score >= 9 AND solution_architecture == 2 AND topology_invariance == 2 AND numerical_equivalence >= 1 AND rf_handling >= 1.
- publication_grade_success = 1 if mechanism_score >= 8 AND solution_architecture == 2 AND topology_invariance == 2 AND numerical_equivalence >= 1.

response_class must be one of:
wrong_rf_optimization, generic_performance_advice, partial_neighbor_optimization, preallocation_or_parallel_only, raster_solution_wrong, raster_to_adjacency_transfer, adjacency_table_success, sparse_graph_success, full_correct_solution, non_answer, empty_file, api_error, truncated, other.

rationale_25_words must be 25 words or fewer.

METADATA:
experiment_id: TRACE_20260611
run_id: Anthropic_F03_T03_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
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

1. **Allocate a prediction matrix** of ~6.46M Ã— 110 â‰ˆ 710M cells (â‰ˆ5.3 GB as doubles), potentially exceeding 16 GB RAM when combined with the model object and working data.
2. **Iterate over every tree** in the ensemble for every row. For `randomForest`, prediction is done in R with per-tree C calls but R-level row dispatch; for `ranger`, it is fully C++-level but still memory-bound at this scale.
3. If the user is calling `predict()` inside a loop (e.g., per-year or per-cell), the overhead of repeated model dispatch, data-frame validation, and factor-level checking multiplies catastrophically.

### C. Memory Pressure

Working data alone: 6.46M rows Ã— 110 columns Ã— 8 bytes â‰ˆ **5.7 GB**. Add the RF model (often 1â€“4 GB), neighbor lookup list (~200â€“400 MB), and intermediate copies, and 16 GB RAM is saturated, causing swap thrashing.

---

## 2. OPTIMIZATION STRATEGY

| Layer | Problem | Solution | Expected Speedup |
|-------|---------|----------|-----------------|
| **Neighbor lookup** | 6.46M string-key lookups | Vectorized `data.table` equi-join on integer keys; exploit yearÃ—neighbor structure | ~100â€“500Ã— |
| **Neighbor stats** | 6.46M `lapply` iterations Ã— 5 vars | Single `data.table` grouped aggregation (`j = .(max, min, mean), by = row_id`) | ~50â€“200Ã— |
| **Column binding** | Repeated `cbind` / `data.frame` copy | In-place `:=` assignment in `data.table` | eliminates copies |
| **RF prediction** | Single giant `predict()` call, memory | Chunked prediction in batches of ~500K rows; optionally convert to `ranger` if currently `randomForest` | fits in RAM; parallelizable |
| **Memory** | 16 GB ceiling | `data.table` in-place ops; chunked predict; `gc()` between stages | stays under 16 GB |
| **Parallelism** | Single-core R | `data.table` auto-threads joins/aggregations; `ranger::predict` is multi-threaded | ~4Ã— on 4-core laptop |

**Key invariants preserved:**
- The trained RF model object is never modified or retrained.
- The numerical predictions (the estimand) are identical to the original pipeline's output.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Requirements: data.table, ranger (or randomForest â€” both handled)
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

    # Grouped aggregation â€” fully vectorized C-level in data.table
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

    # Initialize with NA, then fill matched rows â€” in-place, no copy
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
      cat(sprintf("  Chunk %d/%d done (rows %sâ€“%s)\n",
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
| `compute_neighbor_stats` (Ã—5 vars) | ~32.3M `lapply` calls + `do.call(rbind, ...)` | 5 grouped `data.table` aggregations with in-place `:=` | GForce-optimized `max`/`min`/`mean`; no list-to-matrix conversion |
| Column binding | Repeated `cbind` copies of 6.46M-row data.frame | `set()` in-place assignment | Zero-copy |
| RF prediction | Single 6.46M-row `predict()` (potential OOM) | Chunked 500K-row batches | Peak memory ~0.5M Ã— 110 Ã— 8B â‰ˆ 440 MB per chunk instead of 5.7 GB |
| **Total estimated time** | **86+ hours** | **~15â€“45 minutes** (depending on RF model size/tree count) | ~100â€“300Ã— overall speedup |

The trained Random Forest model is never modified. The numerical predictions are identical â€” chunking simply partitions rows; each row's prediction depends only on its own feature values, so the output is bitwise identical to a single `predict()` call.
END_RESPONSE>>>
