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
run_id: Anthropic_F03_T07_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys for every neighbor of every row
- Named-vector lookup into `idx_lookup` (a named character vector of length 6.46M â€” each lookup is O(n) hash probe on a very large vector)
- NA filtering and `as.integer` coercion

**Cost:** ~6.46M iterations Ã— ~4 neighbors each â‰ˆ 25.8M string-paste + hash-lookup operations on a 6.46M-entry named vector. This alone can take many hours.

**`compute_neighbor_stats`:** Calls `lapply` over 6.46M rows, subsetting a numeric vector, removing NAs, and computing max/min/mean. The `do.call(rbind, result)` at the end binds 6.46M 3-element vectors into a matrix â€” this is a known R anti-pattern that is extremely slow for millions of elements.

**Outer loop:** Calls `compute_neighbor_stats` 5 times (once per variable), so the 6.46M-row lapply + rbind penalty is paid 5 times. Additionally, `compute_and_add_neighbor_features` likely copies the entire `cell_data` data.frame on each assignment (`cell_data <- ...`), which for ~6.46M Ã— 110+ columns is a multi-GB copy each time.

### 1.2 Prediction-Workflow Bottlenecks

- **Model loading:** If the serialized Random Forest is large (hundreds of MB to several GB for 110 predictors on millions of rows), `readRDS` and deserialization is a one-time but significant cost.
- **Prediction in a loop:** If `predict()` is called row-by-row or in small batches rather than on the full matrix at once, overhead per call dominates.
- **Object copying:** R's copy-on-modify semantics mean that modifying `cell_data` inside a loop (adding columns) triggers full data.frame copies.
- **Memory pressure:** 6.46M rows Ã— 110 columns Ã— 8 bytes â‰ˆ 5.7 GB just for the numeric matrix. The Random Forest object, neighbor lookup list, and intermediate copies can easily exceed 16 GB, causing swap thrashing.

### 1.3 Root-Cause Summary

| Bottleneck | Estimated Share | Cause |
|---|---|---|
| `build_neighbor_lookup` (string ops on 6.46M rows) | ~30% | `paste`/named-vector lookups in R loop |
| `compute_neighbor_stats` (lapply + do.call rbind) | ~25% | Per-row R-level loop, slow rbind |
| Data.frame copies in outer loop | ~15% | Copy-on-modify, 5 iterations |
| Prediction loop (if row/batch-wise) | ~20% | R-level predict overhead per call |
| Memory thrashing / GC | ~10% | >16 GB working set on 16 GB machine |

---

## 2. OPTIMIZATION STRATEGY

### Principle: Vectorize, use `data.table`, pre-build integer index matrices, compute all neighbor stats in one pass, predict in one call.

| Problem | Solution |
|---|---|
| String-key lookups in `build_neighbor_lookup` | Build an integer matrix mapping each row to its neighbor-row indices using `data.table` equi-joins â€” no strings, no named vectors |
| Per-row `lapply` in `compute_neighbor_stats` | Use the integer index matrix to do column-wise vectorized aggregation via `data.table` grouping or matrix indexing with `rowMeans`/`pmin`/`pmax` on a pre-extracted neighbor-value matrix |
| `do.call(rbind, ...)` on 6.46M elements | Pre-allocate matrix; or avoid entirely with vectorized path |
| Data.frame copy on each variable | Use `data.table` set-by-reference (`:=`) â€” zero copies |
| Prediction loop | Single `predict(model, newdata)` call on full matrix |
| Memory pressure | Convert to matrix for predict; drop intermediate objects; `gc()` strategically; process neighbor stats for all 5 variables in one pass over the index structure |

**Expected speedup:** From ~86+ hours to ~10â€“30 minutes.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” Feature Preparation + Random Forest Prediction
# =============================================================================
# Requirements: data.table, ranger (or randomForest â€” works with both)
# The trained RF model object is assumed to be on disk as "rf_model.rds".
# cell_data is assumed to be a data.frame/data.table with columns:
#   id, year, ntl, ec, pop_density, def, usd_est_n2, ... (all predictors)
# id_order: integer vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an nb object (list of integer index vectors)
# =============================================================================

library(data.table)

# ---- Step 0: Convert cell_data to data.table in place --------------------
setDT(cell_data)

# ---- Step 1: Build neighbor lookup as an integer-indexed edge list --------
#
# Goal: for every row in cell_data, find the row indices of its rook neighbors
# in the same year.  We avoid all string operations.
#
# Sub-step 1a: Map cell id -> position in id_order (integer vector, not named)

build_neighbor_edgelist <- function(id_order, neighbors) {
  # neighbors is an nb object: neighbors[[i]] gives integer indices into

  # id_order for the neighbors of id_order[i].
  # We expand this into a two-column data.table: (cell_id, neighbor_cell_id)
  n <- length(id_order)
  from <- rep.int(id_order, lengths(neighbors))
  to   <- id_order[unlist(neighbors, use.names = FALSE)]
  data.table(cell_id = from, neighbor_id = to)
}

cat("Building neighbor edge list...\n")
edge_dt <- build_neighbor_edgelist(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %s rows\n", format(nrow(edge_dt), big.mark = ",")))

# Sub-step 1b: Add row-index to cell_data and join
#   We need to map (cell_id, year) -> row_index for both the focal cell and
#   its neighbors.

cell_data[, row_idx := .I]

# Create a keyed lookup: (id, year) -> row_idx
lookup <- cell_data[, .(cell_id = id, year, row_idx)]
setkey(lookup, cell_id, year)

# Sub-step 1c: For each edge (cell_id, neighbor_id), cross with all years
#   of the focal cell, then find the neighbor's row in the same year.
#
#   But expanding 1.37M edges Ã— 28 years = 38.4M rows is feasible and fast.
#
#   More memory-efficient: join edges onto the focal cell's (id, year, row_idx),
#   then join the neighbor side.

cat("Joining edges with years...\n")

# Focal side: every (cell_id, year, focal_row_idx) that exists in the data
focal <- cell_data[, .(cell_id = id, year, focal_row = row_idx)]
setkey(focal, cell_id)

# Join: for each focal row, attach its neighbor cell IDs
# edge_dt is keyed on cell_id
setkey(edge_dt, cell_id)
expanded <- edge_dt[focal, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
# expanded now has columns: cell_id, neighbor_id, year, focal_row

# Now find the neighbor's row index in the same year
setkey(expanded, neighbor_id, year)
setkey(lookup, cell_id, year)
expanded <- lookup[expanded, on = c(cell_id = "neighbor_id", "year"), nomatch = 0L]
# After this join, 'row_idx' is the neighbor's row index, 'focal_row' is the focal row index

neighbor_map <- expanded[, .(focal_row, neighbor_row = row_idx)]

# Clean up large intermediates
rm(focal, expanded, lookup, edge_dt)
gc()

cat(sprintf("  Neighbor map: %s pairs\n", format(nrow(neighbor_map), big.mark = ",")))

# ---- Step 2: Compute neighbor stats (max, min, mean) for all variables ----
#
# Strategy: group neighbor_map by focal_row, extract neighbor values from the
# column, and compute stats â€” all vectorized inside data.table.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))

  # Extract the variable as a plain numeric vector (fast column access)
  vals <- cell_data[[var_name]]

  # Attach neighbor values to the map
  neighbor_map[, nval := vals[neighbor_row]]

  # Compute grouped stats â€” this is the core vectorized aggregation
  stats <- neighbor_map[!is.na(nval),
    .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ),
    by = focal_row
  ]

  # Assign back to cell_data by reference (zero-copy)
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  # Initialize with NA
  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)

  # Fill in computed values
  set(cell_data, i = stats$focal_row, j = max_col,  value = stats$nb_max)
  set(cell_data, i = stats$focal_row, j = min_col,  value = stats$nb_min)
  set(cell_data, i = stats$focal_row, j = mean_col, value = stats$nb_mean)

  rm(stats)
}

# Clean up the temporary column
neighbor_map[, nval := NULL]

cat("Neighbor features complete.\n")
gc()

# ---- Step 3: Load the trained Random Forest model ------------------------

cat("Loading trained Random Forest model...\n")
rf_model <- readRDS("rf_model.rds")

# ---- Step 4: Predict in a single vectorized call -------------------------
#
# Identify the predictor columns the model expects.
# For ranger:
if (inherits(rf_model, "ranger")) {
  pred_vars <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores variable names used during training
  pred_vars <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all predictor columns exist
missing_vars <- setdiff(pred_vars, names(cell_data))
if (length(missing_vars) > 0) {
  stop("Missing predictor columns: ", paste(missing_vars, collapse = ", "))
}

# Extract predictor matrix â€” convert only needed columns to avoid memory bloat
cat("Preparing prediction matrix...\n")
pred_data <- cell_data[, ..pred_vars]

# Predict â€” single call, no loop
cat("Running prediction on all rows...\n")

if (inherits(rf_model, "ranger")) {
  # ranger::predict returns a list; predictions in $predictions
  pred_result <- predict(rf_model, data = pred_data)
  cell_data[, predicted_gdp := pred_result$predictions]
} else {
  # randomForest::predict returns a vector directly
  cell_data[, predicted_gdp := predict(rf_model, newdata = pred_data)]
}

rm(pred_data)
gc()

cat("Prediction complete.\n")

# ---- Step 5 (optional): If memory is tight, batch prediction -------------
#
# If the single predict() call causes an out-of-memory error on a 16 GB
# laptop, use the following batched version instead of Step 4's predict block:

run_batched_prediction <- function(model, dt, pred_vars, batch_size = 500000L) {
  n <- nrow(dt)
  preds <- numeric(n)
  starts <- seq(1L, n, by = batch_size)

  for (k in seq_along(starts)) {
    i1 <- starts[k]
    i2 <- min(i1 + batch_size - 1L, n)
    cat(sprintf("  Batch %d/%d: rows %sâ€“%s\n", k, length(starts),
                format(i1, big.mark = ","), format(i2, big.mark = ",")))

    batch <- dt[i1:i2, ..pred_vars]

    if (inherits(model, "ranger")) {
      preds[i1:i2] <- predict(model, data = batch)$predictions
    } else {
      preds[i1:i2] <- predict(model, newdata = batch)
    }

    rm(batch)
    if (k %% 3 == 0) gc()
  }
  preds
}

# Uncomment the following to use batched prediction instead:
# cell_data[, predicted_gdp := run_batched_prediction(rf_model, cell_data, pred_vars)]

# ---- Step 6: Output / save -----------------------------------------------
cat(sprintf("Rows predicted: %s\n", format(sum(!is.na(cell_data$predicted_gdp)), big.mark = ",")))

# Save results
# fwrite(cell_data, "cell_data_with_predictions.csv")
# or
# saveRDS(cell_data, "cell_data_with_predictions.rds")
```

---

## 4. WHY THIS IS FASTER â€” ITEMIZED

| Original | Optimized | Speedup Factor |
|---|---|---|
| `paste` + named-vector lookup on 6.46M rows in `lapply` | Integer equi-join via `data.table` (radix-sorted merge) | ~100â€“500Ã— |
| `lapply` over 6.46M rows for neighbor stats | `data.table` grouped aggregation (`[, .(max, min, mean), by = focal_row]`) â€” internally parallelized C code | ~50â€“200Ã— |
| `do.call(rbind, list_of_6.46M_vectors)` | Eliminated entirely; stats written directly via `set()` | âˆž (removed) |
| 5 full data.frame copies (`cell_data <- ...`) | Zero copies â€” `:=` and `set()` modify in place | ~5Ã— memory, ~10Ã— time |
| Prediction in a loop (assumed) | Single `predict()` call on full matrix (or large batches) | ~100â€“1000Ã— |
| ~86+ hours estimated | ~10â€“30 minutes expected | ~170â€“500Ã— overall |

**Key invariants preserved:**
- The trained Random Forest model is loaded from disk and never retrained.
- The original numerical estimand (predicted GDP) is identical â€” the same features are computed with the same arithmetic (max, min, mean of non-NA neighbor values), and the same `predict()` function is called on the same model.
- All 15 neighbor-derived features (3 stats Ã— 5 variables) are computed identically to the original logic, including NA handling.
END_RESPONSE>>>
