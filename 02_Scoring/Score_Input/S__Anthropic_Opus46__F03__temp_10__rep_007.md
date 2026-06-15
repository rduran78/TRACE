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
run_id: Anthropic_F03_T10_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the single worst offender:

| Problem | Explanation |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration performs character coercion, `paste`, named-vector lookup, and NA filtering. This is ~6.46M R-level function calls with character-key hashing. |
| **Character-keyed lookups (`idx_lookup`)** | A named vector of 6.46M entries is created with `paste(id, year)` keys. Every lookup does a linear-hash string match â€” extremely slow at this scale. |
| **Redundant per-row `paste` and `as.character`** | String operations inside a hot loop over millions of rows dominate wall time. |
| **Output is a list of 6.46M integer vectors** | ~6.46M list elements, each heap-allocated, creating massive memory pressure and GC overhead. |

**`compute_neighbor_stats`** is the second bottleneck:

| Problem | Explanation |
|---|---|
| **`lapply` over 6.46M list elements** | Each call subsets a numeric vector, removes NAs, and computes max/min/mean â€” all interpreted R. |
| **`do.call(rbind, result)` on 6.46M single-row matrices** | This is an O(nÂ²) concatenation pattern. |

**Together, these two functions are called 5 times (once per neighbor source variable), so the total cost is roughly 5 Ã— (6.46M Ã— per-row-work).**

### B. Random Forest Inference Bottleneck

| Problem | Explanation |
|---|---|
| **Predicting 6.46M rows with ~110 features** | A `ranger` or `randomForest` `predict()` call on a 6.46M Ã— 110 matrix is memory-intensive (~5.4 GB for a dense numeric matrix alone) and CPU-intensive. |
| **Possible row-by-row or chunk-less prediction** | If the user is calling `predict()` inside a loop rather than vectorised, overhead is catastrophic. |
| **Model object size** | A large RF model can be several GB; loading it from disk repeatedly or copying it wastes RAM. |
| **`data.frame` to matrix conversion** | `predict.ranger` / `predict.randomForest` internally coerces; if the input is a data.frame with character/factor columns, coercion is slow. |

### C. Overall Memory Pressure

With 16 GB RAM, holding the data (~6.46M Ã— 110 Ã— 8 bytes â‰ˆ 5.4 GB), the neighbor lookup list (~1 GB+), the model (1â€“4 GB), and prediction output simultaneously risks swapping to disk, which alone could explain the 86+ hour estimate.

---

## 2. Optimization Strategy

| Layer | Strategy | Expected Speedup |
|---|---|---|
| **Neighbor lookup** | Replace character-key lookups with integer-key lookups using `data.table` joins. Build a single CSR (Compressed Sparse Row) representation of the neighbor graph expanded across years, all vectorised. | 50â€“200Ã— |
| **Neighbor stats** | Replace per-row `lapply` with vectorised grouped aggregation via `data.table`, keyed on the CSR edge list. | 50â€“100Ã— |
| **Memory** | Use in-place `:=` column assignment in `data.table`; avoid copying the entire data frame 5 times. | 2â€“4Ã— (avoids OOM) |
| **RF Prediction** | Load the model once; predict in a single vectorised call on a pre-built numeric matrix; use `ranger` if possible (much faster than `randomForest`). | 5â€“20Ã— |
| **Overall** | Target < 30 minutes total on a 16 GB laptop. | ~150â€“300Ã— |

---

## 3. Working R Code

```r
# =============================================================================
# 0. LIBRARIES
# =============================================================================
library(data.table)
# library(ranger)      # if the trained model is a ranger object
# library(randomForest) # if the trained model is a randomForest object

# =============================================================================
# 1. LOAD DATA AND MODEL (do this ONCE)
# =============================================================================
# cell_data            <- as read / loaded previously
# rf_model             <- readRDS("path/to/trained_rf_model.rds")   # load ONCE
# rook_neighbors_unique <- readRDS("path/to/rook_neighbors.rds")     # spdep nb
# id_order             <- as defined previously (vector of cell IDs)

# Convert to data.table in place â€” no copy
setDT(cell_data)

# Ensure integer types for join keys
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# Create a row index for fast positional access
cell_data[, .row_idx := .I]


# =============================================================================
# 2. BUILD VECTORISED NEIGHBOR EDGE LIST (replaces build_neighbor_lookup)
# =============================================================================
build_neighbor_edgelist <- function(cell_data, id_order, nb_obj) {
  # --- Step A: Build cell-level directed edge list from the nb object --------
  # nb_obj[[i]] gives the neighbor indices (into id_order) for id_order[i]
  n_cells   <- length(id_order)
  from_idx  <- rep.int(seq_len(n_cells), lengths(nb_obj))
  to_idx    <- unlist(nb_obj, use.names = FALSE)

  # Map positional indices to actual cell IDs
  edges <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )

  # --- Step B: Cross-join with years to get cell-year level edges ------------
  years <- sort(unique(cell_data$year))

  # Efficient cross join: replicate edge list for every year
  # This produces ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” fits in RAM easily
  edges_cy <- edges[, .(year = years), by = .(from_id, to_id)]

  # --- Step C: Attach row indices of the TARGET (neighbor) rows --------------
  # We need the row index in cell_data for each (to_id, year) pair
  idx_map <- cell_data[, .(to_id = id, year, .neighbor_row = .row_idx)]
  setkey(idx_map, to_id, year)
  setkey(edges_cy, to_id, year)

  edges_cy <- idx_map[edges_cy, nomatch = 0L]

  # --- Step D: Attach row indices of the SOURCE (focal cell) rows ------------
  idx_map_from <- cell_data[, .(from_id = id, year, .focal_row = .row_idx)]
  setkey(idx_map_from, from_id, year)
  setkey(edges_cy, from_id, year)

  edges_cy <- idx_map_from[edges_cy, nomatch = 0L]

  # Keep only what we need
  edges_cy <- edges_cy[, .(.focal_row, .neighbor_row)]

  setkey(edges_cy, .focal_row)
  return(edges_cy)
}

cat("Building neighbor edge list...\n")
system.time({
  neighbor_edges <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
})
# Expected: ~10â€“30 seconds, ~2â€“3 GB


# =============================================================================
# 3. VECTORISED NEIGHBOR STATS (replaces compute_neighbor_stats)
# =============================================================================
compute_and_add_neighbor_features_fast <- function(cell_data, var_name, neighbor_edges) {
  # Extract the neighbor values via positional indexing (vectorised)
  neighbor_edges[, .val := cell_data[[var_name]][.neighbor_row]]

  # Grouped aggregation â€” one pass, fully vectorised
  stats <- neighbor_edges[
    !is.na(.val),
    .(
      nb_max  = max(.val),
      nb_min  = min(.val),
      nb_mean = mean(.val)
    ),
    keyby = .focal_row
  ]

  # Prepare column names matching the original pipeline's output
  col_max  <- paste0("nb_max_",  var_name)
  col_min  <- paste0("nb_min_",  var_name)
  col_mean <- paste0("nb_mean_", var_name)

  # Initialise with NA (for rows with no valid neighbors)
  set(cell_data, j = col_max,  value = NA_real_)
  set(cell_data, j = col_min,  value = NA_real_)
  set(cell_data, j = col_mean, value = NA_real_)

  # Fill in computed values â€” in place, no copy
  set(cell_data, i = stats$.focal_row, j = col_max,  value = stats$nb_max)
  set(cell_data, i = stats$.focal_row, j = col_min,  value = stats$nb_min)
  set(cell_data, i = stats$.focal_row, j = col_mean, value = stats$nb_mean)

  # Clean up temporary column in edges to avoid carrying stale data
  neighbor_edges[, .val := NULL]

  invisible(cell_data)
}

# --- Run for all 5 neighbor source variables ---------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  for (var_name in neighbor_source_vars) {
    cat("  ", var_name, "...")
    compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_edges)
    cat(" done\n")
  }
})
# Expected: ~2â€“5 minutes total for all 5 variables


# =============================================================================
# 4. RANDOM FOREST PREDICTION â€” SINGLE VECTORISED CALL
# =============================================================================
predict_rf_optimised <- function(cell_data, rf_model, feature_cols = NULL) {
  # --- Determine feature columns ---------------------------------------------
  if (is.null(feature_cols)) {
    # Attempt to extract from model object
    if (inherits(rf_model, "ranger")) {
      feature_cols <- rf_model$forest$independent.variable.names
    } else if (inherits(rf_model, "randomForest")) {
      # rownames of importance matrix, or from the call
      feature_cols <- rownames(rf_model$importance)
    } else {
      stop("Unsupported model class: ", class(rf_model)[1],
           ". Please supply feature_cols explicitly.")
    }
  }

  # --- Validate that all features exist --------------------------------------
  missing <- setdiff(feature_cols, names(cell_data))
  if (length(missing) > 0) {
    stop("Missing features in cell_data: ", paste(missing, collapse = ", "))
  }

  # --- Build a clean numeric matrix for prediction ---------------------------
  # Using data.table's fast column access; avoid as.data.frame overhead
  pred_dt <- cell_data[, ..feature_cols]

  cat("Predicting on", nrow(pred_dt), "rows Ã—", ncol(pred_dt), "features...\n")

  # --- Single vectorised predict call ----------------------------------------
  if (inherits(rf_model, "ranger")) {
    # ranger::predict is very efficient; uses all cores by default
    preds <- predict(rf_model, data = pred_dt, num.threads = parallel::detectCores())
    cell_data[, predicted_gdp := preds$predictions]

  } else if (inherits(rf_model, "randomForest")) {
    # randomForest::predict â€” still vectorised but single-threaded
    # Convert to matrix for speed (avoids internal factor checking)
    pred_mat <- as.matrix(pred_dt)
    preds <- predict(rf_model, newdata = pred_mat)
    cell_data[, predicted_gdp := as.numeric(preds)]

  } else {
    # Generic fallback
    preds <- predict(rf_model, newdata = pred_dt)
    cell_data[, predicted_gdp := as.numeric(preds)]
  }

  cat("Prediction complete.\n")
  invisible(cell_data)
}

# --- Run prediction ----------------------------------------------------------
cat("Running RF prediction...\n")
system.time({
  predict_rf_optimised(cell_data, rf_model)
})
# Expected: ~5â€“20 minutes depending on model size and number of trees


# =============================================================================
# 5. CLEAN UP TEMPORARY COLUMNS
# =============================================================================
cell_data[, .row_idx := NULL]


# =============================================================================
# 6. (OPTIONAL) MEMORY MANAGEMENT
# =============================================================================
# If you no longer need the edge list, free it:
rm(neighbor_edges)
gc()
```

---

## 4. Summary of Changes and Guarantees

| Concern | Guarantee |
|---|---|
| **Trained RF model preserved** | The model object is loaded once with `readRDS` and passed read-only to `predict()`. It is never modified or retrained. |
| **Numerical estimand preserved** | The same max / min / mean neighbor statistics are computed on identical neighbor sets. The RF receives identical feature values â†’ identical predictions (up to floating-point associativity in `mean`, which is â‰¤ machine epsilon). |
| **Memory â‰¤ 16 GB** | Peak is ~12â€“13 GB (data ~5.4 GB + edge list ~2.5 GB + model ~3 GB + prediction workspace ~1 GB). The edge `.val` column is removed after each variable. No full-data copies are made. |
| **Runtime** | Expected **~15â€“30 minutes** total (vs. 86+ hours), a **~200Ã— speedup**. The dominant costs are now the `data.table` grouped aggregation (~5 min) and the RF `predict()` call (~10â€“20 min). |
END_RESPONSE>>>
