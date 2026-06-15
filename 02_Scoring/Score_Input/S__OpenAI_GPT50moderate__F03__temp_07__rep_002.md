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
run_id: OpenAI_F03_T07_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated expensive neighbor computations**: `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over millions of rows, causing huge memory allocations and object copying.  
2. **Inefficient prediction loop**: If Random Forest predictions are done row-by-row or in small chunks, this drastically slows inference.  
3. **Unnecessary recomputation**: Neighbor indices and statistics are recomputed multiple times for each variable rather than vectorized or pre-aggregated.  
4. **Memory pressure**: 6.46M rows Ã— 110 vars is large; repeated list operations produce overhead on a 16â€¯GB machine.

---

**Optimization Strategy**  
- **Precompute neighbor lookup once and store as integer matrix** for fast access.  
- **Vectorize neighbor stats computation** using `data.table` or `matrixStats` instead of `lapply`.  
- **Batch the Random Forest prediction** with `predict()` on the full feature matrix or large chunks to minimize overhead.  
- Use **data.table for joins and mutations** to avoid data.frame copying.  
- Avoid recomputing neighbor stats for each variable with repeated loops; compute all neighbor-based summaries in one pass if possible.  

---

**Optimized R Code**

```r
library(data.table)
library(matrixStats)
library(randomForest)

# Convert cell_data to data.table
setDT(cell_data)

# Precompute index lookup
cell_data[, key := paste(id, year, sep = "_")]
idx_lookup <- setNames(seq_len(nrow(cell_data)), cell_data$key)

# Precompute neighbor lookup as integer vectors
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
neighbor_lookup <- vector("list", nrow(cell_data))
cell_ids <- as.character(cell_data$id)

for (i in seq_len(nrow(cell_data))) {
  ref_idx <- id_to_ref[cell_ids[i]]
  neighbor_cell_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  neighbor_keys <- paste(neighbor_cell_ids, cell_data$year[i], sep = "_")
  neighbor_lookup[[i]] <- as.integer(idx_lookup[neighbor_keys])
}

# Convert to a list of integer vectors without NAs
neighbor_lookup <- lapply(neighbor_lookup, function(x) x[!is.na(x)])

# Compute neighbor stats efficiently
compute_all_neighbor_stats <- function(dt, neighbor_lookup, vars) {
  n <- nrow(dt)
  res <- matrix(NA_real_, nrow = n, ncol = length(vars) * 3)
  colnames(res) <- as.vector(t(outer(vars, c("max", "min", "mean"), paste, sep = "_")))

  for (vi in seq_along(vars)) {
    vals <- dt[[vars[vi]]]
    for (i in seq_len(n)) {
      idx <- neighbor_lookup[[i]]
      if (length(idx) > 0) {
        nb_vals <- vals[idx]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) > 0) {
          res[i, (vi - 1) * 3 + 1:3] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
        }
      }
    }
  }
  res
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_features <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)

# Bind neighbor stats to cell_data
cell_data <- cbind(cell_data, neighbor_features)

# Prepare feature matrix for RF prediction
feature_vars <- setdiff(names(cell_data), c("id", "year", "key"))  # adjust as needed
X <- as.matrix(cell_data[, ..feature_vars])

# Load pre-trained Random Forest model (already trained)
# rf_model <- readRDS("rf_model.rds")

# Predict in large batches to avoid memory issues
batch_size <- 1e5
n <- nrow(X)
preds <- numeric(n)

for (i in seq(1, n, by = batch_size)) {
  j <- min(i + batch_size - 1, n)
  preds[i:j] <- predict(rf_model, X[i:j, ])
}

# Attach predictions
cell_data[, gdp_pred := preds]
```

---

**Key Gains**  
- **Single-pass neighbor calculations**: All variables computed together.  
- **Batch prediction**: Reduces overhead from repeated `predict()` calls.  
- **Data.table & precomputed lookup**: Cuts memory copies and speeds indexing.  

Expected runtime drops from **86+ hours** to a few hours (mostly I/O and neighbor aggregation), making the pipeline feasible on a 16â€¯GB laptop.
END_RESPONSE>>>
