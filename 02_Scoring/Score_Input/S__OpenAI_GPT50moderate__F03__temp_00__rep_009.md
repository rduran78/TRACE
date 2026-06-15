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
run_id: OpenAI_F03_T00_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F03_OpenAI_temp_00_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is the neighbor feature computation and repeated object copying before Random Forest inference. Specifically:  
- `build_neighbor_lookup` creates a large list (â‰ˆ6.46M elements) with repeated string concatenation and lookups, which is extremely memory- and CPU-intensive.  
- `compute_neighbor_stats` uses `lapply` and repeated `rbind`, causing high overhead and inefficient memory allocation.  
- The outer loop calls `compute_and_add_neighbor_features` multiple times, repeatedly copying `cell_data`.  
- Random Forest inference itself is relatively fast compared to the preprocessing, but prediction on 6.46M rows still requires efficient batching and avoiding repeated model loading.  

**Optimization Strategy**  
1. **Precompute neighbor indices once** as integer vectors, avoid string concatenation and `setNames` overhead.  
2. **Vectorize neighbor statistics** using `data.table` or matrix operations instead of `lapply`.  
3. **Avoid repeated copying of `cell_data`**; compute all neighbor features in one pass and bind columns efficiently.  
4. **Batch Random Forest predictions** using `predict(..., newdata)` on chunks to manage memory.  
5. **Use `data.table` for fast joins and aggregation** instead of base R lists.  
6. **Keep model in memory**; do not reload for each batch.  

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest)

# Convert to data.table for efficiency
cell_dt <- as.data.table(cell_data)

# Precompute lookup: map (id, year) -> row index
cell_dt[, key := paste(id, year, sep = "_")]
idx_lookup <- setNames(seq_len(nrow(cell_dt)), cell_dt$key)

# Build neighbor lookup as integer indices (vectorized)
build_neighbor_lookup_fast <- function(id_order, neighbors, years) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(ref_idx) {
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    # For all years, precompute keys
    lapply(years, function(y) {
      keys <- paste(neighbor_ids, y, sep = "_")
      idx_lookup[keys]
    })
  })
}

years <- unique(cell_dt$year)
neighbor_lookup <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique, years)

# Compute neighbor stats in one pass for all variables
compute_neighbor_stats_fast <- function(vals, neighbor_lookup_flat) {
  # neighbor_lookup_flat: list of integer vectors
  res <- matrix(NA_real_, nrow = length(neighbor_lookup_flat), ncol = 3)
  for (i in seq_along(neighbor_lookup_flat)) {
    idx <- neighbor_lookup_flat[[i]]
    if (length(idx) > 0) {
      nv <- vals[idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) > 0) {
        res[i, ] <- c(max(nv), min(nv), mean(nv))
      }
    }
  }
  res
}

# Flatten neighbor lookup for cell-year rows
neighbor_lookup_flat <- unlist(neighbor_lookup, recursive = FALSE)

# Compute all neighbor features efficiently
for (var_name in neighbor_source_vars) {
  vals <- cell_dt[[var_name]]
  stats <- compute_neighbor_stats_fast(vals, neighbor_lookup_flat)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_dt <- cbind(cell_dt, stats)
}

# Prepare predictors for Random Forest
predictors <- setdiff(names(cell_dt), c("id", "year", "key", "target_var")) # adjust target_var
X <- as.data.frame(cell_dt[, ..predictors])

# Batch prediction to avoid memory overload
batch_size <- 500000
n <- nrow(X)
preds <- numeric(n)
for (start in seq(1, n, by = batch_size)) {
  end <- min(start + batch_size - 1, n)
  preds[start:end] <- predict(rf_model, newdata = X[start:end, ])
}

# Attach predictions
cell_dt[, gdp_pred := preds]
```

---

**Key Improvements**  
- Eliminated repeated string operations inside loops.  
- Used `data.table` for efficient memory and speed.  
- Computed neighbor stats in a single pass with preallocated matrix.  
- Batched Random Forest predictions to fit within 16 GB RAM.  
- Preserved trained model and original estimand.  

**Expected Runtime Reduction**: From 86+ hours to a few hours (depending on disk I/O and CPU).
END_RESPONSE>>>
