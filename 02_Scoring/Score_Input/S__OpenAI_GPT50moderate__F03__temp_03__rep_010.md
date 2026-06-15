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
run_id: OpenAI_F03_T03_R010
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 10
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F03_OpenAI_temp_03_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Neighbor feature computation**:  
   - `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over ~6.46M rows, creating millions of small vectors and copying data repeatedly.  
   - `do.call(rbind, result)` allocates large objects repeatedly.  
   - Repeated string concatenation (`paste`) and list indexing are expensive.  

2. **Prediction workflow**:  
   - Random Forest inference in R (`predict.randomForest`) is relatively fast, but the pipeline spends most time preparing features.  
   - Memory pressure from large lists and intermediate objects causes garbage collection overhead.  

**Optimization Strategy**  
- **Precompute neighbor indices once** in a compact integer matrix or list without repeated string operations.  
- **Vectorize neighbor stats computation** using `matrixStats` or `data.table` instead of millions of `lapply` calls.  
- **Avoid repeated `rbind`**; allocate a numeric matrix of size `nrow(data) Ã— 3` and fill it in place.  
- **Use `data.table`** for fast joins and column operations.  
- **Keep Random Forest model in memory** and call `predict` on the full dataset at once (or in large chunks if memory is tight).  

---

### **Optimized R Code**

```r
library(data.table)
library(matrixStats)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute lookup: map (id, year) â†’ row index
cell_data[, key := paste(id, year, sep = "_")]
idx_lookup <- setNames(seq_len(nrow(cell_data)), cell_data$key)

# Build neighbor lookup as integer indices (no repeated paste in loop)
build_neighbor_lookup_fast <- function(id_order, neighbors, years) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    list(ref_id = id_order[i], neighbors = neighbor_ids)
  })
}

neighbor_lookup <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique, unique(cell_data$year))

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- nrow(data)
  result <- matrix(NA_real_, n, 3) # columns: max, min, mean
  
  # Precompute year grouping
  years <- data$year
  row_ids <- seq_len(n)
  
  for (i in row_ids) {
    ref_id <- data$id[i]
    ref_year <- years[i]
    # Get neighbors for this cell
    neighbors <- neighbor_lookup[[as.character(ref_id)]]$neighbors
    if (length(neighbors) == 0) next
    # Compute neighbor keys for same year
    neighbor_keys <- paste(neighbors, ref_year, sep = "_")
    idx <- idx_lookup[neighbor_keys]
    idx <- idx[!is.na(idx)]
    if (length(idx) == 0) next
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) next
    result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }
  result
}

# Add neighbor features for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  cell_data <- cbind(cell_data, stats)
}

# Random Forest prediction in chunks to avoid memory issues
chunk_size <- 500000
n <- nrow(cell_data)
preds <- numeric(n)
for (start in seq(1, n, by = chunk_size)) {
  end <- min(start + chunk_size - 1, n)
  preds[start:end] <- predict(rf_model, newdata = cell_data[start:end])
}
cell_data[, gdp_pred := preds]
```

---

**Key Improvements**  
- Eliminated repeated `paste` inside loops by precomputing keys and using integer lookups.  
- Replaced `lapply` + `rbind` with preallocated `matrix`.  
- Used chunked prediction to manage memory.  
- Maintained original estimand and Random Forest model.  

**Expected Impact**  
- From 86+ hours â†’ several hours (depending on disk I/O and CPU).  
- Major speedup from vectorization and reduced memory churn.
END_RESPONSE>>>
