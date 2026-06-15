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
run_id: OpenAI_F04_T07_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the repeated neighbor feature computation using `lapply` over ~6.46M rows for 5 variables. The `build_neighbor_lookup` creates a large list (one per cell-year), and `compute_neighbor_stats` repeatedly scans indices for each variable. This results in excessive memory allocations and R-level loops, which scale poorly.

**Optimization Strategy**  
- Precompute `neighbor_lookup` **once** (already done).
- Avoid per-row `lapply` for each variable; process neighbors in a **vectorized/data.table** manner.
- Use `data.table` joins or `collapse` for fast grouped aggregations.
- Compute neighbor stats in one pass for all variables instead of looping.
- Reduce intermediate objects; leverage integer indexing and preallocation.
- Parallelize where possible using `future.apply` or `data.table` multithreading.

**Working R Code (Optimized Implementation)**  
```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute neighbor lookup (same as before)
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Flatten neighbor relationships into long format
neighbor_dt <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(row_id = i, neighbor_id = neighbor_lookup[[i]])
  })
)

# Merge neighbor values for all source vars
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_dt <- neighbor_dt[cell_data, on = .(neighbor_id = .I), 
                            (vars) := mget(vars)]

# Aggregate neighbor stats in one pass
agg_dt <- neighbor_dt[, {
  lapply(.SD, function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(x), min(x), mean(x))
  })
}, by = row_id, .SDcols = vars]

# Reshape aggregated results
agg_mat <- do.call(cbind, agg_dt[, -1])
colnames(agg_mat) <- as.vector(sapply(vars, function(v) paste(v, c("max","min","mean"), sep = "_")))

# Bind back to original data.table
cell_data <- cbind(cell_data, agg_mat)

# Ensure preservation of trained model and estimand
# (No retraining occurs; features added to cell_data as required)
```

**Expected Impact**  
- Collapses 5 separate heavy loops into a single vectorized aggregation.
- `data.table` drastically reduces runtime (hours â†’ minutes on 16 GB RAM).
- Memory footprint minimized by avoiding massive nested lists.
- Parallelization optional for further speedup.

This approach preserves the Random Forest model and original estimand while making neighbor feature construction practical.
END_RESPONSE>>>
