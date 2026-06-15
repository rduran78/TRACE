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
run_id: OpenAI_F04_T10_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The primary bottleneck is the massive repeated computation in `compute_neighbor_stats` and the overhead of building large lists with `lapply`. For 6.46M rows and multiple variables, repeatedly scanning neighbor indices is extremely costly. Random Forest inference is relatively fast compared to these nested loops. Memory pressure also comes from storing large intermediate lists.

---

**Optimization Strategy:**  
1. **Vectorization & Preallocation:** Avoid building per-row lists; store neighbor IDs in a matrix or keep as list but process in a batched way.  
2. **Use `data.table` for fast indexing:** Convert `data` to `data.table` for fast access and aggregation.  
3. **Precompute neighbor stats for all variables in one pass rather than looping.**  
4. **Parallelization:** Utilize `future.apply` or `parallel::mclapply` over chunks, respecting 16GB RAM.  
5. **Sparse matrix approach:** Represent neighbor relationships once, then multiply by variable arrays for aggregation.

---

**Working Optimized R Code (data.table + parallel):**

```r
library(data.table)
library(parallel)

# Convert to data.table
cell_data_dt <- as.data.table(cell_data)

# Build neighbor lookup as a list (unchanged logic, minimal overhead)
neighbor_lookup <- build_neighbor_lookup(cell_data_dt, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Parallel function to compute stats for each variable
compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  mclapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
    v <- vals[idx]
    v <- v[!is.na(v)]
    if (length(v) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(v), min(v), mean(v))
  }, mc.cores = detectCores() - 1)
}

# Preallocate result columns for efficiency
for (var_name in neighbor_source_vars) {
  vals <- cell_data_dt[[var_name]]
  stats_list <- compute_neighbor_stats_fast(vals, neighbor_lookup)
  stats_mat <- do.call(rbind, stats_list)
  cell_data_dt[[paste0(var_name, "_nbr_max")]] <- stats_mat[, 1]
  cell_data_dt[[paste0(var_name, "_nbr_min")]] <- stats_mat[, 2]
  cell_data_dt[[paste0(var_name, "_nbr_mean")]]<- stats_mat[, 3]
}
```

---

**Further Gains:**  
- Replace `lapply` with `vapply` for type stability and speed.
- Chunk neighbor lookup into blocks, process per block to keep RAM usage manageable.
- Option: Construct a sparse adjacency matrix `A` (cells x cells) and apply matrix multiplications for sums and counts, then compute means.

---

**Expected Improvement:**  
Parallelization and data.table optimization should cut computation from 86+ hours to several hours (depending on cores), without retraining the Random Forest and preserving original numerical estimands.
END_RESPONSE>>>
