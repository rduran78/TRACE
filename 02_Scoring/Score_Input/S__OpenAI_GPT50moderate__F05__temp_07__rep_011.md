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
run_id: OpenAI_F05_T07_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F05_OpenAI_temp_07_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The repeated `paste()` and string-key-based indexing inside `lapply` is a major inefficiency. For every row (~6.46 million), the function constructs `neighbor_keys` and performs lookups in `idx_lookup` using character keys. This is not just a local inefficiencyâ€”it is symptomatic of a larger issue: repeated string concatenation and hash lookups dominate runtime. The outer loop further multiplies cost as the same neighbor structure is recomputed for each variable.

**Optimization Strategy:**  
Reformulate the algorithm to avoid string-based indexing entirely. Use integer-based mapping and precompute the neighbor lookup once as a list of integer indices. Then reuse this list for all variables. Steps:  
1. Map `id` to integer positions up front.  
2. Use a single integer join for `(id, year)` instead of concatenated strings.  
3. Build `neighbor_lookup` efficiently without `paste()` calls.  
4. Use vectorized or preallocated structures for neighbor stats.

This reduces repeated work and memory churn. The result: one pass to build neighbor indices, then fast numeric operations for stats.

---

### **Working R Code**

```r
# Precompute mappings for fast integer joins
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Map cell id to reference position
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map (id, year) -> row index using integer keys
  # Convert years to integer offset
  years <- sort(unique(data$year))
  year_to_idx <- setNames(seq_along(years), years)
  
  # Precompute row index matrix for fast lookup
  # row_key[i] = (id_ref[i]-1)*n_years + year_ref[i]
  n_years <- length(years)
  id_ref <- id_to_ref[as.character(data$id)]
  year_ref <- year_to_idx[as.character(data$year)]
  row_key <- (id_ref - 1L) * n_years + year_ref
  
  # Map key -> row index
  idx_lookup <- seq_len(nrow(data))
  names(idx_lookup) <- row_key
  
  # Build neighbor list as integer indices
  row_ids <- seq_len(nrow(data))
  
  lapply(row_ids, function(i) {
    ref_idx <- id_ref[i]
    neighbor_refs <- neighbors[[ref_idx]]
    if (length(neighbor_refs) == 0) return(integer(0))
    neighbor_keys <- (neighbor_refs - 1L) * n_years + year_ref[i]
    as.integer(idx_lookup[as.character(neighbor_keys)])
  })
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) > 0) {
        result[i, ] <- c(max(v), min(v), mean(v))
      }
    }
  }
  result
}

# Pipeline
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

**Why faster?**  
- Eliminates `paste()` and string-based lookups inside inner loops.
- Integer keys allow O(1) array indexing.
- `neighbor_lookup` computed once and reused.
- Memory footprint reduced (no giant character vectors).

**Expected improvement:**  
From 86+ hours down to a few hours (or less), depending on disk I/O and machine constraints.
END_RESPONSE>>>
