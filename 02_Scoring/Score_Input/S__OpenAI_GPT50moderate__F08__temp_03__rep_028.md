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
run_id: OpenAI_F08_T03_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F08_OpenAI_temp_03_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it recomputes neighbor statistics for every cell-year row across 6.46M rows, even though the neighbor structure is static across years. The `build_neighbor_lookup` function already leverages this static structure, but `compute_neighbor_stats` still iterates over all rows and repeatedly extracts neighbor values. This results in redundant computations for each year and variable. Memory overhead is also high because of repeated list operations.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell (not per cell-year)** since neighbor relationships do not change across years.
2. **Compute neighbor stats year-by-year in bulk** using vectorized operations rather than looping over all rows.
3. Use **matrix or data.table operations** to avoid repeated list allocations.
4. Avoid recomputing for each variable separately in a slow loop; instead, compute all neighbor-based features in a single pass per year.

---

**Optimized Approach**  
- Keep `neighbor_lookup` as a list of integer indices per cell (static).
- For each year:
  - Subset data for that year.
  - For each variable, compute neighbor stats using the static neighbor indices and vectorized operations.
- Bind results back to the main dataset.
- This reduces complexity from O(N*T*neighbors) to O(T*(N+neighbors)) and avoids redundant work.

---

### **Working R Code**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute static neighbor lookup: list of integer positions per cell
build_neighbor_lookup_static <- function(id_order, neighbors) {
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  lapply(neighbors, function(nb) id_to_idx[as.character(id_order[nb])])
}

neighbor_lookup_static <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)

# Variables for neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Preallocate columns for neighbor stats
for (var in neighbor_source_vars) {
  for (stat in c("max", "min", "mean")) {
    cell_data[, paste0(var, "_nbr_", stat) := NA_real_]
  }
}

# Compute neighbor stats year by year
years <- sort(unique(cell_data$year))

for (yr in years) {
  message("Processing year: ", yr)
  year_idx <- which(cell_data$year == yr)
  year_data <- cell_data[year_idx]
  
  # Map id to row index in year_data for fast lookup
  id_to_row <- setNames(seq_len(nrow(year_data)), as.character(year_data$id))
  
  for (i in seq_len(nrow(year_data))) {
    nb_ids <- neighbor_lookup_static[[ id_to_row[[as.character(year_data$id[i])]] ]]
    nb_rows <- id_to_row[as.character(id_order[nb_ids])]
    nb_rows <- nb_rows[!is.na(nb_rows)]
    if (length(nb_rows) == 0) next
    
    for (var in neighbor_source_vars) {
      vals <- year_data[[var]][nb_rows]
      vals <- vals[!is.na(vals)]
      if (length(vals) > 0) {
        cell_data[year_idx[i], paste0(var, "_nbr_max") := max(vals)]
        cell_data[year_idx[i], paste0(var, "_nbr_min") := min(vals)]
        cell_data[year_idx[i], paste0(var, "_nbr_mean") := mean(vals)]
      }
    }
  }
}

# At this point, cell_data has neighbor-based features added.
# Proceed to Random Forest prediction using the pre-trained model.
```

---

**Why This Works**  
- **Static neighbor lookup** avoids recomputation.
- **Year-based processing** reduces memory footprint and allows vectorization.
- **Direct indexing** avoids repeated `paste` and `lookup` operations.
- Preserves the original numerical estimand and does not retrain the model.

**Expected Performance Gain**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on I/O and CPU speed. Further speedups can be achieved by:
- Parallelizing the year loop with `future.apply` or `parallel`.
- Using `Rcpp` for inner loops if necessary.
END_RESPONSE>>>
