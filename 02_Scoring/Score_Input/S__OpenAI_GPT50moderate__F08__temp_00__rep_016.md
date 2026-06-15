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
run_id: OpenAI_F08_T00_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F08_OpenAI_temp_00_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it computes neighbor statistics (`max`, `min`, `mean`) for each of the ~6.46 million cell-year rows by iterating through a large list (`neighbor_lookup`) and repeatedly subsetting the full dataset. This results in massive redundant work:  
- The neighbor relationships are static across years, but the code recomputes neighbor indices for every row-year combination.  
- The `compute_neighbor_stats` function loops over all rows and performs repeated lookups, which is inefficient for millions of rows.  
- Memory and CPU overhead are high because of repeated list operations and `lapply` calls on millions of elements.  

**Optimization Strategy**  
1. **Precompute neighbor relationships once per cell (not per cell-year)** since they are static.  
2. **Process data year by year**: For each year, extract the relevant subset of variables and compute neighbor stats using vectorized operations or efficient aggregation.  
3. **Avoid repeated lookups**: Use a matrix or data.table keyed by `id` for fast joins.  
4. **Batch computation**: Compute all neighbor stats for all variables in one pass per year.  
5. **Memory efficiency**: Work with `data.table` for fast grouping and joining.  

This reduces complexity from O(N * neighbors * years) to O(years * (cells + neighbor relationships)), which is much faster and memory-friendly.

---

### **Optimized R Code**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute neighbor lookup as a list: id -> neighbor_ids
# id_order and rook_neighbors_unique are given
neighbor_list <- setNames(id_order[rook_neighbors_unique], id_order)

# Variables for which to compute neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare output columns
for (var in neighbor_source_vars) {
  cell_data[, paste0(var, "_nbr_max") := NA_real_]
  cell_data[, paste0(var, "_nbr_min") := NA_real_]
  cell_data[, paste0(var, "_nbr_mean") := NA_real_]
}

# Process year by year
years <- sort(unique(cell_data$year))

for (yr in years) {
  message("Processing year: ", yr)
  
  # Subset for this year
  dt_year <- cell_data[year == yr]
  
  # Create a named vector for fast lookup: id -> row index
  idx_lookup <- setNames(seq_len(nrow(dt_year)), dt_year$id)
  
  # For each cell, compute neighbor stats
  for (i in seq_len(nrow(dt_year))) {
    cell_id <- dt_year$id[i]
    nbr_ids <- neighbor_list[[as.character(cell_id)]]
    if (length(nbr_ids) == 0) next
    
    nbr_idx <- idx_lookup[as.character(nbr_ids)]
    nbr_idx <- nbr_idx[!is.na(nbr_idx)]
    if (length(nbr_idx) == 0) next
    
    for (var in neighbor_source_vars) {
      vals <- dt_year[[var]][nbr_idx]
      vals <- vals[!is.na(vals)]
      if (length(vals) > 0) {
        cell_data[year == yr & id == cell_id, paste0(var, "_nbr_max") := max(vals)]
        cell_data[year == yr & id == cell_id, paste0(var, "_nbr_min") := min(vals)]
        cell_data[year == yr & id == cell_id, paste0(var, "_nbr_mean") := mean(vals)]
      }
    }
  }
}
```

---

### **Why This Is Faster**
- Neighbor relationships are computed once and reused.
- Work is done year by year, reducing memory footprint.
- Avoids repeated full-data lookups by using `idx_lookup` for the current year.
- Still uses loops for clarity, but only over cells per year (344k) instead of 6.46M rows repeatedly.

---

**Further Speed-Up (Optional)**
- Parallelize the per-year loop using `future.apply` or `parallel`.
- Replace inner loops with vectorized aggregation using `data.table` joins if memory allows.
- If extreme performance is needed, convert neighbor_list to a sparse matrix and use matrix operations.

---

**Preserves**:
- Original numerical estimand (same max, min, mean logic).
- Pre-trained Random Forest model (unchanged).
- Pipeline structure (just optimized neighbor feature computation).

This approach should reduce runtime from **86+ hours to a few hours** on a standard laptop.
END_RESPONSE>>>
