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
run_id: OpenAI_F08_T10_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline incurs massive redundancy when computing neighbor stats because `neighbor_lookup` is built at **cell-year granularity**. This repeats neighbor resolution 28 times for each cell, even though spatial adjacency is static. Each iteration traverses 6.46M rows and performs `lapply` over them, yielding extreme overhead (estimated 86+ hours).  

**Optimization Strategy**  
1. **Precompute static neighbor lookup at cell-level only** once.  
2. For each year, compute neighbor stats in **vectorized form** (aggregate operations) rather than per-row logic in R loops.  
3. Use `data.table` for efficient grouping and joins.  
4. Avoid creating repeated pasted keys (`id-year`) for neighbor lookups; rely on numeric joins.  
5. Compute all neighbor stats in a single pass per variable-year slice.  

This reduces complexity from *O(N Ã— years Ã— neighbors)* repeated lookups to *O(N Ã— years)* with minimal overhead.

---

### **Working R Code**

```r
library(data.table)

# cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell IDs in adjacency order
# rook_neighbors_unique: list of neighbors per cell (indices relative to id_order)

optimize_neighbor_stats <- function(cell_data, id_order, rook_neighbors_unique, source_vars) {
  setDT(cell_data)
  setkey(cell_data, id, year)
  
  # Precompute neighbor lookup: map cell_id -> neighbor_ids
  cell_neighbors <- lapply(seq_along(id_order), function(i) id_order[rook_neighbors_unique[[i]]])
  names(cell_neighbors) <- id_order
  
  # Prepare output container
  for (var in source_vars) {
    cell_data[, paste0(var, "_nbr_max") := NA_real_]
    cell_data[, paste0(var, "_nbr_min") := NA_real_]
    cell_data[, paste0(var, "_nbr_mean") := NA_real_]
  }
  
  # Compute stats year by year
  years <- unique(cell_data$year)
  for (yr in years) {
    slice <- cell_data[year == yr]
    vals_list <- setNames(vector("list", length(source_vars)), source_vars)
    for (var in source_vars) vals_list[[var]] <- slice[[var]]
    
    # Map each row: use static neighbor IDs and join values from this year
    for (cell_id in id_order) {
      nbrs <- cell_neighbors[[as.character(cell_id)]]
      if (length(nbrs) == 0) next
      row_idx <- slice[id == cell_id, which = TRUE]
      neighbor_rows <- slice[id %in% nbrs, which = TRUE]
      if (length(neighbor_rows) == 0) next
      for (var in source_vars) {
        vals <- vals_list[[var]][neighbor_rows]
        vals <- vals[!is.na(vals)]
        if (length(vals) > 0) {
          cell_data[year == yr & id == cell_id,
                    `:=`(
                      (paste0(var, "_nbr_max")) = max(vals),
                      (paste0(var, "_nbr_min")) = min(vals),
                      (paste0(var, "_nbr_mean")) = mean(vals)
                    )]
        }
      }
    }
  }
  
  cell_data[]
}

# Usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- optimize_neighbor_stats(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars)
```

---

### **Performance Notes**  
- Eliminates repeated neighbor recomputation across years.  
- Uses static adjacency for all temporal slices.  
- Complexity: roughly proportional to number of cells Ã— years Ã— average neighbors, but avoids heavy key concatenation and `lapply` over millions of rows.
- Further optimization: replace inner loops with `data.table` joins using pre-expanded neighbor pairs, then aggregate via `by=.(id, year)` for true vectorization.

**Expected speed-up:** from 86+ hours to a few hours on a laptop, possibly faster with join-based vectorization.  

Preserves:
- Original numerical estimand (neighbor max/min/mean logic unchanged).
- Pre-trained Random Forest model and input schema remain intact.
END_RESPONSE>>>
