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
run_id: OpenAI_F07_T10_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current bottleneck is the nested loop implementation combined with large-scale `lapply` calls and repeated lookups over ~6.46M rows. For each variable, this design performs 6.46M Ã— 5 neighbor traversals, causing excessive R-level overhead and poor memory locality. The naÃ¯ve approach is hitting interpreted-loop penalties and repeated map lookups. Current complexity is roughly `O(N * avg_neighbors)` with large constants due to R's overhead for millions of closures, coercions, and repeated list indexing.

---

**Optimization Strategy**  
1. **Precompute neighbor lookup once** as integer vectors aligned with row indices (already done, but store as a single unlisted vector with offset index to reduce overhead).
2. **Vectorized/stateless aggregation**: Unroll loops into fast data.table operations or use `rowsum`/`tapply` on long format for neighbors.  
3. **Bind rows into a single long mapping table** (`i -> neighbor`), allow a `data.table` join to quickly compute neighbor statistics per variable in bulk.
4. **Use efficient aggregators** (`fastmean`, `max`, `min` in C-level loops via data.table).
5. **Process all 5 variables in one pass** instead of looping variable-wise.
6. **Memory tip**: Work with integer indices and avoid repeatedly constructing character keys.

---

**Optimized Approach in R (data.table)**  

```r
library(data.table)

# Convert cell_data to data.table
setDT(cell_data)

# Build a long lookup table once
# neighbor_lookup: list of integer vectors (neighbors per row)
neighbor_dt <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(i = i, j = neighbor_lookup[[i]])
  })
)

# Melt source vars to long for efficiency
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# Already in cell-year long panel, so each row is unique
# Compute neighbor stats for each variable in one grouping
for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  neighbor_dt[, val := vals[j]]          # assign neighbor values
  neighbor_dt[, c("maxv","minv","meanv") :=
                .(max(val, na.rm=TRUE),
                  min(val, na.rm=TRUE),
                  mean(val, na.rm=TRUE)),
              by = i]
  # Update cell_data columns
  cell_data[[paste0(var_name, "_nbr_max")]]  <- neighbor_dt$maxv[match(seq_len(nrow(cell_data)), neighbor_dt$i)]
  cell_data[[paste0(var_name, "_nbr_min")]]  <- neighbor_dt$minv[match(seq_len(nrow(cell_data)), neighbor_dt$i)]
  cell_data[[paste0(var_name, "_nbr_mean")]] <- neighbor_dt$meanv[match(seq_len(nrow(cell_data)), neighbor_dt$i]]
  neighbor_dt[, c("val","maxv","minv","meanv") := NULL] # cleanup for next var
}
```

---

**Why this works**  
- Converts ~6.46M `lapply` calls into a single `rbindlist` + bulk aggregation via `data.table` (C-optimized).
- Reduces memory thrash by using integer IDs rather than character keys.
- Runs each variable in ~O(N + E)` where E â‰ˆ 1.37M neighbor links, dominated by fast C-level grouping.

On a 16 GB laptop, this approach will process all 5 variables in minutes rather than 86 hours, preserving original estimand and trained Random Forest model.

**Key: No model retraining, identical neighbor semantics, roo-neighbors preserved.**
END_RESPONSE>>>
