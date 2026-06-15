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
run_id: OpenAI_F12_T00_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F12_OpenAI_temp_00_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` is the main bottleneck is **incorrect**. While `rbind` on millions of rows is non-trivial, the deeper bottleneck is the **nested `lapply` over 6.46M rows for each of 5 variables**, repeatedly traversing large lists and performing vector operations in pure R. This results in ~32M function calls and heavy R interpreter overhead. The real issue is the **row-wise neighbor aggregation in R loops**, not the final `rbind`.

---

### **Correct Optimization Strategy**
- **Vectorize neighbor aggregation**: Precompute a long-format edge list of `(source_row, neighbor_row)` pairs once, then use fast grouped operations (e.g., `data.table`) to compute `max`, `min`, and `mean` per source row for all variables in one pass.
- Avoid repeated `lapply` and `do.call(rbind)` by replacing them with efficient joins and aggregations.
- Preserve the trained Random Forest model and the original estimand by ensuring identical numeric results.

---

### **Optimized Approach**
1. Build an **edge table** from `neighbor_lookup`:  
   Each row: `src` (cell-year index), `nbr` (neighbor index).
2. Use `data.table` to join neighbor values and compute stats in bulk.
3. Merge results back to `cell_data`.

---

### **Working R Code**

```r
library(data.table)

# Convert cell_data to data.table
setDT(cell_data)

# Step 1: Build edge list once
edge_list <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(src) {
    nbrs <- neighbor_lookup[[src]]
    if (length(nbrs) == 0) return(NULL)
    data.table(src = src, nbr = nbrs)
  })
)

# Step 2: Compute neighbor stats for all variables in one pass
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (v in vars) {
  # Join neighbor values
  edge_list[, val := cell_data[[v]][nbr]]
  
  # Remove NAs
  edge_list_no_na <- edge_list[!is.na(val)]
  
  # Aggregate stats by src
  stats <- edge_list_no_na[, .(
    max_val = max(val),
    min_val = min(val),
    mean_val = mean(val)
  ), by = src]
  
  # Merge back to cell_data
  setnames(stats, c("src", paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  cell_data[stats$src, (names(stats)[-1]) := stats[, -1], on = .I]
}

# cell_data now has neighbor features for all vars
```

---

### **Why This Works**
- Eliminates **millions of R function calls** by replacing `lapply` with a single `rbindlist` and `data.table` aggregation.
- Avoids repeated `do.call(rbind, ...)` and redundant neighbor lookups.
- Memory-efficient: processes edges (~1.37M) instead of 6.46M rows Ã— 5 loops.
- Expected runtime drops from **86+ hours to minutes** on a 16 GB laptop.

**Bottom line:** The bottleneck is the row-wise R loops, not `rbind`. The correct fix is to vectorize neighbor aggregation using an edge list and `data.table`.
END_RESPONSE>>>
