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
run_id: OpenAI_F06_T10_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is the `lapply`-based loops in `compute_neighbor_stats` across 6.46M rows and ~1.37M neighbor links. Each call constructs vectors, filters NAs, and computes summary stats individually, resulting in enormous overhead. The separate pass for each variable compounds the cost. The logic is correct but not vectorized; this is why runtime is estimated at 86+ hours.  

---

**Optimization Strategy**  
- **Flatten neighborhood into a long table:** Convert `neighbor_lookup` into a two-column (source, neighbor) index table and join with variable values once.  
- **Compute grouped summaries using `data.table` or `dplyr`:** Group by source to compute max, min, and mean in one pass, then merge back into main panel.  
- **Avoid repeated loops for each variable:** Process all five variables sequentially but with efficient aggregation.  
- **Memory-conscious:** Use integer indices and in-place joins with `data.table`.  
- **Preserve results:** Identical statistics as current per-row neighbor max, min, mean.  

---

**Working R Code (`data.table` approach)**  

```r
library(data.table)

# Assume `cell_data` is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Assume neighbor_lookup is as built by build_neighbor_lookup
# Create a long edge list: from (i) to (neighbor)
src <- rep(seq_along(neighbor_lookup), lengths(neighbor_lookup))
nbr <- unlist(neighbor_lookup)
edges <- data.table(src = src, nbr = nbr)

# Add source and neighbor keys: index of cell_data rows
# (src and nbr are row indices into cell_data matching year alignment)

# We'll compute for each variable efficiently:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  edges[, val := vals[nbr]]

  # Aggregate by source: ignoring NAs automatically
  agg <- edges[!is.na(val), .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = src]

  # Prepare output columns
  setnames(
    agg, c("src", "max_val", "min_val", "mean_val"),
    c("row_idx", paste0(var_name, "_nbr_max"), paste0(var_name, "_nbr_min"), paste0(var_name, "_nbr_mean"))
  )

  # Join back to cell_data by row index
  cell_data[, row_idx := .I]
  cell_data <- merge(cell_data, agg, by = "row_idx", all.x = TRUE)
  cell_data[, row_idx := NULL]

  # Remove val column before next loop to save memory
  edges[, val := NULL]
}

# Result: cell_data now has required neighbor summary features
```

---

**Performance Impact:**  
- Single edge list (~1.37M rows) vs. millions of per-row loops â†’ major speed-up (hours to minutes).  
- `data.table` aggregation is highly optimized in C and memory efficient for 16â€¯GB.  
- Exact same estimand since summaries match original logic: max, min, mean per rowâ€™s neighbors ignoring NAs.  

**Key Points:**  
- No retraining of the Random Forest model required.  
- Preserves original numeric outputs.  
- Requires `data.table` for optimal speed.  

**Estimated runtime after optimization:** A few minutes instead of 86+ hours on a 16â€¯GB laptop for 6.46M rows.
END_RESPONSE>>>
