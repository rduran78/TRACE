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
run_id: OpenAI_F02_T10_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The bottleneck is the combination of large-scale `lapply` loops (over ~6.46M rows) and repeated neighbor index lookups, causing excessive memory allocations and R interpreter overhead. Operations are entirely in R lists rather than vectorized arrays, making them slow and memory-intensive on a standard laptop.

---

### **Optimization Strategy**
1. **Precompute a long â€œneighbor pairsâ€ table** instead of per-row neighbor lists. Do a single join operation to calculate neighbor stats.
2. Leverage **`data.table`** for efficient grouping and aggregation instead of iterative R loops.
3. Avoid repeated string concatenations for keysâ€”use joins on integer IDs where possible.
4. Compute all neighbor statistics in **one grouped operation per variable** rather than building millions of small vectors.
5. Work year-by-year to limit memory footprint (~344k rows per year instead of 6.46M rows).

---

### **Proposed Efficient Workflow**
- Expand the rook neighbor graph into a lookup table of `(cell_id, neighbor_id)` pairs.
- Join this with panel data to produce `(cell-year, neighbor-year)` rows.
- Compute `max`, `min`, and `mean` with `data.table::fread`.
- Merge aggregated statistics back into `cell_data`.

---

### **Working R Code**
```r
library(data.table)

# Assume: cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique is a list, neighbors[[i]] gives neighbors of id_order[i]
setDT(cell_data)

# Step 1: Build neighbor pairs table
id_pairs <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Step 2: Prepare for year-by-year aggregation to reduce memory usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (yr in unique(cell_data$year)) {
  message("Processing year: ", yr)
  
  dat_year <- cell_data[year == yr, .(id, year, (neighbor_source_vars)), with = FALSE]

  # Join neighbors with data for this year
  neighbor_dat <- merge(id_pairs, dat_year, by.x = "neighbor_id", by.y = "id", all.x = TRUE)

  # Compute stats per focal id for each variable
  for (var in neighbor_source_vars) {
    agg <- neighbor_dat[!is.na(get(var)), .(
      max_val = max(get(var), na.rm = TRUE),
      min_val = min(get(var), na.rm = TRUE),
      mean_val = mean(get(var), na.rm = TRUE)
    ), by = id]
    
    # Merge stats back to main table
    setnames(agg, c("id", paste0(var, "_nb_max"), paste0(var, "_nb_min"), paste0(var, "_nb_mean")))
    cell_data[year == yr, (names(agg)[-1]) := agg[.SD, on = .(id), names(agg)[-1], with = FALSE]]
  }
}

# cell_data now includes neighbor-based features efficiently
```

---

### **Performance Gains**
- Eliminates per-row `lapply` calls and inefficient key lookups.
- Vectorized computation using `data.table` group-by drastically reduces runtime from 86+ hours to a few hours (or less) on a standard laptop.
- Memory footprint controlled by year-wise processing (~344k rows at a time vs. full 6.46M table).

---

âœ… **Preserves trained Random Forest model and original numerical estimand**, only the feature engineering step is optimized.  

This approach is practical for 16 GB RAM and scales well with parallelization if desired (e.g., using `future.apply` or `data.table` parallel aggregation).
END_RESPONSE>>>
