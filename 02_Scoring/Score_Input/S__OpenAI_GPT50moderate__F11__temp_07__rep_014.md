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
run_id: OpenAI_F11_T07_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F11_OpenAI_temp_07_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s claim is incorrect. The true bottleneck is **building and applying neighbor-based features in pure R using deeply nested `lapply` loops** over ~6.46M rows. This dominates runtime far more than Random Forest inference, which is relatively fast for a trained model.

---

### **Diagnosis**
- `build_neighbor_lookup()` computes a list of neighbor indices for every cell-year. This involves:
  - Translating IDs to indices repeatedly.
  - Constructing and looking up keys (`paste()`) for each of 6.46M rows.
- `compute_neighbor_stats()` applies `lapply()` over 6.46M list elements for each of 5 variables â†’ **~32M neighbor computations**.
- Each iteration does multiple `if` checks, subsetting, and aggregation in R loops.
- On a laptop, this explains the 86+ hour runtime.
- Random Forest `predict()` on 6.46M rows Ã— 110 features is typically minutes (even with disk I/O), so itâ€™s not the main bottleneck.

---

### **Optimization Strategy**
- **Vectorize neighbor aggregation** with `data.table` joins instead of per-row `lapply` loops.
- Precompute long-format neighbor relationships once, then compute min/max/mean by grouping.
- Avoid repeated string concatenation and indexing.

---

### **Optimized Approach (Working R Code)**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Create a long table of neighbors
# rook_neighbors_unique: list of integer vectors, one per cell in id_order
id_map <- data.table(id_order = id_order, ref_idx = seq_along(id_order))
neighbors_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src_id = id_order[i],
             nbr_id = id_order[rook_neighbors_unique[[i]]])
}))

# Expand to cell-year level
years <- unique(cell_data$year)
neighbors_expanded <- CJ(year = years, src_id = id_map$id_order)[
  neighbors_dt, on = .(src_id), allow.cartesian = TRUE]

# Join neighbor values for each source variable and aggregate
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  # Join neighbor values
  neighbors_expanded[, nbr_val := cell_data[.SD, on = .(id = nbr_id, year), get(var_name)]]
  
  # Compute stats by src_id-year
  agg_stats <- neighbors_expanded[, .(
    max_val = if (.N > 0) max(nbr_val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(nbr_val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(nbr_val, na.rm = TRUE) else NA_real_
  ), by = .(src_id, year)]
  
  # Merge back to cell_data
  setnames(agg_stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- merge(cell_data, agg_stats, by.x = c("id", "year"), by.y = c("src_id", "year"), all.x = TRUE)
}
```

---

### **Why This Works**
- Converts millions of small R loops into efficient `data.table` group operations.
- Handles all neighbors in a single pass per variable.
- Eliminates costly `paste()` and repeated lookups.
- Scales to millions of rows with reasonable memory usage on a 16 GB laptop.

---

**Expected Impact:** Reduces feature-engineering time from **86+ hours to a few hours or less**, while preserving the original estimand and trained Random Forest model.
END_RESPONSE>>>
