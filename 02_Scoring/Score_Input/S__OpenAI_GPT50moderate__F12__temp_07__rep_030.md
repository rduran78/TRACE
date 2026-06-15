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
run_id: OpenAI_F12_T07_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F12_OpenAI_temp_07_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottlenecks is **incorrect**. While `do.call(rbind, ...)` has some overhead, it is negligible compared to the actual bottleneck: **repeated full traversal of the 6.46M-row dataset for each of the 5 variables** combined with `lapply` over 6.46M elements in `compute_neighbor_stats()`. This results in ~32 million neighbor lookups and repeated NA filtering in pure R loops, which is extremely slow in R.

**Root Cause:**  
- For each variable, `compute_neighbor_stats()` loops over all rows in `neighbor_lookup` (length â‰ˆ 6.46M).
- Each iteration performs indexing and vector operations in R.
- 5 variables â†’ 5 Ã— 6.46M = 32M iterations.
- This is orders of magnitude more expensive than `do.call(rbind, ...)`.

**Optimization Strategy:**  
- **Vectorize the computation and preallocate.**
- Use a **data.table join or matrix-based approach** to compute neighbor stats in bulk.
- Avoid per-row R loops by melting neighbor relationships into a long table and aggregating with `max`, `min`, and `mean`.
- Compute all neighbor stats for all variables in a single pass, then join back to `cell_data`.

---

### **Optimized Approach**

1. Convert `neighbor_lookup` into an edge list: `(cell_idx, neighbor_idx)`.
2. Reshape `cell_data` into a `data.table` with `row_id` for direct joins.
3. For each variable, join neighbor values and aggregate `max`, `min`, `mean` using `data.table`â€™s fast grouping.
4. Merge back to original `cell_data`.

---

### **Working R Code**

```r
library(data.table)

# Convert cell_data to data.table
dt <- as.data.table(cell_data)
dt[, row_id := .I]

# Build edge list from neighbor_lookup
edges <- rbindlist(lapply(seq_along(neighbor_lookup), function(i) {
  if (length(neighbor_lookup[[i]]) == 0) return(NULL)
  data.table(row_id = i, neighbor_id = neighbor_lookup[[i]])
}))

setkey(dt, row_id)
setkey(edges, neighbor_id)

# Variables for which we compute neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute all neighbor stats in one pass per variable
for (var in neighbor_source_vars) {
  # Join neighbor values
  edges[, neighbor_val := dt[[var]][neighbor_id]]
  
  # Aggregate stats by row_id
  stats <- edges[!is.na(neighbor_val), .(
    max_val = max(neighbor_val),
    min_val = min(neighbor_val),
    mean_val = mean(neighbor_val)
  ), by = row_id]
  
  # Merge back
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var, c("_nb_max", "_nb_min", "_nb_mean")))
  dt <- merge(dt, stats, by = "row_id", all.x = TRUE)
}

# Drop helper column
dt[, row_id := NULL]

# Result
cell_data <- as.data.frame(dt)
```

---

**Why This Is Faster:**  
- Eliminates 32M small R loops in favor of 5 highly optimized `data.table` group aggregations.
- Memory-efficient: uses integer indices and avoids repeated slicing.
- On a 16 GB laptop, this approach should reduce runtime from 86+ hours to **minutes** or low **tens of minutes**.

**Preserves:**  
- Original numerical estimand.
- Trained Random Forest model (no retraining needed).
END_RESPONSE>>>
