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
run_id: OpenAI_F12_T07_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F12_OpenAI_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding are the main bottlenecks is incorrect. The true bottleneck is the repeated R-level looping over 6.46M rows combined with repeated neighbor lookups and data subsetting inside `compute_neighbor_stats()`. Each `lapply` call iterates over millions of elements for each of the 5 variables, resulting in extremely high interpretation overhead and memory churn. `do.call(rbind, result)` is trivial compared to the cost of these millions of function calls and repeated vector allocations.

---

**Correct Optimization Strategy:**  
- Eliminate the per-row R loops by vectorizing the neighbor aggregation.
- Precompute neighbor relationships in a long/tidy format and use a fast aggregation backend (e.g., `data.table` or `dplyr`).
- Compute all neighbor stats in a single grouped operation instead of looping per variable and per row.
- Preserve the trained Random Forest model and the original estimand by keeping the statistics identical (max, min, mean).

---

**Optimized Approach (data.table):**  
1. Convert neighbor relationships to a long table mapping each cell-year to its neighbors.
2. Join neighbor values for all variables in long format.
3. Aggregate max, min, mean using `data.table` grouping by focal cell-year and variable.

---

### **Working R Code**

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Expand neighbor_lookup into long form: (focal_row, neighbor_row)
neighbor_dt <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(focal = i, neighbor = neighbor_lookup[[i]])
  })
)

# Melt the 5 source variables into long format
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_long <- melt(
  cell_data[, .(row_id = .I, id, year, ntl, ec, pop_density, def, usd_est_n2)],
  id.vars = c("row_id", "id", "year"),
  measure.vars = neighbor_source_vars,
  variable.name = "var",
  value.name = "val"
)

# Join neighbor info with long table
neighbor_long <- merge(neighbor_dt, cell_long, by.x = "neighbor", by.y = "row_id", allow.cartesian = TRUE)

# Compute stats by focal cell-year and variable
agg_stats <- neighbor_long[
  !is.na(val),
  .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)),
  by = .(focal, var)
]

# Reshape back to wide and merge into cell_data
agg_wide <- dcast(agg_stats, focal ~ var, value.var = c("nb_max", "nb_min", "nb_mean"))

# Merge back using row_id
cell_data <- cbind(cell_data, agg_wide[order(focal)][, -"focal"])
```

---

**Why this is faster:**  
- Converts expensive R loops (millions of iterations) into compiled C-level aggregation via `data.table`.
- Single pass over data instead of 5 separate full scans.
- Memory-efficient joins and grouping.

**Expected performance:**  
Hours shrink to minutes on a 16 GB laptop because the bottleneck was R-level looping, not `rbind`.  

**Preserves:**  
- Original Random Forest model (unchanged).
- Same numerical estimands (max, min, mean per neighbor set).  

**Summary:** Reject the colleagueâ€™s diagnosis; the bottleneck is Râ€™s per-row loops and repeated subset operations. The fix is vectorized aggregation using `data.table`.
END_RESPONSE>>>
