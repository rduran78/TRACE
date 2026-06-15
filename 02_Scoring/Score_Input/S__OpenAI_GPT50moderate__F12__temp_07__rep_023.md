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
run_id: OpenAI_F12_T07_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F12_OpenAI_temp_07_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s diagnosis is incorrect. The real bottleneck is the repeated **full pass over 6.46M rows Ã— 28 years** for each of 5 variables in `compute_neighbor_stats()`, performing redundant lookups and filtering for every `var_name`. The `do.call(rbind, result)` cost is trivial compared to the sheer number of R function calls (millions of small closures) and repeated NA filtering logic.

### **Correct Diagnosis**
- `neighbor_lookup` is computed once (good), but `compute_neighbor_stats()`:
  - Loops over **every row (6.46M)** for each variable.
  - Performs repeated index-based extraction and NA filtering.
- This results in ~32M iterations of small R functions (6.46M Ã— 5 vars) â†’ enormous interpreter overhead.
- The actual bottleneck: **interpreted R loops over millions of rows**.

### **Optimization Strategy**
- Vectorize neighbor aggregation:
  - Flatten all neighbor relations into a single data frame with `(source, target)` pairs.
  - Join variable values once.
  - Compute grouped `max/min/mean` using `data.table` or `dplyr` in C-optimized code.
- Avoid recalculating per variable â†’ process all 5 variables in one grouped operation.
- This cuts complexity from O(n Ã— vars) interpreted loops to O(|edges| Ã— vars) in compiled code.

---

### **Optimized Approach with `data.table`**

```r
library(data.table)

# Convert cell_data to data.table for speed
setDT(cell_data)

# Precompute neighbor pairs (source: current row index, target: neighbor row index)
neighbor_pairs <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(source = i, target = neighbor_lookup[[i]])
  })
)

# Melt only needed variables into long format
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
long_vals <- melt(
  cell_data[, .(row_id = .I, year, id, (vars)), with = FALSE],
  id.vars = c("row_id"),
  measure.vars = vars,
  variable.name = "var",
  value.name = "val"
)

# Join neighbor values
setkey(long_vals, row_id)
neighbor_vals <- neighbor_pairs[long_vals, on = .(target = row_id)]
# neighbor_vals: source | target | var | val

# Aggregate by source and var
agg_stats <- neighbor_vals[, .(
  max_val = if (.N == 0) NA_real_ else max(val, na.rm = TRUE),
  min_val = if (.N == 0) NA_real_ else min(val, na.rm = TRUE),
  mean_val = if (.N == 0) NA_real_ else mean(val, na.rm = TRUE)
), by = .(source, var)]

# Reshape wide to join back
agg_wide <- dcast(
  agg_stats,
  source ~ var,
  value.var = c("max_val", "min_val", "mean_val")
)

# Merge back into cell_data
cell_data <- cbind(cell_data, agg_wide[order(source)][, -"source"])
```

---

### **Why This Works**
- `neighbor_lookup` is used once to build a flat edges table.
- All 5 variables processed in **one pass** using efficient `data.table` grouping.
- Eliminates millions of interpreted R loops and repeated filtering.
- Memory-efficient because we only keep necessary columns and work with integer indices.

**Expected speed:** From 86+ hours â†’ likely under 1 hour on 16 GB RAM laptop for 6.46M rows.

**Preserves:** Original Random Forest model and all numerical estimands.
END_RESPONSE>>>
