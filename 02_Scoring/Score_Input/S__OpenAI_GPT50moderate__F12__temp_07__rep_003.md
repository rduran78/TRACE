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
run_id: OpenAI_F12_T07_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F12_OpenAI_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` is the main bottleneck is incorrect. While `do.call(rbind, ...)` on millions of rows is non-trivial, the true bottleneck is the *nested lapply-based per-row neighbor traversal and repeated indexing* across ~6.46 million rows Ã— 5 variables. Each call repeatedly slices `vals[idx]` and performs vector filtering, which is extremely expensive in Râ€™s interpreted loop context.

The `build_neighbor_lookup()` itself creates a massive list of integer vectors for all rows, which is necessary but then repeatedly iterated in `compute_neighbor_stats()` for each variable. So the real bottleneck is **repeated R-level interpretation and memory allocation per row Ã— variable** rather than the final `rbind`.

---

### **Optimization Strategy**
- Precompute reusable structures (neighbor index list).
- Replace R loops and repeated indexing with **vectorized or compiled operations** (e.g., matrix-based aggregation or `Rcpp`/`data.table`).
- Avoid per-variable repeated traversal: compute all neighbor summaries in a single pass.
- Use **matrix aggregation** with cumulative sums or `rowsum` for efficiency.

---

### **Optimized Approach**
Compute neighbor max, min, mean for all variables in one pass using `data.table`:

#### **Working R Code**
```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)

# Precompute neighbor pairs (long format)
neighbor_pairs <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(src = i, nbr = neighbor_lookup[[i]])
  })
)

# Source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt relevant columns for efficient join
nbr_vals <- dt[, .SD, .SDcols = neighbor_source_vars]
nbr_vals[, row_id := .I]

# Join neighbor pairs with values
neighbor_pairs <- merge(neighbor_pairs, nbr_vals, by.x = "nbr", by.y = "row_id", all.x = TRUE)

# Compute aggregate stats by src for all variables
agg_stats <- neighbor_pairs[, lapply(.SD, function(x) {
  list(max = if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE),
       min = if (all(is.na(x))) NA_real_ else min(x, na.rm = TRUE),
       mean = if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE))
}), by = src, .SDcols = neighbor_source_vars]

# Reshape columns to wide format: var_max, var_min, var_mean
long_cols <- melt(agg_stats, id.vars = "src", variable.name = "var", value.name = "vals")
long_cols[, c("variable", "stat") := tstrsplit(var, "_", fixed = TRUE)]

# Recast to wide
final_stats <- dcast(long_cols, src ~ variable + stat, value.var = "vals")

# Merge back to original data
dt <- cbind(dt, final_stats[order(src)][, -"src"])

cell_data <- as.data.frame(dt)
```

---

### **Why This Works**
- One pass aggregation using `data.table` avoids millions of small R loops.
- No repeated slicing per variable.
- Minimal memory overhead relative to original approach.
- Fully preserves original numeric estimands (max, min, mean).

**Expected Impact:** Reduces runtime from 86+ hours to a few minutes on 6.5M rows, well within laptop constraints.
END_RESPONSE>>>
