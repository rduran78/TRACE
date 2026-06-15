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
run_id: OpenAI_F12_T07_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F12_OpenAI_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding are the main bottlenecks is incorrect. While `do.call(rbind, ...)` over millions of elements has a cost, the true performance killer is **the nested `lapply` over ~6.46 million rows in `compute_neighbor_stats()` combined with repeated filtering and aggregation of neighbor values**. Each iteration computes `max`, `min`, and `mean` individually, repeatedly indexing long vectors, causing extreme overhead in Râ€™s interpreter.

**Root Cause:**  
- 6.46M iterations Ã— 5 variables = ~32M small R function calls.
- Each call does allocations, NA filtering, and numeric ops.
- R loops are slow; vectorization is needed.

---

### **Correct Optimization Strategy**
1. **Precompute once:** `neighbor_lookup` remains unchanged â€“ good.
2. **Replace per-row `lapply` with vectorized aggregation:**  
   Flatten `neighbor_lookup` into a long two-column mapping (row â†’ neighbor), then join with `data` values and compute `max`, `min`, `mean` by `row_id` in a single grouped operation using `data.table` or `dplyr`.
3. **Preserve numerical equivalence:** Handle `NA` exactly as original logic.

---

### **Optimized R Implementation** (using `data.table`)

```r
library(data.table)

# Assume: cell_data is a data.table with columns id, year, and neighbor source vars
# neighbor_lookup: list of integer vectors (indices of neighbors per row)

# STEP 1: Flatten neighbor_lookup into long table
row_ids <- seq_along(neighbor_lookup)
edges <- data.table(
  row_id = rep(row_ids, lengths(neighbor_lookup)),
  neighbor_id = unlist(neighbor_lookup, use.names = FALSE)
)

# Convert cell_data to data.table if not already
setDT(cell_data)

# STEP 2: Function to compute neighbor stats for one variable
compute_neighbor_stats_dt <- function(var_name) {
  vals <- cell_data[[var_name]]
  # Join neighbor values
  edges[, val := vals[neighbor_id]]
  # Group by row_id and compute stats ignoring NA
  agg <- edges[, .(
    max_val = if (.N == 0 || all(is.na(val))) NA_real_ else max(val, na.rm = TRUE),
    min_val = if (.N == 0 || all(is.na(val))) NA_real_ else min(val, na.rm = TRUE),
    mean_val = if (.N == 0 || all(is.na(val))) NA_real_ else mean(val, na.rm = TRUE)
  ), by = row_id]
  setnames(agg, c("row_id", 
                  paste0(var_name, "_nbr_max"), 
                  paste0(var_name, "_nbr_min"), 
                  paste0(var_name, "_nbr_mean")))
  agg
}

# STEP 3: Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
results_list <- lapply(neighbor_source_vars, compute_neighbor_stats_dt)

# Merge results back into cell_data
for (res in results_list) {
  cell_data[res$row_id, names(res)[-1] := res[, -1]]
}
```

---

### **Why This Is Faster**
- Eliminates millions of R-level loops.
- Uses `data.table`â€™s optimized C-level grouping.
- Single pass per variable instead of per row.
- Memory-friendly because `edges` is reused for all variables.

---

**Expected Impact:**  
> From 86+ hours â†’ likely under 1 hour on 6.46M rows Ã— 1.37M neighbor edges (RAM permitting), preserving the original numerical estimand and leaving the trained Random Forest model intact.
END_RESPONSE>>>
