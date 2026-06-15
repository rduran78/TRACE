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
run_id: OpenAI_F12_T10_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F12_OpenAI_temp_10_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The claim that the main bottleneck is `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` is **incorrect**. The audit indicates that the dominant cost lies in repeatedly iterating 6.46 million times over a large list of neighbor indices while performing `lapply` for each variable. Each iteration executes multiple R-level loops and vector operations in pure R, which is inefficient for this scale. The overhead of `do.call(rbind, result)` is minor compared to the nested R loops and repeated data slicing across millions of rows.

**Deeper bottleneck:**  
- `compute_neighbor_stats()` calls `lapply` on `neighbor_lookup`, which is length **6.46 million**, five times (once per variable).
- Each iteration does vector subset + filtering + `c(max, min, mean)`.  
This is extremely expensive at scale due to R function-call overhead and memory churn.

**Correct optimization strategy:**  
- Move from millions of R-level iterations to vectorized aggregation using **data.table** or **dplyr**.
- Precompute a long-format neighbor table (cell-year â†’ neighbor-cell-year) and directly compute max, min, mean per source variable using grouped summarization in **C-optimized methods**.
- Join the aggregated stats back to `cell_data`.
- Preserve numerical estimand by computing identical summary measures.

---

### **Optimized Approach Using `data.table`**

```r
library(data.table)

# Convert cell_data to data.table
cell_dt <- as.data.table(cell_data)

# Precompute neighbor links (cell-year â†’ neighbor-cell-year)
# Flatten neighbor_lookup into a long table
make_neighbor_table <- function(cell_data, neighbor_lookup) {
  cell_year <- paste(cell_data$id, cell_data$year, sep = "_")
  from <- rep(cell_year, times = lengths(neighbor_lookup))
  to   <- paste(cell_data$id[unlist(neighbor_lookup)],
                cell_data$year[rep(seq_len(nrow(cell_data)),
                                    times = lengths(neighbor_lookup))],
                sep = "_")
  data.table(from = from, to = to)
}

neighbor_dt <- make_neighbor_table(cell_dt, neighbor_lookup)

# Add row ids for join
cell_dt[, key_id := paste(id, year, sep = "_")]
setkey(cell_dt, key_id)
setkey(neighbor_dt, to)

# For each variable, compute neighbor stats efficiently
source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in source_vars) {
  neighbor_dt[, val := cell_dt[to, get(var)]]

  # Compute grouped stats using fast C-level aggregation
  agg_stats <- neighbor_dt[!is.na(val), .(
    max_val = max(val),
    min_val = min(val),
    mean_val = mean(val)
  ), by = .(from)]

  # Join back to cell_dt
  setkey(agg_stats, from)
  cell_dt[agg_stats, paste0(var, "_nbr_max") := i.max_val, on = .(key_id = from)]
  cell_dt[agg_stats, paste0(var, "_nbr_min") := i.min_val, on = .(key_id = from)]
  cell_dt[agg_stats, paste0(var, "_nbr_mean") := i.mean_val, on = .(key_id = from)]

  neighbor_dt[, val := NULL]  # cleanup for next variable
}

# Drop helper column if needed
cell_dt[, key_id := NULL]

# Final optimized cell_data is cell_dt converted back if needed
cell_data <- as.data.frame(cell_dt)
```

---

### **Why This Fixes the Bottleneck**
- Replaces **6.46M Ã— 5 nested R loops** with **vectorized joins + group aggregation** in compiled C code via `data.table`.
- Eliminates repeated slicing and filtering operations inside millions of `lapply` calls.
- Achieves massive speed-up (from hours to minutes on a standard laptop).
- Preserves numerical estimands (`max`, `min`, `mean`) and leaves the trained Random Forest intact.

**Expected runtime improvement:** From ~86 hours to a few minutes to <1 hour depending on I/O, due to efficient grouping and parallelizable operations.
END_RESPONSE>>>
