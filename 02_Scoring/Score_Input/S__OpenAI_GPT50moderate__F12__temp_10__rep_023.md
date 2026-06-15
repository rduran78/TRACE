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
run_id: OpenAI_F12_T10_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F12_OpenAI_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s claim is incorrect. The dominant bottleneck is not `do.call(rbind, result)` or list binding inside `compute_neighbor_stats()`. The real performance sink is the **outer loop running `compute_neighbor_stats()` once per variable across 6.46M observations using nested R-level lapply calls**, resulting in ~32.3M lapply evaluations and repeated vector filtering operations in pure R. This dwarfs the cost of a single `rbind` at the end.

### **Diagnosis**
- `compute_neighbor_stats()` executes `lapply()` for all 6.46M rows Ã— 5 variables.
- Neighbor lookups and NA filtering (`neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`) happen repeatedly in R for each cell-year and variable.
- The per-variable loop prevents vectorization and incurs huge interpreter overhead.
- Memory: 6.4M Ã— 3 stat rows = ~155 MB per run (not fatal), but time is dominated by R loops.

### **Optimization Strategy**
- **Reject the colleagueâ€™s diagnosis**: the expensive part is the *lapply over millions of rows and repeated filtering*, not `rbind()`.
- Use **vectorized, compiled aggregation** across all variables at once.
- Precompute neighbor stats using `data.table` joins or matrix-based aggregation instead of R loops.
- Reduce repeated work by computing for all 5 variables simultaneously.
- Keep output identical for the Random Forest modelâ€™s input.

---

### **Optimized R Code (data.table approach)**

```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)

# Add row index
dt[, row_id := .I]

# Build neighbor lookup as a flat edge list
# neighbor_lookup: list of int vectors per row_id
edges <- rbindlist(lapply(seq_along(neighbor_lookup), function(i) {
  if (length(neighbor_lookup[[i]]) == 0) return(NULL)
  data.table(src = i, dst = neighbor_lookup[[i]])
}))

setkey(edges, dst)

# Variables to process
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Join neighbor values
edge_vals <- dt[edges, on = .(row_id = dst)]
# edge_vals now: src, dst, ntl, ec, ...

# Compute stats per src for all variables at once
agg <- edge_vals[, c(
  lapply(.SD, max, na.rm = TRUE),
  lapply(.SD, min, na.rm = TRUE),
  lapply(.SD, mean, na.rm = TRUE)
), by = src, .SDcols = vars]

# Some rows may have all neighbors NA: set Inf/-Inf back to NA after max/min
replace_invalid <- function(x) {
  x[is.infinite(x)] <- NA_real_
  x
}
agg[, (names(agg)[-1]) := lapply(.SD, replace_invalid), .SDcols = -1]

# Merge aggregated stats back to dt
# This creates columns like ntl_max, ntl_min, ntl_mean, etc.
max_cols <- paste0(vars, "_max")
min_cols <- paste0(vars, "_min")
mean_cols <- paste0(vars, "_mean")

setnames(agg, old = names(agg)[-1],
         new = c(max_cols, min_cols, mean_cols))

dt <- dt[agg, on = .(row_id = src)]

# Drop helper column
dt[, row_id := NULL]

# Final result: dt has original columns + neighbor stats columns
```

---

### **Why this is faster**
- Eliminates 6.46M inner lapply evaluations.
- Computes all 5 variables in **one grouped C-level aggregation**, fully vectorized in `data.table`.
- Memory efficient: edge list (~1.37M rows) is small compared to full data.
- Reduces runtime from 86+ hours to **minutes** on a standard laptop.

**Expected runtime**: dominated by a large join (6.4M rows) + grouped aggregation over 1.37M edges Ã— 5 vars â†’ practically feasible with 16 GB RAM.

---

**Bottom line:** The bottleneck is nested R loops over millions of rows, not `do.call(rbind, result)`. Rewrite the pipeline using set-based operations (`data.table`) to compute all neighbor stats in one pass.
END_RESPONSE>>>
