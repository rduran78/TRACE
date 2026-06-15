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
run_id: OpenAI_F12_T10_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F12_OpenAI_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The evidence does **not** support your colleagueâ€™s claim.  
`do.call(rbind, result)` and small list-binding inside `compute_neighbor_stats()` are negligible relative to the cost of **repeated R loops over 6.46M rows x 5 variables** (â‰ˆ32M iterations) doing repeated subsetting (`vals[idx]`) and NA filtering in interpreted R. The bottleneck is those repeated `lapply` calls with many small operations in pure R, not the final `rbind`.

### **Diagnosis**
- For each variable, `compute_neighbor_stats()` re-loops over **all rows** and repeatedly fetches neighbor indices and subsets `vals` â†’ severe interpretive overhead.
- Core cost is O(N Ã— K) where N â‰ˆ 6.46M and K â‰ˆ mean neighbors (â‰ˆ4), entirely in R loops.
- `build_neighbor_lookup()` is computed once and is relatively cheap.
- `do.call(rbind, â€¦)` at the end of each loop is trivial compared to 6.46MÃ—5 slice/filter operations.

### **Optimization Strategy**
- Flatten neighbor index structure into a vectorized form once and avoid per-row R loops.
- Use **matrix preallocation** and **vectorized operations in C-backed packages** (e.g., `data.table` or `collapse`) or **parallel apply**.
- Compute aggregations (max, min, mean) via `vapply()` or, better, by joining a long-form mapping table and using `data.table` group stats.
- Avoid redundant recomputation for each variable: melt neighbor relations and join instead.

---

### **Working Optimized Code (data.table)**

```r
library(data.table)

# Convert main data to data.table
dt <- as.data.table(cell_data)
setkey(dt, id, year)  # fast joins

# Build long-form neighbor mapping once
neighbor_dt <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(
      row_id = i,
      nb_id  = neighbor_lookup[[i]]
    )
  })
)

# Attach year + IDs for join
neighbor_dt[, year := dt$year[row_id]]
neighbor_dt[, id   := dt$id[nb_id]]

# Join neighbor values and summarize
for (var_name in neighbor_source_vars) {
  temp <- merge(neighbor_dt, dt[, .(id, year, val = get(var_name))],
                by = c("id", "year"), all.x = TRUE)
  
  # Remove NAs and compute per row_id
  agg <- temp[!is.na(val), .(
    max = max(val),
    min = min(val),
    mean = mean(val)
  ), by = row_id]
  
  # Fill missing to keep alignment
  out <- matrix(NA_real_, nrow(dt), 3)
  out[agg$row_id, ] <- as.matrix(agg[, .(max, min, mean)])
  
  # Assign to dt
  nm <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  dt[, (nm) := as.data.table(out)]
}
```

---

### **Why This Fix Works**
- Eliminates 6.46M row-loop per variable â†’ replaced with efficient `data.table` join + group aggregation in compiled C.
- Memory use controlled by reusing `neighbor_dt`.
- Preserves the trained Random Forest and original estimand.
- On a standard laptop, this change can cut runtime from **86+ hours to under ~1 hour** depending on disk I/O and threads.

**Summary:** The true bottleneck is the massive interpreted R loops with repeated subsetting, **not** `do.call(rbind)`. The correct optimization is to vectorize neighbor aggregation and compute across all variables via fast joins and group operations.
END_RESPONSE>>>
