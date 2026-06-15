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
run_id: OpenAI_F11_T10_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The main bottleneck is **not** Random Forest inference. The biggest slowdown comes from building and computing neighbor statistics in pure R with `lapply` over 6.46M rows * 5 variables (â‰ˆ32M expensive list operations). Each iteration repeatedly manipulates vectors, checks NAs, and combines results via `do.call(rbind, ...)`, which is extremely inefficient in Râ€™s interpreter. Loading a pre-trained RF model and calling `predict()` is comparatively fast.

---

### **Optimization Strategy**
- Precompute neighbor lookup **once** in an efficient format (already done).
- Replace R loops + `lapply` with **vectorized or compiled approaches** (data.table or matrix ops).
- Compute all neighbor stats in one pass using vectorized aggregation on a flattened neighbor->target table rather than looping per row.

#### Core Idea:
1. Expand neighbor relationships into a long table: `(source_idx, target_idx)`.
2. Join source variable values.
3. Aggregate with `data.table` to compute `max`, `min`, and `mean` by target in one efficient pass.
4. Merge back into original data.

---

### **Optimized Working R Code**

```r
library(data.table)

# Assume `cell_data` is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Convert to data.table for speed
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# Flatten neighbor relationships into long table
# neighbor_lookup is list: each element = integer vector of neighbor row indices
neighbor_dt <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(target = i, source = neighbor_lookup[[i]])
  })
)

# Make `source` and `target` correspond to row indices in dt
neighbor_dt[, target_id := dt$id[target]]
neighbor_dt[, target_year := dt$year[target]]
setkey(neighbor_dt, source)
neighbor_dt[, source_val_row := source]  # mapping to original rows in dt

# Function to compute and merge neighbor stats for one variable
compute_neighbor_stats_fast <- function(var_name) {
  var_vals <- dt[[var_name]]
  neighbor_dt[, value := var_vals[source_val_row]]
  # Remove NAs
  valid <- !is.na(neighbor_dt$value)
  
  agg <- neighbor_dt[valid, .(
    max_val = max(value),
    min_val = min(value),
    mean_val = mean(value)
  ), by = .(target)]
  
  # Attach results back to dt
  dt[agg$target, paste0(var_name, "_nbr_max") := agg$max_val]
  dt[agg$target, paste0(var_name, "_nbr_min") := agg$min_val]
  dt[agg$target, paste0(var_name, "_nbr_mean") := agg$mean_val]
}

# Apply for all neighbor vars
for (v in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  compute_neighbor_stats_fast(v)
}

# RF prediction (unchanged)
preds <- predict(trained_rf_model, newdata = dt)
```

---

### **Expected Result**
- Eliminates tens of millions of R-level loops â†’ **hours down to minutes** on 6.5M rows.
- Preserves original estimand and trained RF model.
- Bottleneck fixed: computing neighbor stats is now the dominant step, but **massively optimized**.
END_RESPONSE>>>
