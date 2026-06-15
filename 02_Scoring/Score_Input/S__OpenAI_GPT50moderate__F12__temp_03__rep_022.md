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
run_id: OpenAI_F12_T03_R022
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 22
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F12_OpenAI_temp_03_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding in `compute_neighbor_stats()` are the main bottlenecks is incorrect. While `rbind` on millions of rows has some cost, the dominant bottleneck is the repeated full pass over **6.46 million rows Ã— 5 variables** in pure R loops. Each `compute_neighbor_stats()` call iterates through every cell-year row and repeatedly subsets `vals[idx]` using neighbor indices, causing massive overhead in Râ€™s interpreter and memory allocations. This is far more expensive than the final `rbind`.

---

**Correct Optimization Strategy:**  
- Avoid per-row R loops and repeated subsetting.
- Precompute neighbor relationships in a **long format table** and aggregate with **vectorized operations** (e.g., `data.table` or `dplyr`).
- Compute all neighbor stats for all variables in one grouped aggregation rather than looping over variables.
- Preserve the trained Random Forest model and original estimand by keeping the same numeric outputs.

---

### **Optimized Approach Using `data.table`**

1. Expand neighbor relationships into a long table: `(cell_year, neighbor_cell_year)`.
2. Join neighbor values for all source variables.
3. Aggregate with `max`, `min`, and `mean` in a single grouped operation.
4. Merge back into `cell_data`.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute keys for fast joins
cell_data[, cell_year := paste(id, year, sep = "_")]

# Build neighbor lookup in long format
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- setNames(seq_len(nrow(cell_data)), cell_data$cell_year)

# Expand neighbor relationships
neighbor_list <- lapply(seq_len(nrow(cell_data)), function(i) {
  ref_idx <- id_to_ref[as.character(cell_data$id[i])]
  neighbor_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  if (length(neighbor_ids) == 0) return(NULL)
  neighbor_keys <- paste(neighbor_ids, cell_data$year[i], sep = "_")
  valid_neighbors <- neighbor_keys[neighbor_keys %in% names(idx_lookup)]
  if (length(valid_neighbors) == 0) return(NULL)
  data.table(cell_year = cell_data$cell_year[i], neighbor_cell_year = valid_neighbors)
})

neighbor_dt <- rbindlist(neighbor_list, use.names = TRUE, fill = TRUE)

# Join neighbor values
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_dt <- merge(neighbor_dt, cell_data[, c("cell_year", vars), with = FALSE],
                     by.x = "neighbor_cell_year", by.y = "cell_year", all.x = TRUE)

# Aggregate neighbor stats
agg_dt <- neighbor_dt[, .(
  ntl_max = max(ntl, na.rm = TRUE),
  ntl_min = min(ntl, na.rm = TRUE),
  ntl_mean = mean(ntl, na.rm = TRUE),
  ec_max = max(ec, na.rm = TRUE),
  ec_min = min(ec, na.rm = TRUE),
  ec_mean = mean(ec, na.rm = TRUE),
  pop_density_max = max(pop_density, na.rm = TRUE),
  pop_density_min = min(pop_density, na.rm = TRUE),
  pop_density_mean = mean(pop_density, na.rm = TRUE),
  def_max = max(def, na.rm = TRUE),
  def_min = min(def, na.rm = TRUE),
  def_mean = mean(def, na.rm = TRUE),
  usd_est_n2_max = max(usd_est_n2, na.rm = TRUE),
  usd_est_n2_min = min(usd_est_n2, na.rm = TRUE),
  usd_est_n2_mean = mean(usd_est_n2, na.rm = TRUE)
), by = cell_year]

# Merge back into main data
cell_data <- merge(cell_data, agg_dt, by = "cell_year", all.x = TRUE)
```

---

**Why This Works:**  
- Eliminates 6.46M per-row R loops and repeated subsetting.
- Uses a single vectorized aggregation for all variables.
- Reduces runtime from **86+ hours to minutes** on a standard laptop.
- Preserves the original estimand and avoids retraining the model.

**Bottom line:** The real bottleneck is the repeated interpreted loops and subsetting in `compute_neighbor_stats()`, not `do.call(rbind, ...)`. The correct fix is a **vectorized, join-and-aggregate approach** using `data.table`.
END_RESPONSE>>>
