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
run_id: OpenAI_F11_T00_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F11_OpenAI_temp_00_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that Random Forest inference is the main bottleneck is incorrect. The code shows that the pipeline spends significant time in `build_neighbor_lookup` and `compute_neighbor_stats`, both of which use deeply nested `lapply` calls over **6.46 million rows** and repeatedly manipulate large vectors. This is an **O(N Ã— neighbors)** operation done multiple times (once per neighbor source variable), which dominates runtime. Random Forest prediction on 6.46M rows with 110 features is relatively fast compared to these repeated R loops.

---

**Optimization Strategy:**  
- Precompute neighbor indices **once** in a vectorized manner.
- Replace repeated `lapply` loops with **matrix operations** or `data.table` joins.
- Compute all neighbor statistics in a single pass rather than looping over variables.
- Avoid repeated string concatenation and lookups inside loops.

---

**Optimized R Code (data.table approach):**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Add a unique key for each cell-year
cell_data[, cell_year := paste(id, year, sep = "_")]

# Precompute neighbor lookup as a long table
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- setNames(seq_len(nrow(cell_data)), cell_data$cell_year)

# Build neighbor pairs (cell_year -> neighbor_cell_year)
neighbor_pairs <- rbindlist(lapply(seq_len(nrow(cell_data)), function(i) {
  ref_idx <- id_to_ref[as.character(cell_data$id[i])]
  neighbor_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  if (length(neighbor_ids) == 0) return(NULL)
  data.table(
    cell_year = cell_data$cell_year[i],
    neighbor_cell_year = paste(neighbor_ids, cell_data$year[i], sep = "_")
  )
}))

# Join to get neighbor values for all variables in one go
neighbor_pairs <- neighbor_pairs[
  , .(neighbor_idx = idx_lookup[neighbor_cell_year]), by = cell_year
]
neighbor_pairs <- neighbor_pairs[!is.na(neighbor_idx)]

# Melt neighbor source variables for aggregation
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
long_neighbors <- melt(
  cell_data[, c("cell_year", vars), with = FALSE],
  id.vars = "cell_year",
  variable.name = "var_name",
  value.name = "value"
)

# Join neighbor values
neighbor_values <- neighbor_pairs[
  , .(cell_year, neighbor_idx)
][long_neighbors, on = .(neighbor_idx = cell_year)]

# Compute stats by cell_year and var_name
neighbor_stats <- neighbor_values[
  !is.na(value),
  .(max_val = max(value), min_val = min(value), mean_val = mean(value)),
  by = .(cell_year, var_name)
]

# Reshape back to wide and merge
neighbor_stats_wide <- dcast(
  neighbor_stats,
  cell_year ~ var_name,
  value.var = c("max_val", "min_val", "mean_val")
)

cell_data <- merge(cell_data, neighbor_stats_wide, by = "cell_year", all.x = TRUE)
```

---

**Why this works:**  
- Eliminates repeated loops over 6.46M rows Ã— 5 variables.
- Uses `data.table` joins and aggregation for efficient grouping.
- Computes all neighbor statistics in one pass.
- Preserves the trained Random Forest model and original estimand.

**Expected impact:**  
Reduces runtime from **86+ hours to a few hours or less**, depending on I/O and memory.
END_RESPONSE>>>
