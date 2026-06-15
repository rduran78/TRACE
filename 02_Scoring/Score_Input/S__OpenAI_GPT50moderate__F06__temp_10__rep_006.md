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
run_id: OpenAI_F06_T10_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

- The bottleneck is nested `lapply` calls over ~6.46M rows with repeated vector lookups and copying (especially in `compute_neighbor_stats`).
- Each row recomputes neighbor IDs, subset, and aggregates, leading to massive R-level overhead.
- Memory and speed concerns amplify because of panel structure (344K cells Ã— 28 years).
- The neighbor structure is reused, but operations are not vectorized or offloaded to efficient data structures.

---

**Optimization Strategy**

1. **Precompute reproducible indices** once (as is) but flatten neighbor relationships into a long data.table for join-based aggregation.
2. **Leverage data.table** or `dplyr` with grouped summarization instead of iterative R loops.
3. Expand to `(row_id, neighbor_row_id)` pairs for all (cell_year, neighbor cell_year), then compute `max`, `min`, `mean` in grouped fashion. 
4. Avoid re-materializing NA computations inside loopsâ€”handle them during aggregation.
5. Use **chunk-based processing** if memory is tight, but 6.46M rows + ~1.3M neighbors is feasible on a 16 GB machine using optimized joins.

---

**Optimized R Code (data.table approach)**

```r
library(data.table)

# Assume:
# cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell IDs matching rook_neighbors_unique
# rook_neighbors_unique: spdep::nb object
setDT(cell_data)
cell_data[, row_id := .I]

# 1. Build long neighbor pairs once
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- setNames(
  seq_len(nrow(cell_data)),
  paste(cell_data$id, cell_data$year, sep = "_")
)

# Expand neighbor relationships
pairs_list <- vector("list", length = nrow(cell_data))
row_ids <- seq_len(nrow(cell_data))

pairs_list <- lapply(row_ids, function(i) {
  ref_idx <- id_to_ref[as.character(cell_data$id[i])]
  nb_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  if (length(nb_ids) == 0) return(NULL)
  neighbor_keys <- paste(nb_ids, cell_data$year[i], sep = "_")
  neighbor_idx <- idx_lookup[neighbor_keys]
  neighbor_idx <- neighbor_idx[!is.na(neighbor_idx)]
  if (length(neighbor_idx) == 0) return(NULL)
  data.table(row_id = i, nb_row_id = as.integer(neighbor_idx))
})

neighbors_dt <- rbindlist(pairs_list)
rm(pairs_list)  # free memory
setkey(neighbors_dt, nb_row_id)

# 2. Function to compute and merge neighbor stats
compute_and_add_neighbor_features <- function(dt, var_name, neighbors_dt) {
  vals <- dt[, .(nb_row_id = row_id, value = get(var_name))]
  neighbors_dt[vals, value := i.value, on = .(nb_row_id)]
  
  # Aggregate by row_id
  stats <- neighbors_dt[, .(
    n_max = max(value, na.rm = TRUE),
    n_min = min(value, na.rm = TRUE),
    n_mean = mean(value, na.rm = TRUE)
  ), by = row_id]
  
  # Handle cases where all neighbors NA
  stats[is.infinite(n_max), `:=`(n_max = NA, n_min = NA, n_mean = NA)]
  
  # Merge back to main data
  setnames(stats, c("row_id", paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))))
  dt[stats, on = .(row_id), (names(stats)[-1]) := mget(names(stats)[-1])]
}

# 3. Apply for 5 variables efficiently
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  compute_and_add_neighbor_features(cell_data, var_name, neighbors_dt)
}

# Now cell_data has the neighbor features appended
```

---

**Why this is faster**
- Single neighbor mapping built once â†’ no repeated list traversals.
- Vectorized join-based aggregation in C via `data.table` â†’ orders of magnitude speedup.
- Memory management improved by avoiding repetitive intermediate objects.

**Expected Runtime Improvement**
- From 86+ hours â†’ likely **under 2â€“3 hours** on 16 GB RAM (depending on I/O, CPU).
- Scaling dominated by `data.table` grouping and joins, which are highly optimized in C.

This preserves:
- Original numerical estimand (exact same max/min/mean logic).
- The pre-trained Random Forest model (unchanged).
- Data integrity and reproducibility.
END_RESPONSE>>>
