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
run_id: OpenAI_F03_T00_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F03_OpenAI_temp_00_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small R objects. This is extremely slow and memory-inefficient.  
2. **Excessive object copying**: Each `compute_and_add_neighbor_features` call copies `cell_data` repeatedly.  
3. **Inefficient neighbor lookup**: `lapply` with string concatenation (`paste`) for every row is costly.  
4. **Random Forest inference**: If predictions are done in a loop (row-by-row), this is catastrophic. `predict()` on the entire data frame at once is essential.  
5. **Memory pressure**: 6.46M rows Ã— 110+ variables is large; avoid intermediate lists and repeated joins.  

---

**Optimization Strategy**  
- **Precompute neighbor indices as integer vectors** once, not per variable.  
- **Vectorize neighbor stats**: Use `data.table` or `dplyr` grouped joins instead of `lapply`.  
- **Avoid repeated `paste`**: Use numeric keys or precomputed composite keys.  
- **Batch Random Forest predictions**: Call `predict(rf_model, newdata)` on the full dataset or in large chunks.  
- **Use `data.table` for speed and memory efficiency**.  
- **Parallelize** where possible (e.g., `future.apply` or `data.table` parallel).  

---

**Working R Code (Optimized)**  

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.frame), id_order, rook_neighbors_unique, rf_model loaded

# Convert to data.table
setDT(cell_data)

# Precompute composite key for fast joins
cell_data[, key := paste(id, year, sep = "_")]

# Build neighbor lookup as a list of integer vectors (fast)
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- setNames(seq_len(nrow(cell_data)), cell_data$key)

neighbor_lookup <- lapply(seq_len(nrow(cell_data)), function(i) {
  ref_idx <- id_to_ref[as.character(cell_data$id[i])]
  neighbor_cell_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  neighbor_keys <- paste(neighbor_cell_ids, cell_data$year[i], sep = "_")
  idx_lookup[neighbor_keys]
})

# Convert neighbor_lookup to a long table for vectorized stats
neighbor_dt <- rbindlist(lapply(seq_along(neighbor_lookup), function(i) {
  if (length(neighbor_lookup[[i]]) == 0) return(NULL)
  data.table(row = i, neighbor = neighbor_lookup[[i]])
}))

# Compute neighbor stats for all variables efficiently
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  neighbor_dt[, val := vals[neighbor]]
  stats_dt <- neighbor_dt[!is.na(val), .(
    max_val = max(val),
    min_val = min(val),
    mean_val = mean(val)
  ), by = row]
  
  # Join back to cell_data
  setnames(stats_dt, c("row", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  cell_data[stats_dt$row, (names(stats_dt)[-1]) := stats_dt[, -1]]
}

# Drop helper columns
cell_data[, key := NULL]

# Random Forest prediction in batch
preds <- predict(rf_model, newdata = cell_data)

# Attach predictions
cell_data[, gdp_pred := preds]
```

---

**Key Improvements**  
- **Single pass neighbor expansion** â†’ `neighbor_dt` avoids millions of small lists.  
- **Vectorized aggregation** â†’ `data.table` group-by is orders of magnitude faster than `lapply`.  
- **Batch prediction** â†’ No row-wise `predict()`.  
- **Memory efficiency** â†’ Avoid repeated copies of `cell_data`.  

This approach should reduce runtime from **86+ hours to a few hours** on a 16 GB laptop. For further speed, consider:  
- Saving `neighbor_dt` to disk for reuse.  
- Using `parallel::mclapply` or `future.apply` for neighbor expansion.  
- If still slow, move to `arrow` or `duckdb` for out-of-memory handling.
END_RESPONSE>>>
