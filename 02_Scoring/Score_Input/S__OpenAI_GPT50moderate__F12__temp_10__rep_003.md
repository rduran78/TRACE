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
run_id: OpenAI_F12_T10_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F12_OpenAI_temp_10_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The colleagueâ€™s claim that `do.call(rbind, result)` and list-binding inside `compute_neighbor_stats()` is the main bottleneck is incorrect. While `rbind` on millions of rows is not trivial, the much deeper bottleneck is the *nested lapply logic across all cell-years and neighbor sets*. Specifically:  
- `build_neighbor_lookup()` creates a huge list (â‰ˆ6.46M entries, one per row), performing expensive lookups for every cell-year.  
- `compute_neighbor_stats()` repeats aggregation (`max`, `min`, `mean`) on neighbors at row level in pure R loops, millions of times.  
This results in massive interpreter overhead and memory churn, far outweighing the cost of the final `rbind`.  

**Correct Optimization Strategy**  
- Avoid per-row loops; express computation as **vectorized/grouped operations** instead of millions of lapply calls.  
- Convert the neighbor relationships into a long format table mapping `row_id` â†’ `neighbor_row_id`, then join once and compute aggregates using `data.table` or `dplyr`.  
- Precompute all neighbor stats in bulk for all rows and all five variables simultaneously instead of looping over rows and variables.  

**Working R Code (Optimized Approach using `data.table`)**

```r
library(data.table)

# Convert data to data.table for fast aggregation
dt <- as.data.table(cell_data)
dt[, row_id := .I]  # unique row index

# Build neighbor long table (cell-year granularity)
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- setNames(
  seq_len(nrow(dt)),
  paste(dt$id, dt$year, sep = "_")
)

# Expand rook-neighbors to cell-year level
neighbor_pairs <- rbindlist(lapply(seq_len(nrow(dt)), function(i) {
  ref_idx <- id_to_ref[as.character(dt$id[i])]
  neighbor_cell_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  if (length(neighbor_cell_ids) == 0) return(NULL)
  neighbor_keys <- paste(neighbor_cell_ids, dt$year[i], sep = "_")
  neighbor_rows <- idx_lookup[neighbor_keys]
  neighbor_rows <- neighbor_rows[!is.na(neighbor_rows)]
  if (length(neighbor_rows) == 0) return(NULL)
  data.table(row_id = i, neighbor_row_id = neighbor_rows)
}), use.names = TRUE)

setkey(neighbor_pairs, neighbor_row_id)
setkey(dt, row_id)

# Gather both tables for fast joins
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Join neighbor values and aggregate in one pass
agg <- neighbor_pairs[
  dt, on = .(neighbor_row_id = row_id)
][, lapply(.SD, function(x) {
    list(max = max(x, na.rm = TRUE),
         min = min(x, na.rm = TRUE),
         mean = mean(x, na.rm = TRUE))
  }),
  by = row_id, .SDcols = neighbor_vars]

# Reshape so original data gets these features
# Flatten the list columns
flat_cols <- data.table(row_id = agg$row_id)
for (v in neighbor_vars) {
  flat_cols[[paste0(v, "_max")]] <- sapply(agg[[v]], `[[`, "max")
  flat_cols[[paste0(v, "_min")]] <- sapply(agg[[v]], `[[`, "min")
  flat_cols[[paste0(v, "_mean")]] <- sapply(agg[[v]], `[[`, "mean")
}

# Attach to main dt
setkey(flat_cols, row_id)
dt <- dt[flat_cols]

# Result: dt now includes all neighbor-based features efficiently computed
```

**Why This Is Faster**  
- Reduces millions of R-level iterations down to vectorized data.table joins and aggregations.  
- Handles all neighbor stats *once per variable* across all rows, not per-row/per-variable with lapply.  
- Avoids repetitive `do.call(rbind)` entirely.  

**Expected Runtime Gain**  
From 86+ hours to potentially under 1 hour on a standard laptop for ~6.46M rows, given `data.table`'s efficiency with large aggregates. The trained Random Forest model remains usable as variable names and estimands are preserved.
END_RESPONSE>>>
