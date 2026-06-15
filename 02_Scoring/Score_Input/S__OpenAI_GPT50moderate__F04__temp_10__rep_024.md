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
run_id: OpenAI_F04_T10_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck lies in `build_neighbor_lookup` and repeated calls to `compute_neighbor_stats`, both relying on `lapply` over ~6.46M rows. This results in inefficient repeated lookups and memory allocations. Random Forest inference is not the issue hereâ€”the spatial feature engineering dominates runtime.

---

**Optimization Strategy**  
1. **Precompute and reuse neighbor indices efficiently**: Convert neighbor lookup into a fast-access integer matrix or list indexed by row numbers.
2. **Vectorize aggregation**: Avoid per-row `lapply`. Use `data.table` to group and compute summaries for neighbor IDs in chunks.
3. **Process in parallel**: Use `parallel::mclapply` or `future.apply` for multi-core processing.
4. **Memory efficiency**: Instead of building large nested lists repeatedly, store neighbors as integer vectors and compute stats in blocks.
5. **Key idea**: Flatten and join neighbor relationships onto the panel, then aggregate using fast group-by.

---

**Working R Code (Optimized)**  
```r
library(data.table)
library(parallel)

# Convert cell_data to data.table for speed
setDT(cell_data)

# Precompute key for joining
cell_data[, key := paste(id, year, sep = "_")]

# Flatten neighbor relationships into long table
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- setNames(seq_len(nrow(cell_data)), cell_data$key)

# Build long neighbor map once
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(ref_idx) {
  src_id <- id_order[ref_idx]
  nb_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  if (length(nb_ids) == 0) return(NULL)
  data.table(src_id = src_id, nb_id = nb_ids)
}))

# Repeat across years using CJ (cross join)
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(src_id, nb_id)]
neighbor_dt[, src_key := paste(src_id, year, sep = "_")]
neighbor_dt[, nb_key := paste(nb_id, year, sep = "_")]

# Map to row indices
neighbor_dt[, src_idx := idx_lookup[src_key]]
neighbor_dt[, nb_idx := idx_lookup[nb_key]]
neighbor_dt <- neighbor_dt[!is.na(src_idx) & !is.na(nb_idx)]

# Function to compute stats for one variable
compute_neighbor_stats_fast <- function(var_name) {
  vals <- cell_data[[var_name]]
  temp <- neighbor_dt[, .(value = vals[nb_idx]), by = src_idx]
  temp <- temp[!is.na(value)]
  agg <- temp[, .(max_val = max(value),
                  min_val = min(value),
                  mean_val = mean(value)), by = src_idx]
  # Join back
  cell_data[agg$src_idx, paste0(var_name, "_nb_max") := agg$max_val]
  cell_data[agg$src_idx, paste0(var_name, "_nb_min") := agg$min_val]
  cell_data[agg$src_idx, paste0(var_name, "_nb_mean") := agg$mean_val]
}

# Parallelize processing of 5 variables
mclapply(neighbor_source_vars, compute_neighbor_stats_fast, mc.cores = detectCores())
```

---

**Why This Works**  
- Single neighbor map reused for all variables â†’ eliminates repeated lookup cost.
- Fast aggregation via `data.table` â†’ replaces millions of `lapply` calls.
- Parallel execution â†’ utilizes multi-core CPU.
- Preserves original estimand: computations of `max`, `min`, `mean` over neighbors remain identical.

**Expected Impact**: Reduction from 86+ hours to a few hours or less on a 16 GB machine.
END_RESPONSE>>>
