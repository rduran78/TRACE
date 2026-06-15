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
run_id: OpenAI_F04_T07_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck lies in the repeated R `lapply` loops and heavy use of string concatenation and list lookups in `build_neighbor_lookup` and `compute_neighbor_stats`. Generating neighbor features for 6.46M rows across 5 variables involves millions of small object allocations, high overhead for paste-based keys, and repeated filtering. Random Forest inference is fast relative to this step; the spatial neighbor feature construction dominates runtime.

---

**Optimization Strategy**  
1. **Vectorize and Precompute**:
   - Avoid repeated `paste()` calls inside loops by precomputing keys or using numeric indexing.
   - Convert neighbor relationships to integer indices upfront, eliminating costly lookups and string operations.
2. **Use `data.table` for speed and memory efficiency**:
   - Utilize keyed joins instead of looping.
3. **Compute neighbor stats in bulk**:
   - Flatten neighbor relationships into a long table, join values once, and aggregate using fast grouping.
4. **Parallelization**:
   - Use `parallel` or `future.apply` for multi-core execution if possible.

---

**Working R Code** (vectorized + `data.table` approach):

```r
library(data.table)

# Convert data.frame to data.table for efficiency
dt <- as.data.table(cell_data)

# Create a unique numeric key for each cell-year row
dt[, row_id := .I]
dt[, key := paste(id, year, sep = "_")]

# Precompute neighbor lookup as integer indices
id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
dt_idx_lookup <- setNames(dt$row_id, dt$key)

neighbor_lookup_fast <- vector("list", length = nrow(dt))
# Build all neighbor indices without paste inside loop
for (i in seq_len(nrow(dt))) {
  ref_idx <- id_to_idx[as.character(dt$id[i])]
  neighbor_cell_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  neighbor_keys <- paste(neighbor_cell_ids, dt$year[i], sep = "_")
  neighbor_lookup_fast[[i]] <- as.integer(dt_idx_lookup[neighbor_keys])
}

# Flatten neighbor structure into long form
neighbor_dt <- data.table(
  source = rep(seq_along(neighbor_lookup_fast), lengths(neighbor_lookup_fast)),
  target = unlist(neighbor_lookup_fast)
)

# Compute all neighbor stats for selected variables
source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in source_vars) {
  vals <- dt[[var_name]]
  neighbor_dt[, value := vals[target]]
  # Aggregate stats by source row_id
  agg <- neighbor_dt[, .(
    max_val = max(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    mean_val = mean(value, na.rm = TRUE)
  ), by = source]
  # Merge back into dt
  setnames(agg, c("source", paste0(var_name, c("_nbr_max","_nbr_min","_nbr_mean"))))
  dt <- dt[agg, on = .(row_id = source)]
}

# Drop helper columns if needed
dt[, c("row_id", "key") := NULL]

# Updated cell_data with neighbor features
cell_data <- as.data.frame(dt)
```

---

**Expected Gains**  
- Eliminates per-row string concatenation inside nested loops.
- Performs aggregation in bulk using `data.table` grouping instead of millions of R list operations.
- On a 16 GB machine, this approach should reduce runtime from **86+ hours to a few hours or less**, depending on disk I/O and parallelization.

**Model and estimand preserved**: No changes to Random Forest or original prediction targetâ€”only feature engineering is optimized.
END_RESPONSE>>>
