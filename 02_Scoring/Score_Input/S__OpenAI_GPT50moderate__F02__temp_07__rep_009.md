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
run_id: OpenAI_F02_T07_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F02_OpenAI_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the repeated use of `lapply` over 6.46M rows and the creation of large intermediate lists. Each row performs string concatenation and repeated lookups in R environments (e.g., `paste`, `setNames`, `lapply`). This is extremely inefficient in pure R because it causes:

- **High memory overhead**: Large lists, repeated string operations, and multiple copies of vectors.
- **No vectorization**: Everything is row-wise and interpreted, not compiled.
- **Repeated work**: Neighbor lookups are recomputed for each variable.

**Optimization Strategy**  
1. **Precompute all neighbor indices in a vectorized way** using integer joins instead of repeated string pastes.
2. **Avoid lists for per-row lookups**: Store neighbor indices in a fixed-length structure or compressed format.
3. **Use `data.table` for efficient joins**: Map `(id, year)` to row index once, then join.
4. **Compute all neighbor stats in one pass** per variable via grouped operations instead of row-wise `lapply`.
5. **Consider parallelization** with `future.apply` or `data.table` multithreading.
6. **Optional**: Use `Rcpp` for looping if pure R is too slow.

---

### **Optimized Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Ensure keys
setkey(cell_data, id, year)

# Precompute lookup table: (id, year) -> row index
cell_data[, row_idx := .I]

# Build neighbor table (long format)
# rook_neighbors_unique: list of neighbor IDs per id_order position
neighbor_dt <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      id = id_order[i],
      neighbor_id = id_order[rook_neighbors_unique[[i]]]
    )
  })
)

# Expand neighbor relationships across all years
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# Join to get row indices for source and neighbor
neighbor_dt[cell_data, on = .(id, year), source_idx := i.row_idx]
neighbor_dt[cell_data, on = .(neighbor_id = id, year), neighbor_idx := i.row_idx]

# Drop incomplete
neighbor_dt <- neighbor_dt[!is.na(source_idx) & !is.na(neighbor_idx)]

# Compute neighbor stats for each variable efficiently
compute_and_add_neighbor_features <- function(var_name) {
  dt <- neighbor_dt[, .(neighbor_value = cell_data[[var_name]][neighbor_idx]), by = source_idx]
  stats <- dt[, .(
    paste0(var_name, "_nbr_max") := max(neighbor_value, na.rm = TRUE),
    paste0(var_name, "_nbr_min") := min(neighbor_value, na.rm = TRUE),
    paste0(var_name, "_nbr_mean") := mean(neighbor_value, na.rm = TRUE)
  ), by = source_idx]

  # Merge back
  cell_data[stats$source_idx, c(names(stats)[-1]) := stats[, -1]]
}

# Apply for all variables
for (var_name in neighbor_source_vars) {
  compute_and_add_neighbor_features(var_name)
}
```

---

### **Why This is Faster**
- **Single long neighbor table** avoids per-row loops.
- **data.table joins** are highly optimized and in C.
- **Aggregations are vectorized** rather than millions of `lapply` calls.
- **No repeated string concatenation**.
- Handles 6.46M rows and ~1.37M neighbors efficiently in memory (16 GB is sufficient with `data.table`).

---

**Expected Runtime Improvement**  
From 86+ hours to **under 1 hour** on a standard laptop (with `data.table` multithreading enabled).  

**Preserves**: Original estimand and Random Forest model.
END_RESPONSE>>>
