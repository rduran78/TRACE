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
run_id: OpenAI_F04_T03_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F04_OpenAI_temp_03_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The major bottleneck is the repeated `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. For each row, the code performs string concatenation, hash lookups, and list indexing. This results in extremely high overhead. The neighbor relationships are static across years, yet the code redundantly recomputes them per row. Additionally, the pipeline processes each variable sequentially, repeating similar work multiple times.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once for all years**: Expand the neighbor relationships to all years in a vectorized manner rather than row-by-row.
2. **Avoid string concatenation and repeated lookups**: Use integer-based indexing instead of key-based lookups.
3. **Vectorize neighbor statistics computation**: Use `data.table` or `matrix` operations to compute max, min, and mean in bulk.
4. **Parallelize where possible**: Use `parallel::mclapply` or `future.apply` for multi-core execution.
5. **Memory efficiency**: Store neighbor indices in a flat integer matrix or list aligned with row order.

---

**Working R Code**

```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

n_cells <- length(id_order)
years <- sort(unique(cell_data$year))
n_years <- length(years)

# Precompute neighbor indices for all cells (static)
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbors_idx <- lapply(rook_neighbors_unique, function(nbs) id_to_idx[nbs])

# Map each row to (cell_idx, year_idx)
cell_idx <- id_to_idx[as.character(cell_data$id)]
year_idx <- match(cell_data$year, years)

# Build a matrix of row indices: row_map[cell_idx, year_idx] = row number
row_map <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
row_map[cbind(cell_idx, year_idx)] <- seq_len(nrow(cell_data))

# Precompute neighbor row indices for each row
neighbor_lookup <- vector("list", nrow(cell_data))
for (ci in seq_len(n_cells)) {
  nbs <- neighbors_idx[[ci]]
  if (length(nbs) == 0) next
  for (yi in seq_len(n_years)) {
    row_id <- row_map[ci, yi]
    if (is.na(row_id)) next
    nb_rows <- row_map[nbs, yi]
    neighbor_lookup[[row_id]] <- nb_rows[!is.na(nb_rows)]
  }
}

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(var_name) {
  vals <- cell_data[[var_name]]
  mclapply(seq_along(neighbor_lookup), function(i) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) return(c(NA, NA, NA))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, mc.cores = detectCores() - 1) |> do.call(rbind, .)
}

# Apply for all variables
for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(var_name)
  colnames(stats_mat) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

**Why This Is Faster**  
- Eliminates per-row string operations and repeated hash lookups.
- Precomputes neighbor indices once for all years.
- Uses integer indexing and vectorized operations.
- Parallelizes the heavy computation step.
- Reduces redundant work across variables.

**Expected Improvement**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on parallelization and disk I/O.
END_RESPONSE>>>
