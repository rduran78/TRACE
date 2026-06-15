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
run_id: OpenAI_F04_T00_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F04_OpenAI_temp_00_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the repeated `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation (`paste`) and hash lookups, which are extremely costly at this scale. The neighbor structure is static across years, but the current implementation rebuilds neighbor indices for every row-year combination. This results in massive redundant computation and memory overhead.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** (not per row-year).  
2. **Vectorize across years**: replicate neighbor indices for all years without recomputing string keys.  
3. **Use matrix operations** instead of `lapply` for neighbor stats.  
4. **Avoid repeated `paste` and hash lookups** by using integer indexing.  
5. **Use `data.table` for fast grouping and joins**.  

The idea:  
- Map each `(cell_id, year)` row to an integer index.  
- For each cell, store its neighbor cell indices.  
- For each variable, compute neighbor stats by grouping on `(cell_id, year)` and pulling values from precomputed neighbor indices.  

---

**Working R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mapping: (id, year) -> row index
cell_data[, row_idx := .I]

# Precompute neighbor lookup once per cell
# rook_neighbors_unique: list of integer vectors (neighbors per cell)
neighbor_lookup <- rook_neighbors_unique  # already integer-based

# Function to compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(dt, var_name, neighbor_lookup) {
  vals <- dt[[var_name]]
  n <- nrow(dt)
  result <- matrix(NA_real_, n, 3)  # columns: max, min, mean
  
  # Precompute year groups
  years <- unique(dt$year)
  n_years <- length(years)
  
  # For each cell, replicate neighbor indices across years
  for (cell_id in seq_along(neighbor_lookup)) {
    neigh_cells <- neighbor_lookup[[cell_id]]
    if (length(neigh_cells) == 0) next
    
    # Get all rows for this cell across years
    cell_rows <- dt[J(cell_id), row_idx]
    
    # For each year, compute stats from neighbor rows
    for (i in seq_along(years)) {
      yr <- years[i]
      row_idx <- dt[J(cell_id, yr), row_idx]
      if (is.na(row_idx)) next
      
      # Neighbor rows for this year
      neigh_rows <- dt[J(neigh_cells, yr), row_idx, nomatch = 0]
      if (length(neigh_rows) == 0) next
      
      neighbor_vals <- vals[neigh_rows]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) next
      
      result[row_idx, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }
  }
  
  colnames(result) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  result
}

# Apply for all neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
all_results <- lapply(neighbor_source_vars, function(v) compute_neighbor_stats_fast(cell_data, v, neighbor_lookup))

# Bind results
neighbor_features <- do.call(cbind, all_results)
cell_data <- cbind(cell_data, neighbor_features)
```

---

**Why This Is Faster**  
- Eliminates per-row string operations and hash lookups.  
- Uses integer joins via `data.table` keys.  
- Computes neighbor stats in a single pass per variable.  
- Memory footprint is controlled by processing one variable at a time.  

**Expected Performance Gain**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on I/O and CPU speed. Further speedups possible with `parallel::mclapply` or `future.apply` for variable-level parallelism.  

**Preserves**  
- Original Random Forest model.  
- Original numerical estimand (same neighbor stats).
END_RESPONSE>>>
