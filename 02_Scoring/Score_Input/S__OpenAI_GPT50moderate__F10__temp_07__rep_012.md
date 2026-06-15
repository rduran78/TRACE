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
run_id: OpenAI_F10_T07_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- The current implementation repeatedly computes neighbor statistics via `lapply` for each row and each variable, which is extremely inefficient (86+ hours).  
- Neighbor relationships (`rook_neighbors_unique`) are static across years, but `build_neighbor_lookup` is called per full dataset, creating massive repeated lookups.  
- For 6.46M rows, looping in pure R with `lapply` and repeated indexing is a bottleneck.  
- Memory overhead from repeatedly binding results and creating vectors is high.  

---

**Optimization Strategy**  
1. **Precompute Graph Topology Once**: Build a neighbor index matrix that maps each cell to its neighbors (static across years).  
2. **Vectorization**: Use fast aggregation via `data.table` or `matrixStats` across neighbors instead of `lapply`.  
3. **Batch Processing**: Compute stats year-by-year to keep memory usage low while leveraging the static graph.  
4. **Avoid Repeated Lookups**: Convert long panel data to `data.table`, keyed by `(id, year)`, and use integer-based joins.  
5. **Parallelization**: Use `future.apply` or `parallel` for independent yearly computations.  
6. **Preserve Numerical Equivalence**: Same max, min, mean definitions; same handling of `NA`.  

---

**Optimized Working R Code**  

```r
library(data.table)
library(matrixStats)

compute_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique, vars) {
  # Convert to data.table
  setDT(cell_data)
  setkey(cell_data, id, year)
  
  # Build neighbor index for cells (static across years)
  id_to_pos <- setNames(seq_along(id_order), id_order)
  neighbor_list <- lapply(rook_neighbors_unique, function(nb) id_to_pos[nb])
  
  # Precompute neighbor matrix for fast access
  max_neighbors <- max(lengths(neighbor_list))
  neighbor_mat <- matrix(NA_integer_, nrow = length(id_order), ncol = max_neighbors)
  for (i in seq_along(neighbor_list)) {
    nbs <- neighbor_list[[i]]
    neighbor_mat[i, seq_along(nbs)] <- nbs
  }
  
  # Prepare output columns
  for (v in vars) {
    cell_data[, paste0(v, "_nb_max") := NA_real_]
    cell_data[, paste0(v, "_nb_min") := NA_real_]
    cell_data[, paste0(v, "_nb_mean") := NA_real_]
  }
  
  # Process year by year for memory efficiency
  years <- unique(cell_data$year)
  
  for (yr in years) {
    dt_year <- cell_data[year == yr]
    vals_mat <- as.matrix(dt_year[, ..vars])
    
    # Compute neighbor stats for each variable using matrix operations
    for (j in seq_along(vars)) {
      col_vals <- vals_mat[, j]
      # Gather neighbor values for all cells
      n_vals <- matrix(col_vals[neighbor_mat], nrow = nrow(neighbor_mat))
      # Handle NAs
      n_vals[is.na(n_vals)] <- NA_real_
      nb_max <- rowMaxs(n_vals, na.rm = TRUE)
      nb_min <- rowMins(n_vals, na.rm = TRUE)
      nb_mean <- rowMeans(n_vals, na.rm = TRUE)
      nb_max[is.infinite(nb_max)] <- NA_real_
      nb_min[is.infinite(nb_min)] <- NA_real_
      nb_mean[is.nan(nb_mean)] <- NA_real_
      
      # Assign results back
      cell_data[year == yr, (paste0(vars[j], "_nb_max")) := nb_max]
      cell_data[year == yr, (paste0(vars[j], "_nb_min")) := nb_min]
      cell_data[year == yr, (paste0(vars[j], "_nb_mean")) := nb_mean]
    }
  }
  
  return(cell_data)
}

# Usage:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# Predict with pre-trained Random Forest (unchanged)
preds <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Why This Works**  
- **Graph Built Once**: `neighbor_mat` stores all neighbors, reused across years.  
- **Matrix Operations**: `rowMaxs`, `rowMins`, `rowMeans` from `matrixStats` are highly optimized in C.  
- **Memory Efficiency**: Operate year-by-year to avoid loading all 6.46M rows into neighbor matrices at once.  
- **No Model Retraining**: Predictions use the original Random Forest.  

This approach reduces complexity from *O(N Ã— neighbors Ã— vars Ã— years) in R loops* to efficient matrix operations, likely cutting runtime from 86+ hours to **under 1 hour** on a 16 GB laptop.
END_RESPONSE>>>
