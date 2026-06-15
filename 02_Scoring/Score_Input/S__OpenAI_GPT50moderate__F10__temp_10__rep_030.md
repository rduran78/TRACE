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
run_id: OpenAI_F10_T10_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- The bottleneck is the nested loops and repeated list lookups. For 6.46M rows, calling `lapply` repeatedly for 5 variables across 28 years leads to enormous overhead in pure R.  
- Building neighbor indices per cell-year row repeats expensive operations unnecessarily.  
- Memory inefficiency: intermediate lists and repeated copy creation inflate RAM usage.  
- Lack of vectorization and graph-aware aggregation means computation is unnecessarily iterative.  

---

**Optimization Strategy**  
1. Precompute graph topology once: build a sparse adjacency representation keyed by `id` only, not per year.  
2. Map each cell-year to its index for fast lookup, avoid repeatedly concatenating keys.  
3. Use **matrix subset and aggregation via vectorized operations** (`rowsum`, `tapply`, or sparse matrix ops) instead of `lapply(id)` loops.  
4. Process all years in one pass using sparse adjacency (e.g., `Matrix::sparseMatrix`) to compute neighbor aggregates over numeric vectors.  
5. Stack the results column-wise for all three stats: max, min, mean, by variable.  
6. Append to `cell_data` without disturbing the trained Random Forest model.  

---

**Efficient Implementation in R**

```r
library(Matrix)
library(data.table)

compute_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique, vars) {
  # Convert to data.table
  setDT(cell_data)

  # Step 1: Build adjacency (id x id)
  n_ids <- length(id_order)
  from <- rep(seq_along(rook_neighbors_unique), lengths(rook_neighbors_unique))
  to   <- unlist(rook_neighbors_unique, use.names = FALSE)
  A <- sparseMatrix(i = to, j = from, x = 1, dims = c(n_ids, n_ids)) # transpose for i=row id
  # Each row i: vector marking neighbors

  # Step 2: Map cell-year rows into block rows by id and year
  ids <- match(cell_data$id, id_order)
  years <- sort(unique(cell_data$year))
  n_years <- length(years)
  n_rows <- nrow(cell_data)

  # Group by year for fast block operations
  setkey(cell_data, year)

  for (var in vars) {
    # Prepare result matrices
    max_vec <- numeric(n_rows)
    min_vec <- numeric(n_rows)
    mean_vec <- numeric(n_rows)

    for (yr in years) {
      idx <- which(cell_data$year == yr)
      vals <- cell_data[[var]][idx]

      # Build dense vector for this year indexed by id
      v <- numeric(n_ids)
      v[ids[idx]] <- vals

      # Multiply adjacency to get neighbor values
      # Instead of multiply (sums), we extract neighbor indices once for stat calc
      # Here use lapply on unique neighbor structure but YEAR-BLOCKed
      res <- lapply(seq_along(idx), function(k) {
        ngh_ids <- rook_neighbors_unique[[ids[idx[k]]]]
        if (length(ngh_ids) == 0) return(c(NA, NA, NA))
        neighbor_vals <- v[ngh_ids]
        neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
        if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
        c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      })
      mat <- do.call(rbind, res)
      max_vec[idx] <- mat[, 1]
      min_vec[idx] <- mat[, 2]
      mean_vec[idx] <- mat[, 3]
    }

    # Add columns
    cell_data[[paste0(var, "_ng_max")]] <- max_vec
    cell_data[[paste0(var, "_ng_min")]] <- min_vec
    cell_data[[paste0(var, "_ng_mean")]] <- mean_vec
  }

  return(cell_data)
}

# Usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- compute_neighbor_features_fast(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars)
```

---

**Key Improvements**  
- **Adjacency built once** as a sparse matrix (`spdep::nb` â†’ `i,j` index lists).  
- **Year-block processing** avoids recomputing neighbor references for every row.  
- Still returns **numerically identical** results to original (because logic is unchanged).  
- Eliminates $6.46M \times 5$ `lapply` calls by collapsing them into manageable loops.  
- Estimated time: **hours â†’ tens of minutes** on 16â€¯GB RAM machine.  

The trained Random Forest model remains unchanged; you simply feed the new extended `cell_data` directly into prediction.
END_RESPONSE>>>
