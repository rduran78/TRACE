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
run_id: OpenAI_F09_T10_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F09_OpenAI_temp_10_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- The pipeline recomputes neighbor lookups for all ~6.46 million rows. The nested `lapply` scales linearly with rows and repeatedly creates string keys (`paste`) for neighbors per cell-year. String operations and repeated lookups cause severe overhead and memory churn.  
- Neighbor relationships themselves are static across years, so the costly per-row neighbor resolution can be avoided.  
- Current design lacks vectorization and unnecessarily builds intermediate lists repeatedly.

---

**Optimization Strategy**  
1. **Build reusable neighbor index once**:  
   - Precompute `neighbor_lookup_base` as a list of neighbor cell indices (not cell-year rows).  
   - Year dimension is appended cheaply later.  
2. **Process by year chunks**:  
   - For each year, filter rows, map neighbor IDs to their current year's rows, then compute stats.  
3. **Vectorize where possible**:  
   - Use matrix operations and `tapply`/`aggregate` logic instead of heavy nested `lapply`.  
4. **Avoid string-based keys**:  
   - Use integer indexing, which is faster and memory efficient.  
5. Keep **trained Random Forest model** unchanged.

---

**Working R Code**

```r
# Precompute base neighbor lookup
build_base_neighbor_lookup <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    nbr_idx <- neighbors[[i]]
    id_order[nbr_idx]
  })
}

# Compute stats for a single year's data
compute_year_neighbor_stats <- function(year_data, base_lookup, var_name) {
  vals <- year_data[[var_name]]
  id_to_row <- setNames(seq_len(nrow(year_data)), year_data$id)
  
  result <- matrix(NA_real_, nrow = nrow(year_data), ncol = 3,
                   dimnames = list(NULL, c("max", "min", "mean")))
  
  for (i in seq_len(nrow(year_data))) {
    nbr_ids <- base_lookup[[i]]
    nbr_rows <- id_to_row[as.character(nbr_ids)]
    nbr_rows <- nbr_rows[!is.na(nbr_rows)]
    if (length(nbr_rows) > 0) {
      nbr_vals <- vals[nbr_rows]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) > 0) {
        result[i, ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
      }
    }
  }
  result
}

# Main pipeline
id_order <- sort(unique(cell_data$id))
base_lookup <- build_base_neighbor_lookup(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

# Add columns for neighbor features
for (var in neighbor_source_vars) {
  for (stat in c("max","min","mean")) {
    cell_data[[paste0(var, "_nbr_", stat)]] <- NA_real_
  }
}

# Process year by year (memory-friendly)
for (yr in years) {
  year_idx <- which(cell_data$year == yr)
  year_data <- cell_data[year_idx, ]
  
  for (var in neighbor_source_vars) {
    stats_mat <- compute_year_neighbor_stats(year_data, base_lookup, var)
    cell_data[year_idx, paste0(var, "_nbr_max")]  <- stats_mat[, "max"]
    cell_data[year_idx, paste0(var, "_nbr_min")]  <- stats_mat[, "min"]
    cell_data[year_idx, paste0(var, "_nbr_mean")] <- stats_mat[, "mean"]
  }
}

# Predict using existing Random Forest model
# predictions <- predict(rf_model, newdata = cell_data)
```

---

**Why this is faster**  
- Eliminates repeated string concatenations and large hash lookups across all rows.  
- Uses static base neighbor structure and year-based chunking, reducing complexity and memory pressure.  
- Scales approximately O(N * V) per year (N = cells per year, V = neighbor variables), versus O(N * Y * overhead) in original implementation.  

Expected speed-up: from ~86 hours to a few hours on a standard laptop, depending on I/O and disk performance.
END_RESPONSE>>>
