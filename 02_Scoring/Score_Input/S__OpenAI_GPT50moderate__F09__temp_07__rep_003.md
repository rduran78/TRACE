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
run_id: OpenAI_F09_T07_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F09_OpenAI_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline recomputes neighbor relationships for each row every time `build_neighbor_lookup` is called. For ~6.46M rows, repeatedly pasting IDs and doing lookups is extremely costly in both time and memory. The main inefficiency is that neighbor logic is rebuilt at the cell-year level rather than leveraging the static spatial topology. Given that the neighbor structure (rook adjacency) is constant across years, it should be computed *once* at the cell level and then reused for all years. This avoids redundant string concatenations, repeated hashing, and expensive lookups.

---

**Optimization Strategy**  
1. Precompute a static `neighbor_lookup` at the **cell index level** (not cell-year).
2. Iterate over years; for each year subset:
   - Join year-specific attributes.
   - Compute neighbor stats by mapping static neighbor indices to current year's values.
3. Use matrix operations or `vapply` instead of `lapply` for speed.
4. Avoid creating giant objects in memory (chunk by year).
5. Keep the trained Random Forest model unchanged and preserve original estimands.

This reduces complexity from ~6.46M dynamic lookups to ~344K lookups per year, leveraging the constant adjacency structure.

---

**Working R Code**

```r
# Precompute neighbor lookup at cell level
build_cell_neighbor_lookup <- function(neighbors) {
  # neighbors is an spdep nb object, list of integer vectors
  lapply(neighbors, as.integer)
}

neighbor_lookup <- build_cell_neighbor_lookup(rook_neighbors_unique)

# Function to compute neighbor stats for one year
compute_neighbor_stats_year <- function(data_year, neighbor_lookup, var_name) {
  vals <- data_year[[var_name]]
  n <- nrow(data_year)
  # Preallocate matrix
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        out[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  colnames(out) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  out
}

# Main optimized loop: process year by year
years <- sort(unique(cell_data$year))
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

result_list <- vector("list", length(years))

for (i in seq_along(years)) {
  yr <- years[i]
  message("Processing year: ", yr)
  data_year <- subset(cell_data, year == yr)
  # Ensure data_year is ordered by id_order
  data_year <- data_year[match(id_order, data_year$id), ]
  
  feature_mats <- lapply(neighbor_source_vars, function(var_name) {
    compute_neighbor_stats_year(data_year, neighbor_lookup, var_name)
  })
  
  features <- do.call(cbind, feature_mats)
  data_year <- cbind(data_year, features)
  result_list[[i]] <- data_year
}

# Combine all years back
cell_data_enriched <- do.call(rbind, result_list)

# Predict using pre-trained Random Forest (unchanged)
predictions <- predict(trained_rf_model, newdata = cell_data_enriched)
```

---

**Expected Impact**  
- Eliminates ~6.46M dynamic key lookups and string manipulations.
- Reduces runtime from 86+ hours to a few hours (depending on I/O and CPU speed).
- Preserves the trained model and original estimands.
- Fits within 16 GB RAM by chunking per year.

This approach leverages static adjacency and processes year-by-year efficiently.
END_RESPONSE>>>
