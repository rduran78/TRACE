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
run_id: OpenAI_F08_T00_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F08_OpenAI_temp_00_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline recomputes neighbor statistics for each cell-year row by iterating over all rows and looking up neighbors dynamically. This is extremely inefficient because:  
- Neighbor relationships are static across years, yet the lookup is rebuilt for every row-year combination.  
- For 6.46M rows and 5 variables, the nested loops and repeated indexing cause massive overhead.  
- Memory and CPU usage explode due to redundant operations.  

**Optimization Strategy**  
- Precompute a static neighbor index for cells only (not cell-years).  
- For each year, slice the data for that year and compute neighbor stats in a vectorized manner using the static neighbor index.  
- Bind results back to the full panel efficiently.  
- Avoid repeated `lapply` over millions of rows; instead, use matrix operations or `data.table` for speed.  
- Keep the Random Forest model unchanged and preserve original estimands.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute static neighbor index (list of integer positions for each cell)
id_to_pos <- setNames(seq_along(id_order), id_order)
neighbor_index <- lapply(seq_along(id_order), function(i) {
  id_order[rook_neighbors_unique[[i]]]  # neighbor cell IDs
})
names(neighbor_index) <- id_order

# Function to compute neighbor stats for one year
compute_year_stats <- function(year_data, var_name) {
  vals <- year_data[[var_name]]
  res <- matrix(NA_real_, nrow(year_data), 3)
  for (i in seq_len(nrow(year_data))) {
    nbr_ids <- neighbor_index[[as.character(year_data$id[i])]]
    if (length(nbr_ids) > 0) {
      nbr_vals <- vals[match(nbr_ids, year_data$id)]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) > 0) {
        res[i, ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
      }
    }
  }
  res
}

# Apply per year and bind results
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
result_list <- vector("list", length(neighbor_source_vars))

for (var_name in neighbor_source_vars) {
  out_mat <- matrix(NA_real_, nrow(cell_data), 3)
  colnames(out_mat) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  
  # Process year by year
  for (yr in unique(cell_data$year)) {
    year_idx <- which(cell_data$year == yr)
    year_data <- cell_data[year_idx]
    out_mat[year_idx, ] <- compute_year_stats(year_data, var_name)
  }
  
  result_list[[var_name]] <- out_mat
}

# Combine all neighbor features
neighbor_features <- do.call(cbind, result_list)
cell_data <- cbind(cell_data, neighbor_features)

# Now cell_data is ready for Random Forest prediction
# rf_model <- readRDS("trained_rf_model.rds")
# predictions <- predict(rf_model, newdata = cell_data)
```

---

**Why This Is Faster**  
- Neighbor relationships are computed once and reused.  
- Year-by-year processing reduces memory footprint.  
- Avoids building a 6.46M-row neighbor lookup list.  
- Still preserves original estimand and Random Forest model.  

**Expected Performance Gain**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on implementation details and possible further vectorization (e.g., using `data.table` joins or parallelization).  

**Further Optimization**  
- Parallelize the per-year loop using `future.apply` or `parallel`.  
- Replace inner loop with vectorized aggregation using `data.table` joins for large speedups.  
- Consider storing neighbor_index as integer positions for direct indexing.
END_RESPONSE>>>
