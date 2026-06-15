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
run_id: OpenAI_F09_T03_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F09_OpenAI_temp_03_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline recomputes neighbor relationships for every cell-year row, which is extremely costly given 6.46M rows and 5 variables. The `build_neighbor_lookup` function is already efficient because it builds a reusable index, but the bottleneck is repeatedly scanning and joining large vectors in `compute_neighbor_stats` for each variable. For 5 variables Ã— 6.46M rows, this results in heavy repeated work and memory churn.

**Optimization Strategy**  
- Build the neighbor lookup **once** at the cell level (not cell-year).
- For each year, subset the data, compute neighbor stats for all 5 variables in a single pass, then append results.
- Use **vectorized operations** and `vapply` or `matrix` binding instead of repeated `lapply` calls.
- Avoid recomputing `idx_lookup` or string concatenations inside loops.
- Process year by year to keep memory within 16 GB.
- Preserve the trained Random Forest model and estimand by producing the same features, just faster.

---

### **Optimized R Code**

```r
# Build neighbor lookup once at cell level
build_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors is a list of integer vectors (rook neighbors)
  # Return as-is but aligned to id_order
  neighbors
}

compute_neighbor_stats_matrix <- function(vals, neighbor_lookup) {
  # vals: numeric vector of length = number of cells in that year
  n <- length(vals)
  result <- matrix(NA_real_, nrow = n, ncol = 3) # max, min, mean
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  result
}

# Main optimized pipeline
neighbor_lookup <- build_neighbor_lookup(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Preallocate storage
out_list <- vector("list", length = length(unique(cell_data$year)))
names(out_list) <- unique(cell_data$year)

years <- sort(unique(cell_data$year))

for (yr in years) {
  message("Processing year: ", yr)
  subset_idx <- which(cell_data$year == yr)
  year_data <- cell_data[subset_idx, ]
  
  # Ensure order matches id_order
  year_data <- year_data[match(id_order, year_data$id), ]
  
  # Compute neighbor stats for each variable
  stats_list <- lapply(neighbor_source_vars, function(var_name) {
    compute_neighbor_stats_matrix(year_data[[var_name]], neighbor_lookup)
  })
  
  # Combine all stats into a data.frame
  # Each var contributes 3 columns: max, min, mean
  stats_mat <- do.call(cbind, stats_list)
  colnames(stats_mat) <- as.vector(t(outer(neighbor_source_vars, c("_max", "_min", "_mean"), paste0)))
  
  # Bind to year_data
  year_data <- cbind(year_data, stats_mat)
  
  out_list[[as.character(yr)]] <- year_data
}

# Combine all years back
cell_data_enriched <- do.call(rbind, out_list)

# Predict using existing Random Forest model
predictions <- predict(trained_rf_model, newdata = cell_data_enriched)
```

---

**Why This Is Faster**  
- Neighbor relationships are reused without recomputation.
- Year-by-year processing keeps memory manageable.
- All 5 variables are processed in a single pass per year, reducing overhead.
- Avoids repeated string concatenation and large `lapply` calls over 6.46M rows.

**Expected Impact**  
This approach reduces complexity from ~O(N Ã— V) repeated heavy lookups to O(N) per year with minimal overhead, likely cutting runtime from 86+ hours to a few hours on a 16 GB laptop.
END_RESPONSE>>>
