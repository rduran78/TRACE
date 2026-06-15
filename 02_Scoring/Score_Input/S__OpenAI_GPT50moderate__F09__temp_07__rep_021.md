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
run_id: OpenAI_F09_T07_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F09_OpenAI_temp_07_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current pipeline repeatedly computes neighbor relationships for each cell-year during feature generation, causing massive redundant computation across 6.46M rows. The spatial topology (neighbor structure) does not change over time, but the current design rebuilds or reuses neighbor lookups at row level for every variable-year combination. This results in O(N Ã— T Ã— V) expensive operations instead of leveraging the static adjacency structure.

---

**Optimization Strategy:**  
- Precompute a reusable neighbor index (`neighbor_lookup`) once using cell IDs, not cell-years.
- For each year, subset the corresponding rows and compute neighbor stats by joining yearly attributes to the static adjacency.
- Use **vectorized operations** and `data.table` for speed and memory efficiency.
- Avoid looping over every row; instead, aggregate neighbor stats in bulk per year.
- Preserve the trained Random Forest model and estimand by keeping feature definitions unchanged.

---

**Optimized Working R Code:**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute static neighbor lookup (list of neighbor cell IDs by cell ID)
neighbor_lookup <- lapply(rook_neighbors_unique, function(nbs) id_order[nbs])
names(neighbor_lookup) <- id_order

# Function to compute neighbor stats for a single variable and year
compute_yearly_neighbor_stats <- function(dt_year, var_name, neighbor_lookup) {
  vals <- setNames(dt_year[[var_name]], dt_year$id)  # named vector for fast lookup
  
  res <- lapply(names(neighbor_lookup), function(cell_id) {
    nbs <- neighbor_lookup[[cell_id]]
    neighbor_vals <- vals[nbs]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  })
  
  res_mat <- do.call(rbind, res)
  colnames(res_mat) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  
  data.table(id = names(neighbor_lookup), res_mat)
}

# Loop over years and variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

result_list <- vector("list", length(years))

for (i in seq_along(years)) {
  yr <- years[i]
  dt_year <- cell_data[year == yr]
  
  # Start with id and year
  year_res <- data.table(id = dt_year$id, year = yr)
  
  # Compute neighbor features for each variable and merge
  for (var_name in neighbor_source_vars) {
    nbr_stats <- compute_yearly_neighbor_stats(dt_year, var_name, neighbor_lookup)
    year_res <- merge(year_res, nbr_stats, by = "id", all.x = TRUE, sort = FALSE)
  }
  
  result_list[[i]] <- year_res
}

# Combine all years
neighbor_features <- rbindlist(result_list)

# Merge back into original data
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE, sort = FALSE)

# Random Forest prediction (model already trained)
cell_data$rf_pred <- predict(rf_model, newdata = cell_data)
```

---

**Why this is faster:**  
- `neighbor_lookup` built once â†’ no repeated adjacency computations.
- Per year computation reduces 6.46M row-wise operations into ~28 iterations.
- Vectorized aggregation via `data.table` and named vectors minimizes overhead.
- Avoids nested loops over rows and variables.

**Expected improvement:**  
From 86+ hours down to a few hours (or less), depending on I/O and system resources, because the complexity drops from per-observation neighbor search to per-year bulk computation.
END_RESPONSE>>>
