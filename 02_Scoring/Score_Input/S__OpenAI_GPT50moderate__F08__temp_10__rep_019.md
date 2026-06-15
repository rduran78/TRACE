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
run_id: OpenAI_F08_T10_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline recalculates neighbor max, min, and mean individually for every cell-year row, repeatedly filtering neighbors and extracting values. This causes **massive redundant computation** because the neighbor graph is static (344K cells with ~1.37M relationships) while only the variables change across years. Computing neighbor stats in an on-the-fly row-wise fashion scales poorly: 6.46 million iterations Ã— multiple variables Ã— neighbor lookups leads to hours of runtime (estimated 86+ hrs). Additionally, using `lapply` row-by-row greatly amplifies overhead.

**Optimization Strategy**  
- **Separate static and dynamic components:**  
  - Precompute a static neighbor index keyed by `cell_id` (not cell-year).  
  - For each year, compute neighbor max/min/mean for all cells using **vectorized operations**, avoiding per-row `lapply`.  
- **Batch process by year:** For each variable and each year, aggregate values for all cells and map neighbor relationships in one go.  
- **Avoid row-wise loops:** Use `vapply`, matrix operations, or `data.table` joins for efficiency.  
- **Preserve estimands:** Ensure neighbor aggregates correspond exactly to original definitionsâ€”max, min, mean across same-year neighbors.

**Working R Code (Optimized Version)**  
```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute static neighbor lookup: list keyed by cell_id
neighbor_lookup_static <- lapply(seq_along(id_order), function(i) id_order[rook_neighbors_unique[[i]]])
names(neighbor_lookup_static) <- id_order

# Function to compute neighbor stats for one variable and one year
compute_year_neighbor_stats <- function(dt_year, var_name, neighbor_lookup) {
  vals <- dt_year[[var_name]]
  # Named vector for fast lookup
  val_lookup <- setNames(vals, dt_year$id)
  
  res <- t(vapply(names(neighbor_lookup), function(cid) {
    n_ids <- neighbor_lookup[[cid]]
    if (length(n_ids) == 0) return(c(NA, NA, NA))
    neighbor_vals <- val_lookup[n_ids]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, numeric(3L)))
  
  data.table(id = names(neighbor_lookup),
             year = dt_year$year[1],
             paste0(var_name, "_nb_max") := res[,1],
             paste0(var_name, "_nb_min") := res[,2],
             paste0(var_name, "_nb_mean") := res[,3])
}

# Main loop: process by year in bulk
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
result_list <- vector("list", length(neighbor_source_vars) * length(unique(cell_data$year)))

for (var_name in neighbor_source_vars) {
  idx <- 1
  for (yr in sort(unique(cell_data$year))) {
    dt_year <- cell_data[year == yr, .(id, year, (var_name))]
    neighbor_stats <- compute_year_neighbor_stats(dt_year, var_name, neighbor_lookup_static)
    result_list[[idx]] <- neighbor_stats
    idx <- idx + 1
  }
}

neighbor_features <- rbindlist(result_list, use.names = TRUE, fill = TRUE)

# Merge back to main dataset (efficient join)
setkey(cell_data, id, year)
setkey(neighbor_features, id, year)
cell_data <- cell_data[neighbor_features]

# Random Forest prediction step remains unchanged
# predict(random_forest_model, newdata = cell_data)
```

**Performance Impact**  
- Reduces complexity from **O(N Ã— T Ã— neighbors)** row-wise to **O(T Ã— (N + neighbors))** batch-wise.  
- Avoids recomputing neighbor relationships repeatedly.  
- Eliminates per-row overhead; designed for scalability on 16GB RAM laptop.  
- Expected runtime drops from **86+ hours to a few hours or less** with `data.table` and vectorization.

This preserves the original numerical estimand (same max/min/mean across same-year neighbors) and keeps the trained Random Forest untouched.
END_RESPONSE>>>
