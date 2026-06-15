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
run_id: OpenAI_F08_T03_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F08_OpenAI_temp_03_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because `compute_neighbor_stats` iterates over all ~6.46M rows and repeatedly scans vectors for each neighbor set. This results in massive redundant computation:  
- The neighbor structure is static across years, but the code recalculates neighbor indices for every row-year combination.  
- For each variable, neighbor stats are computed row by row, causing repeated lookups and aggregation.  
- The entire process is repeated for 5 variables across 6.46M rows, leading to >86 hours runtime.  

**Optimization Strategy**  
- Precompute a **static neighbor index map by cell ID** (not by cell-year).  
- For each year and variable, compute neighbor stats in **vectorized chunks** using matrix operations.  
- Avoid repeated `lapply` over millions of rows; instead, process per-year slices (28 iterations instead of 6.46M).  
- Use `data.table` for fast grouping and joins.  
- Preserve the Random Forest model and estimand by producing identical features, just computed efficiently.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute static neighbor lookup by cell ID
build_static_neighbor_lookup <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    neighbor_ids
  }) |> setNames(id_order)
}

neighbor_lookup_static <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

# Function to compute neighbor stats for one variable in one year
compute_neighbor_stats_year <- function(dt_year, var_name, neighbor_lookup) {
  vals <- setNames(dt_year[[var_name]], dt_year$id)
  res <- lapply(names(neighbor_lookup), function(cell_id) {
    n_ids <- neighbor_lookup[[cell_id]]
    n_vals <- vals[n_ids]
    n_vals <- n_vals[!is.na(n_vals)]
    if (length(n_vals) == 0) return(c(NA, NA, NA))
    c(max(n_vals), min(n_vals), mean(n_vals))
  })
  res_mat <- do.call(rbind, res)
  colnames(res_mat) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  data.table(id = names(neighbor_lookup), res_mat)
}

# Main optimized loop
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
result_list <- vector("list", length = length(neighbor_source_vars) * length(unique(cell_data$year)))

years <- sort(unique(cell_data$year))
idx <- 1

for (yr in years) {
  dt_year <- cell_data[year == yr]
  for (var_name in neighbor_source_vars) {
    stats_dt <- compute_neighbor_stats_year(dt_year, var_name, neighbor_lookup_static)
    stats_dt[, year := yr]
    result_list[[idx]] <- stats_dt
    idx <- idx + 1
  }
}

# Combine all neighbor stats
neighbor_stats_all <- Reduce(function(x, y) merge(x, y, by = c("id", "year"), all = TRUE), result_list)

# Merge back into main dataset
cell_data <- merge(cell_data, neighbor_stats_all, by = c("id", "year"), all.x = TRUE)
```

---

### **Why This Is Fast**
- **Static neighbor map**: Built once for 344k cells instead of 6.46M rows.  
- **Per-year processing**: Only 28 iterations Ã— 5 variables instead of 6.46M Ã— 5.  
- **Vectorized aggregation**: `lapply` over 344k cells per year (manageable) instead of 6.46M rows.  
- **Memory efficient**: Uses `data.table` joins instead of repeated row-wise operations.  

**Expected runtime**: From 86+ hours down to a few hours (or less) on a 16 GB laptop.  

**Preserves**:  
- Same neighbor stats (max, min, mean).  
- Same Random Forest model and estimand.  
- No change to prediction logic, only preprocessing speed.
END_RESPONSE>>>
