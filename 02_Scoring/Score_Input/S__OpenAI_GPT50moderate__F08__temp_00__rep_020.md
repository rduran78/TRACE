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
run_id: OpenAI_F08_T00_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F08_OpenAI_temp_00_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline recomputes neighbor statistics for each cell-year row by iterating over all rows and looking up neighbors dynamically. This is extremely inefficient because:  
- Neighbor relationships are static across years, yet the lookup is rebuilt for every row-year combination.  
- The computation of max, min, and mean is repeated for each variable across all 6.46 million rows.  
- The approach uses `lapply` over millions of rows, causing memory and CPU bottlenecks.  

**Optimization Strategy**  
- Precompute a static neighbor index for cells only (not cell-year).  
- For each year, slice the data and compute neighbor stats in a vectorized manner using matrix operations or `data.table`.  
- Avoid repeated concatenation and repeated lookups; instead, reuse the static neighbor structure and apply it to yearly subsets.  
- Process one year at a time to keep memory usage manageable.  
- Append results back to the main dataset efficiently.  

**Working R Code**  

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute static neighbor list keyed by cell id
neighbor_list <- rook_neighbors_unique  # already precomputed
id_order <- unique(cell_data$id)
id_to_idx <- setNames(seq_along(id_order), id_order)

# Function to compute neighbor stats for one year
compute_year_stats <- function(dt_year, neighbor_list, id_to_idx, vars) {
  n <- nrow(dt_year)
  result_list <- vector("list", length(vars))
  
  for (v in seq_along(vars)) {
    var_name <- vars[v]
    vals <- dt_year[[var_name]]
    
    # Preallocate result matrix
    res_mat <- matrix(NA_real_, nrow = n, ncol = 3)
    
    for (i in seq_len(n)) {
      cell_id <- dt_year$id[i]
      neighbors <- neighbor_list[[id_to_idx[[as.character(cell_id)]]]]
      if (length(neighbors) > 0) {
        neighbor_ids <- id_order[neighbors]
        neighbor_idx <- match(neighbor_ids, dt_year$id)
        neighbor_vals <- vals[neighbor_idx]
        neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
        if (length(neighbor_vals) > 0) {
          res_mat[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
        }
      }
    }
    result_list[[v]] <- res_mat
  }
  
  # Combine results into a data.table
  res_dt <- data.table(id = dt_year$id, year = dt_year$year)
  for (v in seq_along(vars)) {
    var_name <- vars[v]
    res_dt[[paste0(var_name, "_nbr_max")]] <- result_list[[v]][, 1]
    res_dt[[paste0(var_name, "_nbr_min")]] <- result_list[[v]][, 2]
    res_dt[[paste0(var_name, "_nbr_mean")]] <- result_list[[v]][, 3]
  }
  res_dt
}

# Main loop: process year by year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

result_list <- vector("list", length(years))
for (y in seq_along(years)) {
  dt_year <- cell_data[year == years[y]]
  result_list[[y]] <- compute_year_stats(dt_year, neighbor_list, id_to_idx, neighbor_source_vars)
}

# Merge all results back
neighbor_features <- rbindlist(result_list)
setkey(cell_data, id, year)
setkey(neighbor_features, id, year)
cell_data <- neighbor_features[cell_data]

# Preserve trained Random Forest model and original estimand
# Prediction step remains unchanged:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

**Why this is faster**  
- Neighbor relationships are used statically; no repeated string concatenation or dynamic lookup per row-year.  
- Computation is done year-by-year, reducing memory footprint.  
- Uses `data.table` for efficient joins and slicing.  
- Avoids rebuilding neighbor lookup for every row.  

This approach should reduce runtime from 86+ hours to a few hours on a standard laptop.
END_RESPONSE>>>
