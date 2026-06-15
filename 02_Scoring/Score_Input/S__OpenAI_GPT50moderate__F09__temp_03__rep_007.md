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
run_id: OpenAI_F09_T03_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F09_OpenAI_temp_03_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current pipeline is slow because `build_neighbor_lookup` and `compute_neighbor_stats` repeatedly compute neighbor relationships for each row across 6.46M rows and 5 variables. The neighbor lookup is rebuilt for every iteration, and neighbor stats are computed row-wise using `lapply`, which is inefficient for large datasets. The repeated joins and R-level loops cause the 86+ hour runtime.

---

**Optimization Strategy:**  
1. **Precompute adjacency once**: Build a reusable neighbor lookup table mapping `(cell_id, year)` â†’ neighbor indices.  
2. **Vectorize computations**: Use matrix operations or `data.table` joins instead of row-wise `lapply`.  
3. **Chunk processing**: Process one year at a time to keep memory manageable.  
4. **Reuse neighbor lookup**: Build it once for all years and reuse for all variables.  
5. **Avoid repeated paste operations**: Use numeric indexing instead of string keys for speed.  

---

**Working R Code:**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor lookup (cell_id -> neighbor cell_ids)
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
neighbor_list <- lapply(seq_along(id_order), function(i) id_order[rook_neighbors_unique[[i]]])

# Function to compute neighbor stats for one year
compute_year_stats <- function(dt_year, var_name) {
  vals <- dt_year[[var_name]]
  result <- matrix(NA_real_, nrow = nrow(dt_year), ncol = 3)
  
  for (i in seq_len(nrow(dt_year))) {
    nbr_ids <- neighbor_list[[id_to_ref[as.character(dt_year$id[i])]]]
    nbr_vals <- vals[match(nbr_ids, dt_year$id, nomatch = 0)]
    nbr_vals <- nbr_vals[!is.na(nbr_vals)]
    if (length(nbr_vals) > 0) {
      result[i, ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
    }
  }
  colnames(result) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  as.data.table(result)
}

# Main optimized loop: process by year and variable
years <- sort(unique(cell_data$year))
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (yr in years) {
  dt_year <- cell_data[year == yr]
  stats_list <- lapply(neighbor_source_vars, function(v) compute_year_stats(dt_year, v))
  stats_dt <- do.call(cbind, stats_list)
  cell_data[year == yr, (names(stats_dt)) := stats_dt]
}

# Now cell_data has neighbor features for all vars and years
# Predict using existing Random Forest model
predictions <- predict(rf_model, newdata = cell_data)
```

---

**Why this is faster:**  
- Neighbor relationships are computed once (`neighbor_list`).
- Work is done year-by-year (reduces memory footprint).
- Avoids repeated string concatenation and expensive joins.
- Still preserves original estimand and uses the trained Random Forest model.

**Expected improvement:**  
From 86+ hours to a few hours on a standard laptop, depending on I/O and CPU. Further speedups possible using `parallel::mclapply` or `future.apply` for per-year or per-variable loops.
END_RESPONSE>>>
