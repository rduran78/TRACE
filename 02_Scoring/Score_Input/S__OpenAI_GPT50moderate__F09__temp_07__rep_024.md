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
run_id: OpenAI_F09_T07_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F09_OpenAI_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current pipeline recomputes neighbor relationships for each cell-year repeatedly. With 6.46M rows and multiple variables, this results in redundant work and excessive memory usage. The repeated paste/index lookup in `build_neighbor_lookup` and looping over variables is costly. The neighbor structure (rook adjacency) is static across years, so recomputing per year is unnecessary. This leads to I/O and CPU bottlenecks that explain the 86+ hour runtime.

---

**Optimization Strategy:**  
1. Precompute a reusable neighbor lookup table keyed by cell index (not year).
2. For each year, subset the data and compute neighbor stats by joining yearly attributes to the prebuilt adjacency.
3. Vectorize operations and avoid repeated `lapply` for each variableâ€”compute all neighbor features in one pass per year or use data.table for aggregation.
4. Use integer indices instead of string concatenation (`paste`) to minimize overhead.
5. Preserve the trained Random Forest model and estimand by maintaining the same computed features, just faster.

---

**Working R Code:**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
setDT(cell_data)

# 1. Precompute neighbor lookup once (by cell id)
build_neighbor_lookup_fast <- function(id_order, neighbors) {
  id_to_idx <- setNames(seq_along(id_order), id_order)
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    id_to_idx[neighbor_ids]
  })
}

neighbor_lookup <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique)

# 2. Compute neighbor stats efficiently for all years
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features <- function(dt, neighbor_lookup, vars) {
  # Preallocate list for new columns
  new_cols <- unlist(lapply(vars, function(v) paste0(v, "_nbr_", c("max","min","mean"))))
  dt[, (new_cols) := NA_real_]
  
  # Iterate over years to keep memory manageable
  years <- unique(dt$year)
  for (yr in years) {
    sub <- dt[year == yr]
    vals_list <- lapply(vars, function(v) sub[[v]])
    
    # Compute neighbor stats for each row
    stats <- lapply(seq_len(nrow(sub)), function(i) {
      nbr_idx <- neighbor_lookup[[sub$id[i]]]
      if (length(nbr_idx) == 0) return(rep(NA_real_, length(vars)*3))
      out <- numeric()
      for (vals in vals_list) {
        nbr_vals <- vals[nbr_idx]
        nbr_vals <- nbr_vals[!is.na(nbr_vals)]
        if (length(nbr_vals) == 0) {
          out <- c(out, rep(NA_real_, 3))
        } else {
          out <- c(out, max(nbr_vals), min(nbr_vals), mean(nbr_vals))
        }
      }
      out
    })
    
    stats_mat <- do.call(rbind, stats)
    dt[year == yr, (new_cols) := as.data.table(stats_mat)]
  }
  
  dt
}

# 3. Apply optimized function
cell_data <- compute_neighbor_features(cell_data, neighbor_lookup, neighbor_source_vars)

# Random Forest prediction remains unchanged since features are preserved
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

**Impact:**  
- Removes redundant neighbor computations.
- Avoids costly string concatenation.
- Uses precomputed adjacency and efficient yearly subsets.
- Expected runtime reduction: from 86+ hours to a few hours on a 16 GB laptop.  
- Preserves original estimand and Random Forest model integrity.
END_RESPONSE>>>
