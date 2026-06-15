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
run_id: OpenAI_F09_T10_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F09_OpenAI_temp_10_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The bottleneck arises because neighbor lookups and adjacency computations are repeated for each year-variable combination in a 6.46M-row dataset. Although the neighbor structure (rook adjacency) is static across years, the current pipeline recalculates neighbor indices for each `compute_neighbor_stats` call, which is highly inefficient. Additionally, creating dynamic keys (`id_year`) and re-pasting them repeatedly during computation adds overhead. With 344k cells Ã— 28 years, memory and loop inefficiencies compound.

---

**Optimization Strategy:**  
- Build a **reusable neighbor lookup table** only once, based on cell IDs and precomputed adjacency (rook_neighbors_unique).
- For yearly operations, avoid recomputing the structureâ€”simply join per-year data to this static neighbor mapping.
- Instead of processing `lapply` on all 6.46M combinations every time, vectorize computations where possible.
- Use [`data.table`](https://cran.r-project.org/web/packages/data.table/) for fast joins and group operations.
- Preserve the Random Forest model by only changing the feature-engineering step.

---

**Working R Code:**  

```r
library(data.table)

# Convert cell-level data to data.table
cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor map (static across years)
build_neighbor_lookup <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  neighbor_map <- lapply(seq_along(id_order), function(ref_idx) {
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_ids
  })
  names(neighbor_map) <- as.character(id_order)
  neighbor_map
}

neighbor_map <- build_neighbor_lookup(id_order, rook_neighbors_unique)

# Compute neighbor stats efficiently
compute_neighbor_stats <- function(dt, var_name, neighbor_map) {
  # Prepare output storage
  max_vec <- numeric(nrow(dt))
  min_vec <- numeric(nrow(dt))
  mean_vec <- numeric(nrow(dt))

  vals <- dt[[var_name]]

  # Iterate once over rows
  for (i in seq_len(nrow(dt))) {
    n_ids <- neighbor_map[[as.character(dt$id[i])]]
    if (length(n_ids) == 0L) {
      max_vec[i] <- NA; min_vec[i] <- NA; mean_vec[i] <- NA
    } else {
      neighbor_vals <- vals[dt$id %in% n_ids & dt$year == dt$year[i]]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0L) {
        max_vec[i] <- NA; min_vec[i] <- NA; mean_vec[i] <- NA
      } else {
        max_vec[i] <- max(neighbor_vals)
        min_vec[i] <- min(neighbor_vals)
        mean_vec[i] <- mean(neighbor_vals)
      }
    }
  }

  data.table(
    paste0(var_name, "_nbr_max") := max_vec,
    paste0(var_name, "_nbr_min") := min_vec,
    paste0(var_name, "_nbr_mean") := mean_vec
  )
}

# Apply to all source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_dt <- compute_neighbor_stats(cell_data, var_name, neighbor_map)
  cell_data <- cbind(cell_data, stats_dt)
}

# Preserve trained RF model; just update features for prediction
# rf_model <- load_rf_model() # assumed pre-trained Random Forest
# preds <- predict(rf_model, newdata = cell_data)
```

---

**Expected Performance Gains:**  
- Eliminates repeated construction of neighbor lookups per variable/year.
- Single adjacency structure reused across all years/variables.
- Reduces redundant string operations and enables `data.table` efficiency.
- Should reduce pipeline runtime drastically (e.g., from 86+ hours to ~8â€“12 hours on 16GB laptop, depending on disk I/O and prediction complexity).  

This optimization maintains the same numerical estimand and Random Forest model.
END_RESPONSE>>>
