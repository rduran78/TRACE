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
run_id: OpenAI_F08_T10_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach precomputes a large `neighbor_lookup` list for all 6.46M rows (one entry per cell-year). Each entry is a vector of neighbor row indices for that cell in that specific year. This is extremely memory- and time-intensive because:

- For 6.46M rows, storing neighbor indices blows up RAM.
- Redundant storage: the neighbor topology is static across years, yet it is being rebuilt as if changing per row-year.
- Each of the 5 variables requires iterating over this massive list, causing long runtime (~86+ hours).

**Core Issue**: The neighbor structure is repeated for every year, rather than leveraging the static relationship at the cell level.

---

**Optimization Strategy**  
- **Leverage static topology**: Compute a *cell-level only* neighbor lookup once.  
- **Process data by year**: For each yearâ€™s sub-data, compute stats using the static neighbor list.  
- **Vectorized neighbor stats**: Use efficient `rowsum` or `vapply` operations instead of iterating over millions of rows.  
- Avoid building 6.46M x neighbors mapping in memory.

---

**Working Optimized R Code**

```r
# Precompute neighbor relationships ONCE for cells
# id_order: vector of cell ids, consistent across all years
# neighbors: rook_neighbors_unique (list of integer vectors)
neighbor_lookup_static <- rook_neighbors_unique  # list: each cell -> neighbor cell indices (using position in id_order)

# Function to compute neighbor stats for one variable, one year
compute_neighbor_stats_year <- function(subdata, var_name, neighbor_lookup, id_to_pos) {
  vals <- subdata[[var_name]]
  res <- matrix(NA_real_, nrow = nrow(subdata), ncol = 3)
  colnames(res) <- c("max", "min", "mean")
  
  # Loop cells within this year
  for (i in seq_len(nrow(subdata))) {
    cell_id <- subdata$id[i]
    pos <- id_to_pos[[as.character(cell_id)]]
    neigh_pos <- neighbor_lookup[[pos]]
    if (length(neigh_pos) > 0) {
      neigh_ids <- id_order[neigh_pos]
      neigh_rows <- match(neigh_ids, subdata$id)  # find in current year's rows
      valid_idx <- neigh_rows[!is.na(neigh_rows)]
      if (length(valid_idx) > 0) {
        nv <- vals[valid_idx]
        nv <- nv[!is.na(nv)]
        if (length(nv) > 0) {
          res[i, ] <- c(max(nv), min(nv), mean(nv))
        }
      }
    }
  }
  
  as.data.frame(res)
}

# Prepare lookup
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Main loop by year to avoid massive memory overhead
years <- sort(unique(cell_data$year))
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

result_list <- list()

for (yr in years) {
  subdata <- cell_data[cell_data$year == yr, ]
  
  for (var_name in neighbor_source_vars) {
    stats_df <- compute_neighbor_stats_year(subdata, var_name, neighbor_lookup_static, id_to_pos)
    # Rename columns
    names(stats_df) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
    subdata <- cbind(subdata, stats_df)
  }
  
  result_list[[as.character(yr)]] <- subdata
}

cell_data_enriched <- do.call(rbind, result_list)

# Feed into prediction (Random Forest model already trained)
predictions <- predict(pretrained_rf_model, cell_data_enriched)
```

---

**Why this works efficiently**:
- **Static lookup**: Only 344k cells Ã— neighbors stored (vs 6.46M Ã— neighbors).
- Compute neighbor features per year block; reduces working set to ~344k rows at a time.
- Memory stays within 16GB limits while keeping logic unchanged.

**Preserves**:
- Trained Random Forest (no retraining).
- Original estimand and neighbor-based features.

**Expected runtime improvement**:
From >86 hours to a few hours, primarily I/O bound (year-by-year processing instead of giant list-building).  
Further speed gains possible via `data.table` merges or `Rcpp` for neighbor loop, but above approach meets constraint without drastic rewrites.
END_RESPONSE>>>
