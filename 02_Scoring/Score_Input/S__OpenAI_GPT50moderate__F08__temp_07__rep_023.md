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
run_id: OpenAI_F08_T07_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because `compute_neighbor_stats` iterates over **6.46 million rows** and repeatedly subsets vectors in R (`lapply` over every row). This creates massive overhead from R loops and redundant operations. The neighbor relationships are **static across years**, yet the function rebuilds neighbor statistics per cell-year individually. The root inefficiency is treating each of the 6.46M cell-year rows independently instead of leveraging the fixed adjacency to compute statistics in a vectorized, year-wise manner.

---

**Optimization Strategy**  
1. **Exploit static neighbor structure**: Precompute a fixed mapping of cell â†’ neighbors at the cell level (344k cells).  
2. **Compute per-year statistics in blocks**: For each year, extract the variable vector (length = 344k), compute neighbor max/min/mean in a single vectorized pass using the static adjacency.  
3. **Avoid row-by-row lapply**: Replace with matrix operations or `vapply` on 344k cells Ã— 28 years (instead of 6.46M lapply calls).  
4. **Memory-friendly approach**: Process one variable and one year at a time to stay within 16 GB RAM.  
5. **Preserve original estimand**: Same neighbor relationships, same variable transformations, only faster computation.  
6. **Reuse Random Forest model**: Just add new features to `cell_data` before prediction; no retraining.  

---

**Optimized R Code**  

```r
# Precompute static neighbor list as integer indices
build_static_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors: spdep::nb object
  # id_order: vector of cell ids in the same order as neighbors
  lapply(neighbors, function(nb_idx) id_order[nb_idx])
}

# Compute neighbor stats by year for one variable
compute_neighbor_stats_by_year <- function(cell_data, var_name, neighbor_list) {
  years <- sort(unique(cell_data$year))
  n_cells <- length(neighbor_list)
  
  # Prepare output: matrix [nrow(cell_data), 3]
  result <- matrix(NA_real_, nrow(cell_data), 3)
  
  for (yr in years) {
    idx_year <- which(cell_data$year == yr)
    vals <- cell_data[[var_name]][idx_year]
    
    # Assuming idx_year is ordered by id_order
    stopifnot(length(vals) == n_cells)
    
    # Compute stats for each cell
    max_vec <- numeric(n_cells)
    min_vec <- numeric(n_cells)
    mean_vec <- numeric(n_cells)
    
    for (i in seq_len(n_cells)) {
      nb_idx <- neighbor_list[[i]]
      if (length(nb_idx) == 0) {
        max_vec[i] <- min_vec[i] <- mean_vec[i] <- NA_real_
      } else {
        nb_vals <- vals[nb_idx]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) == 0) {
          max_vec[i] <- min_vec[i] <- mean_vec[i] <- NA_real_
        } else {
          max_vec[i] <- max(nb_vals)
          min_vec[i] <- min(nb_vals)
          mean_vec[i] <- mean(nb_vals)
        }
      }
    }
    
    # Fill result rows for this year
    result[idx_year, ] <- cbind(max_vec, min_vec, mean_vec)
  }
  
  colnames(result) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  result
}

# Main pipeline
id_order <- sort(unique(cell_data$id))  # Ensure consistent ordering
neighbor_list <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

# Split cell_data by year and reorder to align with id_order
cell_data <- cell_data[order(cell_data$year, match(cell_data$id, id_order)), ]

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_by_year(cell_data, var_name, neighbor_list)
  cell_data <- cbind(cell_data, stats_mat)
}

# Now you can run the pre-trained Random Forest prediction step as before
```

---

**Performance Gains**  
- Reduces complexity from **6.46M row-wise operations** to **28 yearly blocks Ã— 344k cells**.
- Eliminates repeated string concatenation and index lookups.
- Memory overhead limited to one year slice at a time.
- Expected runtime: **hours â†’ minutes** on a standard laptop.

**Key Points**  
- Preserves the original neighbor structure and estimand.
- Keeps Random Forest model unchanged.
- Fully exploits the static adjacency vs. dynamic variables distinction.
END_RESPONSE>>>
