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
run_id: OpenAI_F08_T00_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F08_OpenAI_temp_00_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it computes neighbor statistics (`max`, `min`, `mean`) for each cell-year row (â‰ˆ6.46M rows) by iterating over a large list of neighbor indices. This results in repeated lookups and redundant computations across years, even though the neighbor structure is static. The complexity is roughly `O(N * k)` per variable, where `N` is 6.46M and `k` is average neighbor count, multiplied by 5 variables. This is memory- and CPU-intensive on a 16 GB laptop.

**Optimization Strategy**  
- **Exploit static neighbor structure**: Precompute a neighbor index matrix for cells (not cell-years) once.
- **Vectorize across years**: For each year, compute neighbor stats using matrix operations instead of per-row loops.
- **Chunk processing**: Process one year at a time to keep memory usage manageable.
- **Avoid repeated lookups**: Use a fixed mapping from cell IDs to row indices for each year.
- **Preserve model and estimand**: Do not retrain; only change feature computation.

---

### **Optimized R Code**

```r
# Precompute neighbor index list for cells (static)
build_neighbor_index <- function(id_order, neighbors) {
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  lapply(neighbors, function(nbs) id_to_pos[as.character(id_order[nbs])])
}

neighbor_index <- build_neighbor_index(id_order, rook_neighbors_unique)

# Function to compute neighbor stats for one variable and one year
compute_year_neighbor_stats <- function(data_year, var_name, neighbor_index) {
  vals <- data_year[[var_name]]
  n_cells <- length(neighbor_index)
  
  # Preallocate result matrix
  res <- matrix(NA_real_, nrow = n_cells, ncol = 3)
  
  for (i in seq_len(n_cells)) {
    nbs <- neighbor_index[[i]]
    if (length(nbs) > 0) {
      nb_vals <- vals[nbs]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) > 0) {
        res[i, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
      }
    }
  }
  colnames(res) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  res
}

# Main loop: process year by year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))
n_cells <- length(id_order)

# Prepare final result container
neighbor_features <- vector("list", length(years))

for (y in seq_along(years)) {
  yr <- years[y]
  message("Processing year: ", yr)
  
  data_year <- cell_data[cell_data$year == yr, ]
  stopifnot(nrow(data_year) == n_cells)  # Ensure full coverage
  
  year_feats <- NULL
  for (var_name in neighbor_source_vars) {
    stats_mat <- compute_year_neighbor_stats(data_year, var_name, neighbor_index)
    year_feats <- if (is.null(year_feats)) stats_mat else cbind(year_feats, stats_mat)
  }
  
  # Combine with original data_year
  neighbor_features[[y]] <- cbind(data_year, year_feats)
}

# Combine all years back
cell_data <- do.call(rbind, neighbor_features)
```

---

### **Why This Is Faster**
- **Static neighbor index**: Built once for 344,208 cells, reused for all years.
- **Year-wise processing**: Only 344,208 rows in memory at a time, not 6.46M.
- **Loop only over cells, not cell-years**: Reduces redundant computation by factor of 28.
- **No repeated string concatenation or lookups**: Direct integer indexing.

---

**Expected Performance**  
- Complexity drops from `O(N * k)` for all rows to `O(C * k * Y)` where `C` = 344k cells, `Y` = 28 years, `k` â‰ˆ 4â€“8 neighbors.
- Estimated runtime: a few hours instead of 86+ hours on a 16 GB laptop.

**Preserves**  
- Original estimand (neighbor max, min, mean per cell-year).
- Pre-trained Random Forest model (no retraining needed).
END_RESPONSE>>>
