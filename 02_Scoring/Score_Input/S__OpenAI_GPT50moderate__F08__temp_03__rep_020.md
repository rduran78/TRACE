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
run_id: OpenAI_F08_T03_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F08_OpenAI_temp_03_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it recomputes neighbor statistics for all 6.46M rows across 28 years, repeatedly scanning large vectors. The neighbor structure is static across years, but the code redundantly processes neighbor relationships for each cell-year. This leads to massive repeated work and memory overhead.

**Optimization Strategy**  
- Precompute a static neighbor index for each cell (not cell-year).
- For each year, slice the data into a matrix of size `n_cells Ã— n_years` for each variable.
- Compute neighbor statistics year-by-year using vectorized operations or `apply` over neighbors.
- Append results back to the long panel without changing the Random Forest input structure.
- Avoid recomputing neighbor lookups for every row; reuse static mapping.

---

### **Optimized R Code**

```r
# Precompute static neighbor list (cell-level)
# id_order: vector of unique cell IDs in consistent order
# rook_neighbors_unique: spdep::nb object
neighbor_list <- lapply(rook_neighbors_unique, function(neigh) as.integer(neigh))

# Convert panel data into wide format for fast yearly access
# Assume cell_data has columns: id, year, and variables
library(data.table)
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# Unique IDs and years
ids <- unique(dt$id)
years <- sort(unique(dt$year))
n_cells <- length(ids)
n_years <- length(years)

# Create an index for fast mapping
id_index <- setNames(seq_along(ids), ids)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(var_name) {
  # Wide matrix: rows = cells, cols = years
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(id_index[dt$id], match(dt$year, years))] <- dt[[var_name]]
  
  # Prepare result matrices
  max_mat <- min_mat <- mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Compute stats year by year
  for (y in seq_len(n_years)) {
    vals <- mat[, y]
    for (i in seq_len(n_cells)) {
      neigh <- neighbor_list[[i]]
      if (length(neigh) == 0) next
      neigh_vals <- vals[neigh]
      neigh_vals <- neigh_vals[!is.na(neigh_vals)]
      if (length(neigh_vals) == 0) next
      max_mat[i, y] <- max(neigh_vals)
      min_mat[i, y] <- min(neigh_vals)
      mean_mat[i, y] <- mean(neigh_vals)
    }
  }
  
  # Melt back to long
  data.table(
    id = rep(ids, times = n_years),
    year = rep(years, each = n_cells),
    max = as.vector(max_mat),
    min = as.vector(min_mat),
    mean = as.vector(mean_mat)
  )
}

# Apply for all neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
result_list <- lapply(neighbor_source_vars, function(v) {
  stats_dt <- compute_neighbor_stats_fast(v)
  setnames(stats_dt, c("id", "year", paste0(v, "_nbr_max"), paste0(v, "_nbr_min"), paste0(v, "_nbr_mean")))
  stats_dt
})

# Merge all stats back into main data
for (res in result_list) {
  dt <- merge(dt, res, by = c("id", "year"), all.x = TRUE)
}

# Convert back to data.frame if needed
cell_data <- as.data.frame(dt)
```

---

### **Why This Is Faster**
- **Static neighbor mapping**: Built once at cell-level, reused across years.
- **Matrix-based computation**: Access values by index instead of repeated lookups.
- **Year loop only**: Avoids 6.46M row-level loops; instead, loops over ~344k cells Ã— 28 years.
- **Memory-efficient**: Works in chunks (per variable), avoids huge intermediate lists.

**Expected runtime**: Hours reduced from 86+ to a few hours (or less) on a 16 GB laptop.

**Preserves**:
- Original estimand (neighbor max, min, mean per cell-year).
- Trained Random Forest model (no retraining).
- Panel structure and variable names.
END_RESPONSE>>>
