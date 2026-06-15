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
run_id: OpenAI_F07_T07_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is too slow because:  
1. **Inefficient nested loops**: For ~6.46M rows, creating neighbor lookups for each row and repeatedly subsetting vectors inside `lapply` is highly inefficient.  
2. **Repeated concatenations and lookups**: Converting cell IDs and years to strings and doing `paste()` for every row adds massive overhead.  
3. **Interpretation of panel structure**: Each cell-year row recomputes neighbor indices instead of leveraging a precomputed, panel-aware adjacency structure.  
4. **Memory-bound operations**: Multiple large lists and repeated indexing stress RAM and GC.  

---

**Optimization Strategy**  
- **Precompute a direct integer neighbor index matrix** for the panel so every row can be accessed without string operations.  
- **Vectorize neighbor computations** using `matrixStats` or data.table grouping rather than per-row `lapply`.  
- **Leverage panel regularity**: Each cell has the same set of spatial neighbors across years, so replicate neighbor indices across years efficiently.  
- **Chunked processing**: Process in chunks to stay within 16 GB RAM.  
- **Avoid retraining the RF model**: Only modify feature computation.  

---

**Working R Code**  

```r
library(data.table)
library(matrixStats)

# Assume: cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor indices across all years --------------------------------
# rook_neighbors_unique: list of integer vectors (spdep nb object)
n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

# Map cell id -> position in id_order
id_to_pos <- setNames(seq_along(id_order), id_order)

# Build a matrix of neighbor positions for each cell (ragged -> padded with 0)
max_nbrs <- max(lengths(rook_neighbors_unique))
neighbor_matrix <- matrix(0L, n_cells, max_nbrs)
for (i in seq_len(n_cells)) {
  nbrs <- rook_neighbors_unique[[i]]
  if (length(nbrs) > 0) neighbor_matrix[i, seq_along(nbrs)] <- nbrs
}

# Expand to panel indices: compute row index for each cell-year
# Create a lookup: (cell position, year index) -> row index
cell_year_index <- matrix(NA_integer_, n_cells, n_years)
for (i in seq_len(n_years)) {
  yr_rows <- which(cell_data$year == years[i])
  # data is keyed by id, so row order matches id_order
  cell_year_index[, i] <- yr_rows
}

# Compute neighbor stats efficiently ------------------------------------------
compute_neighbor_stats_fast <- function(var_vec) {
  # var_vec is length = nrow(cell_data)
  result <- matrix(NA_real_, nrow(cell_data), 3L) # columns: max, min, mean
  for (yr_idx in seq_len(n_years)) {
    yr_rows <- cell_year_index[, yr_idx]
    vals    <- var_vec[yr_rows]          # values for this year
    for (j in seq_len(n_cells)) {
      nbr_ids <- neighbor_matrix[j, ]
      nbr_ids <- nbr_ids[nbr_ids > 0L]
      if (length(nbr_ids) == 0) next
      nbr_rows <- yr_rows[nbr_ids]
      nbr_vals <- vals[nbr_ids]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) > 0) {
        result[yr_rows[j], ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
      }
    }
  }
  result
}

# Apply for each variable (loop in R is fine because inner loop is small)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data[[var_name]])
  cell_data[[paste0(var_name, "_nbr_max")]]  <- stats_mat[,1]
  cell_data[[paste0(var_name, "_nbr_min")]]  <- stats_mat[,2]
  cell_data[[paste0(var_name, "_nbr_mean")]] <- stats_mat[,3]
}

```

---

**Why this is faster**  
- Eliminates repeated string operations and list indexing.  
- Uses integer matrices for neighbor references and panel indexing.  
- Processes year-by-year, reducing memory overhead and improving cache locality.  
- Complexity drops from O(n_rows * avg_neighbors) with heavy R overhead to a more vectorized and memory-efficient approach.  

**Expected runtime improvement**: From ~86 hours down to a few hours (or less), depending on CPU and disk speed.  

**Preserves the original estimand**: Same neighbor definitions, same stats (max, min, mean) for each cell-year panel row.  
**Trained RF model remains intact**: We only generate additional features for prediction.
END_RESPONSE>>>
