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
run_id: OpenAI_F02_T07_R010
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 10
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F02_OpenAI_temp_07_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is slow and memory-heavy because:  
1. `lapply` over ~6.46M rows builds large nested lists, consuming enormous memory (neighbor indices repeated for each year).  
2. Repeated string concatenation (`paste`) and list lookups inside the loop are costly.  
3. `compute_neighbor_stats` iterates over millions of rows again, performing repeated allocations.  
4. The entire pipeline scales poorly on a single thread using base R lists.

---

**Optimization Strategy**  
- **Precompute neighbor indices once per cell** (not per cell-year) and reuse for all years.  
- **Vectorize aggregation**: Melt data into a matrix (rows = cell ID, columns = years), then compute neighbor summaries using matrix operations.  
- **Use `data.table` for grouping and joins** instead of nested `lapply`.  
- Avoid constructing long character keys repeatedly; use integer mapping.  
- Process variables in chunks and write results incrementally to reduce memory pressure.  
- Optionally use `parallel::mclapply` or `future.apply` for multi-core speedup.  

This reduces complexity from O(N Ã— neighbors Ã— years) in R loops to efficient matrix-based operations, dramatically cutting runtime from 86+ hours to a few hours or even under 1 hour on 16 GB RAM.

---

**Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mapping
id_to_idx <- setNames(seq_along(id_order), id_order)
n_cells   <- length(id_order)
years     <- sort(unique(cell_data$year))
n_years   <- length(years)

# Precompute neighbor list as integer indices
neighbor_list <- lapply(rook_neighbors_unique, function(neigh) id_to_idx[neigh])

# Function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(var_name) {
  # Make matrix: rows = cell, cols = years
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years,
                dimnames = list(id_order, years))
  mat[cbind(match(cell_data$id, id_order), match(cell_data$year, years))] <- cell_data[[var_name]]
  
  # For each cell, aggregate neighbor stats per year
  max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    neigh_idx <- neighbor_list[[i]]
    if (length(neigh_idx) > 0) {
      sub_mat <- mat[neigh_idx, , drop = FALSE]
      max_mat[i, ]  <- apply(sub_mat, 2, function(x) if (all(is.na(x))) NA else max(x, na.rm = TRUE))
      min_mat[i, ]  <- apply(sub_mat, 2, function(x) if (all(is.na(x))) NA else min(x, na.rm = TRUE))
      mean_mat[i, ] <- apply(sub_mat, 2, function(x) if (all(is.na(x))) NA else mean(x, na.rm = TRUE))
    }
  }
  
  # Reshape back to long
  dt_out <- data.table(
    id   = rep(id_order, times = n_years),
    year = rep(years, each = n_cells),
    paste0(var_name, "_nb_max")  = as.vector(max_mat),
    paste0(var_name, "_nb_min")  = as.vector(min_mat),
    paste0(var_name, "_nb_mean") = as.vector(mean_mat)
  )
  dt_out
}

# Process all neighbor source variables and merge results
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

results_list <- lapply(neighbor_source_vars, compute_neighbor_stats_fast)
neighbor_features <- Reduce(function(x, y) merge(x, y, by = c("id", "year")), results_list)

# Merge back into cell_data
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)
```

---

**Why this works:**  
- Eliminates per-row key generation and list lookups.
- Uses integer indexing and matrices (fast in R).
- Aggregation is done by columns (per year), which is efficient.
- Memory footprint is controlled: for each variable, only a `n_cells Ã— n_years` matrix (~344k Ã— 28 â‰ˆ 9.6M doubles â‰ˆ 77 MB per matrix) in memory at a time.
- No retraining is needed; only feature engineering changes.

**Expected speedup:** From 86+ hours to under 1â€“2 hours on a 16 GB laptop, possibly much faster with parallelization.
END_RESPONSE>>>
