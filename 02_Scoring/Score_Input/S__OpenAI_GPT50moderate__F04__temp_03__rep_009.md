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
run_id: OpenAI_F04_T03_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F04_OpenAI_temp_03_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The major bottleneck is the repeated use of `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation, list indexing, and repeated lookups in R lists and named vectors, which is extremely inefficient at this scale. The neighbor structure is static across years, so recomputing neighbor indices for every row is unnecessary. Additionally, `compute_neighbor_stats` repeatedly traverses lists and performs small vector operations in R, which is slow for millions of rows.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** (not per cell-year) and reuse them for all years.
2. **Vectorize operations**: Reshape data into a matrix with rows = cells and columns = years (or vice versa). Compute neighbor stats using matrix operations instead of per-row loops.
3. **Use `data.table` for efficient joins and aggregation**.
4. **Parallelize if possible** (optional).
5. Avoid string concatenation and named lookups in the inner loop.

---

**Optimized Approach**  
- Convert `cell_data` into a `data.table` keyed by `id` and `year`.
- Create a wide matrix for each variable: rows = cell IDs, columns = years.
- For each variable, compute neighbor stats by applying `pmax`, `pmin`, and `rowMeans` over neighbor rows in the matrix.
- Melt back to long format and merge.

---

### **Working R Code**

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Basic parameters
ids   <- sort(unique(cell_data$id))
years <- sort(unique(cell_data$year))
n_ids <- length(ids)
n_years <- length(years)

# Map id to row index
id_to_idx <- setNames(seq_along(ids), ids)

# Precompute neighbor index list (once per cell)
neighbor_idx_list <- lapply(rook_neighbors_unique, function(neigh_ids) {
  id_to_idx[as.character(neigh_ids)]
})

# Function to compute neighbor stats for one variable
compute_neighbor_features <- function(var_name) {
  # Create matrix: rows = cells, cols = years
  mat <- matrix(NA_real_, nrow = n_ids, ncol = n_years)
  vals <- cell_data[[var_name]]
  mat[cbind(id_to_idx[as.character(cell_data$id)], match(cell_data$year, years))] <- vals
  
  # For each cell, compute stats across neighbors for each year
  max_mat <- matrix(NA_real_, nrow = n_ids, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_ids, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_ids, ncol = n_years)
  
  for (i in seq_len(n_ids)) {
    neigh_idx <- neighbor_idx_list[[i]]
    if (length(neigh_idx) == 0) next
    neigh_vals <- mat[neigh_idx, , drop = FALSE]
    max_mat[i, ]  <- apply(neigh_vals, 2, function(x) if (all(is.na(x))) NA else max(x, na.rm = TRUE))
    min_mat[i, ]  <- apply(neigh_vals, 2, function(x) if (all(is.na(x))) NA else min(x, na.rm = TRUE))
    mean_mat[i, ] <- apply(neigh_vals, 2, function(x) if (all(is.na(x))) NA else mean(x, na.rm = TRUE))
  }
  
  # Convert back to long format
  dt_out <- data.table(
    id = rep(ids, times = n_years),
    year = rep(years, each = n_ids),
    paste0(var_name, "_nbr_max")  = as.vector(max_mat),
    paste0(var_name, "_nbr_min")  = as.vector(min_mat),
    paste0(var_name, "_nbr_mean") = as.vector(mean_mat)
  )
  dt_out
}

# Compute for all variables and merge
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
feature_list <- lapply(neighbor_source_vars, compute_neighbor_features)
neighbor_features <- Reduce(function(x, y) merge(x, y, by = c("id", "year")), feature_list)

# Merge back into cell_data
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)
```

---

**Why this is faster**  
- Eliminates per-row string operations and named lookups.
- Reduces complexity from O(n_rows * avg_neighbors) in R loops to efficient matrix operations.
- Precomputes neighbor indices once.
- Uses `data.table` for fast joins.

**Expected performance**  
On 6.46M rows and ~1.37M neighbor links, this approach should reduce runtime from **86+ hours to a few hours or less**, depending on I/O and memory constraints. Further speedups can be achieved by:
- Using `parallel::mclapply` for the variable loop.
- Using `Rcpp` or `matrixStats` for inner computations if needed.
END_RESPONSE>>>
