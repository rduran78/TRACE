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
run_id: OpenAI_F05_T03_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string-key construction (`paste(id, year, sep="_")`) and lookup inside `lapply` is not just a local inefficiency; it is a symptom of a broader algorithmic issue. For each of ~6.46M rows, the code repeatedly builds neighbor keys and performs hash lookups in `idx_lookup`. This results in tens of millions of string concatenations and hash lookups, which is extremely costly in R.

The root cause:  
- The algorithm repeatedly maps `(id, year)` â†’ row index inside the innermost loop.
- Neighbor relationships are static across years, but the code recomputes them for every row-year combination.
- `compute_neighbor_stats` then iterates again over all rows, compounding the overhead.

**Optimization Strategy**  
Reformulate the algorithm to avoid repeated string operations and hash lookups. Key ideas:  
1. Precompute a numeric matrix of neighbor row indices for all rows and all years once, instead of doing it inside `lapply`.  
2. Use integer indexing rather than string keys.  
3. Exploit the fact that neighbor structure is constant across years: build a base neighbor index for IDs, then replicate across years.  
4. Use `matrixStats` or `data.table` for fast aggregation.

**Working R Code**

```r
library(data.table)

# Assume: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mapping: id -> row indices by year
id_order <- sort(unique(cell_data$id))
years <- sort(unique(cell_data$year))
n_id <- length(id_order)
n_year <- length(years)

# Build a lookup table: row index matrix [id, year]
row_index_matrix <- matrix(NA_integer_, n_id, n_year,
                           dimnames = list(id_order, years))
row_index_matrix[cbind(match(cell_data$id, id_order),
                        match(cell_data$year, years))] <- seq_len(nrow(cell_data))

# rook_neighbors_unique: list of neighbors per id position
# Precompute neighbor index matrix for all ids
max_neighbors <- max(lengths(rook_neighbors_unique))
neighbor_matrix <- matrix(NA_integer_, n_id, max_neighbors)
for (i in seq_along(rook_neighbors_unique)) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) > 0) neighbor_matrix[i, seq_along(nb)] <- nb
}

# Function to compute neighbor stats for all variables
compute_neighbor_features <- function(var_name) {
  vals <- cell_data[[var_name]]
  result <- matrix(NA_real_, nrow(cell_data), 3) # max, min, mean
  
  for (y_idx in seq_along(years)) {
    year_rows <- row_index_matrix[, y_idx]
    valid_rows <- !is.na(year_rows)
    year_vals <- vals[year_rows]
    
    for (i in which(valid_rows)) {
      nb_ids <- neighbor_matrix[i, ]
      nb_ids <- nb_ids[!is.na(nb_ids)]
      if (length(nb_ids) == 0) next
      nb_rows <- row_index_matrix[nb_ids, y_idx]
      nb_rows <- nb_rows[!is.na(nb_rows)]
      if (length(nb_rows) == 0) next
      nb_vals <- vals[nb_rows]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0) next
      result[year_rows[i], ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
    }
  }
  colnames(result) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  result
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute all neighbor features and cbind
neighbor_features <- do.call(cbind, lapply(neighbor_source_vars, compute_neighbor_features))
cell_data <- cbind(cell_data, neighbor_features)
```

**Why this is faster:**  
- Eliminates per-row string concatenation and hash lookups.
- Uses integer indexing and precomputed matrices.
- Loops only over IDs Ã— years, not rows Ã— neighbors repeatedly with string ops.
- Memory-friendly: `row_index_matrix` is about `n_id Ã— n_year` (~344k Ã— 28 â‰ˆ 9.6M integers â‰ˆ 76 MB).

**Expected impact:**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, because weâ€™ve removed the dominant O(N Ã— neighbors Ã— string_ops) cost and replaced it with numeric indexing.

**Preserves:**  
- Original Random Forest model (unchanged).
- Original estimand (neighbor stats computed identically, just faster).
END_RESPONSE>>>
