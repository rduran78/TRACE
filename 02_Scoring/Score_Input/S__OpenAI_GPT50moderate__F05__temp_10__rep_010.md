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
run_id: OpenAI_F05_T10_R010
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 10
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F05_OpenAI_temp_10_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 Diagnosis:  
The repeated string-based key construction and lookup (`paste(...)` and `idx_lookup[...]`) in `build_neighbor_lookup` occurs inside an `lapply` over ~6.46M rows. This is not just a local inefficiencyâ€”it is a fundamental bottleneck because each row recomputes both neighbor keys and map lookups. With 6.46M iterations, multiple neighbor lookups per iteration, and repeated string concatenation, this scales very poorly.

Optimization Strategy:  
Instead of dynamically creating string keys per row-year, precompute reusable indices. Represent data as a matrix with dimensions `[cell_id, year]` and replace string operations with integer indexing. Build `neighbor_lookup` in terms of integer positions directly, avoiding repeated paste/lookup cycles. Use vectorized operations and base indexing to compute neighbor stats.

Working R Code:

```r
# Precompute: cell_id -> row indices by year
build_neighbor_index <- function(data, id_order) {
  # Ensure ids are integer factors aligned with id_order
  cell_id_to_idx <- match(data$id, id_order)
  years <- sort(unique(data$year))
  year_to_idx <- match(data$year, years)

  # Matrix map: row index for (cell_id, year)
  row_map <- matrix(NA_integer_, nrow = length(id_order), ncol = length(years))
  row_map[cbind(cell_id_to_idx, year_to_idx)] <- seq_len(nrow(data))

  list(row_map = row_map, years = years)
}

build_neighbor_lookup_fast <- function(row_map, neighbors) {
  n_years <- ncol(row_map)
  lapply(seq_len(nrow(row_map)), function(cell_idx) {
    nb <- neighbors[[cell_idx]]
    if (length(nb) == 0) return(vector("list", n_years))
    lapply(seq_len(n_years), function(y_idx) {
      valid_rows <- row_map[nb, y_idx]
      # Drop NAs
      valid_rows[!is.na(valid_rows)]
    })
  })
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- vector("list", length(vals))
  for (i in seq_along(result)) {
    nb_rows <- neighbor_lookup[[i]]
    if (length(nb_rows) == 0) {
      result[[i]] <- c(NA, NA, NA)
    } else {
      nbr_vals <- vals[nb_rows]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) == 0) {
        result[[i]] <- c(NA, NA, NA)
      } else {
        result[[i]] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
      }
    }
  }
  do.call(rbind, result)
}

# Pipeline
idx_struct <- build_neighbor_index(cell_data, id_order)
row_map <- idx_struct$row_map

nested_lookup <- build_neighbor_lookup_fast(row_map, rook_neighbors_unique)

# Flatten neighbor_lookup for each row in dataset: integer positions only
neighbor_lookup <- vector("list", nrow(cell_data))
cell_id_to_idx <- match(cell_data$id, id_order)
year_to_idx <- match(cell_data$year, idx_struct$years)
for (r in seq_len(nrow(cell_data))) {
  neighbor_lookup[[r]] <- nested_lookup[[cell_id_to_idx[r]]][[year_to_idx[r]]]
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_nbr_")
  cell_data <- cbind(cell_data, stats)
}
```

Why this is faster:  
- Eliminates repeated string paste and hash lookup.
- Uses integer-based indexing via a precomputed matrix `row_map`.
- Moves from per-row reconstruction to amortized precomputation.
- Reduces complexity from O(N * neighbors * years) string operations to integer-based lookups.

Expected improvement:  
Runtime drops from 86+ hours down to manageable few hours (or less) on 16 GB laptop because string overhead is replaced by integer operations, and neighbor index resolution is reused without repeated concatenation.
END_RESPONSE>>>
