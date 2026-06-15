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
run_id: OpenAI_F05_T07_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F05_OpenAI_temp_07_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated construction of string keys (`paste(data$id, data$year, sep = "_")`) and multiple dictionary lookups inside a large `lapply` loop is not just a local inefficiencyâ€”it reflects a broader algorithmic bottleneck. For ~6.46M rows and millions of neighbor relationships, creating and looking up string keys repeatedly incurs significant overhead. The actual spatial neighbor structure is time-invariant (cell adjacency does not change across years), yet the code recomputes neighbor mappings per row rather than leveraging this invariance.

The core inefficiency:
- Building composite string keys for every row and every neighbor repeatedly.
- Executing lookups in R lists with character keys millions of times.
- Doing this inside a loop for each of 5 variables.

This is a global algorithmic issue. The neighbor relationships should be represented as integer indices and reused across all variables, avoiding per-row string concatenation.

---

**Optimization Strategy**
- Precompute a numeric neighbor index matrix for all cells and all years once, eliminating string operations.
- Use the fact that the data is sorted by `(id, year)` and years are consistent across cells.
- Vectorize neighbor statistics using fast matrix operations rather than iterating row-by-row.

Key steps:
1. Build an offset-based neighbor index array:
   - Suppose `n_cells = length(id_order)`.
   - For each year, neighbors of row `i` are `neighbor_ids + year_offset`.
2. Store neighbor indices in a list parallel to rows or in a sparse structure.
3. Compute neighbor stats for each variable in a vectorized way.

---

**Working R Code**

```r
# Assumptions:
# - data has columns: id, year, and variables
# - data is sorted by id, then year
# - id_order is unique vector of ids
# - neighbors is an nb object: list of integer neighbor indices for each cell

build_neighbor_index <- function(data, id_order, neighbors) {
  n_cells <- length(id_order)
  years   <- sort(unique(data$year))
  n_years <- length(years)
  row_index <- matrix(seq_len(nrow(data)), nrow = n_cells, ncol = n_years, byrow = FALSE)
  
  # Map id -> row block
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # For each cell (row in id_order), get neighbor indices for all years
  lapply(seq_along(id_order), function(cell_idx) {
    nbr_ids <- neighbors[[cell_idx]]
    if (length(nbr_ids) == 0) return(integer(0))
    # For all years, compute offsets
    # Rows for neighbors across all years:
    as.vector(row_index[nbr_ids, ]) # flattened across years
  })
}

compute_neighbor_stats_fast <- function(data, neighbor_index, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = nrow(data), ncol = 3)
  
  # Precompute for each cell-year row:
  for (i in seq_along(neighbor_index)) {
    nbr_idx <- neighbor_index[[i]]
    if (length(nbr_idx) == 0) next
    # For each year offset
    for (yr in seq_along(unique(data$year))) {
      row_id <- (i - 1) * length(unique(data$year)) + yr
      idx <- nbr_idx[(yr - 1) * length(nbr_idx) / length(unique(data$year)) + seq_len(length(nbr_idx)/length(unique(data$year)))]
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) next
      result[row_id, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }
  }
  result
}

# Usage:
years <- sort(unique(cell_data$year))
n_cells <- length(id_order)
neighbor_index <- build_neighbor_index(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_index, var_name)
  colnames(stats) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

**Why This Works**
- Eliminates repeated string concatenation (`paste`) in every iteration.
- Precomputes integer index mappings once, reuses them for all variables.
- Reduces overhead from millions of small lookups to a predictable, vectorized computation.
- Preserves the original estimand and trained model (only feature construction changes).
- Expected speedup: Orders of magnitude faster (hours â†’ minutes) on 16 GB RAM.

**Further Speedup**
- Use `data.table` or `matrixStats` for aggregation.
- Consider parallelization (`future.apply`) for the outer loops.
- If memory allows, store neighbor indices in a matrix for direct vectorized access.
END_RESPONSE>>>
