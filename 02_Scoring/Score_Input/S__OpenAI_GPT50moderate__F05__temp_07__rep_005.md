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
run_id: OpenAI_F05_T07_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F05_OpenAI_temp_07_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The repeated string-based key construction and lookup (`paste(...)` and `idx_lookup[...]`) within the inner `lapply` is a **local inefficiency**, but it is also symptomatic of a broader algorithmic issue: the current design performs high-cost string manipulations and repeated lookups for every cell-year row (â‰ˆ 6.46M times), even though the mapping between `(id, year)` and row index is fully deterministic and could be vectorized or precomputed once.

### **Diagnosis**
- For each of 6.46M rows, the code:
  1. Pastes `neighbor_cell_ids` and `year` into keys.
  2. Looks up those keys in `idx_lookup` (a named vector).
- This happens inside an `lapply` over all rows, creating **millions of string operations and hash lookups**.
- Additionally, `compute_neighbor_stats` repeats neighbor value aggregation per variable, multiplying cost by 5 variables.
- Current pattern is essentially O(n Ã— avg_neighbors) string work, where n = 6.46M.

### **Optimization Strategy**
- Replace string-based lookup with integer-based indexing:
  - Precompute a `(id Ã— year) â†’ row_index` integer matrix or use direct integer mapping.
  - Instead of building `paste(...)` keys repeatedly, compute neighbor row indices via arithmetic or a join.
- Vectorize neighbor statistics computation:
  - Store neighbor indices in an integer matrix or list once.
  - Then apply computations without string work.
- Avoid recomputing neighbor lookup per year-variable combination. Build it once, then reuse.

### **Algorithmic Reformulation**
1. Create a mapping from `(id, year)` â†’ row index using fast integer operations.
2. Compute neighbor lookups as integer vectors *once*.
3. Use vectorized aggregation for neighbor stats.

---

### **Working R Code**

```r
# Precompute id-year to row index mapping
build_neighbor_lookup_optimized <- function(data, id_order, neighbors) {
  # Ensure ids and years are integer or factor for fast mapping
  id_to_row <- match(data$id, id_order)     # maps each row's id to its ref index
  year_vals <- as.integer(as.factor(data$year))  # map years to integers
  
  # Build matrix for (id_ref, year_int) -> row index
  n_ids <- length(id_order)
  n_years <- length(unique(data$year))
  row_index_matrix <- matrix(NA_integer_, nrow = n_ids, ncol = n_years)
  for (i in seq_len(nrow(data))) {
    row_index_matrix[id_to_row[i], year_vals[i]] <- i
  }
  
  # Build neighbor lookup: list of integer vectors
  lapply(seq_len(nrow(data)), function(i) {
    ref_idx <- id_to_row[i]
    neighbor_refs <- neighbors[[ref_idx]]
    if (length(neighbor_refs) == 0) return(integer(0))
    year_idx <- year_vals[i]
    # Direct integer lookup from matrix
    idxs <- row_index_matrix[neighbor_refs, year_idx]
    idxs[!is.na(idxs)]
  })
}

# Compute neighbor stats with vectorization
compute_neighbor_stats_optimized <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) > 0) {
        result[i, ] <- c(max(v), min(v), mean(v))
      }
    }
  }
  result
}

# Outer loop unchanged except for optimized functions
neighbor_lookup <- build_neighbor_lookup_optimized(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_optimized(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Expected Gains**
- Eliminates millions of string concatenations and hash lookups.
- Uses integer arithmetic and matrix indexing (O(1) per lookup).
- Memory: `row_index_matrix` is `n_ids Ã— n_years` (â‰ˆ 344,208 Ã— 28 â‰ˆ 9.6M integers, ~40 MB), feasible on 16 GB RAM.
- Runtime improvement: from **86+ hours** to **a few hours or less**, dominated by aggregation and I/O.

**Preserves original estimand and Random Forest model.**
END_RESPONSE>>>
