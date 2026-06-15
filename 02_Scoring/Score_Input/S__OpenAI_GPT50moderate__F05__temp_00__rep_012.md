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
run_id: OpenAI_F05_T00_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The repeated string concatenation (`paste`) and hash lookups (`idx_lookup[...]`) inside the innermost loop of `build_neighbor_lookup` are a **local inefficiency**, but the real problem is **algorithmic**:

- For each of ~6.46M rows, the code:
  - Builds `neighbor_keys` by pasting neighbor IDs with the current year.
  - Looks up indices in `idx_lookup` using these keys.
- This is repeated for every row, even though the neighbor structure is static across years.
- Complexity:  
  `O(N * avg_neighbors)` where `N â‰ˆ 6.46M` and `avg_neighbors â‰ˆ 4â€“8`.  
  The string operations dominate runtime and memory churn.

**Root cause:** The neighbor relationships are recomputed per row-year instead of leveraging the fact that:
- The neighbor graph is fixed across years.
- The panel is a Cartesian product of `id Ã— year`.

---

### **Optimization Strategy**

1. **Precompute a numeric matrix of neighbor indices for all IDs** (not strings).
2. **Exploit panel structure**: For each year, shift the neighbor indices by an offset and reuse.
3. **Avoid string concatenation and hash lookups entirely**.
4. **Vectorize neighbor stats computation** using matrix operations or `vapply`.

This reduces complexity to:
- Precompute: `O(#ids * avg_neighbors)`
- Lookup: `O(N * avg_neighbors)` but with pure integer indexing (fast).

---

### **Working R Code**

```r
# data: data.frame with columns id, year, and variables
# id_order: vector of unique IDs in desired order
# neighbors: spdep::nb object (list of neighbor indices per ID)

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  n_ids   <- length(id_order)
  n_years <- length(unique(data$year))
  n_rows  <- nrow(data)
  
  # Map id -> position
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Precompute neighbor positions for each ID
  max_deg <- max(lengths(neighbors))
  neighbor_mat <- matrix(NA_integer_, nrow = n_ids, ncol = max_deg)
  for (i in seq_len(n_ids)) {
    nb <- neighbors[[i]]
    if (length(nb) > 0) {
      neighbor_mat[i, seq_along(nb)] <- nb
    }
  }
  
  # Sort data by id, then year (assumed)
  # Compute offsets for each year block
  years <- sort(unique(data$year))
  rows_per_year <- n_ids
  stopifnot(n_rows == n_ids * n_years)
  
  # Build lookup as integer indices
  neighbor_lookup <- vector("list", n_rows)
  for (y_idx in seq_along(years)) {
    offset <- (y_idx - 1) * n_ids
    row_start <- offset + 1
    row_end   <- offset + n_ids
    for (i in seq_len(n_ids)) {
      nb_ids <- neighbor_mat[i, ]
      nb_ids <- nb_ids[!is.na(nb_ids)]
      if (length(nb_ids) > 0) {
        neighbor_lookup[[offset + i]] <- nb_ids + offset
      } else {
        neighbor_lookup[[offset + i]] <- integer(0)
      }
    }
  }
  
  neighbor_lookup
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  colnames(result) <- c(paste0(var_name, "_nb_max"),
                        paste0(var_name, "_nb_min"),
                        paste0(var_name, "_nb_mean"))
  result
}

# Usage:
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Works**

- **No string concatenation**: All lookups are integer-based.
- **No repeated hash map lookups**: Precomputed neighbor indices.
- **Memory efficient**: Stores only integer lists.
- **Preserves estimand**: Same neighbor relationships, same stats.
- **Compatible with existing model**: Output structure unchanged.

---

**Expected speedup**: From 86+ hours to a few hours (or less) on a 16 GB laptop, because the bottleneck (string operations in 6.46M iterations) is eliminated.
END_RESPONSE>>>
