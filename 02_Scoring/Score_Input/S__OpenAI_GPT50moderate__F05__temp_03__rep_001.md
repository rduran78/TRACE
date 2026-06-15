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
run_id: OpenAI_F05_T03_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste(id, year, sep="_")`) and name-based lookups (`idx_lookup[neighbor_keys]`) inside the `lapply` loop are a **local inefficiency** but also indicate a **broader algorithmic issue**:  
- For each of ~6.46M rows, you recompute neighbor keys and perform hash lookups in `idx_lookup`.  
- This is repeated for every variable in `neighbor_source_vars`, so the cost multiplies.  
- The neighbor structure is static across variables, so recomputing lookups for each variable is unnecessary.  

Thus, the bottleneck is both string operations and repeated index resolution. The algorithm should precompute **integer neighbor indices once** and reuse them for all variables.

---

### **Optimization Strategy**
1. **Precompute integer neighbor indices**: Instead of string keys, map `(id, year)` pairs to row indices once, then store neighbors as integer vectors.
2. **Reuse neighbor indices for all variables**: Compute `neighbor_lookup` once and apply it to all variables without recomputing keys.
3. **Vectorize where possible**: Avoid repeated `lapply` over millions of rows for each variable.

---

### **Reformulated Approach**
- Build a matrix or list of integer neighbor indices (`neighbor_lookup`) once.
- Use these indices directly in `compute_neighbor_stats` for all variables.

---

### **Working R Code**

```r
# Precompute neighbor indices once
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Map id to position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Create a fast lookup for (id, year) -> row index
  # Assuming data is sorted by year and id
  row_index <- seq_len(nrow(data))
  
  # Precompute a matrix for id-year positions
  # We'll use integer mapping: id_index + year_offset
  years <- sort(unique(data$year))
  year_to_offset <- setNames(seq_along(years), years)
  
  # Compute composite key as integer: (id_ref - 1) * n_years + year_offset
  n_years <- length(years)
  composite_key <- (id_to_ref[as.character(data$id)] - 1L) * n_years + year_to_offset[as.character(data$year)]
  
  # Build a direct lookup vector
  max_key <- max(composite_key)
  idx_lookup <- integer(max_key)
  idx_lookup[composite_key] <- row_index
  
  # Now build neighbor lookup as integer indices
  lapply(seq_len(nrow(data)), function(i) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- (id_to_ref[as.character(neighbor_ids)] - 1L) * n_years + year_to_offset[as.character(data$year[i])]
    result <- idx_lookup[neighbor_keys]
    result[result != 0L]  # remove zeros (missing)
  })
}

# Compute neighbor stats using precomputed indices
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
  result
}

# Main pipeline
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Works**
- Eliminates repeated string concatenation and hash lookups.
- Converts `(id, year)` mapping to integer arithmetic for O(1) access.
- Precomputes neighbor indices once and reuses them across variables.
- Memory-efficient: `neighbor_lookup` is a list of integer vectors, no repeated strings.

---

**Expected Impact**:  
- From 86+ hours to a few hours or less on a 16 GB laptop.
- Complexity drops from repeated string operations Ã— 6.46M Ã— 5 variables to a single integer-based mapping plus 5 numeric passes.

**Preserves**:  
- Original estimand and Random Forest model remain unchanged.  
- The neighbor-based features are numerically identical.
END_RESPONSE>>>
