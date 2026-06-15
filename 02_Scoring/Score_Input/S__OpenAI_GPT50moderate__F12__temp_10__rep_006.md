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
run_id: OpenAI_F12_T10_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F12_OpenAI_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleague's claim that `do.call(rbind, result)` and repeated list-binding inside `compute_neighbor_stats()` is the main bottleneck is incorrect. The real bottleneck is deeper:  
- **Repeated scanning through large lists and strings in `build_neighbor_lookup`** for each of ~6.46 million rows is extremely expensive.  
- It constructs neighbor keys per row using `paste()` and does repeated lookups in hash-based named vectors (`id_to_ref`, `idx_lookup`), causing massive overhead.  
- `compute_neighbor_stats()` itself is relatively light; the aggregation on 5 variables only runs on a list of length equal to rows and is dominated by earlier neighbor resolution.

Thus, the main bottleneck is key-based neighbor retrieval done per-row in `build_neighbor_lookup`, not `do.call(rbind)`.

---

### **Correct Optimization Strategy**
- Avoid repeated string creation (`paste()`) and named lookups inside a per-row `lapply`.
- Precompute direct integer indices instead of string keys.
- Represent neighbor relationships as integer indices, so we can process them with vectorized operations or apply using pre-built mapping.
- For computing stats, avoid building large intermediate lists and instead use matrix operations or `vapply` which is faster and memory-friendly.

Given constraints (Random Forest model preserved, same estimand, same data), focus on:
- Building a compact integer neighbor index list once.
- Performing neighbor aggregates in a vectorized/`vapply` fashion instead of full `lapply` + bind.

---

### **Optimized Working Code**

```r
# Optimized build_neighbor_lookup
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Map cell IDs to row positions directly
  id_to_row <- match(id_order, data$id)
  
  # Split data by year for quick access
  rows_by_year <- split(seq_len(nrow(data)), data$year)
  
  lapply(seq_len(nrow(data)), function(i) {
    # For each row, get its id and year
    cell_id <- data$id[i]
    year    <- data$year[i]
    
    # Get rook neighbors by their index in id_order
    ref_idx    <- match(cell_id, id_order)
    neigh_ids  <- neighbors[[ref_idx]]
    
    # Get row indices for neighbors in the same year using pre-split
    if (length(neigh_ids) == 0) return(integer(0))
    
    neigh_id_vals <- id_order[neigh_ids]
    year_rows <- rows_by_year[[as.character(year)]]
    year_ids  <- data$id[year_rows]
    
    year_rows[match(neigh_id_vals, year_ids, nomatch = 0)]
  })
}

# Optimized compute_neighbor_stats
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  vapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, numeric(3))
  # Transpose result for rbind-like output
  t(.)
}

# Example pipeline usage
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste0(var_name, c("_max", "_min", "_mean"))
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Works**
- Eliminates redundant string operations and repeated named lookup maps.
- Uses `match` and pre-splitting by year for direct integer access.
- `vapply` with fixed-length output instead of large lists + `rbind`.
- Complexity is dominated by neighbor size, not by row count, achieving drastic time reduction (from 86+ hours to a scale suitable for a laptop).

**Expected outcome:** Similar estimand, same trained model preserved, massive speedup with memory efficiency.
END_RESPONSE>>>
