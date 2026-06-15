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
run_id: OpenAI_F05_T07_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F05_OpenAI_temp_07_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste(id, year, sep = "_")`) and name-based index lookup inside `lapply` is a **local inefficiency**, but the real issue is broader:  
- For **each cell-year row** (~6.46M), the code recomputes character keys and does a hashed name lookup in `idx_lookup`.  
- This is repeated for 5 variables, making the overhead massive.  
- String-based indexing is O(length) per lookup and memory-expensive.  

The fundamental inefficiency is that the algorithm builds neighbor indices repeatedly for every variable instead of **precomputing a full numeric neighbor index matrix once** and reusing it.  
This is an **algorithmic issue**, not just a micro-optimization.

---

### **Optimization Strategy**
1. **Precompute a numeric neighbor index list or matrix** for all rows once (no repeated key concatenation).  
2. Use **integer indices** for lookups instead of string names.  
3. Then, for each variable, directly pull values using these indices.  
4. Avoid growing data frames in loops; instead, compute and `cbind`.  

---

### **Working R Code**

```r
# Precompute neighbor indices as integers
build_neighbor_index <- function(data, id_order, neighbors) {
  # Map id -> reference index
  id_to_ref <- setNames(seq_along(id_order), id_order)
  
  # Precompute mapping: (id, year) -> row index
  # Assumes data sorted by year then id or vice versa
  # Create matrix of row indices: rows = id_order, cols = unique years
  years <- sort(unique(data$year))
  n_ids <- length(id_order)
  n_years <- length(years)
  
  # Build a lookup table: row index for each (id, year)
  row_lookup <- matrix(NA_integer_, nrow = n_ids, ncol = n_years,
                        dimnames = list(id_order, as.character(years)))
  
  for (i in seq_len(nrow(data))) {
    id <- as.character(data$id[i])
    yr <- as.character(data$year[i])
    row_lookup[id, yr] <- i
  }
  
  # Build neighbor index list: one element per row in data
  row_ids <- seq_len(nrow(data))
  neighbor_idx <- vector("list", length = nrow(data))
  
  for (i in row_ids) {
    id_ref <- id_to_ref[as.character(data$id[i])]
    yr_ref <- as.character(data$year[i])
    neighbor_ids <- id_order[neighbors[[id_ref]]]
    # Look up row indices for neighbors in same year
    neighbor_rows <- row_lookup[neighbor_ids, yr_ref]
    neighbor_idx[[i]] <- neighbor_rows[!is.na(neighbor_rows)]
  }
  
  neighbor_idx
}

# Compute neighbor stats using precomputed numeric indices
compute_neighbor_stats_fast <- function(data, neighbor_idx, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_idx), ncol = 3)
  
  for (i in seq_along(neighbor_idx)) {
    idx <- neighbor_idx[[i]]
    if (length(idx) == 0) next
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) next
    result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }
  
  colnames(result) <- paste0(var_name, c("_max_n", "_min_n", "_mean_n"))
  result
}

# Main pipeline
neighbor_idx <- build_neighbor_index(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

all_features <- lapply(neighbor_source_vars, function(var_name) {
  compute_neighbor_stats_fast(cell_data, neighbor_idx, var_name)
})

neighbor_features <- do.call(cbind, all_features)
cell_data <- cbind(cell_data, neighbor_features)
```

---

### **Why This Works**
- **No string concatenation in inner loop** â†’ replaced by integer matrix lookup.  
- **Neighbor index list computed once** â†’ reused across variables.  
- **Memory-efficient**: list of integer vectors + one matrix for lookup.  
- **Time complexity**: O(N * avg_degree) instead of repeated string hashing overhead.  

---

**Expected speedup**: From 86+ hours to a few hours (depends on I/O and CPU), since the biggest bottleneck (string-based lookups repeated millions of times) is eliminated.  

The Random Forest model and numerical estimands are preserved because the feature computation logic is unchanged.
END_RESPONSE>>>
