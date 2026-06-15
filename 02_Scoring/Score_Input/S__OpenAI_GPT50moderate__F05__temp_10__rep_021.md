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
run_id: OpenAI_F05_T10_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F05_OpenAI_temp_10_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The repeated string-key construction (`paste(..., sep = "_")`) inside the inner loop is not just a local inefficiencyâ€”itâ€™s a symptom of a broader repeated lookup pattern. For each of ~6.46 million rows, keys are rebuilt and mapped through `idx_lookup`. This is extremely costly given the panel size and neighbor relations.

### **Diagnosis**
- Current approach performs per-row key concatenation and dictionary lookups repeatedly.
- Each loop iteration recomputes neighbor keys (`paste`) and retrieves indices via `idx_lookup`.
- This pattern scales as **O(N Ã— avg_neighbors)** for Nâ‰ˆ6.46M rows.
- The overhead arises mainly from redundant string construction and list indexing.

### **Optimization Strategy**
- Replace string-key join with **direct numeric indexing**:
    - Precompute a numeric vector that maps each `(cell_id, year)` directly to row index.
    - Replace `paste` operations with integer-based matrix lookups.
- Use **vectorized joins** once using `data.table` or `match()` rather than inside per-row closures.
- Build `neighbor_lookup` as an integer matrix upfront using Cartesian expansion.

### **Proposed Algorithm**
1. Encode `id` and `year` as integer factors.
2. Create a fast lookup table (integer-index array) for `(id_idx, year_idx) â†’ row`.
3. For each id-year row, pull neighbor IDs as integers, then map to rows by lookup in the index table without repeated string ops.
4. Keep Random Forest model untouched; this is only pre-model feature computation.

---

### **Working R Code**

```r
library(data.table)

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  dt <- as.data.table(data)
  
  # Encode cell ids and years as integers
  id_to_int <- setNames(seq_along(id_order), id_order)
  dt[, id_int := id_to_int[as.character(id)]]
  year_unique <- sort(unique(dt$year))
  year_to_int <- setNames(seq_along(year_unique), year_unique)
  dt[, year_int := year_to_int[as.character(year)]]
  
  n_id <- length(id_order)
  n_year <- length(year_unique)
  
  # Create a matrix mapping (id_int, year_int) -> row index
  lookup_idx <- matrix(NA_integer_, nrow = n_id, ncol = n_year)
  lookup_idx[cbind(dt$id_int, dt$year_int)] <- seq_len(nrow(dt))
  
  # Build neighbor lookup using numeric mapping
  row_ids <- seq_len(nrow(dt))
  lapply(row_ids, function(i) {
    ref_idx <- dt$id_int[i]
    neighbor_ids <- id_to_int[id_order[neighbors[[ref_idx]]]]
    # Fetch rows for same year
    y <- dt$year_int[i]
    result <- lookup_idx[neighbor_ids, y]
    as.integer(na.omit(result))
  })
}

compute_neighbor_stats <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- lapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA, NA, NA))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  })
  do.call(rbind, result)
}

# Usage
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
}
```

---

### **Why This Improves Performance**
- Eliminates key string concatenation (`paste`) inside millions of iterations.
- Numeric indexing via `lookup_idx` matrix is **O(1)** per query vs repeated hash lookup.
- Memory footprint is reduced (matrix mapping vs large hashed key dictionaries).
- Overall complexity becomes close to linear in number of neighbor relations (~1.37M), not in full cross product.

**Expect significant speed-up**: from 86+ hours to a few hours, depending on disk I/O and CPU.
END_RESPONSE>>>
