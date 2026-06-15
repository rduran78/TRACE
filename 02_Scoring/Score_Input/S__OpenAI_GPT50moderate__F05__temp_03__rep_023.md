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
run_id: OpenAI_F05_T03_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string-key construction (`paste(...)`) and lookup (`idx_lookup[...]`) inside the `lapply` loop is a **local inefficiency**, but the real issue is **algorithmic**:  
- For 6.46M rows, `build_neighbor_lookup` iterates over every row and repeatedly constructs neighbor keys and performs named lookups.  
- This results in **O(N Ã— avg_neighbors)** string concatenations and hash lookups, which is extremely costly for millions of rows.  
- The inefficiency compounds because the neighbor structure is **static across years**, yet the code rebuilds keys per row-year.  

**Optimization Strategy**  
- Precompute a **numeric index matrix** for neighbors instead of string-based lookups.  
- Exploit the fact that the neighbor graph is constant across years:  
  - For each cell, store its neighbor indices (by cell ID, not by string).  
  - Then, for each year, compute offsets to map cell-year rows to neighbor-year rows using vectorized arithmetic.  
- Avoid repeated `paste` and `setNames` entirely.  
- Use `matrix` or `list` of integer vectors for neighbor indices.  

**Algorithmic Reformulation**  
Let:  
- `n_cells = length(id_order)`  
- `n_years = length(unique(data$year))`  
- Rows are ordered by `(id, year)` so row index = `(cell_index - 1) * n_years + year_index`.  

Then:  
- Precompute `neighbor_ids` as a list of integer vectors (cell-level).  
- For each row, compute neighbor row indices as `neighbor_cell_index * n_years + year_index`.  

This reduces complexity from **string hashing per row** to **simple integer arithmetic**.

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Assumes data is sorted by id, then year
  years <- sort(unique(data$year))
  n_years <- length(years)
  n_cells <- length(id_order)
  
  # Map id -> position in id_order
  id_to_pos <- match(data$id[seq(1, nrow(data), by = n_years)], id_order)
  
  # Precompute neighbor cell indices
  neighbor_ids <- lapply(neighbors, function(nb) id_to_pos[nb])
  
  # Precompute year index for each row
  year_index <- match(data$year, years)
  
  # Compute neighbor lookup as list of integer vectors
  row_ids <- seq_len(nrow(data))
  cell_index <- ((row_ids - 1) %/% n_years) + 1
  
  lapply(row_ids, function(i) {
    ci <- cell_index[i]
    yi <- year_index[i]
    nb_cells <- neighbor_ids[[ci]]
    if (length(nb_cells) == 0) return(integer(0))
    # Convert to row indices
    ((nb_cells - 1) * n_years) + yi
  })
}
```

**Compute neighbor stats (unchanged):**

```r
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
```

---

### **Why This Works**
- Eliminates string concatenation and hash lookups (major bottleneck).  
- Uses integer math and precomputed mappings.  
- Memory footprint: neighbor_lookup is a list of integer vectors (same as before, but computed 50â€“100Ã— faster).  

**Expected Speedup:**  
- From 86+ hours to a few hours or less, depending on I/O and downstream steps.  
- Complexity: `O(N Ã— avg_neighbors)` remains, but with negligible per-iteration overhead.  

**Preserves:**  
- Original estimand  
- Trained Random Forest model  
- Panel structure and neighbor relationships  

This is the recommended approach for large panel neighbor feature construction.
END_RESPONSE>>>
