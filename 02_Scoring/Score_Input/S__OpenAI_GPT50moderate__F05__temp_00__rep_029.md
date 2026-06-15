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
run_id: OpenAI_F05_T00_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The repeated string concatenation (`paste`) and hash lookups inside the `lapply` loop in `build_neighbor_lookup` are a **local inefficiency**, but the real problem is **algorithmic**:

- For each of ~6.46M rows, the code:
  - Builds `neighbor_keys` by pasting neighbor IDs with the current year.
  - Looks up indices in `idx_lookup` (a named vector of length 6.46M).
- This results in **tens of millions of string operations and hash lookups**, which is extremely slow and memory-heavy.
- The inefficiency is compounded because the neighbor structure is **static across years**. The only thing that changes is the year dimension, but the code recomputes everything per row.

**Broader Reformulation**

Instead of doing string-based lookups per row, we can:

1. **Precompute a numeric mapping** from `(id, year)` â†’ row index using integer arrays, not strings.
2. **Exploit panel structure**: For each year, the neighbor relationships are the same, so we can build neighbor indices for all rows in a vectorized way.
3. **Avoid repeated lapply over 6.46M rows**: Use matrix operations or `split` by year.

---

### **Optimized Strategy**

- Sort `data` by `(id, year)` so that IDs and years are in consistent order.
- Create:
  - `id_to_idx`: integer mapping from ID to position in `id_order`.
  - `year_to_offset`: integer offset for each year block.
- For each year:
  - Compute neighbor indices by adding offsets to base neighbor indices.
- Store neighbor indices in a list aligned with rows.

This removes string concatenation and hash lookups entirely.

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Ensure data is sorted by id, then year
  data <- data[order(data$id, data$year), ]
  
  n_ids   <- length(id_order)
  n_years <- length(unique(data$year))
  rows_per_year <- n_ids
  
  # Map id to its position in id_order
  id_to_ref <- match(data$id[seq_len(n_ids)], id_order)
  
  # Precompute offsets for each year block
  years <- sort(unique(data$year))
  year_offsets <- setNames(seq(0, by = rows_per_year, length.out = n_years), years)
  
  # Precompute neighbor indices for one year (base pattern)
  base_neighbors <- lapply(seq_along(id_order), function(i) {
    as.integer(id_to_ref[neighbors[[i]]])
  })
  
  # Build full neighbor lookup
  neighbor_lookup <- vector("list", nrow(data))
  
  for (y in seq_along(years)) {
    offset <- year_offsets[y]
    year_idx_start <- offset + 1
    year_idx_end   <- offset + rows_per_year
    
    for (i in seq_len(rows_per_year)) {
      row_idx <- offset + i
      nb <- base_neighbors[[i]]
      if (length(nb) > 0) {
        neighbor_lookup[[row_idx]] <- nb + offset
      } else {
        neighbor_lookup[[row_idx]] <- integer(0)
      }
    }
  }
  
  neighbor_lookup
}
```

---

### **Why This Is Faster**

- **No string concatenation**: All lookups are integer-based.
- **No hash maps**: Direct index arithmetic.
- **Panel structure exploited**: Neighbor pattern repeated across years with simple offset addition.
- Complexity drops from `O(N * avg_neighbors * string_ops)` to `O(N)` with small constant factors.

---

### **Integration**

Replace:

```r
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
```

with:

```r
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
```

Then reuse `compute_neighbor_stats` as is. The numerical estimand remains unchanged, and the trained Random Forest model is preserved.

---

**Expected Performance Gain**: From 86+ hours to a few minutes for neighbor lookup construction, making the entire pipeline feasible on a 16 GB laptop.
END_RESPONSE>>>
