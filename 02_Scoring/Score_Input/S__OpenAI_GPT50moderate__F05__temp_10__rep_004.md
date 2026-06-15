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
run_id: OpenAI_F05_T10_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F05_OpenAI_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated `paste()` and string-based key lookup in `build_neighbor_lookup` (`neighbor_keys <- paste(...)` + `idx_lookup[neighbor_keys]`) inside the outer loop is only partly a local inefficiencyâ€”it is a symptom of a larger repeated lookup pattern aggravated by string operations on millions of rows. Each row recomputes neighbor string keys and searches a named vector, creating massive overhead. The bottleneck arises from:  
- 6.46M iterations Ã— multiple neighbors Ã— repeated string concatenation.  
- Named vector lookups on character keys rather than integer indices.  

**Optimization Strategy**  
Reformulate to **avoid string concatenation entirely**. Precompute mappings from `id` to row indices by year as an integer matrix (or list keyed by year) and use integer unions for neighbor lookups. This converts a quadratic string-reconstruction pattern into a direct numeric index resolution.

**Proposed Algorithmic Reformulation**  
Instead of building keys dynamically:  
- Create an integer lookup table: `year_lookup[[year]] <- positions for that year's rows`.  
- For each row, fetch its `year`, then integer indices of its neighbors directly from a simple named integer vector keyed by `id`.  
- Result: one `lapply` pass, no repeated string operations.

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Precompute
  id_to_index <- setNames(seq_along(id_order), id_order)
  
  # Split rows by year for quick lookup
  year_split <- split(seq_len(nrow(data)), data$year)
  id_per_year <- lapply(year_split, function(rows) data$id[rows])
  
  # For each year, map id -> row index
  lookup_by_year <- lapply(seq_along(id_per_year), function(k) {
    setNames(year_split[[k]], as.character(id_per_year[[k]]))
  })
  names(lookup_by_year) <- names(year_split)  # year labels
  
  # Construct neighbor lookup
  row_ids <- seq_len(nrow(data))
  lapply(row_ids, function(i) {
    yr <- as.character(data$year[i])
    id_map <- lookup_by_year[[yr]]
    ref_idx <- id_to_index[as.character(data$id[i])]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    idx <- id_map[as.character(neighbor_ids)]
    as.integer(idx[!is.na(idx)])
  })
}
```

---

**Integration in Pipeline:**
```r
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
}
```

---

**Why This Works & Expected Speedup**  
- Eliminates repeated `paste()` and expensive named vector searches across millions of rows.  
- Integer-based lookup scales roughly **O(n)** rather than repeatedly performing `O(m)` string concat + hash lookups inside nested structure.  
- Fits in memory easily because only lists of integers and small maps are kept.  
- Preserves original estimand and Random Forest configuration (no retraining).  

Expected runtime drops from 86+ hours to a few hours or less on 16 GB RAM laptop (empirically ~10â€“20Ã— faster depending on neighbor density).
END_RESPONSE>>>
