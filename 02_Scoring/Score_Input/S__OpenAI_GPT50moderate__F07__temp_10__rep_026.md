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
run_id: OpenAI_F07_T10_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

- Current method uses an R `lapply` loop over **6.46 million rows**, repeatedly subsetting vectors, generating character keys (`paste`), and doing list lookups.  
- This is highly inefficient due to:
  - Repeated string construction and hash table lookups (`paste` + named vectors).
  - Function call overhead per row.
  - `lapply` over millions of rows without vectorization or efficient joins.
- Memory pressure: repeated intermediate objects on a **16 GB machine**.
- Complexity approximates **O(N Ã— avg_neighbors)** but with very high constant factors due to R-level overhead.

---

### **Optimization Strategy**
- **Precompute all joins as integer mappings** (avoid string concatenation).
- **Use `data.table` for vectorized grouping operations**, leveraging fast joins.
- Pivot computation from "lookup per row" to "group stats by neighbor relationship".
- Apply **parallelization** (`data.table` + `multithread`) or packages like `collapse` or `dplyr` with `data.table` backend.
- Compute neighbor stats once per variable using **long format** â†’ aggregate â†’ join back.
- Memory-efficient processing in chunks if necessary, but full `data.table` should handle ~6.5M rows.

---

### **Optimized Approach**

1. Flatten neighbor structure into a **long edge table** (`from` â†’ `to` with `year` attached).
2. Join this edge list to the main panel data for both source (`from`) and neighbor variables.
3. Aggregate (max, min, mean) by `from` cell-year over neighbor values.
4. Join aggregated results back to `cell_data`.

---

### **Working R Code**

```r
library(data.table)

# Assume cell_data: data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Flatten rook_neighbors_unique into data.table mapping cell -> neighbors
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(from = id_order[i], to = id_order[rook_neighbors_unique[[i]]])
}))

# Expand across years
years <- unique(cell_data$year)
edge_dt <- edges[ , .(year = years), by = .(from, to)]

# Merge neighbor values
setkey(cell_data, id, year)
setkey(edge_dt, to, year)
edge_dt <- edge_dt[cell_data, on = .(to = id, year), nomatch = 0]

# Compute stats for each source cell-year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  
  # Aggregate max, min, mean of neighbor values for each source (from, year)
  stats_dt <- edge_dt[ , .(
    max_val = max(get(var), na.rm = TRUE),
    min_val = min(get(var), na.rm = TRUE),
    mean_val = mean(get(var), na.rm = TRUE)
  ), by = .(from, year)]
  
  # Handle all-NA neighbors
  stats_dt[!is.finite(max_val), c("max_val","min_val","mean_val") := NA_real_]
  
  # Join back to cell_data
  setkey(stats_dt, from, year)
  cell_data <- stats_dt[cell_data, on = .(from = id, year)]
  
  # Rename joined columns
  setnames(cell_data,
           old = c("max_val","min_val","mean_val"),
           new = paste0(var, c("_nb_max","_nb_min","_nb_mean")))
}

# cell_data now has neighbor stats efficiently computed
```

---

### **Performance Gains**
- Replaces **6.46M lapply iterations** with **vectorized joins and aggregations** in `data.table`.
- Eliminates string concatenation overhead.
- Expected runtime on 16 GB machine: **minutes, not hours** (~10â€“20 min depending on IO and CPU).
- Preserves:
  - Trained Random Forest model.
  - Original estimation logic: rook adjacency, same temporal alignment.
- Memory-safe (edges â‰ˆ 1.37M, expanded across 28 years â‰ˆ 38M rows, which is manageable for `data.table` on 16 GB if processed in chunks if needed).

---

**Final Note:** For datasets just above the memory threshold, chunk on `year` (process one year at a time) and `rbindlist` results.

**Result:** Same numerical estimand, correct rook-neighbor relationships, and efficient computation pipeline suitable for large panel datasets on a standard laptop.
END_RESPONSE>>>
