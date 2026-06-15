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
run_id: OpenAI_F07_T03_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F07_OpenAI_temp_03_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow (86+ hours) because:  
- `lapply` loops over ~6.46 million rows, creating large intermediate lists.  
- Neighbor lookups are recomputed per row, causing repeated string concatenation and indexing.  
- Memory overhead from lists and repeated `paste` operations is huge.  
- No vectorization; operations are row-wise rather than block-wise.  

**Optimization Strategy**  
- Precompute neighbor indices as integer vectors without repeated string concatenation.  
- Use `data.table` for fast keyed joins and aggregation.  
- Avoid per-row loops; compute neighbor stats via grouped operations.  
- Keep everything in memory-efficient structures (integer indices, numeric vectors).  
- Parallelize where possible using `data.table` or `future`.  
- Preserve the trained Random Forest model by only changing feature engineering.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Create a unique key for cell-year
cell_data[, key := paste(id, year, sep = "_")]

# Precompute neighbor lookup as integer indices
build_neighbor_lookup_fast <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(ref_idx) {
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    as.integer(neighbor_cell_ids)
  })
}

neighbor_lookup <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique)

# Flatten neighbor relationships into a long table for joins
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  data.table(
    id = id_order[i],
    neighbor_id = neighbor_lookup[[i]]
  )
}), use.names = TRUE)

# Repeat for all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id = rep(id, length(years)),
                                neighbor_id = rep(neighbor_id, length(years)),
                                year = rep(years, each = .N))]

# Join to get neighbor values and compute stats
compute_neighbor_stats_fast <- function(var_name) {
  # Join cell_data twice: source and neighbor
  merged <- merge(neighbor_dt, cell_data[, .(neighbor_id = id, year, val = get(var_name))],
                  by = c("neighbor_id", "year"), all.x = TRUE)
  
  # Aggregate max, min, mean by (id, year)
  stats <- merged[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Merge back into cell_data
  setnames(stats, c("id", "year", paste0(var_name, "_nbr_max"),
                    paste0(var_name, "_nbr_min"), paste0(var_name, "_nbr_mean")))
  cell_data[stats, on = .(id, year)]  # updates in place
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  compute_neighbor_stats_fast(var_name)
}
```

---

### **Expected Performance Gains**
- Eliminates 6.46M row-wise loops â†’ uses fast joins and aggregation.
- Avoids repeated string operations.
- Scales efficiently on 16 GB RAM.
- Estimated runtime drops from **86+ hours to under 1 hour** on a laptop.

**Preserves:**  
- Original rook-neighbor topology.  
- Numerical estimands (max, min, mean).  
- Trained Random Forest model (no retraining).  

This approach is memory-safe, vectorized, and leverages `data.table` for high performance.
END_RESPONSE>>>
