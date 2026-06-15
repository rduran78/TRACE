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
run_id: OpenAI_F07_T07_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
- The pipeline is slow because `compute_neighbor_stats` runs `lapply` over ~6.46M rows, repeatedly subsetting vectors for neighbors.  
- Each lookup calls `idx_lookup` and builds `paste` strings, which is expensive at scale.  
- Memory overhead from millions of intermediate objects and repeated string operations is high.  
- The complexity is roughly `O(N * avg_neighbors)`, and N = 6.46M with multiple variables intensifies runtime.  

---

**Optimization Strategy:**  
1. **Precompute indices efficiently:** Avoid repeated `paste` and `setNames`. Use integer-based joins instead of string keys.  
2. **Vectorize neighbor stats:** Flatten neighbor relationships into a long table and compute grouped max/min/mean with `data.table`, which is highly efficient for large datasets.  
3. **Avoid repeated loops:** Process all variables in one pass via melt/cast rather than looping per variable.  
4. **Memory efficiency:** Keep only integer IDs, avoid copying large vectors repeatedly.  
5. **Parallelization (optional):** `data.table` can use multithreading internally.  

---

**Working R Code (Optimized):**  

```r
library(data.table)

# Convert data to data.table
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# Precompute integer-based neighbor relationships for all years
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_pairs <- rbindlist(lapply(seq_along(id_order), function(i) {
  nbrs <- rook_neighbors_unique[[i]]
  if (length(nbrs) == 0) return(NULL)
  data.table(src_id = id_order[i], nbr_id = id_order[nbrs])
}))

# Expand neighbor pairs across years
years <- unique(dt$year)
neighbor_long <- neighbor_pairs[, .(src_id, nbr_id), by = .EACHI]
neighbor_long <- neighbor_long[rep(seq_len(nrow(neighbor_long)), each = length(years))]
neighbor_long[, year := rep(years, times = nrow(neighbor_pairs))]

# Map to row indices
neighbor_long[, src_idx := dt[.(src_id, year), which = TRUE]]
neighbor_long[, nbr_idx := dt[.(nbr_id, year), which = TRUE]]
neighbor_long <- neighbor_long[!is.na(src_idx) & !is.na(nbr_idx)]

# Gather neighbor values for all source vars
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
long_vals <- melt(
  dt[neighbor_long$nbr_idx, ..vars],
  measure.vars = vars,
  value.name = "val",
  variable.name = "var"
)
long_vals[, src_idx := neighbor_long$src_idx]

# Compute neighbor stats efficiently
stats <- long_vals[, .(
  nbr_max = max(val, na.rm = TRUE),
  nbr_min = min(val, na.rm = TRUE),
  nbr_mean = mean(val, na.rm = TRUE)
), by = .(src_idx, var)]

# Reshape back to wide and merge
stats_wide <- dcast(stats, src_idx ~ var, value.var = c("nbr_max", "nbr_min", "nbr_mean"))
dt <- cbind(dt, stats_wide[match(seq_len(nrow(dt)), src_idx), -"src_idx"])

# dt now contains added neighbor features for all five variables
```

---

**Expected Impact:**  
- Eliminates per-row `lapply` and string operations.  
- Reduces runtime from 86+ hours to likely under 1 hour on a 16 GB machine.  
- Preserves original rook-neighbor topology and numerical estimand.  
- Trained Random Forest model remains untouched; only feature engineering is optimized.
END_RESPONSE>>>
