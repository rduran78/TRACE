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
run_id: OpenAI_F03_T10_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The major bottleneck is the neighbor feature computation step feeding into Random Forest inference. Specifically:  
- `build_neighbor_lookup` constructs a large list of 6.46M elements, each storing neighbor indices. This causes extreme memory overhead and object copying.  
- `compute_neighbor_stats` repeatedly iterates over this list for every variable, leading to multiple full passes (5x over 6.46M rows), creating heavy loop overhead.  
- Using `lapply` and `do.call(rbind, ...)` results in large intermediate objects and expensive concatenation.  
- Prediction itself is fast for Random Forest in R when applied in batch via `predict()`. The delay mostly comes from inefficient preprocessing.  

**Optimization Strategy**  
1. **Precompute neighbor feature stats in a vectorized/data.table approach, not per-row loops.**  
2. Restructure neighbor relationships into an edge list, join for aggregation (max/min/mean) via `data.table` group operations on `year`.  
3. Avoid storing giant lists; work with numeric keys and hashing in columns for performance.  
4. Keep Random Forest inference batched: use `predict(model, newdata)` on the complete 6.46M dataset instead of looping predictions.  
5. Use `data.table` for fast joins and aggregations in memory-efficient manner.  

---

### **Working R Code (Optimized Implementation)**

```r
library(data.table)
library(randomForest)

# Assume: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique is the neighbor list for spatial IDs (vector or nb object)
# id_order: vector of all ids in consistent order
# rf_model: pre-trained Random Forest object

# Convert cell_data to data.table for speed
setDT(cell_data)

# Build edge list of neighbors: (id -> neighbor_id)
edges <- rbindlist(lapply(seq_along(id_order), function(i) {
  neighs <- rook_neighbors_unique[[i]]
  if (length(neighs) > 0) {
    return(data.table(id = id_order[i], neighbor_id = id_order[neighs]))
  } else {
    return(NULL)
  }
}))

# Expand by year: join cell-years for id and neighbor_id
years <- unique(cell_data$year)

# Repeat edges for all years and merge with cell_data
edges_expanded <- edges[, .(year = years), by = .(id, neighbor_id)]
edges_expanded <- merge(edges_expanded, cell_data[, .(neighbor_id = id, year, ntl, ec, pop_density, def, usd_est_n2)], 
                        by = c("neighbor_id", "year"), all.x = TRUE)

# Compute neighbor stats for each (id, year)
neighbor_stats <- edges_expanded[, .(
  ntl_max = max(ntl, na.rm = TRUE),
  ntl_min = min(ntl, na.rm = TRUE),
  ntl_mean = mean(ntl, na.rm = TRUE),
  ec_max = max(ec, na.rm = TRUE),
  ec_min = min(ec, na.rm = TRUE),
  ec_mean = mean(ec, na.rm = TRUE),
  pop_max = max(pop_density, na.rm = TRUE),
  pop_min = min(pop_density, na.rm = TRUE),
  pop_mean = mean(pop_density, na.rm = TRUE),
  def_max = max(def, na.rm = TRUE),
  def_min = min(def, na.rm = TRUE),
  def_mean = mean(def, na.rm = TRUE),
  usd_max = max(usd_est_n2, na.rm = TRUE),
  usd_min = min(usd_est_n2, na.rm = TRUE),
  usd_mean = mean(usd_est_n2, na.rm = TRUE)
), by = .(id, year)]

# Merge back to cell_data
cell_data <- merge(cell_data, neighbor_stats, by = c("id", "year"), all.x = TRUE)

# Now perform Random Forest prediction in one batch:
predictions <- predict(rf_model, newdata = cell_data)

```

---

### **Why this is Faster**
- **Single pass over edges per year, aggregated with data.table**, instead of 6 passes of lapply on millions of rows.
- Avoid huge lists; everything handled as tables and joins.
- Prediction is vectorized and runs in seconds to minutes on millions of rows, assuming RF model is large but fixed.
- Memory stays within feasible limits with `data.table` grouping and merges instead of object proliferation.

Estimated speed: reduces preprocessing from 86+ hours to **well under 1 hour** on a 16GB laptop while preserving all modeling integrity.
END_RESPONSE>>>
