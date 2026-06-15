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
run_id: OpenAI_F09_T07_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F09_OpenAI_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The bottleneck is redundant computation: for each of ~6.46M cell-year rows, the current code repeatedly looks up neighbors and computes stats in R loops. This leads to excessive memory allocations and function-call overhead. Building the neighbor lookup once is good, but the computation of neighbor statistics is still done row-wise in R, which is very slow for millions of rows.

**Optimization Strategy:**  
- Precompute a reusable neighbor adjacency table (cell â†’ neighbors) once at the cell level.  
- Expand it across years using a fast join instead of per-row `lapply`.  
- Use `data.table` for vectorized operations and grouping rather than looping.  
- Compute neighbor `max`, `min`, and `mean` in batch for each variable using grouped aggregation.  
- Avoid repeated indexing of large lists; keep everything in flat tables for joins.  

---

### **Optimized R Code**

```r
library(data.table)

# Assume cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Convert to data.table
setDT(cell_data)

# Build adjacency table once at cell level
id_order <- unique(cell_data$id)
neighbor_pairs <- data.table(
  cell = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor = unlist(rook_neighbors_unique)
)

# Expand adjacency across years by joining on year
years <- unique(cell_data$year)
adj_dt <- CJ(year = years, cell = id_order)[
  neighbor_pairs, on = .(cell), allow.cartesian = TRUE
]

# Add neighbor id to join
setnames(adj_dt, "neighbor", "neighbor_cell")

# Join neighbor attributes for all variables
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_cols <- paste0("neighbor_", vars)

# Join attributes from cell_data for neighbor cells
adj_dt[cell_data, on = .(neighbor_cell = id, year), (neighbor_cols) := mget(vars)]

# Compute neighbor stats grouped by cell-year
stat_dt <- adj_dt[, .(
  ntl_max = max(neighbor_ntl, na.rm = TRUE),
  ntl_min = min(neighbor_ntl, na.rm = TRUE),
  ntl_mean = mean(neighbor_ntl, na.rm = TRUE),
  ec_max = max(neighbor_ec, na.rm = TRUE),
  ec_min = min(neighbor_ec, na.rm = TRUE),
  ec_mean = mean(neighbor_ec, na.rm = TRUE),
  pop_density_max = max(neighbor_pop_density, na.rm = TRUE),
  pop_density_min = min(neighbor_pop_density, na.rm = TRUE),
  pop_density_mean = mean(neighbor_pop_density, na.rm = TRUE),
  def_max = max(neighbor_def, na.rm = TRUE),
  def_min = min(neighbor_def, na.rm = TRUE),
  def_mean = mean(neighbor_def, na.rm = TRUE),
  usd_est_n2_max = max(neighbor_usd_est_n2, na.rm = TRUE),
  usd_est_n2_min = min(neighbor_usd_est_n2, na.rm = TRUE),
  usd_est_n2_mean = mean(neighbor_usd_est_n2, na.rm = TRUE)
), by = .(cell, year)]

# Merge stats back to main data
cell_data <- merge(cell_data, stat_dt, by.x = c("id", "year"), by.y = c("cell", "year"), all.x = TRUE)

# Predict with existing Random Forest model (unchanged)
preds <- predict(rf_model, newdata = cell_data)
```

---

**Why this is faster:**  
- Only one neighbor table is built.  
- No per-row `lapply`; everything uses efficient `data.table` joins and aggregations.  
- Operates on ~1.37M neighbor relationships Ã— 28 years (~38M rows) in batch, which is feasible in memory on 16 GB using `data.table`.  
- Reduces estimated time from 86+ hours to under an hour (often minutes) on a modern laptop.  

**Preserves:**  
- Original grid IDs and years.  
- Original estimand and trained Random Forest model.
END_RESPONSE>>>
