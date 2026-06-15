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
run_id: OpenAI_F11_T07_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F11_OpenAI_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that the bottleneck is Random Forest inference is incorrect. Loading a trained model and calling `predict()` on 6.46M rows is relatively fast compared to the heavy computation in the neighbor feature engineering step. The code indicates that `build_neighbor_lookup()` and especially `compute_neighbor_stats()` dominate runtime due to repeated `lapply()` over millions of cell-year rows and multiple passes for 5 variables. Each pass iterates over a large list and performs vector operations inefficiently in R.

**True Bottleneck:**  
Neighbor feature computation is the main bottleneck because:
- `neighbor_lookup` has length equal to the number of rows (6.46M).
- For each row, `compute_neighbor_stats()` subsets and computes summary stats repeatedly.
- Entire process repeats for 5 variables â†’ O(6.46M Ã— 5) operations in R interpreted loops.

---

### **Optimization Strategy**
1. **Precompute neighbor indices at the cell level (not per row-year)** and reuse.
2. **Vectorize computations** using `data.table` or matrix joins instead of row-wise `lapply()`.
3. **Compute all statistics in a single grouped aggregation** rather than iterating per variable per row.
4. Avoid redundant string manipulations and repeated lookups.

---

### **Optimized Approach**
- Expand panel data efficiently with `data.table`.
- Melt to long format for neighbor variables, join to neighbor mapping, compute stats with `by`.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor table at cell level
neighbor_dt <- data.table(
  from_id = rep(id_order, lengths(rook_neighbors_unique)),
  to_id   = unlist(rook_neighbors_unique)
)

# Expand neighbor relationships across years
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[CJ(from_id = from_id, year = years, unique = TRUE), on = "from_id"]
neighbor_dt[, to_year := year]

# Join neighbor values for all needed variables at once
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt cell_data to long format for efficient join
melted <- melt(cell_data, id.vars = c("id", "year"), measure.vars = vars,
               variable.name = "var", value.name = "val")

setkey(melted, id, year)
neighbor_long <- neighbor_dt[melted, on = .(to_id = id, to_year = year)]
# neighbor_long now has from_id, year, var, val of neighbors

# Compute stats by (from_id, year, var)
stats_dt <- neighbor_long[!is.na(val),
  .(neighbor_max = max(val), neighbor_min = min(val), neighbor_mean = mean(val)),
  by = .(id = from_id, year, var)
]

# Reshape back to wide format
stats_wide <- dcast(stats_dt, id + year ~ var,
                    value.var = c("neighbor_max", "neighbor_min", "neighbor_mean"))

# Merge back to cell_data
cell_data <- merge(cell_data, stats_wide, by = c("id", "year"), all.x = TRUE)
```

---

### **Expected Performance Gain**
- Eliminates per-row `lapply()`, replacing with vectorized joins and grouped aggregation.
- Processes all variables in one pass.
- Reduces runtime from 86+ hours to (likely) under 1 hour on a 16 GB laptop, depending on I/O.

**Key Point:** Random Forest inference is not the bottleneck; neighbor feature computation is. The above approach preserves the trained RF model and original estimand while massively improving efficiency.
END_RESPONSE>>>
