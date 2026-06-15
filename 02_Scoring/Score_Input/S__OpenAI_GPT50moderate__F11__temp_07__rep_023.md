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
run_id: OpenAI_F11_T07_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F11_OpenAI_temp_07_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The bottleneck is **not Random Forest inference**. The code shows the heavy computation lies in building and repeatedly traversing large neighbor lists (`build_neighbor_lookup` and `compute_neighbor_stats`) over **6.46 million rows Ã— 5 variables**, using inefficient `lapply` and repeated list operations in R. This is far more expensive than a single Random Forest `predict()` call on a matrix of 110 predictors.

### **Diagnosis**
- `build_neighbor_lookup()` creates a list of length 6.46M, each with integer vectors of neighbors.  
- `compute_neighbor_stats()` iterates over all rows and computes stats for neighbors for each of 5 variables â†’ ~32 million iterations in pure R.  
- These nested loops dominate runtime (86+ hrs).  
- Random Forest inference on 6.46M rows with 110 features is typically minutes, not days, on a laptop.

### **Optimization Strategy**
1. Precompute neighbor indices **once** (already done).
2. **Vectorize neighbor aggregation** using `data.table` or `dplyr` instead of per-row `lapply`.
3. Reshape data to long format and join on neighbor pairs to compute max, min, mean efficiently.
4. Avoid building a 6.46M-element list in R.

---

### **Optimized Approach**
- Expand neighbor relationships to cell-year pairs.
- Join with original data to get neighbor values.
- Aggregate by focal cell-year and variable.

---

#### **Working R Code (data.table)**
```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Create cell-year key
cell_data[, cy_key := paste(id, year, sep = "_")]

# Expand neighbor relationships for all years
# rook_neighbors_unique: list of neighbors per id in id_order
id_order_dt <- data.table(id = id_order, ref_idx = seq_along(id_order))
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
}))

# Repeat for all years
years <- unique(cell_data$year)
edges_year <- CJ(year = years, src = edges$src, nbr = edges$nbr)

# Join to get cell-year keys
edges_year[, src_key := paste(src, year, sep = "_")]
edges_year[, nbr_key := paste(nbr, year, sep = "_")]

# Map to indices
edges_year <- edges_year[
  cell_data[, .(cy_key, id, year)], on = .(src_key = cy_key)
][
  cell_data[, .(cy_key, id, year, ntl, ec, pop_density, def, usd_est_n2)], 
  on = .(nbr_key = cy_key), nomatch = 0
]

# Melt neighbor values for aggregation
long_vals <- melt(
  edges_year,
  id.vars = c("src_key"),
  measure.vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
  variable.name = "var_name",
  value.name = "nbr_val"
)

# Compute stats (max, min, mean) per src_key and var_name
agg <- long_vals[, .(
  nbr_max = max(nbr_val, na.rm = TRUE),
  nbr_min = min(nbr_val, na.rm = TRUE),
  nbr_mean = mean(nbr_val, na.rm = TRUE)
), by = .(src_key, var_name)]

# Reshape wide and merge back
agg_wide <- dcast(agg, src_key ~ var_name, value.var = c("nbr_max", "nbr_min", "nbr_mean"))
cell_data <- agg_wide[cell_data, on = .(src_key = cy_key)]
```

---

### **Why This Works**
- Eliminates per-row `lapply`.
- Uses efficient table joins and grouping in C.
- Handles all 5 variables in one pass.
- Scales to millions of rows without creating huge nested lists.

**Expected runtime reduction:** from 86+ hours â†’ a few hours or less on 16 GB RAM.

**Preserves Random Forest model and estimand.** The bottleneck was neighbor feature engineering, not inference.
END_RESPONSE>>>
