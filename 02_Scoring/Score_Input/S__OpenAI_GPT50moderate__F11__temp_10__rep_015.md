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
run_id: OpenAI_F11_T10_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The main bottleneck is **not Random Forest inference**, but the neighbor-feature computation in the R loop over ~6.46M rows Ã— 5 variables. The functions `build_neighbor_lookup` and especially `compute_neighbor_stats` rely on deeply nested `lapply` calls and repeated vector subsetting, which is extremely slow and memory-inefficient for tens of millions of lookups. Random Forest `predict()` on ~6.5M rows and 110 predictors is large but feasible compared to 86+ hours; the overwhelming cost is in the iterative neighbor-aggregation step.

---

### **Optimization Strategy**
1. **Precompute neighbor index map once** â€“ already done by `build_neighbor_lookup`.
2. **Replace `lapply` loops with vectorized matrix ops** using `data.table` for speed.
3. **Compute all 5 neighbor-based stats (max, min, mean) for all variables at once** using efficient joins instead of looping variables.
4. Use **long format** transformation: stack cell-year rows per variable, join to neighbors, aggregate in `data.table` (fast C backend).

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...

# Expand lookup table into long neighbor pairs
# neighbor_lookup is list of integer vectors, same order as cell_data rows
pairs <- data.table(
  from = rep(seq_along(neighbor_lookup), lengths(neighbor_lookup)),
  to   = unlist(neighbor_lookup)
)

# Add year and neighbor variables: join cell_data[to] onto pairs
pairs[, year := cell_data$year[from]]

# Keep source variables
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt cell_data for fast join
long_vals <- melt(cell_data,
                  id.vars = c("id", "year"),
                  measure.vars = vars,
                  variable.name = "var",
                  value.name = "value")

# Join neighbor values
pairs_long <- merge(
  pairs[, .(from, to, year)],
  long_vals, by.x = c("to", "year"), by.y = c("id", "year"),
  allow.cartesian = TRUE
)

# Aggregate neighbor stats per source row and variable
agg <- pairs_long[!is.na(value),
                  .(nb_max = max(value),
                    nb_min = min(value),
                    nb_mean = mean(value)),
                  by = .(from, var)]

# Reshape wide and add to cell_data
features <- dcast(agg, from ~ var,
                  value.var = c("nb_max", "nb_min", "nb_mean"))

# Bind features back
cell_data <- cbind(cell_data, features[order(from)][, -"from"])
```

---

### **Why Faster?**
- `data.table` applies fast joins and aggregation in C.
- Eliminates 6.4M Ã— 5 nested `lapply` calls and repeated R-level subsetting.
- Memory is controlled via long format without creating multiple large intermediate lists.
- Computation per neighbor only once per variable, heavily optimized.

---

**Bottom line:** Neighbor feature construction dominates runtime; replacing the R-loop with vectorized and parallelized data.table solution reduces hours to minutes. Random Forest inference can remain unchanged.
END_RESPONSE>>>
