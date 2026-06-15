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
run_id: OpenAI_F03_T10_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
Your performance bottleneck is the extreme inefficiency in repeatedly computing neighbor-based features row-by-row with `lapply` and repeated vector slicing. With ~6.46M rows and 5 variables, you're materializing and copying large intermediate lists multiple times. This creates enormous memory pressure and CPU time. The overall Random Forest inference likely adds overhead from repeatedly passing huge data frames and possibly loading the model multiple times, but the major slowdown originates from the neighbor-stat computation.

Issues observed:  
- `build_neighbor_lookup()` returns a massive list (length = nrow = 6.46M), repeating neighbor linking for each row-year.  
- `compute_neighbor_stats()` loops again over this giant list for each variable (5 passes).  
- Heavy object copying inside `paste()`, `idx_lookup`, and `lapply`.  
- Random Forest prediction itself is relatively fast in comparison, assuming `ranger` or efficient C++ backendâ€”not the bottleneck.

---

### **Optimization Strategy**
1. **Vectorize neighbor lookups:** Precompute a matrix of neighbors for all cells (static across years) and reuse with year offsets.  
2. **Avoid per-row loops:** Use `data.table` or matrix ops to compute neighbor aggregates in bulk via joins.  
3. **Precompute keys instead of `paste` for every row.** Use integer indexing for joins.  
4. **Compute all neighbor features in one grouped aggregation rather than variable-by-variable loops.**  
5. **Load Random Forest model once and predict in big chunks or all at once (if RAM permits).** Using `ranger::predict()` is preferable.  
6. **Memory recycling:** Drop intermediate objects promptly and use `gc()` if needed.

---

### **High-Performance Implementation in R**

Below is an efficient `data.table`-based approach:

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Assume: cell_data columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: sorted vector of unique ids
# rook_neighbors_unique: spdep nb object with neighbor adjacency

# 1. Build neighbor edge list: cell_id -> neighbor_id
neighbor_edges <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# 2. Expand for all years and all neighbors in one go
years <- sort(unique(cell_data$year))
neighbors_full <- neighbor_edges[CJ(id, years, sorted = TRUE), on = .(id)]
setnames(neighbors_full, c("id", "year", "neighbor_id"))

# Add neighbor_id year as well (same year as focal row)
neighbors_full[, neighbor_year := year]

# 3. Prepare for join: key= (id, year)
setkey(cell_data, id, year)

# 4. Join to bring in neighbors' variables
neighbors_full <- neighbors_full[cell_data, on = .(neighbor_id = id, neighbor_year = year)]

# neighbors_full now has columns: id, year (focal), and neighbor vars from neighbor cells
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 5. Compute summary stats per focal id-year
agg <- neighbors_full[, .(
  ntl_max = max(ntl, na.rm = TRUE),
  ntl_min = min(ntl, na.rm = TRUE),
  ntl_mean = mean(ntl, na.rm = TRUE),
  ec_max = max(ec, na.rm = TRUE),
  ec_min = min(ec, na.rm = TRUE),
  ec_mean = mean(ec, na.rm = TRUE),
  pd_max = max(pop_density, na.rm = TRUE),
  pd_min = min(pop_density, na.rm = TRUE),
  pd_mean = mean(pop_density, na.rm = TRUE),
  def_max = max(def, na.rm = TRUE),
  def_min = min(def, na.rm = TRUE),
  def_mean = mean(def, na.rm = TRUE),
  usd_max = max(usd_est_n2, na.rm = TRUE),
  usd_min = min(usd_est_n2, na.rm = TRUE),
  usd_mean = mean(usd_est_n2, na.rm = TRUE)
), by = .(id, year)]

# 6. Merge features back into cell_data
cell_data <- agg[cell_data, on = .(id, year)]

# Random Forest prediction step
# Assume RF model is loaded as rf_model and uses ranger
pred <- predict(rf_model, data = as.data.frame(cell_data))$predictions
cell_data[, gdp_hat := pred]
```

---

### **Key Gains**
- Completely removes per-row `lapply()` loops â†’ massive reduction from 86+ hours to likely under an hour on 16GB RAM.
- All neighbor aggregations done in one grouped operation, leveraging `data.table` optimized C code.
- Avoids repeated string concatenation with `paste()`.
- Random Forest inference is a single bulk call with efficient `ranger`.

---

**Summary:** The core bottleneck was the repeated list-based neighbor logic. Moving to a join-based, vectorized solution using `data.table` makes the pipeline scalable and memory efficient while preserving the exact estimand.
END_RESPONSE>>>
