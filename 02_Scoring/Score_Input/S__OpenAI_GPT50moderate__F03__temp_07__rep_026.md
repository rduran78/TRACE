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
run_id: OpenAI_F03_T07_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main performance bottlenecks are:  
1. **Repeated model loading and per-row prediction loops** â€” Random Forest prediction should be fully vectorized, not row-wise.  
2. **Neighbor feature computation** â€” `lapply` over 6.46M rows with repeated object copying is extremely slow and memory-heavy.  
3. **Inefficient neighbor lookup** â€” building and reusing large lists repeatedly is expensive.  
4. **Memory pressure** â€” handling ~6.5M rows * 110+ features with R lists can easily exhaust RAM if not vectorized.  

---

### **Optimization Strategy**
- **Load model once** and keep it in memory.  
- **Vectorize neighbor feature computation**:  
  - Flatten neighbor relationships into a long format and compute aggregates with `data.table` or `dplyr`.  
  - Avoid per-row `lapply` and repeated `rbind`.  
- **Use data.table** for all joins and aggregations.  
- **Batch prediction**: `predict(model, newdata)` on the full data or in large chunks, not in row loops.  
- **Precompute neighbor stats in one pass** for all variables.  

---

### **Optimized Workflow**
1. Convert `cell_data` to `data.table`.  
2. Create a lookup table of `(cell_id, neighbor_id)` expanded for all years.  
3. Join neighbor values for all variables in long format, aggregate `max`, `min`, `mean`.  
4. Merge aggregated features back.  
5. Batch-predict with Random Forest.  

---

### **Working R Code**

```r
library(data.table)
library(randomForest)

# Assume: cell_data (id, year, predictors), rook_neighbors_unique (list), rf_model loaded
setDT(cell_data)

# 1. Expand neighbor relationships to all years
id_order <- sort(unique(cell_data$id))
neighbors_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))
# Cartesian join with all years
years <- unique(cell_data$year)
neighbors_dt <- neighbors_dt[, .(year = years), by = .(id, neighbor_id)]

# 2. Merge neighbor values for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt cell_data for fast joins
long_dt <- melt(cell_data, id.vars = c("id", "year"), 
                measure.vars = neighbor_source_vars, 
                variable.name = "var_name", value.name = "value")

# Merge neighbor values
neighbors_long <- merge(neighbors_dt, long_dt, by.x = c("neighbor_id", "year"), 
                        by.y = c("id", "year"), allow.cartesian = TRUE)

# 3. Compute neighbor stats by (id, year, var_name)
agg_stats <- neighbors_long[, .(
  nb_max = max(value, na.rm = TRUE),
  nb_min = min(value, na.rm = TRUE),
  nb_mean = mean(value, na.rm = TRUE)
), by = .(id, year, var_name)]

# Handle all-NA cases (replace Inf/-Inf with NA)
agg_stats[is.infinite(nb_max), nb_max := NA]
agg_stats[is.infinite(nb_min), nb_min := NA]
agg_stats[is.nan(nb_mean), nb_mean := NA]

# 4. Reshape wide and merge back
agg_wide <- dcast(agg_stats, id + year ~ var_name, 
                  value.var = c("nb_max", "nb_min", "nb_mean"))
cell_data <- merge(cell_data, agg_wide, by = c("id", "year"), all.x = TRUE)

# 5. Random Forest prediction in batch
predictions <- predict(rf_model, newdata = cell_data)

# Attach predictions
cell_data[, gdp_pred := predictions]
```

---

### **Key Gains**
- **From 86+ hours â†’ a few hours or less**, dominated by aggregation and join steps.
- **Memory usage optimized** via `data.table` and avoiding giant lists.
- **No model retraining**, preserves estimand.
- **Fully vectorized prediction**: no per-row loops.

---

**Summary:** Replace `lapply` loops with `data.table` joins and grouped aggregations, compute all neighbor stats in one pass, and predict in batch. This approach dramatically reduces runtime and memory overhead while maintaining correctness.
END_RESPONSE>>>
