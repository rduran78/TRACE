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
run_id: OpenAI_F03_T10_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The main performance bottleneck arises from:  
- **Repeated `lapply` loops with millions of iterations** during neighbor feature computation (extremely slow in R).
- Building and applying `neighbor_lookup` for 6.46M rows using R-level loops with heavy object copying.
- Inefficient repeated index lookups and rbind calls.
- Random Forest inference is likely also slow because predictions are being done on large data chunks serially.

---

### **Optimization Strategy**
1. **Precompute neighbor lookups at the `id` level**, not per row. Avoid duplication across years since neighbors don't change by year.
2. **Vectorize neighbor statistics computation:**  
   - Use `data.table` or `dplyr` for grouped joins.
   - Compute for all years in one pass, exploiting neighbor relationships once.
3. **Reduce memory pressure:**  
   - Reuse intermediate objects and avoid `rbind` in loops.
4. **Batch predictions or use `predict` on large chunks**, minimizing repeated object marshalling.
5. **Optionally parallelize** both feature computation and prediction.

---

### **Optimized Workflow in R**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor pairs (id-level)
# rook_neighbors_unique: list of neighbors by id index
neighbor_pairs <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(id = id_order[i],
             neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

# Expand neighbor pairs across all years by joining to cell_data
years <- unique(cell_data$year)
neighbor_pairs_full <- neighbor_pairs[, .(id, neighbor_id)][, .(year = years), by = .(id, neighbor_id)]

# Join neighbor data values
setkey(cell_data, id, year)
setkey(neighbor_pairs_full, neighbor_id, year)

# Function to compute and merge neighbor stats for one variable
compute_neighbor_stats_fast <- function(var_name) {
  neighbor_vals <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  np <- neighbor_pairs_full[neighbor_vals, on = .(neighbor_id, year)]
  # Now np has id, neighbor_id, year, val
  stats <- np[!is.na(val),
              .(max_val = max(val, na.rm = TRUE),
                min_val = min(val, na.rm = TRUE),
                mean_val = mean(val, na.rm = TRUE)),
              by = .(id, year)]
  # Merge back
  setnames(stats, c("id", "year", paste0(var_name, c("_nbr_max","_nbr_min","_nbr_mean"))))
  cell_data[stats, on = .(id, year), `:=`(
    paste0(var_name, "_nbr_max") = i.get(paste0(var_name, "_nbr_max")),
    paste0(var_name, "_nbr_min") = i.get(paste0(var_name, "_nbr_min")),
    paste0(var_name, "_nbr_mean") = i.get(paste0(var_name, "_nbr_mean"))
  )]
}

# Apply for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in neighbor_source_vars) {
  compute_neighbor_stats_fast(v)
}

# Prediction: batch for memory efficiency
library(randomForest) # Assuming model is rf_model
batch_size <- 500000
n <- nrow(cell_data)
pred <- numeric(n)
for (i in seq(1, n, by = batch_size)) {
  end <- min(i + batch_size - 1, n)
  pred[i:end] <- predict(rf_model, newdata = cell_data[i:end])
}
cell_data[, pred_gdp := pred]
```

---

### **Performance Gains**
- Neighbor computation reduces from per-cell-year loops to vectorized joins (~minutes instead of hours).
- Random Forest predictions handled in large chunks minimize overhead (RAM fits in 16 GB).
- Preserves original Random Forest model and numerical estimand. 

**Expected runtime:** Hours â†’ Tens of minutes depending on disk speed and CPU cores.
END_RESPONSE>>>
