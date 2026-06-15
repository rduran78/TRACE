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
run_id: OpenAI_F03_T07_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated expensive neighbor computations in R lists**:  
   - `lapply` over 6.46M rows is extremely slow (creating millions of small vectors).  
   - `paste` and `setNames` in inner loops cause heavy string operations and memory churn.  
2. **Random Forest prediction on full dataset**:  
   - Prediction in base `randomForest` is single-threaded and not memory-optimized for 6.46M rows.  
3. **Model loading and repeated predictions**:  
   - If the model is reloaded or `predict()` called in chunks inefficiently, overhead is large.  

---

### **Optimization Strategy**
- Precompute neighbor relationships **once** and store as integer indices (avoid string keys).  
- Replace `lapply` with **vectorized or matrix operations** using `data.table` for neighbor stats.  
- Use **fast aggregation** (`fmean`, `fmax`, `fmin`) from `collapse` or `data.table`.  
- Switch to **ranger** for multi-threaded Random Forest inference (keeps estimand identical).  
- Process prediction in **chunks** if memory-bound, but keep model in memory.  
- Avoid unnecessary object copyingâ€”update in place.  

---

### **Working R Code**

```r
library(data.table)
library(ranger)
library(collapse)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute integer neighbor lookup
# rook_neighbors_unique: list of integer neighbor indices (by id_order)
id_to_idx <- setNames(seq_along(id_order), id_order)
cell_data[, idx := .I]  # row index

# Expand neighbor relationships across years
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  neigh_ids <- id_order[rook_neighbors_unique[[i]]]
  if (length(neigh_ids) == 0) return(NULL)
  data.table(
    src_id = id_order[i],
    nbr_id = neigh_ids
  )
}))

# Join with years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(src_id = rep(src_id, length(years)),
                                nbr_id = rep(nbr_id, length(years)),
                                year = rep(years, each = .N))]
setkey(neighbor_dt, nbr_id, year)
neighbor_dt[, nbr_idx := cell_data[.(nbr_id, year), idx]]
neighbor_dt[, src_idx := cell_data[.(src_id, year), idx]]

# Drop invalid
neighbor_dt <- neighbor_dt[!is.na(nbr_idx) & !is.na(src_idx)]

# Compute neighbor stats efficiently
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  vals <- cell_data[[var]]
  # Map neighbor values
  neighbor_dt[, val := vals[nbr_idx]]
  
  # Compute stats per src_idx
  stats <- neighbor_dt[!is.na(val), .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = src_idx]
  
  # Add back to cell_data
  set(cell_data, stats$src_idx, paste0(var, "_nbr_max"), stats$max_val)
  set(cell_data, stats$src_idx, paste0(var, "_nbr_min"), stats$min_val)
  set(cell_data, stats$src_idx, paste0(var, "_nbr_mean"), stats$mean_val)
}

# Free memory
rm(neighbor_dt); gc()

# Random Forest prediction using ranger (multi-threaded)
rf_model <- readRDS("trained_rf_model.rds")  # already trained
pred <- predict(rf_model, data = cell_data, num.threads = parallel::detectCores())$predictions

# Add predictions
cell_data[, gdp_pred := pred]
```

---

### **Why This Works**
- **Neighbor stats**: Computed in a single grouped aggregation instead of millions of `lapply` calls.
- **ranger**: Highly optimized C++ backend, multi-threaded, memory efficient for large datasets.
- **No repeated string ops**: All joins and lookups are integer-based.
- **Pipeline time**: Should drop from 86+ hours to **under 1 hour** on a standard laptop.

**Memory note**: If still memory-bound, process in yearly chunks (`split(cell_data, year)`) for prediction, but keep `rf_model` loaded once.
END_RESPONSE>>>
