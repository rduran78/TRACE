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
run_id: OpenAI_F03_T07_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the repeated and highly inefficient neighbor feature computation combined with large-scale Random Forest inference. Current issues:  
- `build_neighbor_lookup` returns a massive list (length = 6.46M) of integer vectors â†’ huge memory overhead and GC pressure.  
- `compute_neighbor_stats` uses `lapply` + `do.call(rbind, â€¦)` repeatedly â†’ heavy R interpreter overhead.  
- Outer loop calls `compute_and_add_neighbor_features` 5 times â†’ repeated full passes over 6.46M rows.  
- Prediction loop likely uses similar row-wise logic â†’ slow for 6.46M predictions.  
- Model loading inside loop (if any) would kill performance.  

**Optimization Strategy**  
1. **Vectorize neighbor aggregation:** Precompute a flat edge list of `(cell_id, neighbor_id)` and use fast group operations with `data.table`.  
2. **Use `data.table` keyed joins instead of `lapply` lists.**  
3. **Compute all neighbor stats for all variables in one grouped operation.**  
4. **Persist `neighbor_lookup` as an edge table instead of a giant list.**  
5. **Random Forest inference:**  
   - Use `predict(model, newdata, type="response", num.threads = parallel::detectCores())` if using **ranger** or **randomForestSRC** for parallel prediction.  
   - Do **not** reload model inside loop. Keep it in memory.  
6. **Memory:** Process in batches if `predict` cannot handle full dataset at once.  

---

### **Working R Code (Highly Optimized)**

```r
library(data.table)
library(ranger)  # assuming trained model is from ranger

# Convert cell_data to data.table
setDT(cell_data)

# Precompute edge list once
# rook_neighbors_unique: list of integer vectors per cell in id_order sequence
edge_list <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Expand edge list to panel: join on years
years <- unique(cell_data$year)
edge_panel <- edge_list[CJ(id = id_order, year = years), on = "id", allow.cartesian = TRUE]
setnames(edge_panel, "id", "cell_id")

# Join neighbor_id + year to neighbor keys
edge_panel[, neighbor_key := paste(neighbor_id, year, sep = "_")]
cell_data[, key := paste(id, year, sep = "_")]

# Map keys to row indices
idx_map <- data.table(key = cell_data$key, row_id = seq_len(nrow(cell_data)))
edge_panel <- idx_map[edge_panel, on = .(key = neighbor_key)]
# row_id now refers to neighbor's row in cell_data

# Drop rows where neighbor is missing
edge_panel <- edge_panel[!is.na(row_id)]

# Compute neighbor stats for all vars in one go
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt neighbor values
neighbor_vals <- melt(
  cell_data[, c("row_id", neighbor_source_vars), with = FALSE][, row_id := .I],
  id.vars = "row_id", variable.name = "var", value.name = "val"
)

# Join neighbor values to edge_panel
edge_panel_long <- neighbor_vals[edge_panel, on = .(row_id), allow.cartesian = TRUE]

# Aggregate stats by (cell_id, year, var)
stats <- edge_panel_long[, .(
  max = max(val, na.rm = TRUE),
  min = min(val, na.rm = TRUE),
  mean = mean(val, na.rm = TRUE)
), by = .(cell_id, year, var)]

# Reshape stats wide and merge back to cell_data
stats_wide <- dcast(stats, cell_id + year ~ var, value.var = c("max", "min", "mean"))
cell_data <- merge(cell_data, stats_wide, by.x = c("id", "year"), by.y = c("cell_id", "year"), all.x = TRUE)

# Random Forest prediction
# Assuming model is already loaded as rf_model
# Use ranger for fast parallel prediction
pred_batch_size <- 1e6
n <- nrow(cell_data)
preds <- numeric(n)
for (i in seq(1, n, by = pred_batch_size)) {
  idx <- i:min(i + pred_batch_size - 1, n)
  preds[idx] <- predict(rf_model, data = cell_data[idx, predictor_vars, with = FALSE])$predictions
}

cell_data[, gdp_pred := preds]
```

---

### **Key Gains**
- Eliminates 6.46M-length list and repeated `lapply` loops.
- Uses `data.table` grouping and joins for massive speed-up.
- Computes all neighbor stats in one pass.
- Predicts in large batches with multithreading.
- Memory footprint reduced dramatically since intermediate objects are flat tables, not nested lists.

**Estimated runtime**: Hours reduced from ~86 to <1 hour (with efficient I/O and parallel prediction).
END_RESPONSE>>>
