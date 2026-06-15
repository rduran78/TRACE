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
run_id: OpenAI_F03_T00_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F03_OpenAI_temp_00_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small vectors and lists. This is extremely slow and memory-heavy.  
2. **Excessive object copying**: Each `compute_and_add_neighbor_features` likely copies `cell_data` repeatedly.  
3. **Inefficient neighbor lookup**: Current approach builds a full list of neighbors for every row, even though neighbors are static across years.  
4. **Random Forest inference**: If predictions are done in a loop (row-by-row), this is catastrophic. `predict()` should be vectorized on the full data or large chunks.  
5. **Memory pressure**: 6.46M rows Ã— 110+ variables is large; repeated intermediate objects worsen this.  

---

**Optimization Strategy**  
- **Precompute neighbor relationships once per cell** (not per cell-year).  
- **Use vectorized joins instead of nested `lapply`**: Reshape data and compute neighbor stats via `data.table` or `dplyr`.  
- **Chunked prediction**: Use `predict(rf_model, newdata, type="response")` on large batches, not per row.  
- **Avoid repeated copies**: Compute all neighbor features in one grouped operation.  
- **Memory efficiency**: Use `data.table` for in-place updates.  

---

**Optimized Workflow**  
1. Convert `cell_data` to `data.table`.  
2. Expand neighbor relationships across years via a join.  
3. Compute max, min, mean in a single grouped aggregation.  
4. Merge back to main table.  
5. Predict in chunks.  

---

### **Working R Code**

```r
library(data.table)
library(randomForest)

# Assume: cell_data (id, year, predictors), rook_neighbors_unique (list of neighbors per id)
setDT(cell_data)

# 1. Build neighbor table once
neighbor_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(id = i, neighbor_id = rook_neighbors_unique[[i]])
}))

# 2. Expand across years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id = rep(id, length(years)),
                                neighbor_id = rep(neighbor_id, length(years)),
                                year = rep(years, each = .N)), by = .(id, neighbor_id)]
# Drop extra grouping
neighbor_dt <- neighbor_dt[, .(id, neighbor_id, year)]

# 3. Melt source vars for aggregation
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Join neighbor values
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)
neighbor_vals <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]

# 4. Compute stats for each var in one pass
agg_list <- lapply(neighbor_source_vars, function(v) {
  neighbor_vals[, .(
    max = max(get(v), na.rm = TRUE),
    min = min(get(v), na.rm = TRUE),
    mean = mean(get(v), na.rm = TRUE)
  ), by = .(id, year)]
})

# Combine results
for (i in seq_along(neighbor_source_vars)) {
  var <- neighbor_source_vars[i]
  stats <- agg_list[[i]]
  setnames(stats, c("id", "year", paste0(var, "_nbr_max"), paste0(var, "_nbr_min"), paste0(var, "_nbr_mean")))
  cell_data <- stats[cell_data, on = .(id, year)]
}

# 5. Random Forest prediction in chunks
chunk_size <- 500000
n <- nrow(cell_data)
preds <- numeric(n)

for (i in seq(1, n, by = chunk_size)) {
  idx <- i:min(i + chunk_size - 1, n)
  preds[idx] <- predict(rf_model, newdata = cell_data[idx], type = "response")
}

cell_data[, prediction := preds]
```

---

**Why this is faster**  
- Eliminates per-row `lapply` â†’ replaced with vectorized joins and grouped aggregation.  
- Avoids repeated copying of `cell_data`.  
- Uses `data.table` for memory efficiency and speed.  
- Predicts in large chunks instead of row-by-row.  

**Expected improvement**: From 86+ hours to a few hours (or less) on a 16 GB laptop.
END_RESPONSE>>>
