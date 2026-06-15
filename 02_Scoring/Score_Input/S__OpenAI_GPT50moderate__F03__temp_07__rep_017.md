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
run_id: OpenAI_F03_T07_R017
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 17
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 ### **Diagnosis**
1. **Prediction bottleneck**  
   - Random Forest inference over ~6.46M rows can be slow if predictions are done in small chunks or with repeated model loads.
   - `predict()` in `randomForest` or `ranger` can handle large batches but is memory-intensive if not optimized.

2. **Major inefficiency in feature preparation**  
   - `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over 6.46M rows â†’ extremely slow and memory heavy.
   - Multiple `paste()`, `setNames()`, and repeated indexing operations inside loops create unnecessary object copies.
   - Neighbor statistics computed separately for each variable multiplies runtime.

3. **Memory pressure**  
   - 6.46M rows Ã— ~110 variables Ã— repeated intermediate objects â†’ blows up RAM usage on 16GB system.

---

### **Optimization Strategy**
- **Avoid per-row loops**. Use **vectorized joins** or **data.table** for neighbor aggregation.
- **Precompute neighbor relationships once** as a long table (`from`, `to`).
- **Compute all neighbor stats in one grouped operation** using `data.table` or `dplyr`.
- **Batch predictions**: Use `predict()` in chunks of e.g., 500k rows to fit in memory.
- Keep RF model in memory, do not reload for each batch.

---

### **Fast Approach**
- Convert `cell_data` to `data.table`.
- Expand neighbor relations into a long table, join features, aggregate max/min/mean for each variable in one pass.
- Merge aggregated stats back to main table.
- Predict in large batches with `ranger` (optimized C++ backend).

---

### **Working R Code**

```r
library(data.table)
library(ranger)

# Assume: cell_data (id, year, predictors), id_order, rook_neighbors_unique, rf_model loaded

# Convert to data.table
setDT(cell_data)

# Build neighbor long table: (from_id, to_id)
from_ids <- rep(id_order, lengths(rook_neighbors_unique))
to_ids   <- unlist(rook_neighbors_unique)
neighbor_dt <- data.table(from_id = from_ids, to_id = to_ids)

# Expand across years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id = from_id, neighbor_id = to_id), by = 1:nrow(neighbor_dt)]
neighbor_dt <- neighbor_dt[, .(id = rep(id, length(years)), 
                               neighbor_id = rep(neighbor_id, length(years)), 
                               year = rep(years, each = .N))]

# Join neighbor values
neighbor_dt <- merge(neighbor_dt, cell_data, 
                     by.x = c("neighbor_id", "year"), 
                     by.y = c("id", "year"),
                     all.x = TRUE, suffixes = c("", "_nbr"))

# Aggregate neighbor stats for all source vars in one go
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg_exprs <- lapply(neighbor_source_vars, function(v) {
  list(max = as.name(v), min = as.name(v), mean = as.name(v))
})

agg_cols <- unlist(lapply(neighbor_source_vars, function(v) {
  c(paste0(v, "_nbr_max"), paste0(v, "_nbr_min"), paste0(v, "_nbr_mean"))
}))

neighbor_stats <- neighbor_dt[, 
  c(lapply(.SD, function(x) list(max = max(x, na.rm = TRUE),
                                 min = min(x, na.rm = TRUE),
                                 mean = mean(x, na.rm = TRUE)))),
  .SDcols = neighbor_source_vars,
  by = .(id, year)
]

# Flatten column names
setnames(neighbor_stats, old = names(neighbor_stats)[-(1:2)], new = agg_cols)

# Merge back to main data
cell_data <- merge(cell_data, neighbor_stats, by = c("id", "year"), all.x = TRUE)

# Prediction in batches
batch_size <- 5e5
n <- nrow(cell_data)
preds <- numeric(n)

for (i in seq(1, n, by = batch_size)) {
  idx <- i:min(i + batch_size - 1, n)
  preds[idx] <- predict(rf_model, data = cell_data[idx])$predictions
}

cell_data[, prediction := preds]
```

---

### **Key Gains**
- **No per-row lapply**: replaced with vectorized joins and grouped aggregations.
- **Neighbor stats computed in one pass**, not per variable.
- **Batch prediction** avoids RAM overflow.
- Expected runtime reduction: from **86+ hours â†’ under a few hours** on a 16 GB laptop.

---

**Summary:** Replace nested loops with `data.table` joins and grouped operations, batch predictions, and keep model in memory. This preserves the original estimand and the trained RF model.
END_RESPONSE>>>
