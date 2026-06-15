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
run_id: OpenAI_F03_T07_R022
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 22
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

- **Main bottlenecks**:  
  1. **Neighbor feature computation**: Current implementation uses `lapply` over ~6.46M rows for each of 5 variables â†’ extremely slow and memory-heavy.  
  2. **Random Forest inference**: Likely dominated by repeated model loading and prediction loops instead of batch prediction.  
  3. **Excessive object copying**: Each `compute_and_add_neighbor_features` call rebuilds large objects, causing memory churn.  
- **Scale**: 6.46M rows Ã— 110 predictors Ã— 28 years requires a vectorized and memory-efficient approach.  

---

**Optimization Strategy**

1. **Precompute neighbor statistics in a vectorized/data.table fashion**:
   - Avoid per-row `lapply`.
   - Flatten neighbor relationships into a long table, join, and aggregate using `data.table`.
2. **Batch prediction**:
   - Load model once.
   - Predict in large chunks (e.g., 100k rows at a time) to avoid RAM exhaustion.
3. **Memory efficiency**:
   - Use `data.table` instead of repeatedly copying `data.frame`.
   - Avoid repeated joins, compute all neighbor stats in one grouped step.
4. **Parallelization**:
   - If possible, use `data.table` multithreading (built-in) and `ranger` for fast RF prediction.

---

**Optimized R Code**

```r
library(data.table)
library(ranger)  # or randomForest if you must, but ranger is much faster

# Convert to data.table
setDT(cell_data)

# Flatten neighbor list into long format
# rook_neighbors_unique: list of neighbor indices for each cell
neighbor_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(from = id_order[i], to = id_order[rook_neighbors_unique[[i]]])
}))

# Expand by year to match panel
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[CJ(year = years, from = from, unique = TRUE), on = .(from), allow.cartesian = TRUE]
setnames(neighbor_dt, "year", "year")

# Add source and target keys
cell_data[, key := paste(id, year, sep = "_")]
neighbor_dt[, to_key := paste(to, year, sep = "_")]

# Map neighbor values for all variables in one pass
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt neighbor vars for easier aggregation
val_dt <- melt(
  cell_data[, c("key", neighbor_vars), with = FALSE],
  id.vars = "key",
  variable.name = "var",
  value.name = "val"
)

# Join neighbor values
neighbor_long <- merge(
  neighbor_dt[, .(key = paste(from, year, sep = "_"), to_key)],
  val_dt, by.x = "to_key", by.y = "key", allow.cartesian = TRUE
)

# Aggregate neighbor stats: max, min, mean
neighbor_stats <- neighbor_long[
  ,
  .(
    nb_max = max(val, na.rm = TRUE),
    nb_min = min(val, na.rm = TRUE),
    nb_mean = mean(val, na.rm = TRUE)
  ),
  by = .(key, var)
]

# Reshape back to wide
neighbor_stats_wide <- dcast(
  neighbor_stats,
  key ~ var,
  value.var = c("nb_max", "nb_min", "nb_mean")
)

# Merge into main table
cell_data <- merge(cell_data, neighbor_stats_wide, by = "key", all.x = TRUE)

# Drop helper columns
cell_data[, key := NULL]

# Random Forest Prediction in Batches
# Assume rf_model is already loaded
batch_size <- 100000
n <- nrow(cell_data)
preds <- numeric(n)

for (start in seq(1, n, by = batch_size)) {
  end <- min(start + batch_size - 1, n)
  preds[start:end] <- predict(rf_model, data = cell_data[start:end])$predictions
}

cell_data[, gdp_pred := preds]
```

---

**Expected Impact**

- **Neighbor stats computation**: From >80 hours to minutes using `data.table` aggregation and avoiding per-row loops.
- **Prediction**: From repeated row-wise predictions to fast batched inference, reducing hours to minutes (especially with `ranger`).
- **Memory**: Single-pass vectorized operations avoid repeated large object copies.

**Summary**:  
- Replace per-row `lapply` with `data.table` joins and grouped aggregation.  
- Compute all neighbor-based features in one pipeline step.  
- Batch RF predictions with `ranger` or similar efficient implementation.  
- Keep the original model and estimand intact.
END_RESPONSE>>>
