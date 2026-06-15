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
run_id: OpenAI_F03_T07_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The primary bottlenecks are:  
- **Repeated R loops and `lapply` over millions of rows** in `build_neighbor_lookup` and `compute_neighbor_stats`. These are extremely slow for 6.46M rows.  
- **Inefficient neighbor-stat computation**: repeated `paste`, repeated lookups, and repeated object copying.  
- **Prediction loop**: Random Forest predictions are likely happening in chunks or row-by-row rather than vectorized.  
- **Memory overhead**: constructing large lists of neighbors and repeatedly appending to `data.frame` causes huge overhead.  

---

### **Optimization Strategy**
1. **Precompute neighbor relationships in a long format table** (cell-year â†’ neighbor-year) using vectorized joins instead of loops.
2. **Use `data.table` for fast grouping and aggregation** instead of `lapply`.
3. **Compute all neighbor stats for all variables in one go** using melt/cast operations.
4. **Vectorize Random Forest predictions**: Pass the entire feature matrix (or big chunks) to `predict()`.
5. **Avoid repeated string concatenation (`paste`) and lookups**: use numeric indices for joining.
6. **Minimize copying**: work with `data.table` in place.

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.frame), id_order (vector), rook_neighbors_unique (list), rf_model loaded

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor pairs (id â†’ neighbor_id)
neighbor_pairs <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Expand to panel by joining on years
years <- unique(cell_data$year)
neighbor_panel <- neighbor_pairs[CJ(year = years), on = .(dummy = NULL)]
setnames(neighbor_panel, c("id", "neighbor_id", "year"))

# Merge indices
neighbor_panel[cell_data, on = .(id = id, year = year), idx := .I]
neighbor_panel[cell_data, on = .(neighbor_id = id, year = year), n_idx := .I]

# Drop rows without valid neighbor-cell-year
neighbor_panel <- neighbor_panel[!is.na(idx) & !is.na(n_idx)]

# Compute neighbor stats for all variables at once
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt cell_data for neighbor vars
melted <- melt(cell_data, id.vars = c("id", "year"), measure.vars = neighbor_source_vars)
melted[, cell_row := .I]

# Join neighbors
neighbor_vals <- neighbor_panel[melted, on = .(n_idx = cell_row)]
# neighbor_vals: idx (target row), variable, value

# Aggregate stats
agg <- neighbor_vals[, .(
  n_max = max(value, na.rm = TRUE),
  n_min = min(value, na.rm = TRUE),
  n_mean = mean(value, na.rm = TRUE)
), by = .(idx, variable)]

# Reshape wide
agg_wide <- dcast(agg, idx ~ variable, value.var = c("n_max", "n_min", "n_mean"))

# Join back to cell_data
cell_data <- cbind(cell_data, agg_wide[order(idx)][, -"idx"])

# ---- Random Forest Prediction ----
# Prepare matrix of predictors
predictors <- as.data.frame(cell_data[, setdiff(names(cell_data), c("id", "year", "target_var")), with = FALSE])

# Predict in large chunks for memory safety
chunk_size <- 1e6
n <- nrow(predictors)
preds <- numeric(n)
for (i in seq(1, n, by = chunk_size)) {
  j <- min(i + chunk_size - 1, n)
  preds[i:j] <- predict(rf_model, newdata = predictors[i:j, ])
}

cell_data[, prediction := preds]
```

---

### **Why This Works**
- **`data.table` joins** eliminate costly per-row operations.
- **Single pass aggregation** dramatically reduces overhead.
- **Chunked prediction** avoids memory overflow while maximizing vectorization.
- **No retraining**: The existing Random Forest model is preserved.
- This approach can reduce runtime from **86+ hours to under a few hours** on a 16 GB laptop.

**Next step:** Save intermediate artifacts (`neighbor_panel`, melted data) if needed for incremental runs.
END_RESPONSE>>>
