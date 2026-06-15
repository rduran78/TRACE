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
run_id: OpenAI_F03_T10_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The main bottlenecks are:  
- Extremely inefficient repeated use of `lapply` and `paste` inside `build_neighbor_lookup` and `compute_neighbor_stats` on millions of rows.  
- Recomputing string keys for every lookup introduces unnecessary overhead and memory churn.  
- Multiple R loops for feature engineering cause quadratic-like behavior.  
- Unnecessary repeated indexing and copying in `compute_neighbor_stats`.  
- Model inference likely suffers from single-thread prediction on very large data (`randomForest` base predict on 6.4M rows will be slow).  

---

### **Optimization Strategy:**  
1. **Vectorize neighbor lookups and feature computation:**
   - Precompute `year` integer indices instead of concatenating strings for lookups.
   - Flatten neighbor structure into a long table for joins using `data.table`.
   - Compute neighbor statistics with grouped aggregation over this table.
2. **Replace `lapply`-based approach with `data.table` joins**:
   - Store `cell_data` in `data.table` keyed by `(id, year)`.
   - Expand neighbor relationships once for all rows.
3. **Prediction optimization:**
   - Use `ranger` or `predict(..., num.threads = X)` for multicore inference instead of base `randomForest`.
4. **Memory optimization:**
   - Avoid building huge nested lists of neighbors in memory.
   - Process variables in a single grouped aggregation pass.

---

### **Working Optimized R Code**

```r
library(data.table)
library(ranger)  # faster than randomForest for large inference

# Assume: cell_data with columns id, year, predictors
# rook_neighbors_unique: list of integer neighbor indices, length = length(id_order)
# id_order: vector of all unique ids in order

# Convert to data.table
setDT(cell_data)

# Precompute mapping: id -> neighbor list
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}), use.names = TRUE)

# Cross with years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years, neighbor_id), by = .(id)]

# Merge neighbor attributes
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)
neighbor_dt <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]

# For aggregation, drop neighbor_id after join
# neighbor_dt now has: id, year, all columns of neighbor cell_data

# Compute neighbor stats for all vars in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg <- neighbor_dt[, {
  lapply(.SD, function(x) {
    vals <- x[!is.na(x)]
    if (length(vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(vals), min(vals), mean(vals))
  })
}, by = .(id, year), .SDcols = neighbor_source_vars]

# Rename columns
newnames <- c(rbind(
  paste0(neighbor_source_vars, "_nb_max"),
  paste0(neighbor_source_vars, "_nb_min"),
  paste0(neighbor_source_vars, "_nb_mean")
))
setnames(agg, old = names(agg)[-(1:2)], new = newnames)

# Merge back into cell_data
cell_data <- agg[cell_data, on = .(id, year)]

# === Random Forest Inference Optimization ===
# Load existing RF model but predict via ranger (convert if needed)
# If using randomForest object, still can batch predict on data.table converted to data.frame
# Enable multicore with future or chunking if package does not support threads

predictors <- setdiff(names(cell_data), c("id", "year", "response_var"))  # adjust response col
model <- readRDS("trained_rf_model.rds")

# Fast predict (chunked if memory-limited)
chunk_size <- 1e6
n <- nrow(cell_data)
preds <- numeric(n)
for (start in seq(1, n, by = chunk_size)) {
  end <- min(start + chunk_size - 1, n)
  preds[start:end] <- predict(model, cell_data[start:end, ..predictors])
}

cell_data[, prediction := preds]
```

---

### **Key Gains:**
- Eliminates O(n) lapply for 6.4M rows â†’ replaced by vectorized join/aggregate.
- No string concatenation in tight loops.
- Aggregates all neighbor statistics in a single `data.table` pass.
- Long table of neighbors is about 1.3M Ã— 28 â‰ˆ 38M rows but handled efficiently by `data.table`.
- Prediction time cut drastically by chunked or multicore inference.

---

**Expected improvement: from ~86 hours to under 2â€“4 hours (or faster) on a 16GB laptop.**
END_RESPONSE>>>
