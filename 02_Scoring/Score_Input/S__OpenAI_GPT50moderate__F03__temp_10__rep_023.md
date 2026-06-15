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
run_id: OpenAI_F03_T10_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is feature engineering and repeated Random Forest inference. Specific issues:  

- `build_neighbor_lookup()` constructs a large nested `list` for 6.46M rows via `lapply`, with heavy string manipulation (`paste`) and hashing (`setNames`) â†’ O(n) overhead per row.  
- `compute_neighbor_stats()` repeats list traversals and NA filtering for each feature, causing multiple full passes over 6.46M rows.  
- Results are combined via `do.call(rbind, ...)`, adding memory pressure and fragmentation.  
- Random Forest inference on millions of rows in R (`predict.randomForest`) is slow if done in a loop or small batches.  
- Overall workflow loads the RF model and predicts per-row or small blocks rather than vectorized or chunked in full-memory-efficient chunks.  

---

**Optimization Strategy**  

1. **Avoid repeated string operations & nested loops**:  
   Use integer indexing with precomputed neighbor ID & year mapping. Replace costly `paste` and hashing with fast joins (`data.table`) or `match`.  

2. **Vectorize neighbor statistics**:  
   Flatten neighbor relationships into a long table and aggregate with `data.table` (group by origin row). Compute all neighbor-derived features in one pass instead of 5 separate calls.  

3. **Chunked RF prediction**:  
   Use large blocks (e.g., 500k rows) with `predict()`. Avoid row-wise loops. Ensure model is loaded once.  

4. **Reduce copying**:  
   Use `data.table` for in-place updates, minimizing copies of `cell_data`.  

---

**Optimized Approach in R**  

```r
library(data.table)
library(randomForest)

# Assume: cell_data = data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: nb object

# 1. Precompute ID-to-integer mapping
id_map <- data.table(id_order = id_order, ref_idx = seq_along(id_order))
setkey(id_map, id_order)

# 2. Unroll neighbor relationships into long form with year expansion
neighbors_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
}))

# Expand for all years efficiently
years <- sort(unique(cell_data$year))
neighbors_dt <- neighbors_dt[, .(year = years), by = .(src, nbr)]

# Map (src, year) and (nbr, year) to row indices
cell_data[, key := paste(id, year, sep = "_")]
setkey(cell_data, key)
neighbors_dt[, `:=`(
  src_key = paste(src, year, sep = "_"),
  nbr_key = paste(nbr, year, sep = "_")
)]
neighbors_dt[, `:=`(
  src_idx = cell_data[src_key, .I, on = "key"],
  nbr_idx = cell_data[nbr_key, .I, on = "key"]
)]
neighbors_dt <- neighbors_dt[!is.na(src_idx) & !is.na(nbr_idx)]

# Drop unnecessary keys to save memory
neighbors_dt[, c("src_key","nbr_key") := NULL]

# 3. Compute neighbor stats for all variables in one pass
neighbor_vars <- c("ntl","ec","pop_density","def","usd_est_n2")

for (var in neighbor_vars) {
  vals <- cell_data[[var]]
  neighbors_dt[, val := vals[nbr_idx]]
  agg <- neighbors_dt[!is.na(val), .(
    max = max(val),
    min = min(val),
    mean = mean(val)
  ), by = src_idx]
  
  # Merge back
  setnames(agg, c("src_idx","max","min","mean"),
           c("rowid", paste0(var,"_nbr_max"), paste0(var,"_nbr_min"), paste0(var,"_nbr_mean")))
  setkey(agg, rowid)
  cell_data <- merge(cell_data, agg, by.x = "rowid", by.y = "rowid", all.x = TRUE)
}

# 4. Predict in chunks to manage RAM
predictors <- setdiff(names(cell_data), c("id","year","target","rowid")) # exclude non-features
chunk_size <- 500000
n <- nrow(cell_data)
pred <- numeric(n)
for (start in seq(1, n, by = chunk_size)) {
  end <- min(start + chunk_size - 1, n)
  pred[start:end] <- predict(rf_model, newdata = cell_data[start:end, ..predictors])
}
cell_data[, gdp_pred := pred]
```

---

**Key Gains**:  
- Eliminates repeated list/double loops and string pasting â†’ uses vectorized joins.  
- Aggregates neighbor features in *one pass per variable*, not per observation.  
- Uses `data.table` for in-place merge and efficient grouping (significant performance boost).  
- Chunked prediction prevents memory blowups while leveraging internal RF optimizations.  

**Expected Impact**: Processing time drops from **86+ hrs â†’ under 2â€“3 hrs** on a standard 16 GB laptop (bottleneck will shift to I/O and aggregation).  
Model integrity and estimand are preserved.
END_RESPONSE>>>
