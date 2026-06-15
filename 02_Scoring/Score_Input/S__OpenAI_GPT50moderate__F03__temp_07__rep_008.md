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
run_id: OpenAI_F03_T07_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 ### **Diagnosis**
1. **Bottleneck is not Random Forest inference itself**:  
   - Random Forest prediction in R (`predict.randomForest`) is relatively fast for millions of rows if data is in a matrix and model is preloaded.  
   - The main performance issue is **feature engineering**, particularly the repeated use of `lapply` and `do.call(rbind, ...)` across 6.46M rows Ã— 5 variables.

2. **Problems in current workflow**:  
   - `build_neighbor_lookup` returns a list of length *nrow(data)* (~6.46M elements), each itself a vector â†’ **huge memory overhead** and slow indexing.  
   - `compute_neighbor_stats` repeatedly loops over this list for every variable â†’ **O(N Ã— V)** complexity (N = 6.46M, V = 5).  
   - Multiple object copying in `cell_data <- compute_and_add_neighbor_features(...)`.  
   - Neighbor stats computed row-by-row instead of vectorized.

---

### **Optimization Strategy**
- **Precompute neighbor pairs once in long format** (like an edge list): `(row_id, neighbor_id)`.  
- Use **`data.table`** for fast joins and aggregations in memory.  
- Compute all neighbor statistics in one pass using group operations instead of per-row `lapply`.  
- Convert `Random Forest` prediction to matrix input and **predict in batches** if RAM is tight.  
- **Do not store giant lists**; store as integer vectors and aggregated tables.  

---

### **Optimized Workflow**
1. Flatten neighbors into a table: `(cell_row, neighbor_row)`.  
2. Join neighbor variables for all 5 vars in one `data.table`.  
3. Aggregate by `cell_row` to compute `max`, `min`, `mean`.  
4. Bind results back to `cell_data`.  
5. Predict using `predict(rf_model, newdata = as.matrix(...))` in chunks.

---

### **Working R Code**

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.frame), rook_neighbors_unique (list), id_order (vector), rf_model loaded

setDT(cell_data)  # convert to data.table
cell_data[, row_id := .I]  # index for referencing

# --- 1. Build neighbor edge list once ---
id_to_row <- setNames(cell_data$row_id, paste(cell_data$id, cell_data$year, sep = "_"))

edge_list <- rbindlist(lapply(seq_along(id_order), function(i) {
  this_id <- id_order[i]
  neigh_ids <- rook_neighbors_unique[[i]]
  if (length(neigh_ids) == 0) return(NULL)
  # Expand for each year
  years <- unique(cell_data$year)
  CJ(year = years, id = this_id)[, {
    ref_key <- paste(id, year, sep = "_")
    src <- id_to_row[ref_key]
    res <- rbindlist(lapply(neigh_ids, function(nid) {
      neigh_key <- paste(nid, year, sep = "_")
      neighbor_row <- id_to_row[neigh_key]
      if (!is.na(neighbor_row)) data.table(cell_row = src, neighbor_row = neighbor_row) else NULL
    }))
    res
  }]
}), fill = TRUE)

setkey(edge_list, neighbor_row)

# --- 2. Compute neighbor stats for all vars in one pass ---
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_vars) {
  vals <- cell_data[, .(neighbor_row = row_id, val = get(var))]
  stats <- merge(edge_list, vals, by = "neighbor_row")[, .(
    max = max(val, na.rm = TRUE),
    min = min(val, na.rm = TRUE),
    mean = mean(val, na.rm = TRUE)
  ), by = cell_row]

  # Append new columns
  setnames(stats, c("cell_row", paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  cell_data[stats$cell_row, c(paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean"))):=stats[, -1]]
}

# --- 3. Prepare matrix for RF prediction ---
predictors <- setdiff(names(cell_data), c("id", "year", "row_id"))  # exclude non-features
X <- as.matrix(cell_data[, ..predictors])

# --- 4. Predict in batches if needed ---
batch_size <- 500000
pred <- numeric(nrow(X))
for (i in seq(1, nrow(X), by = batch_size)) {
  idx <- i:min(i + batch_size - 1, nrow(X))
  pred[idx] <- predict(rf_model, newdata = X[idx, , drop = FALSE])
}

cell_data[, prediction := pred]
```

---

### **Key Gains**
- Eliminates **6.46M-length neighbor lists** â†’ massive memory savings.
- Aggregates all neighbor stats via `data.table` group operations â†’ reduces complexity from **O(N Ã— V)** to roughly **O(E)** (E â‰ˆ 1.37M edges Ã— years).
- Avoids repeated object copies.
- Predicts in **matrix mode** with optional batching to fit RAM.
- End-to-end runtime should drop **from 86+ hours to under 2â€“3 hours** on a 16GB laptop.

---

âœ… **Preserves trained model and original estimand**  
âœ… **Scales to millions of rows and precomputed neighbor graph**  
âœ… **Uses only base + data.table (no retraining)**
END_RESPONSE>>>
