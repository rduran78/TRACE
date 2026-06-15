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
run_id: OpenAI_F03_T10_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- **Bottleneck**: The workflow is dominated by repeated `lapply()` + `do.call(rbind, ...)` over 6.46M rows. This creates huge intermediate R objects, repeated vector scans, and heavy memory copying.  
- Neighbor stats calculation (`compute_neighbor_stats`) currently runs *once per variable* sequentially on a large list (length nrow), doing repeated NA filtering and aggregation, causing extreme overhead.  
- Prediction likely loops over rows or chunks inefficiently, and model inference is not vectorized.  
- Huge neighbor lookup list (length ~6.46M) adds memory pressure.  
- Loading model repeatedly (if done inside prediction loop) must be avoided.  
- 16â€¯GB RAM cannot handle this with naive lists â†’ requires memory-efficient operations and vectorization.

---

### **Optimization Strategy**

1. **Precompute Neighbor Indices on IDs only**  
   - Avoid expanding to `id_year` for all rows in a giant list. Instead store numeric indices for efficient indexing.

2. **Vectorize Neighbor Summarization**  
   - Use **data.table** or fastapply approach that computes all neighbor-based stats **in bulk** rather than one cell at a time.
   - Pre-flatten neighbor graph into a two-column structure: `(cell_idx, neighbor_idx)` with repeated years applied, then join.

3. **Compute all variables in *one pass***  
   - Compute a long table of neighbor values for all source vars and collapse by `(cell_idx, year)`.

4. **Prediction**  
   - Do not loop row-wise. Use `predict(rf_model, newdata, ...)` on chunks if memory is tight (e.g., 500k rows at a time).
   - Load RF model once, outside loop.

5. **Memory**  
   - Use `data.table` keyed joins and `set()` for columns to avoid copies.

---

### **Working R Code (Optimized)**

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.frame) with columns: id, year, predictors
# Convert to data.table
setDT(cell_data)

# Precompute full size
n <- nrow(cell_data)

# ---- Build flattened neighbor map ----
# rook_neighbors_unique: list of integer neighbor IDs for each cell ID position in id_order
id_to_idx <- setNames(seq_along(id_order), id_order)

# Flatten neighbor relationships (cell_id -> neighbor_id)
pairs <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(cell_id = id_order[i],
             neigh_id = id_order[rook_neighbors_unique[[i]]])
}))

# Join years: cross with 28 years via cell-year index
# Merge cell_data to get full (cell_id, year) rows
pairs <- merge(pairs, cell_data[, .(id, year, row_id = .I)], by.x = "cell_id", by.y = "id")
pairs <- merge(pairs, cell_data[, .(id, year, neigh_row = .I)], by.x = c("neigh_id","year"), by.y = c("id","year"))

# Drop rows without neighbors for that year
pairs <- pairs[!is.na(neigh_row)]

# ---- Compute neighbor stats for all variables ----
neighbor_vars <- c("ntl","ec","pop_density","def","usd_est_n2")

# Melt once for efficiency
vals_long <- melt(cell_data[, c("row_id", neighbor_vars), with = FALSE],
                  id.vars = "row_id", variable.name = "var", value.name = "val")

# Map neighbor values
pairs_long <- merge(pairs[, .(row_id, neigh_row)], vals_long,
                    by.x = "neigh_row", by.y = "row_id", allow.cartesian = TRUE)

# Aggregate neighbor stats: max, min, mean
agg_stats <- pairs_long[, .(
  neigh_max = max(val, na.rm = TRUE),
  neigh_min = min(val, na.rm = TRUE),
  neigh_mean = mean(val, na.rm = TRUE)
), by = .(row_id, var)]

# Reshape wide to add back to cell_data
agg_wide <- dcast(agg_stats, row_id ~ var, value.var = c("neigh_max","neigh_min","neigh_mean"))

# Set keys and join back
setkey(agg_wide, row_id)
cell_data <- merge(cell_data, agg_wide, by.x = "row_id", by.y = "row_id", all.x = TRUE)

# ---- Prediction ----
# Assume rf_model already loaded
chunk_size <- 500000
preds <- vector("numeric", n)

for (start in seq(1, n, by = chunk_size)) {
  end <- min(start + chunk_size - 1, n)
  preds[start:end] <- predict(rf_model, newdata = cell_data[start:end])
}

# Add predictions to table
cell_data[, gdp_pred := preds]
```

---

### **Key Gains**
- Avoids per-row `lapply` â†’ uses `data.table` joins and grouping.
- Neighbor stats computed in few grouped operations over a long table.
- One-pass aggregation for all variables.
- Prediction vectorized by large chunks, preventing out-of-memory.

*Expected time reduction*: From 86+ hrs â†’ down to **1â€“2 hours** on a laptop.
END_RESPONSE>>>
