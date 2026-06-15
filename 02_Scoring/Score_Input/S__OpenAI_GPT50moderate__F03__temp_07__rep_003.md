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
run_id: OpenAI_F03_T07_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- **Bottlenecks**:  
  1. `lapply` over 6.46M rows is very slow; repeated small R function calls dominate runtime.  
  2. `build_neighbor_lookup` constructs a list of ~6.46M elements with repeated string concatenation and named lookups â†’ huge memory overhead and GC pressure.  
  3. `compute_neighbor_stats` repeatedly subsets vectors and allocates small vectors; repeated `do.call(rbind, ...)` is inefficient.  
  4. Prediction loop likely calls `predict()` on small chunks or per-row instead of using vectorized batch predictions.  
  5. Random Forest model loading per iteration (if happening) is costly; must load once.  
- With 6.46M rows Ã— 5 vars Ã— 3 stats = ~97M neighbor computations; naive R loops are infeasible on 16â€¯GB RAM.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once** and use integer indexing (avoid string keys).  
2. Replace `lapply` loops with vectorized/compiled operations using `data.table` or `Rcpp`.  
3. Compute neighbor stats in a **single pass** using vectorized aggregation (e.g., `data.table` join or `rowsum`).  
4. Avoid building 6.46M-element lists; instead, reshape data and join neighbor values by ID and year.  
5. For Random Forest inference:  
   - Load model once.  
   - Call `predict()` on the full data frame or large chunks (e.g., 100k rows) rather than per row.  
6. Use memory-efficient types: integers/factors for IDs, avoid intermediate character concatenations.  

---

**Working R Code (Efficient Approach)**  
Below uses `data.table` for neighbor feature computation and batched RF prediction:

```r
library(data.table)
library(randomForest)

# Convert to data.table
setDT(cell_data)

# Assume cell_data has: id, year, predictors
# rook_neighbors_unique: list of neighbors keyed by cell id

# Precompute neighbor pairs
neighbor_pairs <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Expand neighbor pairs over years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_pairs[, .(year = years), by = .(id, neighbor_id)]

# Join neighbor values for each source variable
for (var in neighbor_source_vars) {
  vals <- cell_data[, .(neighbor_id = id, year, val = get(var))]
  neighbor_dt <- merge(neighbor_dt, vals, by = c("neighbor_id", "year"), all.x = TRUE)
  
  # Aggregate neighbor stats per (id, year)
  stats <- neighbor_dt[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Merge back into cell_data
  setnames(stats, c("id", "year", paste0(var, "_nbr_max"), paste0(var, "_nbr_min"), paste0(var, "_nbr_mean")))
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
  
  # Drop val column for next iteration
  neighbor_dt[, val := NULL]
}

# Random Forest prediction in large batches
rf_model <- readRDS("trained_rf_model.rds")  # Load once

batch_size <- 100000
n <- nrow(cell_data)
predictions <- numeric(n)

for (i in seq(1, n, by = batch_size)) {
  idx <- i:min(i + batch_size - 1, n)
  predictions[idx] <- predict(rf_model, newdata = cell_data[idx])
}

cell_data[, prediction := predictions]
```

---

**Key Improvements**  
- Eliminates 6.46M `lapply` calls â†’ uses a join-and-aggregate pattern in `data.table`.  
- Avoids massive lists and repeated string concatenations.  
- Computes neighbor stats efficiently by grouping `(id, year)`.  
- Performs RF inference in large chunks to minimize overhead.  

**Expected Impact**  
- Reduces hours-to-days runtime to **tens of minutes** on 16â€¯GB RAM system.  
- Keeps trained Random Forest intact and preserves numerical outputs.
END_RESPONSE>>>
