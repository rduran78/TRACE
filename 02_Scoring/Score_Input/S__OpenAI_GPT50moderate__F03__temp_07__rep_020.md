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
run_id: OpenAI_F03_T07_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The major bottlenecks are in the neighbor feature computation and repeated Random Forest predictions across ~6.46M rows. Specific issues:  
1. **`lapply` within `build_neighbor_lookup` and `compute_neighbor_stats`**: pure R loops on millions of elements create heavy overhead.  
2. **Repeated object copying**: Each call to `compute_and_add_neighbor_features` rebuilds large vectors and uses `do.call(rbind, â€¦)` repeatedly, which is expensive.  
3. **Non-vectorized neighbor operations**: Every row recomputes neighbor keys and indices in R lists instead of leveraging efficient joins or matrix operations.  
4. **Memory pressure**: Storing large lists of neighbor indices and intermediate objects strains 16â€¯GB RAM.  
5. **Random Forest inference**: `predict` on millions of rows can be slow in R. If using `randomForest` package, itâ€™s single-threaded and memory-heavy.  

---

**Optimization Strategy**  
- **Precompute neighbor features once in a fully vectorized way**:  
  - Convert neighbor relationships into a long table (edges) and join to compute aggregate stats (`max`, `min`, `mean`) via `data.table`.  
- **Avoid per-row loops**: Replace `lapply` with `data.table` group operations.  
- **Efficient model prediction**:  
  - Use `ranger::predict` (fast C++ backend, multi-threaded) with `num.threads` > 1.  
  - Feed all rows in chunks if memory limits hit.  
- **Memory efficiency**:  
  - Store features in `data.table` to avoid unnecessary copies.  
  - Drop unused columns before prediction.  

---

**Working R Code** (high-performance, vectorized):

```r
library(data.table)
library(ranger)  # For fast Random Forest inference

# Assume: cell_data (with columns id, year, ntl, ec, pop_density, def, usd_est_n2)
# rook_neighbors_unique: list of neighbor ids per cell in id_order
# rf_model: pre-trained ranger model

setDT(cell_data)

# Build neighbor edge table
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
}))

# Expand edges across years
years <- unique(cell_data$year)
edges <- edges[CJ(year = years, src = src), on = .(src), allow.cartesian = TRUE]
setnames(edges, "src", "id")
edges[, nbr_year := year]

# Join target and neighbor values efficiently
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (v in neighbor_vars) {
  # Join neighbor variable
  edges <- merge(edges,
                 cell_data[, .(nbr = id, year, val = get(v))],
                 by.x = c("nbr", "nbr_year"), by.y = c("nbr", "year"),
                 all.x = TRUE)
  
  # Compute aggregates per (id, year)
  agg <- edges[, .(max_v = max(val, na.rm = TRUE),
                   min_v = min(val, na.rm = TRUE),
                   mean_v = mean(val, na.rm = TRUE)),
               by = .(id, year)]
  
  # Replace NaN with NA
  for (col in names(agg)[-(1:2)]) set(agg, which(is.nan(agg[[col]])), col, NA)
  
  # Merge back into main table
  setnames(agg, c("id", "year", paste0(v, "_nbr_max"), paste0(v, "_nbr_min"), paste0(v, "_nbr_mean")))
  cell_data <- merge(cell_data, agg, by = c("id", "year"), all.x = TRUE)
  
  # Drop val column for next iteration
  edges[, val := NULL]
}

# Drop unnecessary columns before prediction to save memory
predictors <- setdiff(names(cell_data), c("id", "year"))

# Fast prediction with ranger (multi-threaded)
preds <- predict(rf_model, data = cell_data[, ..predictors], num.threads = parallel::detectCores())$predictions

# Attach predictions
cell_data[, gdp_pred := preds]
```

---

**Key Gains**  
- **Vectorization**: Eliminates millions of R-level loops.  
- **`data.table` join + group by**: Handles 6.46M rows efficiently.  
- **`ranger::predict`**: Multi-threaded prediction dramatically faster than `randomForest`.  

**Expected Result**: Reduce runtime from 86â€¯h to a few hours or less on a 16â€¯GB laptop.
END_RESPONSE>>>
