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
run_id: OpenAI_F03_T07_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 ### **Diagnosis**
The main bottlenecks are:

1. **Inefficient neighbor feature computation**  
   - `build_neighbor_lookup` and `compute_neighbor_stats` repeatedly create large lists and perform `lapply` loops for millions of rows.
   - Heavy use of `paste`, `setNames`, and repeated lookups creates excessive overhead and memory churn.
   - `do.call(rbind, ...)` on huge lists is extremely slow.

2. **Prediction workflow inefficiency**  
   - Random Forest inference in R (`ranger` or `randomForest`) is fast enough for 6.46M rows if features are precomputed efficiently, but the current preprocessing dominates runtime.
   - Object copying inside the loop for each variable (`compute_and_add_neighbor_features`) adds further overhead.

3. **Memory constraints**  
   - 6.46M Ã— 110 variables â‰ˆ 700M entries â†’ ~5 GB for numeric matrix alone; repeated copying of `data.frame` makes it worse.

---

### **Optimization Strategy**
- **Key idea:** Replace expensive R loops with **vectorized joins/data.table aggregation** and **precompute neighbor relationships once**.
- Use `data.table` for fast keyed operations.
- Represent neighbor relationships as a long table `(cell_id, year, neighbor_id)`, then `merge` to bring neighbor values and compute stats in bulk.
- Eliminate repeated `paste` and list structures.
- Use `ranger::predict()` on a `data.table` or `matrix` after feature computation, avoiding data frame copying.
- If possible, store intermediate features on disk in chunks and predict in chunks to fit in memory.

---

### **Optimized R Code**

```r
library(data.table)
library(ranger)

# Assume cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, etc.
setDT(cell_data)

# Convert rook_neighbors_unique to an edge list
# rook_neighbors_unique: list of neighbors for each id_order element
neighbor_pairs <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Build long table of (cell_id, year, neighbor_id)
years <- unique(cell_data$year)
neighbor_dt <- neighbor_pairs[, .(id = from, neighbor_id = to)]
neighbor_dt <- neighbor_dt[CJ(year = years), on = .(dummy = NULL)]
neighbor_dt[, year := rep(years, each = nrow(neighbor_pairs))]

# Join neighbor values for all variables at once
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Merge neighbor_dt with cell_data to get neighbor values
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)
neighbor_dt <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), 
                            nomatch = 0, allow.cartesian = TRUE]

# neighbor_dt now has columns: id (target), year, neighbor_id, [vars]
# Compute stats by (id, year)
agg_list <- lapply(vars, function(v) {
  list(max = as.name(v), min = as.name(v), mean = as.name(v))
})
names(agg_list) <- vars

neighbor_stats <- neighbor_dt[, {
  res <- list()
  for (v in vars) {
    vals <- get(v)
    res[[paste0(v, "_nbr_max")]]  <- max(vals, na.rm = TRUE)
    res[[paste0(v, "_nbr_min")]]  <- min(vals, na.rm = TRUE)
    res[[paste0(v, "_nbr_mean")]] <- mean(vals, na.rm = TRUE)
  }
  res
}, by = .(id, year)]

# Merge back into cell_data
cell_data <- neighbor_stats[cell_data, on = .(id, year)]

# Convert to matrix for prediction
predictors <- setdiff(names(cell_data), c("id", "year", "gdp"))  # adjust target name
X <- as.matrix(cell_data[, ..predictors])

# Load trained Random Forest model (already available)
# model <- readRDS("rf_model.rds")

# Predict in chunks to save memory
chunk_size <- 1e6
n <- nrow(X)
preds <- numeric(n)
for (i in seq(1, n, by = chunk_size)) {
  idx <- i:min(i+chunk_size-1, n)
  preds[idx] <- predict(model, data = X[idx, , drop = FALSE])$predictions
}

# Add predictions
cell_data[, gdp_pred := preds]
```

---

### **Expected Gains**
- Replaces millions of `lapply` calls with a few highly optimized `data.table` operations.
- Eliminates repeated string concatenations and object copying.
- Handles all neighbor variables in **one pass**.
- Chunked prediction avoids memory overflow.
- Runtime reduction from **86+ hours to a few hours or less**, depending on disk speed and CPU.

---

**Summary:** The critical fix is restructuring the neighbor feature computation into a long-format join + grouped aggregation using `data.table`, then using efficient chunked prediction with the existing Random Forest model.
END_RESPONSE>>>
