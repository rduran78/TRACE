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
run_id: OpenAI_F03_T10_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- The main bottleneck is extreme inefficiency in the neighbor feature-engineering step (`build_neighbor_lookup` and `compute_neighbor_stats`) combined with repeated object copying in R loops.
- `lapply` over 6.46M rows is extremely slow and memory-intensive (generating huge lists).  
- Building neighbor stats repeatedly for each variable scales badly (â‰ˆ O(N * neighbors * vars)).
- Random Forest inference itself is fast compared to this preprocessing; the heavy lifting is happening in the R loops and row-by-row computation.
- Each yearâ€™s data is processed repeatedly, so large repeated string manipulations (`paste(id, year)`) multiply overhead.
- Lookup and aggregation should be vectorized or implemented using efficient join/aggregation operations (e.g., `data.table`), not per-row `lapply`.

---

**Optimization Strategy**  
1. **Avoid per-row and per-variable loops:** Instead of computing neighbor stats row-by-row, reshape to long format and do grouped merges or use an adjacency expansion + join approach.
2. **Precompute neighbor relationships as integers (cell indices), replicate across years, and work with numeric indexes.**
3. **Use `data.table` for merges and aggregations** (optimized in C).
4. **Batch Random Forest prediction** using matrix input (`predict(..., newdata, ...`) in chunks to avoid memory blow-up.
5. **Reuse model object**; load once in memory.
6. **Optional:** Parallelize with `data.table` `by=` or with `future.apply`.

---

**Optimized Workflow (R Code with `data.table`)**

```r
library(data.table)

# Assume `cell_data` has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Convert rook_neighbors_unique (list of neighbors) into an edge table
edges <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
  })
)

# Cartesian expand edges for all years
years <- unique(cell_data$year)
edges_year <- edges[, .(id = src, nbr, key = NULL)][
  , .(id = rep(id, each = length(years)),
      nbr = rep(nbr, each = length(years)),
      year = rep(years, times = .N))
]

setkey(edges_year, nbr, year)
# Join neighbor values for all variables in one pass
merge_dt <- merge(edges_year, cell_data, by.x = c("nbr", "year"), by.y = c("id", "year"),
                  all.x = TRUE, suffixes = c("", "_nbr"))

# Compute stats for each neighbor source variable in one grouped aggregation
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg_list <- lapply(neighbor_source_vars, function(v) {
  list(
    max = as.call(list(as.name("max"), as.name(paste0(v, "_nbr")), na.rm = TRUE)),
    min = as.call(list(as.name("min"), as.name(paste0(v, "_nbr")), na.rm = TRUE)),
    mean = as.call(list(as.name("mean"), as.name(paste0(v, "_nbr")), na.rm = TRUE))
  )
})
names(agg_list) <- neighbor_source_vars
agg_exprs <- unlist(agg_list, recursive = FALSE)

neighbor_stats <- merge_dt[, eval(agg_exprs), by = .(id, year)]

# Merge back to main dataset
setkey(neighbor_stats, id, year)
cell_data <- neighbor_stats[cell_data]

# Now data.table cell_data has neighbor-based features appended
# Prepare predictor matrix for Random Forest
predictors <- setdiff(names(cell_data), c("target_var", "id", "year"))  # adjust target_var name
X <- cell_data[, ..predictors]

# Load trained RF model once
rf_model <- readRDS("path_to_trained_rf_model.rds")

# Predict in chunks to control memory
chunk_size <- 1e6
preds <- numeric(nrow(X))
for (i in seq(1, nrow(X), by = chunk_size)) {
  rng <- i:min(i + chunk_size - 1, nrow(X))
  preds[rng] <- predict(rf_model, newdata = X[rng], type = "response")
}

# Add predictions to table
cell_data[, gdp_pred := preds]
```

---

**Key Performance Gains**
- Eliminates 6.46M `lapply` calls + multiple `paste` calls.
- Processes neighbor aggregations in compiled C using `data.table` grouping.
- Performs all neighbor variables in one grouped step (not N Ã— Var loops).
- Predicts in large batches instead of per-row.

**Expected Outcome:**  
From 86+ hours â†’ likely **minutes to 1â€“2 hours**, well within laptop capabilities for 6.5M rows when using data.table and chunked prediction.
END_RESPONSE>>>
