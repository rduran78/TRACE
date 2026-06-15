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
run_id: OpenAI_F03_T03_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F03_OpenAI_temp_03_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated expensive neighbor computations**: `build_neighbor_lookup` and `compute_neighbor_stats` are pure R loops over ~6.46M rows, creating millions of small vectors and lists. This causes heavy memory allocation and garbage collection.  
2. **Inefficient repeated feature computation**: For each of 5 variables, `compute_neighbor_stats` loops again over all rows.  
3. **Prediction inefficiency**: Random Forest inference in R (`randomForest` or `ranger`) is usually fast, but if predictions are done in small chunks or with repeated model loading, it becomes slow.  
4. **Memory pressure**: Storing large lists of indices and intermediate objects in R is costly on a 16GB machine.  

**Optimization Strategy**  
- **Vectorize neighbor feature computation**: Instead of looping row-by-row, precompute neighbor relationships in a long format and aggregate with `data.table`.  
- **Precompute neighbor stats for all variables in one pass**.  
- **Use `ranger` for fast prediction** (if not already). Predict in large chunks (e.g., 1M rows per batch).  
- **Avoid repeated model loading**: Load the model once.  
- **Minimize object copying**: Use `data.table` for in-place joins and aggregations.  

---

### **Optimized Workflow**

1. Convert `cell_data` to `data.table`.  
2. Expand neighbor relationships into a long table: `(cell_id, year, neighbor_id)`.  
3. Join neighbor values for all variables, compute `max`, `min`, `mean` in one grouped aggregation per variable.  
4. Merge back to `cell_data`.  
5. Predict with `ranger` in large batches.

---

### **Working R Code**

```r
library(data.table)
library(ranger)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2, and predictors
setDT(cell_data)

# Expand neighbor relationships into long format
# rook_neighbors_unique: list of integer vectors, length = length(id_order)
neighbor_dt <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      id = id_order[i],
      neighbor_id = id_order[rook_neighbors_unique[[i]]]
    )
  })
)

# Cross with years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# Join neighbor values
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)
neighbor_dt <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), 
                           nomatch = 0, allow.cartesian = TRUE]

# Now neighbor_dt has: id, neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2
# Compute stats by (id, year)
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
agg_exprs <- lapply(vars, function(v) {
  list(
    as.name(paste0(v, "_nb_max")) := max(get(v), na.rm = TRUE),
    as.name(paste0(v, "_nb_min")) := min(get(v), na.rm = TRUE),
    as.name(paste0(v, "_nb_mean")) := mean(get(v), na.rm = TRUE)
  )
})

neighbor_stats <- neighbor_dt[, eval(as.call(c(quote(.()), agg_exprs))), by = .(id, year)]

# Merge back to cell_data
cell_data <- neighbor_stats[cell_data, on = .(id, year)]

# Prepare predictors
predictors <- setdiff(names(cell_data), c("id", "year", "target_var")) # adjust target_var name

# Load trained Random Forest model (assume it's a ranger object)
rf_model <- readRDS("trained_rf_model.rds")

# Predict in batches
batch_size <- 1e6
n <- nrow(cell_data)
preds <- numeric(n)

for (i in seq(1, n, by = batch_size)) {
  idx <- i:min(i + batch_size - 1, n)
  preds[idx] <- predict(rf_model, data = cell_data[idx, ..predictors])$predictions
}

cell_data[, prediction := preds]
```

---

**Why this is faster**  
- Eliminates millions of small R loops by using `data.table` joins and grouped aggregation.  
- Computes all neighbor stats in one pass.  
- Uses efficient batch prediction with `ranger`.  
- Avoids repeated model loading and unnecessary object copies.  

**Expected improvement**: From 86+ hours to a few hours (depending on disk I/O and CPU).
END_RESPONSE>>>
