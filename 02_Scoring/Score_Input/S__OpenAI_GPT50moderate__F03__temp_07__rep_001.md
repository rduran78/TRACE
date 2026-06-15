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
run_id: OpenAI_F03_T07_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck lies in:  
1. **Repeated `lapply` and object copying** over 6.46M rows for neighbor stats. This is highly inefficient in R due to memory allocations.  
2. **Building neighbor lookup repeatedly** (list of indices per row). For large panels, this consumes excessive memory and slows computation.  
3. **Random Forest inference**: `predict()` on millions of rows in small chunks repeatedly can be slow; loading the model multiple times is worse.  
4. **Data structure**: Using lists and repeated `paste()` calls creates huge overhead.

---

### **Optimization Strategy**
- **Precompute neighbor indices once** in a flat, vectorized format (e.g., `data.table` or matrix form).  
- **Vectorize neighbor feature computation** using `data.table` joins instead of looping.  
- **Use memory-efficient `data.table` for panel data** instead of base R.  
- **Batch prediction**: Load the Random Forest model once and predict in large chunks (e.g., 500k rows per batch).  
- Avoid repeatedly copying large data frames; update in place.  
- Consider parallelization via `parallel` or `future.apply`.  

---

### **Working R Code**

```r
library(data.table)
library(randomForest)

# Convert to data.table
cell_dt <- as.data.table(cell_data)

# Precompute key for fast joining
cell_dt[, key := paste(id, year, sep = "_")]
setkey(cell_dt, key)

# Flatten neighbors: for each id, store neighbors
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  ref_id <- id_order[i]
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = ref_id,
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand to panel: join with years
years <- unique(cell_dt$year)
neighbor_panel <- neighbor_dt[, .(neighbor_id = neighbor_id), by = .(id)]
neighbor_panel <- neighbor_panel[, .(id = rep(id, length(years)),
                                     year = rep(years, each = .N),
                                     neighbor_id = neighbor_id), by = id]
neighbor_panel[, key := paste(neighbor_id, year, sep = "_")]

# Join to get neighbor values
compute_neighbor_stats_dt <- function(var) {
  vals <- cell_dt[, .(key, val = get(var))]
  merged <- neighbor_panel[vals, on = "key"]
  merged <- merged[!is.na(val)]
  # Aggregate stats
  merged[, .(max = max(val), min = min(val), mean = mean(val)), by = .(id, year)]
}

# Compute all neighbor stats efficiently
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_dt(var_name)
  setnames(stats, c("id", "year", paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))))
  cell_dt[stats, on = .(id, year), `:=`(
    (paste0(var_name, "_nb_max")) = i[[paste0(var_name, "_nb_max")]],
    (paste0(var_name, "_nb_min")) = i[[paste0(var_name, "_nb_min")]],
    (paste0(var_name, "_nb_mean")) = i[[paste0(var_name, "_nb_mean")]]
  )]
}

# Prepare predictors
predictors <- setdiff(names(cell_dt), c("id", "year", "gdp")) # adjust target name

# Load Random Forest model once
rf_model <- readRDS("rf_model.rds")

# Batch prediction
batch_size <- 500000
n <- nrow(cell_dt)
preds <- numeric(n)
for (start in seq(1, n, by = batch_size)) {
  end <- min(start + batch_size - 1, n)
  preds[start:end] <- predict(rf_model, newdata = cell_dt[start:end, ..predictors])
}

cell_dt[, gdp_pred := preds]
```

---

### **Why This Works**
- **Joins instead of nested lapply** reduce complexity from O(n Ã— neighbors) loops to efficient indexed operations.  
- **`data.table` aggregation** avoids repeated copying and uses optimized C backend.  
- **Random Forest predict** in chunks prevents memory overflow while minimizing repeated overhead.  

This approach should reduce runtime from **86+ hours to a few hours** depending on disk speed and CPU.
END_RESPONSE>>>
