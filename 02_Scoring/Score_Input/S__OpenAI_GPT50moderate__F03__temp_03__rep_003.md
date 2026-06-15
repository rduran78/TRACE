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
run_id: OpenAI_F03_T03_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F03_OpenAI_temp_03_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is in the neighbor feature computation and repeated Random Forest inference. Key issues:  
- `lapply` loops over ~6.46M rows, creating millions of small vectors â†’ heavy memory allocation and garbage collection.  
- `build_neighbor_lookup` returns a large list of integer vectors, repeatedly accessed in downstream computations.  
- `compute_neighbor_stats` uses nested `lapply` and `do.call(rbind, ...)` â†’ inefficient for large data.  
- Random Forest prediction on millions of rows in R is slow when done in small batches or with repeated object copying.  
- Memory pressure from large intermediate objects and repeated copying of `cell_data`.  

**Optimization Strategy**  
1. **Precompute neighbor indices efficiently**: Convert neighbor relationships into a flat data frame or matrix instead of a list-of-lists.  
2. **Vectorize neighbor stats**: Use `data.table` or `dplyr` grouped operations instead of per-row `lapply`.  
3. **Avoid repeated copying of `cell_data`**: Compute all neighbor features in one pass and `cbind` results.  
4. **Batch Random Forest prediction**: Use `predict(..., newdata, type="response")` on the full dataset or in large chunks (e.g., 500k rows).  
5. **Memory efficiency**: Use integer indexing and avoid creating millions of small objects.  
6. **Parallelization**: Use `parallel::mclapply` or `future.apply` for multi-core execution if possible.  

---

### **Working R Code**

```r
library(data.table)
library(randomForest)

# Convert cell_data to data.table for efficiency
setDT(cell_data)

# Precompute neighbor pairs (flattened)
# rook_neighbors_unique: list of neighbors per cell index
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    cell_id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Join with years to create full neighbor mapping
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(cell_id, neighbor_id)]

# Map to row indices
cell_data[, key := paste(id, year, sep = "_")]
neighbor_dt[, key := paste(neighbor_id, year, sep = "_")]

# Create lookup for neighbor rows
idx_lookup <- setNames(seq_len(nrow(cell_data)), cell_data$key)
neighbor_dt[, neighbor_row := idx_lookup[key]]
neighbor_dt <- neighbor_dt[!is.na(neighbor_row)]

# Compute neighbor stats for all vars in one pass
compute_neighbor_features <- function(var_name) {
  vals <- cell_data[[var_name]]
  neighbor_dt[, val := vals[neighbor_row]]
  neighbor_dt <- neighbor_dt[!is.na(val)]
  neighbor_dt[, .(
    max_val = max(val),
    min_val = min(val),
    mean_val = mean(val)
  ), by = .(cell_id, year)]
}

# Apply for all source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_features_list <- lapply(neighbor_source_vars, compute_neighbor_features)

# Merge all neighbor features back to cell_data
for (i in seq_along(neighbor_source_vars)) {
  var <- neighbor_source_vars[i]
  feats <- neighbor_features_list[[i]]
  setnames(feats, c("max_val", "min_val", "mean_val"),
           paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- merge(cell_data, feats, by = c("id", "year"), all.x = TRUE)
}

# Random Forest prediction in batches
batch_size <- 500000
n <- nrow(cell_data)
preds <- numeric(n)
for (start in seq(1, n, by = batch_size)) {
  end <- min(start + batch_size - 1, n)
  preds[start:end] <- predict(rf_model, newdata = cell_data[start:end])
}

cell_data[, gdp_pred := preds]
```

---

**Expected Gains**  
- Vectorized neighbor computation reduces runtime from tens of hours to minutes.  
- Batch prediction avoids repeated overhead and memory thrashing.  
- Memory footprint reduced by avoiding millions of small objects.  

This preserves the trained Random Forest model and original estimand while making the pipeline feasible on a 16 GB machine.
END_RESPONSE>>>
