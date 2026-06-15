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
run_id: OpenAI_F03_T00_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F03_OpenAI_temp_00_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small R objects. This is extremely slow and memory-inefficient.  
2. **Excessive object copying**: Each `compute_and_add_neighbor_features` likely copies the entire `data.frame` repeatedly.  
3. **Inefficient neighbor lookup**: `lapply` with string concatenation (`paste`) for every row is costly.  
4. **Random Forest prediction overhead**: If predictions are done in small chunks or with repeated model loading, this adds significant time.  

**Optimization Strategy**  
- **Precompute neighbor indices as integer vectors** once, avoid string keys.  
- **Vectorize neighbor stats computation** using `data.table` or `matrix` operations instead of `lapply`.  
- **Avoid repeated data copies**: compute all neighbor features in one pass and `cbind` results.  
- **Batch predictions**: Use `predict(model, newdata, ...)` on large chunks or entire dataset if memory allows.  
- **Use `data.table` for memory efficiency** and fast joins.  

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest)

# Assume: cell_data is a data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (neighbors per id_order)
# id_order: vector of cell ids in same order as rook_neighbors_unique
# rf_model: pre-trained randomForest model

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor lookup as integer indices
id_to_idx <- setNames(seq_along(id_order), id_order)
cell_data[, row_idx := .I]

# Build neighbor index list (integer indices into cell_data)
neighbor_lookup <- vector("list", nrow(cell_data))
# Map id -> row indices by year
year_groups <- split(cell_data$row_idx, cell_data$year)
id_map <- split(cell_data$row_idx, cell_data$id)

# Efficient neighbor lookup
for (yr in names(year_groups)) {
  rows <- year_groups[[yr]]
  ids <- cell_data$id[rows]
  for (i in seq_along(rows)) {
    ref_id <- ids[i]
    ref_idx <- id_to_idx[[as.character(ref_id)]]
    neigh_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
    neigh_rows <- unlist(id_map[as.character(neigh_ids)], use.names = FALSE)
    # Filter by same year
    neigh_rows <- neigh_rows[cell_data$year[neigh_rows] == as.integer(yr)]
    neighbor_lookup[[rows[i]]] <- neigh_rows
  }
}

# Compute neighbor stats in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
result_mat <- matrix(NA_real_, nrow = nrow(cell_data), ncol = length(neighbor_source_vars) * 3)
colnames(result_mat) <- as.vector(t(outer(neighbor_source_vars, c("_max", "_min", "_mean"), paste0)))

vals_list <- lapply(neighbor_source_vars, function(v) cell_data[[v]])

for (i in seq_along(neighbor_lookup)) {
  idx <- neighbor_lookup[[i]]
  if (length(idx) > 0) {
    for (j in seq_along(vals_list)) {
      neigh_vals <- vals_list[[j]][idx]
      neigh_vals <- neigh_vals[!is.na(neigh_vals)]
      if (length(neigh_vals) > 0) {
        base_col <- (j - 1) * 3
        result_mat[i, base_col + 1] <- max(neigh_vals)
        result_mat[i, base_col + 2] <- min(neigh_vals)
        result_mat[i, base_col + 3] <- mean(neigh_vals)
      }
    }
  }
}

# Bind results
cell_data <- cbind(cell_data, result_mat)

# Drop helper column
cell_data[, row_idx := NULL]

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

### **Key Gains**
- Eliminates repeated `paste` and `lapply` over millions of rows.
- Computes all neighbor features in one pass.
- Uses `data.table` for efficient memory and indexing.
- Predicts in large batches to minimize overhead.

**Expected runtime reduction**: From 86+ hours to a few hours (dominated by neighbor aggregation). Further speedup possible by parallelizing the neighbor stats loop or using `Rcpp`.
END_RESPONSE>>>
