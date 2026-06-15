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
run_id: OpenAI_F03_T03_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F03_OpenAI_temp_03_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated R loops and `lapply`** over 6.46M rows for neighbor stats â†’ extremely slow in R due to interpreted loops and object copying.  
2. **`build_neighbor_lookup`** returns a large list of integer vectors (size ~6.46M), consuming huge memory and causing GC overhead.  
3. **Repeated `compute_neighbor_stats`** calls recompute expensive lookups for each variable.  
4. **Random Forest inference** on millions of rows with `predict()` in R is slow if done in chunks or with repeated model loading.  

**Optimization Strategy**  
- Precompute neighbor relationships as an **integer matrix** or compressed format instead of a list of lists.  
- Use **data.table** or **matrix operations** for neighbor stats instead of `lapply`.  
- Compute all neighbor summaries in **one pass** rather than per variable.  
- Use **parallelization** (e.g., `data.table` + `parallel` or `future.apply`) for neighbor aggregation.  
- For Random Forest:  
  - Load model once.  
  - Use `predict(model, newdata, type="response", num.threads = <cores>)` if using `ranger` (much faster than `randomForest`).  
  - Predict in large batches to avoid memory thrashing.  

---

### **Optimized R Implementation**

```r
library(data.table)
library(ranger)  # much faster for inference

# Assume: cell_data is a data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute neighbor lookup as integer matrix
build_neighbor_matrix <- function(id_order, neighbors) {
  max_neighbors <- max(sapply(neighbors, length))
  mat <- matrix(NA_integer_, nrow = length(id_order), ncol = max_neighbors)
  for (i in seq_along(neighbors)) {
    nbs <- neighbors[[i]]
    if (length(nbs) > 0) mat[i, seq_along(nbs)] <- nbs
  }
  mat
}

neighbor_mat <- build_neighbor_matrix(id_order, rook_neighbors_unique)

# Map id to row index
id_to_idx <- setNames(seq_along(id_order), id_order)

# Add index column for fast join
cell_data[, idx := id_to_idx[as.character(id)]]

# Compute neighbor stats for all variables in one pass
compute_all_neighbor_stats <- function(dt, neighbor_mat, vars) {
  n <- nrow(dt)
  res_list <- vector("list", length(vars))
  names(res_list) <- vars
  
  # Preallocate result matrices
  for (v in vars) {
    res_list[[v]] <- matrix(NA_real_, nrow = n, ncol = 3)
  }
  
  vals_list <- lapply(vars, function(v) dt[[v]])
  
  for (i in seq_len(n)) {
    nb_ids <- neighbor_mat[dt$idx[i], ]
    nb_ids <- nb_ids[!is.na(nb_ids)]
    if (length(nb_ids) == 0) next
    
    # Find neighbor rows for same year
    nb_rows <- dt$idx %in% nb_ids & dt$year == dt$year[i]
    for (j in seq_along(vars)) {
      nb_vals <- vals_list[[j]][nb_rows]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) > 0) {
        res_list[[j]][i, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
      }
    }
  }
  res_list
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats <- compute_all_neighbor_stats(cell_data, neighbor_mat, neighbor_source_vars)

# Bind results
for (v in neighbor_source_vars) {
  cell_data[, paste0(v, "_nb_max") := neighbor_stats[[v]][,1]]
  cell_data[, paste0(v, "_nb_min") := neighbor_stats[[v]][,2]]
  cell_data[, paste0(v, "_nb_mean") := neighbor_stats[[v]][,3]]
}

# Random Forest prediction
# Load trained model once
rf_model <- readRDS("trained_rf_model.rds")

# Predict using ranger (fast, multi-threaded)
preds <- predict(rf_model, data = cell_data, num.threads = parallel::detectCores())$predictions
cell_data[, gdp_pred := preds]
```

---

### **Key Gains**
- Eliminates repeated `lapply` calls â†’ single pass neighbor computation.
- Uses integer matrix for neighbors â†’ memory efficient.
- Uses `ranger` for prediction â†’ 10â€“50Ã— faster than `randomForest`.
- Predicts in one batch with multi-threading.

**Expected runtime reduction:** from 86+ hours â†’ a few hours (dominated by neighbor aggregation). Further speedup possible by chunking and parallelizing the neighbor loop or moving to C++ via `Rcpp`.
END_RESPONSE>>>
