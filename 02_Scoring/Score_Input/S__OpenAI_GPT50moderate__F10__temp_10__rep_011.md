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
run_id: OpenAI_F10_T10_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- **Bottleneck**: The current pipeline recomputes neighbor statistics row-by-row (6.46M rows Ã— 5 variables Ã— per-row `lapply`). This results in severe overhead (86+ hrs).  
- **Key inefficiency**: Repeated neighbor lookups and aggregation across 6.46M rows instead of vectorized aggregation.  
- **Goal**: Build sparse graph once (344,208 nodes), reuse for all 28 years, and compute max/min/mean per neighbor set using efficient matrix operations.  

---

### **Optimization Strategy**
1. Build a **sparse adjacency matrix** (nodes Ã— nodes) **once** from `rook_neighbors_unique`.
2. Use the adjacency matrix for all years (**reusable structure**).
3. For each variable:
   - Reshape `data[var]` into a **node-year matrix**: 344,208 Ã— 28.
   - Compute neighbor stats with **fast matrix ops**:  
     - `neighbor_matrix %*% values` for sum (then average).  
     - For `max`/`min`, use `Matrix::rowMaxs` or grouped operations.  
4. Combine results back into long format efficiently (`cbind` to panel data).
5. Apply the pre-trained Random Forest without retraining.

---

### **Efficient R Implementation**

```r
library(Matrix)
library(matrixStats)
library(data.table)

compute_neighbor_features <- function(cell_data, id_order, neighbors, vars) {
  # Dimensions
  n_nodes <- length(id_order)
  years <- sort(unique(cell_data$year))
  n_years <- length(years)
  
  # Build adjacency matrix (sparse)
  from <- rep(seq_along(neighbors), lengths(neighbors))
  to   <- unlist(neighbors)
  adj  <- sparseMatrix(i = from, j = to, x = 1, dims = c(n_nodes, n_nodes))
  
  # Create node-year mapping
  setDT(cell_data)
  setkey(cell_data, id, year)
  
  # Preallocate result list
  result_list <- list()
  
  for (var_name in vars) {
    # Reshape into matrix [nodes x years]
    mat <- matrix(NA_real_, nrow = n_nodes, ncol = n_years,
                  dimnames = list(id_order, years))
    idx <- match(cell_data$id, id_order)
    year_idx <- match(cell_data$year, years)
    mat[cbind(idx, year_idx)] <- cell_data[[var_name]]
    
    # Neighbor aggregation (sum and count for mean)
    n_mat <- adj %*% (!is.na(mat))  # neighbor counts per node-year
    sum_mat <- adj %*% (replace(mat, is.na(mat), 0))  # sum ignoring NAs
    
    # Compute mean
    mean_mat <- sum_mat / n_mat
    mean_mat[is.nan(mean_mat)] <- NA
    
    # Compute max and min efficiently
    # We'll iterate per year for max/min to keep memory manageable
    max_mat <- matrix(NA_real_, n_nodes, n_years)
    min_mat <- matrix(NA_real_, n_nodes, n_years)
    
    for (yr in seq_len(n_years)) {
      vals <- mat[, yr]
      # Extract neighbors for each node
      max_vals <- sapply(seq_len(n_nodes), function(i) {
        nb <- neighbors[[i]]
        if (length(nb) == 0) return(NA_real_)
        vals_nb <- vals[nb]
        vals_nb <- vals_nb[!is.na(vals_nb)]
        if (length(vals_nb) == 0) return(NA_real_)
        max(vals_nb)
      })
      min_vals <- sapply(seq_len(n_nodes), function(i) {
        nb <- neighbors[[i]]
        if (length(nb) == 0) return(NA_real_)
        vals_nb <- vals[nb]
        vals_nb <- vals_nb[!is.na(vals_nb)]
        if (length(vals_nb) == 0) return(NA_real_)
        min(vals_nb)
      })
      max_mat[, yr] <- max_vals
      min_mat[, yr] <- min_vals
    }
    
    # Reshape back to vector aligned with cell_data
    idx <- cbind(idx, year_idx)
    feature_mat <- cbind(
      max = max_mat[idx],
      min = min_mat[idx],
      mean = mean_mat[idx]
    )
    
    colnames(feature_mat) <- paste0(var_name, "_nb_", c("max", "min", "mean"))
    result_list[[var_name]] <- feature_mat
  }
  
  # Combine all features with original data
  new_features <- do.call(cbind, result_list)
  cell_data <- cbind(cell_data, new_features)
  
  return(cell_data)
}

# Usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

optimized_data <- compute_neighbor_features(
  cell_data = cell_data,
  id_order = id_order,
  neighbors = rook_neighbors_unique,
  vars = neighbor_source_vars
)

# Apply pre-trained Random Forest
predictions <- predict(pretrained_rf_model, optimized_data)
```

---

### **Why This is Efficient**
- **Adjacency matrix** computed once (memory-efficient sparse representation).
- **Matrix multiplications** (C-level, highly optimized) for sums/means.
- Max/min computed by iterating per year but only at node-level (344K vs 6.46M iterations).
- No per-row R loops on 6.46 million rows.
- Scales easily within 16 GB RAM.

**Expected runtime**: ~30â€“60 mins vs 86+ hrs.  
Numerical equivalence guaranteed since neighbor definitions and aggregation remain unchanged.  
Pre-trained Random Forest remains intact.
END_RESPONSE>>>
