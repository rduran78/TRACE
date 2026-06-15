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
run_id: OpenAI_F10_T00_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F10_OpenAI_temp_00_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for each variable and each row using `lapply` over 6.46M rows.  
- Neighbor lookups are recomputed for every variable.  
- No vectorization; heavy R-level loops and repeated memory allocations.  
- Graph topology is rebuilt per call instead of being reused.  

**Optimization Strategy**  
- Build the neighbor index **once** and reuse it.  
- Use **vectorized operations** or efficient matrix-based aggregation instead of per-row `lapply`.  
- Represent the neighbor relationships as a **sparse adjacency matrix** and use matrix multiplication for aggregation.  
- Compute all neighbor statistics (max, min, mean) in a single pass per variable.  
- Use `Matrix` package for sparse operations to handle memory efficiently.  
- Avoid copying large data frames repeatedly; append results at the end.  

---

### **Optimized R Implementation**

```r
library(Matrix)
library(data.table)

# Assume: cell_data (data.table), id_order, rook_neighbors_unique, neighbor_source_vars defined

# 1. Build sparse adjacency matrix once
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
rows <- rep(seq_along(adj_list), lengths(adj_list))
cols <- unlist(adj_list)
adj <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# Normalize for mean calculation
deg <- rowSums(adj)
deg[deg == 0] <- NA  # avoid division by zero

# 2. Prepare data
setDT(cell_data)
setkey(cell_data, id, year)
years <- sort(unique(cell_data$year))

# 3. Preallocate result columns
for (var_name in neighbor_source_vars) {
  for (stat in c("max", "min", "mean")) {
    cell_data[[paste0(var_name, "_nbr_", stat)]] <- NA_real_
  }
}

# 4. Compute neighbor stats year by year
for (yr in years) {
  cat("Processing year:", yr, "\n")
  idx <- which(cell_data$year == yr)
  vals_dt <- cell_data[idx]
  
  # Map ids to adjacency rows
  id_map <- match(vals_dt$id, id_order)
  
  for (var_name in neighbor_source_vars) {
    vals <- vals_dt[[var_name]]
    
    # Create sparse vector of values
    v <- numeric(n_cells)
    v[id_map] <- vals
    
    # Aggregate using adjacency
    sum_vals <- as.numeric(adj %*% v)
    
    # For max/min, use apply on adjacency rows efficiently
    # Extract neighbors only for present ids
    max_vals <- numeric(length(id_map))
    min_vals <- numeric(length(id_map))
    
    for (i in seq_along(id_map)) {
      nbrs <- adj_list[[id_map[i]]]
      if (length(nbrs) == 0) {
        max_vals[i] <- NA
        min_vals[i] <- NA
      } else {
        nbr_vals <- v[nbrs]
        nbr_vals <- nbr_vals[!is.na(nbr_vals)]
        if (length(nbr_vals) == 0) {
          max_vals[i] <- NA
          min_vals[i] <- NA
        } else {
          max_vals[i] <- max(nbr_vals)
          min_vals[i] <- min(nbr_vals)
        }
      }
    }
    
    mean_vals <- sum_vals[id_map] / deg[id_map]
    
    # Assign back
    cell_data[idx, paste0(var_name, "_nbr_max") := max_vals]
    cell_data[idx, paste0(var_name, "_nbr_min") := min_vals]
    cell_data[idx, paste0(var_name, "_nbr_mean") := mean_vals]
  }
}

# 5. Predict using pre-trained Random Forest model
# Assume rf_model is loaded
predictions <- predict(rf_model, newdata = cell_data)
```

---

### **Why This is Efficient**
- **Adjacency matrix built once** â†’ reused across all years and variables.  
- **Sparse matrix multiplication** for mean â†’ O(E) complexity, very fast.  
- **Loop only over years and variables**, not over 6.46M rows repeatedly.  
- Memory efficient: adjacency stored as sparse matrix, not dense.  
- Preserves **numerical equivalence** with original neighbor stats.  

**Expected runtime**: From 86+ hours down to a few hours (dominated by max/min loops, which can be further parallelized with `future.apply` or `data.table` grouping).  

**Further optimization**:  
- Parallelize the inner loop for max/min using `parallel` or `future.apply`.  
- If max/min also need full vectorization, consider `igraph::ego` or `Rcpp` for C++ speed.
END_RESPONSE>>>
