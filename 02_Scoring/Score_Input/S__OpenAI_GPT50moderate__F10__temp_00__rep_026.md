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
run_id: OpenAI_F10_T00_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F10_OpenAI_temp_00_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for ~6.46M rows using `lapply` and `do.call(rbind, ...)`, which is inefficient in R.  
- Neighbor lookup is rebuilt for every variable, and aggregation is done row-wise rather than vectorized.  
- Memory overhead from repeated list-to-matrix conversions and function calls is high.  

**Optimization Strategy**  
- Build the neighbor index **once** and reuse it across all variables and years.  
- Use **vectorized matrix operations** instead of per-row `lapply`.  
- Represent the panel as a matrix where rows = cell-year, columns = variables.  
- Use **sparse adjacency matrix** (from `Matrix` package) to compute neighbor aggregates via matrix multiplication.  
- Compute max, min, and mean in a single pass per variable using efficient grouping.  
- Avoid copying large objects repeatedly; preallocate result columns.  

---

### **Efficient Implementation in R**

```r
library(Matrix)
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs in consistent order
# rook_neighbors_unique: spdep::nb object
# neighbor_source_vars: c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

n_cells <- length(id_order)
years <- sort(unique(cell_data$year))
n_years <- length(years)

# Build sparse adjacency matrix (directed)
adj_list <- rook_neighbors_unique
i_idx <- rep(seq_along(adj_list), lengths(adj_list))
j_idx <- unlist(adj_list)
A <- sparseMatrix(i = i_idx, j = j_idx, x = 1, dims = c(n_cells, n_cells))

# Precompute row sums for mean calculation
neighbor_counts <- rowSums(A)

# Create mapping from (id, year) to row index in big matrix
cell_index <- match(cell_data$id, id_order)
year_index <- match(cell_data$year, years)
row_index <- (year_index - 1) * n_cells + cell_index

# Build big sparse block-diagonal adjacency for all years
A_big <- kronecker(Diagonal(n_years), A)  # block diagonal adjacency
neighbor_counts_big <- rep(neighbor_counts, n_years)

# Prepare result columns
for (var_name in neighbor_source_vars) {
  cell_data[[paste0(var_name, "_nbr_max")]] <- NA_real_
  cell_data[[paste0(var_name, "_nbr_min")]] <- NA_real_
  cell_data[[paste0(var_name, "_nbr_mean")]] <- NA_real_
}

# Compute neighbor stats efficiently
for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  
  # Vector of length n_cells * n_years
  vals_big <- vals[order(year_index, cell_index)]
  
  # Compute sums for mean
  sum_vals <- as.numeric(A_big %*% vals_big)
  mean_vals <- sum_vals / neighbor_counts_big
  mean_vals[neighbor_counts_big == 0] <- NA
  
  # For max and min, do chunked aggregation by year
  max_vals <- numeric(length(vals_big))
  min_vals <- numeric(length(vals_big))
  
  for (y in seq_len(n_years)) {
    idx <- ((y - 1) * n_cells + 1):(y * n_cells)
    sub_vals <- vals_big[idx]
    # For each node, get neighbors
    for (i in seq_len(n_cells)) {
      nbrs <- adj_list[[i]]
      if (length(nbrs) == 0) {
        max_vals[idx[i]] <- NA
        min_vals[idx[i]] <- NA
      } else {
        nbr_vals <- sub_vals[nbrs]
        nbr_vals <- nbr_vals[!is.na(nbr_vals)]
        if (length(nbr_vals) == 0) {
          max_vals[idx[i]] <- NA
          min_vals[idx[i]] <- NA
        } else {
          max_vals[idx[i]] <- max(nbr_vals)
          min_vals[idx[i]] <- min(nbr_vals)
        }
      }
    }
  }
  
  # Map back to cell_data
  cell_data[[paste0(var_name, "_nbr_max")]] <- max_vals[row_index]
  cell_data[[paste0(var_name, "_nbr_min")]] <- min_vals[row_index]
  cell_data[[paste0(var_name, "_nbr_mean")]] <- mean_vals[row_index]
}

# At this point, cell_data has neighbor features added
# Apply pre-trained Random Forest model (already loaded)
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

### **Key Improvements**
- **Adjacency built once** and reused across all variables and years.
- **Block-diagonal sparse matrix** allows fast mean computation via matrix multiplication.
- **Chunked loop for max/min** avoids repeated list allocations.
- Preserves **numerical equivalence** with original method.
- Expected runtime: **hours â†’ minutes** on 16 GB RAM laptop.

This approach scales efficiently for 6.46M rows and 1.37M edges while preserving the original estimand and using the pre-trained Random Forest model without retraining.
END_RESPONSE>>>
