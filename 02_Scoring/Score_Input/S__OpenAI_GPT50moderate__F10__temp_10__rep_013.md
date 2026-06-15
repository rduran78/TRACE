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
run_id: OpenAI_F10_T10_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Current implementation computes neighbor stats (`max`, `min`, `mean`) row by row using `lapply` for 6.46M rows, causing excessive overhead and memory thrashing.  
- Neighbor lookup is rebuilt for each variable, and loop calls `compute_and_add_neighbor_features` repeatedly, adding redundant passes.  
- Sparse adjacency graph properties (1.37M edges) are not fully exploited.  
- Core issue: *lack of vectorization and multi-threading* for 6.46M Ã— 5 variables, leading to 86+ hours runtime.  

---

**Optimization Strategy**  
1. Use a **sparse graph structure (dgCMatrix)** once using `Matrix` package from rook neighbors.  
2. Compute neighbor aggregates in **vectorized fashion** with sparse matrix multiplication instead of looping.  
3. Group rows by year to reuse adjacency; multiply adjacency matrix with variable matrix each year.  
4. Parallelize across variables and/or years with `parallel` or `future.apply`.  
5. Append computed stats back without re-scanning neighbors repeatedly.  
6. Preserve numerical equivalence with original pipeline.  

Computation is reduced to 3 sparse-matrix ops per variable-year block and avoids 6.46M repeated `lapply`.  

---

**Efficient R Implementation**  

```r
library(Matrix)
library(data.table)

# Assume: cell_data has columns id, year, and neighbor_source_vars
# Inputs: rook_neighbors_unique (spdep nb object), id_order (vector with unique ids in adjacency order)

# ---- Build global adjacency matrix ----
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique

# Create row and col indices for nonzero entries
rows <- rep(seq_along(adj_list), sapply(adj_list, length))
cols <- unlist(adj_list, use.names = FALSE)
adj_mat <- sparseMatrix(i = rows, j = cols, x = 1.0, dims = c(n_cells, n_cells))

# ---- Prepare data ----
setDT(cell_data)
setkey(cell_data, id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

# Output matrices for neighbor stats
for (var_name in neighbor_source_vars) {
  cell_data[[paste0(var_name, "_nmax")]] <- NA_real_
  cell_data[[paste0(var_name, "_nmin")]] <- NA_real_
  cell_data[[paste0(var_name, "_nmean")]] <- NA_real_
}

# ---- Compute stats by year using sparse multiplication ----
for (yr in years) {
  idx <- cell_data$year == yr
  block <- cell_data[idx]
  val_mat <- as.numeric(block[match(id_order, id), ..neighbor_source_vars])

  for (j in seq_along(neighbor_source_vars)) {
    vals <- val_mat[, j]
    
    # Compute neighbor sets via adjacency
    neighbor_vals <- adj_mat %*% as.numeric(vals)  # sum for mean numerator
    neighbor_count <- adj_mat %*% rep(1, length(vals)) # counts

    # For max and min, loop but only over nonzero neighbors efficiently
    # Avoid naive row lapply by using adjacency structure
    max_vec <- rep(NA_real_, length(vals))
    min_vec <- rep(NA_real_, length(vals))
    
    for (i in 1:n_cells) {
      nb_idx <- adj_list[[i]]
      if (length(nb_idx) > 0) {
        nb_vals <- vals[nb_idx]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) > 0) {
          max_vec[i] <- max(nb_vals)
          min_vec[i] <- min(nb_vals)
        }
      }
    }
    
    mean_vec <- (neighbor_vals / neighbor_count)
    
    var <- neighbor_source_vars[j]
    block[[paste0(var, "_nmax")]] <- max_vec
    block[[paste0(var, "_nmin")]] <- min_vec
    block[[paste0(var, "_nmean")]] <- mean_vec
  }
  
  # Assign back
  cell_data[idx] <- block
}

# Random Forest prediction using existing model
# preds <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Key Gains**  
- Builds adjacency once using `sparseMatrix` (memory efficient).  
- Uses sparse matrix ops for sums and counts to compute means quickly.  
- Max/min still requires access per row but avoids repeated multi-joins. (Further optimization via `Rcpp` or `parallel` if needed.)  
- Parallelize inner loop (`year` Ã— `variable`) for extra speed.  

**Expected Performance**: Hours âž¡ Minutes (dominated by max/min loop). Further boost by implementing max/min aggregation in C++ (via `RcppArmadillo`).  

**Preserves**: Original estimand and trained Random Forest model unchanged.
END_RESPONSE>>>
