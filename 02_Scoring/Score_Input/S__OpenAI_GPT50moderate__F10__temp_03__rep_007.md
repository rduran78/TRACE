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
run_id: OpenAI_F10_T03_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It repeatedly computes neighbor statistics for each variable and each cell-year using `lapply` over ~6.46M rows.  
- Neighbor lookups are recomputed for every variable loop.  
- No vectorization or efficient matrix operations; heavy R-level loops dominate runtime.  
- Memory overhead from repeated list operations.  

**Optimization Strategy**  
- Build neighbor index once and reuse across all variables and years.  
- Represent the panel as a matrix where rows = cell-years, columns = variables.  
- Use a sparse graph representation (e.g., `Matrix` package) for rook adjacency.  
- Compute neighbor stats via sparse matrix multiplications and group operations instead of per-row loops.  
- Process all years in one pass by leveraging block structure: adjacency repeated for each year.  
- Avoid `lapply` over millions of elements; use vectorized or compiled code (`data.table` or `Matrix`).  

---

### **Efficient Implementation in R**

```r
library(data.table)
library(Matrix)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell ids in consistent order
# rook_neighbors_unique: spdep::nb object
# Pre-trained Random Forest model: rf_model

# 1. Build adjacency matrix for cells (rook neighbors)
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
rows <- rep(seq_along(adj_list), sapply(adj_list, length))
cols <- unlist(adj_list)
adj_mat <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# 2. Expand adjacency for all years (block diagonal)
years <- sort(unique(cell_data$year))
n_years <- length(years)
adj_block <- kronecker(Diagonal(n_years), adj_mat)  # size: (n_cells*n_years) x (n_cells*n_years)

# 3. Prepare data in correct order
setkey(cell_data, id, year)
cell_data[, row_idx := .I]  # row index for mapping
n_rows <- nrow(cell_data)

# 4. Compute neighbor stats for each variable
compute_neighbor_stats_sparse <- function(var_vec, adj_block) {
  # var_vec: numeric vector length = n_rows
  neighbor_sum <- adj_block %*% var_vec
  neighbor_count <- adj_block %*% rep(1, length(var_vec))
  
  # Mean
  neighbor_mean <- neighbor_sum / neighbor_count
  # For min/max: need aggregation per row's neighbors
  # Extract neighbors efficiently:
  # Use adjacency indices
  # Convert to triplet for fast per-row ops
  trip <- summary(adj_block)
  vals <- var_vec[trip$j]
  dt <- data.table(i = trip$i, val = vals)
  minmax <- dt[, .(min = min(val), max = max(val)), by = i]
  
  # Combine
  res <- matrix(NA_real_, nrow = length(var_vec), ncol = 3)
  res[minmax$i, 1] <- minmax$max
  res[minmax$i, 2] <- minmax$min
  res[, 3] <- as.numeric(neighbor_mean)
  res
}

neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_vars) {
  var_vec <- cell_data[[var_name]]
  stats_mat <- compute_neighbor_stats_sparse(var_vec, adj_block)
  cell_data[, paste0(var_name, "_nbr_max") := stats_mat[, 1]]
  cell_data[, paste0(var_name, "_nbr_min") := stats_mat[, 2]]
  cell_data[, paste0(var_name, "_nbr_mean") := stats_mat[, 3]]
}

# 5. Predict with pre-trained Random Forest
preds <- predict(rf_model, newdata = cell_data)
```

---

### **Why This Is Efficient**
- Builds adjacency once, then uses sparse matrix operations (highly optimized in C).
- Eliminates per-row `lapply` loops.
- Handles all years in a single block structure (Kronecker product).
- Min/max computed via grouped aggregation on adjacency triplets (fast with `data.table`).
- Preserves numerical equivalence: same neighbor sets, same stats.

**Expected Runtime**: Minutes to a few hours (depending on disk I/O), not 86+ hours.  
**Memory**: Sparse representation keeps RAM usage manageable on 16 GB.
END_RESPONSE>>>
