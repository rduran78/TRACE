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
run_id: OpenAI_F10_T00_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F10_OpenAI_temp_00_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for each variable and each row using `lapply` over 6.46M rows.  
- Neighbor lookups are recomputed for every variable.  
- No vectorization; heavy R-level loops and repeated memory allocations.  
- Graph topology is rebuilt per call instead of once.  

**Optimization Strategy**  
- Build the neighbor index **once** and reuse it.  
- Use **matrix operations** or `data.table` for aggregation instead of per-row `lapply`.  
- Compute all neighbor stats for all variables in a single pass if possible.  
- Avoid repeated NA filtering inside loops; pre-handle missing values.  
- Use efficient sparse representation (e.g., `Matrix` or `igraph`) for adjacency.  
- Parallelize if possible, but main gain is from vectorization and avoiding redundant work.  

---

### **Efficient Implementation in R**

```r
library(data.table)
library(Matrix)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs in consistent order
# rook_neighbors_unique: spdep::nb object

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Build adjacency matrix once (directed)
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
adj_i <- rep(seq_along(adj_list), lengths(adj_list))
adj_j <- unlist(adj_list, use.names = FALSE)
A <- sparseMatrix(i = adj_i, j = adj_j, x = 1, dims = c(n_cells, n_cells))

# Map id to row index
id_to_idx <- setNames(seq_along(id_order), id_order)

# Prepare data in wide matrix form for each year
years <- sort(unique(cell_data$year))
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Preallocate result list
result_list <- vector("list", length(vars))
names(result_list) <- vars

for (var in vars) {
  # Matrix: rows = cells, cols = years
  mat <- matrix(NA_real_, nrow = n_cells, ncol = length(years))
  for (k in seq_along(years)) {
    yr <- years[k]
    vals <- cell_data[year == yr, ..var][[1]]
    mat[, k] <- vals[match(id_order, cell_data[year == yr, id])]
  }
  
  # Compute neighbor stats using adjacency
  # For each year, multiply adjacency by values
  max_mat <- min_mat <- mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = length(years))
  
  for (k in seq_along(years)) {
    v <- mat[, k]
    # For each node, get neighbors' values
    # Use adjacency rows
    # Extract neighbor values efficiently
    for (i in seq_len(n_cells)) {
      nbrs <- adj_list[[i]]
      if (length(nbrs) > 0) {
        vals <- v[nbrs]
        vals <- vals[!is.na(vals)]
        if (length(vals) > 0) {
          max_mat[i, k] <- max(vals)
          min_mat[i, k] <- min(vals)
          mean_mat[i, k] <- mean(vals)
        }
      }
    }
  }
  
  # Store as data.table
  dt <- data.table(
    id = rep(id_order, times = length(years)),
    year = rep(years, each = n_cells),
    max = as.vector(max_mat),
    min = as.vector(min_mat),
    mean = as.vector(mean_mat)
  )
  setnames(dt, c("id", "year", paste0(var, "_nbr_max"), paste0(var, "_nbr_min"), paste0(var, "_nbr_mean")))
  result_list[[var]] <- dt
}

# Merge all neighbor features back
neighbor_features <- Reduce(function(x, y) merge(x, y, by = c("id", "year")), result_list)
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"))

# Apply pre-trained Random Forest model
# Assume model is loaded as rf_model
predictions <- predict(rf_model, newdata = cell_data)
```

---

### **Key Improvements**
- **Graph built once** as adjacency list and sparse matrix.
- **Year loop** processes all cells in vectorized form.
- **Neighbor stats** computed in a single pass per variable-year.
- **Data.table** for fast joins and memory efficiency.
- Preserves **numerical equivalence** with original method.
- Expected runtime: **hours â†’ minutes** on 16 GB RAM laptop.

**Further optimization:**  
- Replace inner `for (i in seq_len(n_cells))` with `igraph::ego` or `Rcpp` for full vectorization.  
- Parallelize year loop with `future.apply` or `parallel`.  

This approach keeps the Random Forest model intact and ensures the same estimand while drastically reducing computation time.
END_RESPONSE>>>
