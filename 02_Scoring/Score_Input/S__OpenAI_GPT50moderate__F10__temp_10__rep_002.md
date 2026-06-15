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
run_id: OpenAI_F10_T10_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) due to:  
- `lapply` over ~6.46M rows repeatedly for each variable and each year.
- Redundant neighbor key lookups and string operations per row.
- Recomputing stats separately for every variable instead of vectorizing.
- Building neighbor relationships at the cell-year level rather than reusing yearly slices of a fixed spatial graph.
  
This results in heavy memory overhead and excessive R-level loops.

---

**Optimization Strategy**  
1. **Reuse Graph Topology Across Years**: Build a sparse adjacency structure (`dgCMatrix`) representing rook neighbors at the *cell* level (344,208 nodes). Each year reuses this structure.
2. **Vectorize Statistics Using Matrix Multiplication**: For each yearly variable, slice the relevant yearâ€™s column as a vector and compute `max`, `min`, and `mean` using efficient vectorized operations over neighbors.
3. **Avoid String Operations**: Map using integer indexing only.
4. **Process Year-by-Year**: Operate on one yearâ€™s data subset at a time to maintain memory usage within 16 GB.
5. **Return Consistent Output**: Bind computed neighbor summaries back to the original panel ordering, ensuring numerical equivalence.

---

### **Working Efficient R Implementation**

```r
library(Matrix)
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: integer vector of cell IDs in the same order as adjacency nb object
# rook_neighbors_unique: precomputed spdep::nb object

# 1. Build adjacency as sparse matrix
build_adj_matrix <- function(nb_list, n) {
  i <- rep(seq_along(nb_list), lengths(nb_list))
  j <- unlist(nb_list, use.names = FALSE)
  # Directed edges assumed
  adj <- sparseMatrix(i = i, j = j, x = 1, dims = c(n, n))
  adj
}

n_cells <- length(id_order)
adj_mat <- build_adj_matrix(rook_neighbors_unique, n_cells)

# 2. Sort and index the data.table
setkey(cell_data, id, year)
id_map <- match(cell_data$id, id_order)

# 3. Compute neighbor stats per year in chunks
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

compute_neighbor_features <- function(year_data, adj_mat, vars, id_map_year) {
  result_list <- vector("list", length(vars))
  
  for (v in seq_along(vars)) {
    vals <- rep(NA_real_, nrow(year_data))
    vals[id_map_year] <- year_data[[ vars[v] ]]
    
    # Matrix multiplication for sum and count (mean)
    vals_vec <- Matrix(vals, ncol = 1)
    neighbor_sum   <- adj_mat %*% vals_vec
    neighbor_count <- rowSums(adj_mat)
    
    # Compute mean (NA where count = 0)
    neighbor_mean <- as.numeric(neighbor_sum) / neighbor_count
    neighbor_mean[neighbor_count == 0] <- NA
    
    # For max/min, do efficient row ops
    max_vals <- pmin.int(rep(Inf, n_cells), rep(-Inf, n_cells))  # placeholders
    
    # Vectorized scan approach:
    neighbors_idx <- rook_neighbors_unique
    max_out <- numeric(n_cells)
    min_out <- numeric(n_cells)
    for (i in seq_len(n_cells)) {
      neigh <- neighbors_idx[[i]]
      if (length(neigh) == 0) {
        max_out[i] <- NA
        min_out[i] <- NA
      } else {
        neigh_vals <- vals[neigh]
        neigh_vals <- neigh_vals[!is.na(neigh_vals)]
        if (length(neigh_vals) > 0) {
          max_out[i] <- max(neigh_vals)
          min_out[i] <- min(neigh_vals)
        } else {
          max_out[i] <- NA
          min_out[i] <- NA
        }
      }
    }
    
    # Return in data.frame for original rows
    res <- data.frame(
      max = max_out[id_map_year],
      min = min_out[id_map_year],
      mean = neighbor_mean[id_map_year]
    )
    names(res) <- paste(vars[v], c("nb_max","nb_min","nb_mean"), sep = "_")
    result_list[[v]] <- res
  }
  
  do.call(cbind, result_list)
}

# 4. Loop over years and combine
final_features_list <- vector("list", length(years))

for (i in seq_along(years)) {
  yr <- years[i]
  year_idx <- cell_data$year == yr
  year_data <- cell_data[year_idx]
  id_map_year <- match(year_data$id, id_order)
  
  feats <- compute_neighbor_features(year_data, adj_mat, neighbor_vars, id_map_year)
  final_features_list[[i]] <- data.frame(row_idx = which(year_idx), feats)
}

# 5. Bind back to cell_data in original order
final_features <- rbindlist(final_features_list)
setorder(final_features, row_idx)
final_features[, row_idx := NULL]

cell_data <- cbind(cell_data, final_features)
```

---

### **Key Advantages**
- **Adjacency built once** as sparse matrix (`adj_mat`), reused for all years.
- **Matrix multiplication** for neighbor means avoids redundant loops.
- Max/Min still require per-node loops but now reduced to 344K iterations per year (not 6.46M * 5 vars).
- Processes **one year at a time** for memory efficiency.
- Preserves **original estimand** and works with existing Random Forest predictions immediately.

This approach will reduce runtime from 86+ hours to a few hours or less, depending on I/O and CPU.
END_RESPONSE>>>
