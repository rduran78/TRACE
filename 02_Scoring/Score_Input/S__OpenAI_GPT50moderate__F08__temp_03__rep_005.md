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
run_id: OpenAI_F08_T03_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F08_OpenAI_temp_03_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it computes neighbor statistics (`max`, `min`, `mean`) for each variable across all 6.46M rows by repeatedly scanning neighbor indices. This results in redundant computations across years and variables. The neighbor relationships are static across years, but the code recomputes neighbor lookups and stats for every row and variable, leading to high time complexity and memory pressure.

**Optimization Strategy**  
- **Precompute neighbor lookup once** (already done).
- **Exploit static neighbor structure**: For each year, compute neighbor stats in a vectorized way using matrix operations rather than looping over rows.
- **Chunk by year**: Process one year at a time to keep memory usage manageable.
- **Avoid repeated lapply for each variable**: Compute all neighbor-based stats in one pass per year using the same neighbor index structure.
- Use **sparse adjacency matrix** or **list-of-indices** for fast aggregation.
- Preserve the trained Random Forest model and original estimand.

---

### **Optimized R Code**

```r
library(Matrix)
library(data.table)

# Assume: cell_data has columns id, year, and predictor variables
# id_order: vector of unique cell IDs in fixed order
# rook_neighbors_unique: list of neighbor indices (spdep::nb)

# 1. Build adjacency matrix once
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
rows <- rep(seq_along(adj_list), lengths(adj_list))
cols <- unlist(adj_list)
adj_mat <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# 2. Convert cell_data to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# 3. Precompute mapping from id to row index
id_to_idx <- setNames(seq_along(id_order), id_order)

# 4. Function to compute neighbor stats for all vars in one pass per year
compute_neighbor_stats_year <- function(dt_year, vars, adj_mat, id_to_idx) {
  # dt_year: data for one year
  idx <- id_to_idx[as.character(dt_year$id)]
  result_list <- vector("list", length(vars))
  
  for (v in seq_along(vars)) {
    vals <- dt_year[[vars[v]]]
    # Vector of length n_cells with NA where no data
    full_vec <- rep(NA_real_, n_cells)
    full_vec[idx] <- vals
    
    # Compute neighbor sums and counts
    neighbor_sum <- adj_mat %*% full_vec
    neighbor_count <- adj_mat %*% (!is.na(full_vec))
    
    # Compute neighbor max and min using apply on adjacency
    # For max/min, use lapply on adj_list (sparse)
    neighbor_max <- sapply(adj_list, function(nei) max(full_vec[nei], na.rm = TRUE))
    neighbor_min <- sapply(adj_list, function(nei) min(full_vec[nei], na.rm = TRUE))
    
    # Compute mean
    neighbor_mean <- as.numeric(neighbor_sum / neighbor_count)
    
    # Extract only for current year's rows
    result_list[[v]] <- data.table(
      paste0(vars[v], "_nb_max") = neighbor_max[idx],
      paste0(vars[v], "_nb_min") = neighbor_min[idx],
      paste0(vars[v], "_nb_mean") = neighbor_mean[idx]
    )
  }
  
  cbind(dt_year, do.call(cbind, result_list))
}

# 5. Apply by year and combine
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- unique(cell_data$year)

result_list <- lapply(years, function(y) {
  dt_year <- cell_data[year == y]
  compute_neighbor_stats_year(dt_year, neighbor_source_vars, adj_mat, id_to_idx)
})

cell_data_with_neighbors <- rbindlist(result_list)

# cell_data_with_neighbors now has neighbor features added
```

---

### **Why This is Faster**
- **Adjacency matrix** allows O(n) aggregation instead of nested loops.
- **Chunking by year** keeps memory under control.
- **Single pass per year for all variables** reduces redundant lookups.
- **Sparse matrix** avoids huge memory overhead.

---

**Expected Performance**  
Instead of 86+ hours, this approach should complete in **minutes to a few hours** on a 16 GB laptop, depending on I/O and CPU speed, because:
- Each year processes ~344k rows (manageable in memory).
- Matrix multiplications and sparse operations are highly optimized in R.

**Preserves**:
- Original numerical estimand.
- Trained Random Forest model (unchanged).
- Neighbor structure and computation logic.
END_RESPONSE>>>
