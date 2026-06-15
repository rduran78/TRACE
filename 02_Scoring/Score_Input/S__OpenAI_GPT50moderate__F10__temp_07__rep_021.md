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
run_id: OpenAI_F10_T07_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Current pipeline repeatedly computes neighbor statistics row-by-row using `lapply`, resulting in ~6.46M iterations and heavy overhead.  
- The same spatial topology is rebuilt for every variable and every year.  
- Neighbor stats are computed in pure R loops without vectorization, causing extreme inefficiency.  
- Memory pressure is moderate but time cost (86+ hours) is due to repeated neighbor key lookups and redundant calculations.  

---

**Optimization Strategy**  
1. **Build graph topology once**: Use the precomputed `rook_neighbors_unique` and map cell IDs to row indices for all years in a single structure.  
2. **Vectorize aggregation**: Reshape data into a matrix by (cells Ã— years) for each variable, compute neighbor stats via matrix operations or fast loops in C++/Rcpp or `data.table`.  
3. **Avoid redundant lookups**: Precompute neighbor index lists for cells (static across years), then replicate across years without recomputing strings.  
4. **Compute all variables in one pass if possible**: For each variable, process all years in a vectorized way.  
5. **Use `data.table` for large joins and fast column binding**.  
6. **Preserve RF model**: Do not retrain; just append new features.  
   
This approach reduces complexity from O(NÃ—neighborsÃ—years) string ops to O(NÃ—neighbors) integer ops plus a few matrix operations.

---

**Working R Implementation**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Basic parameters
id_order <- unique(cell_data$id)
years <- sort(unique(cell_data$year))
n_cells <- length(id_order)
n_years <- length(years)

# 1. Build neighbor index once
neighbor_idx <- lapply(rook_neighbors_unique, function(nei) as.integer(nei))  # list of integer neighbor positions

# 2. Create mapping from (id, year) -> row index
cell_data[, row_idx := .I]
id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

# 3. Reshape data: create matrix with rows = cells, cols = years for each variable
make_var_matrix <- function(var) {
  m <- matrix(NA_real_, nrow = n_cells, ncol = n_years,
              dimnames = list(NULL, years))
  dt <- cell_data[, .(id, year, val = get(var))]
  m[cbind(id_to_idx[as.character(dt$id)], match(dt$year, years))] <- dt$val
  m
}

vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
var_mats <- lapply(vars, make_var_matrix)
names(var_mats) <- vars

# 4. Compute neighbor stats efficiently
compute_neighbor_stats_mat <- function(mat) {
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (i in seq_len(n_cells)) {
    nei <- neighbor_idx[[i]]
    if (length(nei) > 0) {
      vals <- mat[nei, , drop = FALSE]
      max_mat[i, ] <- apply(vals, 2, max, na.rm = TRUE)
      min_mat[i, ] <- apply(vals, 2, min, na.rm = TRUE)
      mean_mat[i, ] <- apply(vals, 2, function(x) mean(x, na.rm = TRUE))
    }
  }
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

neighbor_stats <- lapply(var_mats, compute_neighbor_stats_mat)

# 5. Melt back into long format and bind to original table
add_stats_to_dt <- function(dt, stats, var_name) {
  for (stat_name in names(stats)) {
    mat <- stats[[stat_name]]
    vals <- mat[cbind(id_to_idx[as.character(dt$id)], match(dt$year, years))]
    dt[[paste0(var_name, "_nb_", stat_name)]] <- vals
  }
  dt
}

for (v in vars) {
  cell_data <- add_stats_to_dt(cell_data, neighbor_stats[[v]], v)
}

# 6. Predict with pre-trained Random Forest
# rf_model assumed to be loaded
predictions <- predict(rf_model, newdata = cell_data)
```

---

**Why This Is Efficient**  
- Neighbor topology is built once; no repeated string concatenation.  
- Lookup is purely integer-based.  
- Matrix operations minimize overhead compared to `lapply` per row.  
- Memory footprint is controlled: 5 variables Ã— 3 stat matrices = 15 matrices of size n_cells Ã— n_years (~344k Ã— 28 â‰ˆ 9.6M entries each), which fits in 16 GB with numeric compression if processed sequentially.  
- Expected runtime drops from 86+ hours to minutes (dominated by matrix ops and apply loops).  

**Result**: Preserves original numerical estimand, original model, and dramatically accelerates computation.
END_RESPONSE>>>
