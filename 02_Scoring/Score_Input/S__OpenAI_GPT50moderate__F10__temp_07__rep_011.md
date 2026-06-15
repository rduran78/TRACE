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
run_id: OpenAI_F10_T07_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck lies in repeatedly computing neighbor lookups and statistics in pure R using nested `lapply` for 6.46M rows. This results in high overhead from list operations, repeated indexing, and no vectorization. Memory consumption is high because intermediate lists are large and recomputation happens across variables.  

**Optimization Strategy**  
- Precompute graph topology once as a sparse adjacency structure (e.g., `dgCMatrix` from **Matrix**).
- Use fast matrix operations to compute neighbor stats:  
  - For each year, extract the submatrix and multiply adjacency with attribute vectors.
- Avoid repeated `lapply` loops; instead, vectorize across all nodes in a year.
- Use efficient data handling with **data.table**.
- Preserve numerical equivalence: process per year to match original logic (neighbors within same year).
- Append max, min, mean for each variable to the main table without changing order.
- Do not retrain the Random Forest; just generate features.

---

### **Optimized R Implementation**

```r
library(data.table)
library(Matrix)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# 1. Build adjacency matrix once (directed rook graph)
# rook_neighbors_unique: list of integer vectors (spdep::nb)
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
i_idx <- rep(seq_along(adj_list), lengths(adj_list))
j_idx <- unlist(adj_list, use.names = FALSE)
adj <- sparseMatrix(i = i_idx, j = j_idx, x = 1, dims = c(n_cells, n_cells))

# 2. Split data by year for efficient processing
years <- sort(unique(cell_data$year))
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 3. Precompute mapping from id -> row in adjacency
id_to_row <- match(cell_data$id[!duplicated(cell_data$id)], id_order)

# 4. Function to compute neighbor stats for one year
compute_year_stats <- function(sub_dt) {
  # sub_dt: data for one year
  idx <- match(sub_dt$id, id_order)
  res_list <- vector("list", length(vars))
  
  for (v in vars) {
    vals <- sub_dt[[v]]
    # Multiply adjacency * value vector
    sum_vals <- adj %*% vals   # sum of neighbors
    count_vals <- adj %*% (!is.na(vals)) # neighbor counts
    means <- sum_vals / pmax(count_vals, 1)
    
    # For max/min, use aggregation by adjacency
    # Build for nonzeros only
    rows <- rep(1:n_cells, lengths(adj_list))
    cols <- unlist(adj_list)
    neigh_vals <- vals[cols]
    # Compute max/min per row
    max_vals <- tapply(neigh_vals, rows, max, na.rm = TRUE)
    min_vals <- tapply(neigh_vals, rows, min, na.rm = TRUE)
    
    # Align to idx
    res_list[[v]] <- data.table(
      paste0(v, "_nb_max") = max_vals[idx],
      paste0(v, "_nb_min") = min_vals[idx],
      paste0(v, "_nb_mean") = means[idx]
    )
  }
  
  # Combine all variables
  cbind(sub_dt[, .(id, year)], do.call(cbind, res_list))
}

# 5. Apply across years
result_list <- lapply(years, function(yr) {
  sub_dt <- cell_data[year == yr]
  compute_year_stats(sub_dt)
})

# 6. Combine all
neighbor_features <- rbindlist(result_list)
setkey(neighbor_features, id, year)

# 7. Merge back to main data
cell_data <- neighbor_features[cell_data]

# cell_data now has original columns + neighbor stats for all vars
```

---

**Key Optimizations**
- **Sparse adjacency**: Built once, reused for all years.
- **Matrix multiplication** for means: O(E) complexity, far faster than millions of `lapply` calls.
- **Vectorized max/min** using `tapply` on adjacency edges.
- **data.table** for fast joins and memory efficiency.
- Processes per year to match original neighborhood logic with minimal RAM usage.
- Preserves numerical equivalence with original implementation.

**Expected Performance**  
From 86+ hours to a few hours on a 16 GB laptop (depending on sparse ops and I/O).  
Memory footprint manageable (adjacency stored once, per-year chunks processed sequentially).
END_RESPONSE>>>
