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
run_id: OpenAI_F07_T07_R010
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 10
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow (86+ hours) because:  
- `lapply` over 6.46M rows repeatedly for each variable is highly inefficient in R.  
- Neighbor lookups are computed in pure R lists, causing massive overhead.  
- No vectorization and repeated object expansions inside the loop.  
- No memory-efficient strategy for 6.46M Ã— 3 stats Ã— 5 variables.  

**Optimization Strategy**  
1. **Precompute neighbor indices once** as an integer matrix or list of integer vectors.  
2. **Use `matrixStats` or `data.table` for fast row-level aggregation** instead of R loops.  
3. **Batch process variables** to avoid repeated passes and minimize intermediate allocations.  
4. **Use `vapply` or vectorized apply** and avoid `do.call(rbind, ...)` for millions of rows.  
5. **Consider sparse representation** since neighbor count is small relative to dataset size.  

**Core Idea:**  
- Convert `neighbor_lookup` into a fixed-length integer matrix (pad NAs).  
- Use fast row-wise operations on numeric matrices for max, min, mean.  

---

### **Optimized Working Code**

```r
library(matrixStats)
library(data.table)

# Precompute neighbor matrix
build_neighbor_matrix <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  row_ids <- seq_len(nrow(data))
  max_nbrs <- max(lengths(neighbors))
  
  neighbor_mat <- matrix(NA_integer_, nrow = length(row_ids), ncol = max_nbrs)
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    nbr_ids <- id_order[neighbors[[ref_idx]]]
    nbr_keys <- paste(nbr_ids, data$year[i], sep = "_")
    nbr_idx <- idx_lookup[nbr_keys]
    nbr_idx <- as.integer(nbr_idx[!is.na(nbr_idx)])
    
    if (length(nbr_idx) > 0) {
      neighbor_mat[i, seq_along(nbr_idx)] <- nbr_idx
    }
  }
  neighbor_mat
}

neighbor_mat <- build_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)

# Compute stats in one pass for each variable
compute_neighbor_stats_fast <- function(data, neighbor_mat, var_name) {
  vals <- data[[var_name]]
  res <- matrix(NA_real_, nrow = nrow(neighbor_mat), ncol = 3)
  
  for (i in seq_len(nrow(neighbor_mat))) {
    idx <- neighbor_mat[i, ]
    idx <- idx[!is.na(idx)]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        res[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  colnames(res) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  res
}

# Efficiently add all neighbor features
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
all_features <- lapply(neighbor_source_vars, function(v) compute_neighbor_stats_fast(cell_data, neighbor_mat, v))
all_features <- do.call(cbind, all_features)

# Bind to original data.table for efficient handling
cell_data <- cbind(cell_data, all_features)
```

---

**Key Improvements:**  
- **One-time neighbor matrix build** â†’ avoids repeated list lookups.  
- **Vectorized loops with preallocated matrices** â†’ eliminates `lapply` overhead.  
- Reduces complexity from *O(N Ã— vars Ã— neighbors)* with high R overhead to a much tighter loop.  

**Estimated Runtime:**  
- Neighbor matrix build: ~30â€“40 min for 6.46M rows (mostly I/O bound).  
- Stats computation: ~1â€“2 hours total (vs. 86+ hours previously).  

**Preserves:**  
- Original rook-neighbor topology.  
- Exact max, min, mean estimands.  
- Trained Random Forest model remains unchanged.
END_RESPONSE>>>
