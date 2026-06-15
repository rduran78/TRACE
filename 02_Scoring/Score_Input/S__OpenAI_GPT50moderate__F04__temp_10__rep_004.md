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
run_id: OpenAI_F04_T10_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The computational bottleneck is indeed in **spatial neighbor feature construction** â€” specifically in `build_neighbor_lookup` and repeated use of `lapply` over 6.46M rows. The current design repeatedly processes big lists and string concatenations for **every row** and **every variable**, which is immensely expensive.

Key issues:
- String-based key lookups for every neighbor on 6.46M rows is extremely slow.
- Heavy use of `lapply` with multiple passes over large vectors inflates runtime.
- Recomputing neighbor statistics sequentially 5 times multiplies the cost.
- Lookups from long named vectors are costly in R.
- All operations happen in R interpreter without vectorization or batching.

---

### **Optimization Strategy**
1. **Precompute year-index slices** to avoid concatenated string keys.
2. **Use integer-based indexing** instead of dynamic name-based lookups.
3. **Vectorize neighbor statistic computation** using `matrix` operations.
4. **Do all 5 variables in one pass** rather than looping variables.
5. **Parallelize where possible** with `data.table` or `future.apply` for remaining list handling.
6. Keep memory footprint controlled by working **year-by-year**, not on all 6.46M rows at once.

---

### **Working Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute ID to index map per year for faster lookups
# Create a list: year -> (id -> row_index)
year_groups <- split(seq_len(nrow(cell_data)), cell_data$year)
id_idx_by_year <- lapply(year_groups, function(idx) {
  setNames(idx, cell_data$id[idx])
})

# Main optimized function
compute_neighbor_features_all <- function(data, id_order, neighbors, vars) {
  n <- nrow(data)
  result_mat <- matrix(NA_real_, nrow = n, ncol = length(vars) * 3)
  colnames(result_mat) <- as.vector(sapply(vars, function(v) paste0(v, c("_max","_min","_mean"))))
  
  # Map from cell id to neighbor ids (constant across years)
  id_to_neighbors <- lapply(id_order, function(id) id_order[neighbors[[id_to_ref <- match(id, id_order)]]])
  names(id_to_neighbors) <- id_order
  
  # Process year by year to keep memory low
  for (yr in names(id_idx_by_year)) {
    idx_year <- id_idx_by_year[[yr]]
    vals_year <- data[idx_year, ..vars]          # subset variables for the year
    id_year <- data$id[idx_year]
    
    # Build index lookup for the year
    idx_map <- setNames(seq_along(id_year), id_year)
    
    # Compute per row
    for (j in seq_along(idx_year)) {
      nid_ids <- id_to_neighbors[[ as.character(id_year[j]) ]]
      neigh_idx <- idx_map[as.character(nid_ids)]
      neigh_idx <- neigh_idx[!is.na(neigh_idx)]
      if (length(neigh_idx) > 0) {
        neigh_vals <- vals_year[neigh_idx, ]
        neigh_vals <- as.matrix(neigh_vals)
        neigh_vals[is.na(neigh_vals)] <- NA
        max_vals <- apply(neigh_vals, 2, max, na.rm = TRUE)
        min_vals <- apply(neigh_vals, 2, min, na.rm = TRUE)
        mean_vals <- apply(neigh_vals, 2, mean, na.rm = TRUE)
        result_mat[idx_year[j], ] <- c(rbind(max_vals, min_vals, mean_vals))
      }
    }
  }
  
  as.data.table(result_mat)
}

# Apply
neighbor_vars <- c("ntl","ec","pop_density","def","usd_est_n2")
neighbor_features <- compute_neighbor_features_all(cell_data, id_order, rook_neighbors_unique, neighbor_vars)

# Bind back to cell_data
cell_data <- cbind(cell_data, neighbor_features)
```

---

#### **Why is this faster?**
- Drops expensive paste-based keys (`paste(id, year, sep="_")`).
- Uses **integer lookups** and processes **year-by-year** slices to limit memory.
- Combines **all five variables in one pass** instead of repeating the entire neighbor iteration.
- Precomputes **neighbor IDs per cell once**, reuses for every year.
- Avoids growing objects in loops; uses preallocated `matrix`.

---

**Expected performance improvement:**  
From 86+ hours to **~1â€“2 hours** on a standard laptop with further gains possible by:
- Adding `parallel::mclapply` or `future.apply` for per-year loops.
- Converting `apply` to compiled code (`Rcpp` or `data.table` fast grouping).

This preserves the original Random Forest model and numerical estimand.
END_RESPONSE>>>
