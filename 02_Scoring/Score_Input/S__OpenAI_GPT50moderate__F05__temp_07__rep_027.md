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
run_id: OpenAI_F05_T07_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F05_OpenAI_temp_07_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string-key operations (`paste()`, `setNames()`, and `idx_lookup[...]`) inside `lapply` represent a **symptom of a larger repeated lookup pattern** across *all* rows and variables. For each of the ~6.46M rows, you reconstruct string keys for neighbor cells and perform hash lookups, even though the grid structure and neighbor relationships are invariant across years and variables. This leads to severe overhead (string concatenation, hashing) and repeated memory allocations.

**Optimization Strategy**  
- Precompute **neighbor indices for all rows once**, using integer operations only.
- Avoid repeated `paste()` and named indexing during feature computation.
- Store neighbor indices in an `IntegerList` (or list of integer vectors) aligned with row order.
- Use these precomputed indices for all neighbor source variables without recomputing keys.
- The neighbor relationships are deterministic given `data$id`, `data$year`, and `id_order`; leverage this to flatten the nested lookups into a single integer join.

**Algorithmic Reformulation**  
Instead of building keys like `"cellID_year"` repeatedly, create:
- A **fast mapping** from `(id, year)` to row index via an integer matrix or environment once.
- Generate `neighbor_lookup` as a list of integer vectors of row indices for all rows in one pass.
- Then compute neighbor stats in pure numeric space.

---

### **Working Optimized R Code**

```r
# Precompute (id, year) -> row index mapping as a matrix
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  n <- nrow(data)
  years <- sort(unique(data$year))
  n_years <- length(years)
  
  # Map id to reference index
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Precompute row lookup: create a matrix [id_ref, year_idx] -> global row
  row_lookup <- integer(length(id_order) * n_years)
  dim(row_lookup) <- c(length(id_order), n_years)
  
  # Fill row_lookup
  year_to_idx <- setNames(seq_along(years), years)
  for (i in seq_len(n)) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    y_idx   <- year_to_idx[as.character(data$year[i])]
    row_lookup[ref_idx, y_idx] <- i
  }
  
  # Build neighbor lookup as list of integer vectors
  row_ids <- seq_len(n)
  neighbor_lookup <- vector("list", n)
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    y_idx   <- year_to_idx[as.character(data$year[i])]
    neighbor_ref_ids <- neighbors[[ref_idx]]
    
    if (length(neighbor_ref_ids) > 0) {
      neighbor_rows <- row_lookup[neighbor_ref_ids, y_idx]
      neighbor_lookup[[i]] <- neighbor_rows[neighbor_rows > 0]
    } else {
      neighbor_lookup[[i]] <- integer(0)
    }
  }
  
  neighbor_lookup
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  
  res_mat <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(res_mat) <- c("max", "min", "mean")
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        res_mat[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  
  res_mat
}

# Usage:
# Precompute neighbor lookup ONCE
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute features for each variable efficiently
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats_mat) <- paste0(var_name, "_nbr_", c("max", "min", "mean"))
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

### **Expected Performance Gain**
- Removes 6.46M Ã— 5 Ã— (string concatenation + hashing) operations.
- Converts repeated hash lookups into **integer matrix indexing**, which is O(1) and vectorized.
- Memory footprint remains manageable: `neighbor_lookup` holds ~6.46M lists of small integer vectors, similar to original but without repeated key computation.
- Anticipated runtime drop from **86+ hours to a few hours or less**, depending on I/O and CPU.

This preserves:
- Original numeric estimand.
- Trained Random Forest model.
- Panel structure and neighbor logic.

**Summary:** The inefficiency is global, not local. Fix by precomputing integer neighbor indices once and using them for all variables.
END_RESPONSE>>>
