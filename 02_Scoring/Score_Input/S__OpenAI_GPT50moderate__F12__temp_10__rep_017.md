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
run_id: OpenAI_F12_T10_R017
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 17
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F12_OpenAI_temp_10_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The evidence suggests the colleagueâ€™s diagnosis is incorrect. The dominant bottleneck is **not** `do.call(rbind, result)`, but rather the repeated **full scan and recomputation in `compute_neighbor_stats()` for each variable**. Specifically:

- `compute_neighbor_stats()` traverses the entire `neighbor_lookup` list for **each of the 5 variables**, doing millions of small `c()` allocations and repeatedly filtering `NA`s.
- Each pass touches ~6.46M rows and ~1.37M neighbor lists â†’ O(n Ã— k Ã— vars) cost.
- `do.call(rbind, result)` is O(n) and done only once per variable; negligible compared to repeated lapply logic.

### Correct Diagnosis
The real bottleneck is **computing neighbor aggregates 5 times instead of once**. The neighbor lookup never changes, so recomputing stats separately per variable is wasteful.

---

## Optimization Strategy
- Vectorize across all 5 variables in a **single pass** over `neighbor_lookup`.
- Pre-extract all relevant variable columns into a matrix.
- Compute neighbor stats for all variables at once and then bind results.
- Avoid repeated `lapply` loops and redundant NA filtering.

---

## Fast Implementation

```r
compute_all_neighbor_stats <- function(data, neighbor_lookup, vars) {
  # Extract columns of interest as matrix
  val_mat <- as.matrix(data[vars])
  
  n <- length(neighbor_lookup)
  p <- length(vars)
  
  # Preallocate result: 3 stats (max, min, mean) Ã— p variables
  result <- matrix(NA_real_, nrow = n, ncol = p * 3)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      # Get all neighbor values for current observation across all vars
      neighbor_vals <- val_mat[idx, , drop = FALSE]
      
      # Remove rows with all NA
      neighbor_vals <- neighbor_vals[rowSums(is.na(neighbor_vals)) < ncol(neighbor_vals), , drop=FALSE]
      
      if (nrow(neighbor_vals) > 0) {
        result[i, ] <- c(
          apply(neighbor_vals, 2, max, na.rm = TRUE),
          apply(neighbor_vals, 2, min, na.rm = TRUE),
          colMeans(neighbor_vals, na.rm = TRUE)
        )
      }
    }
  }
  
  # Assign column names
  colnames(result) <- as.vector(
    outer(vars, c("max", "min", "mean"), paste, sep = "_neighbor_")
  )
  
  result
}

# Usage:
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

neighbor_features <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_vars)

# Bind to main dataset
cell_data <- cbind(cell_data, neighbor_features)
```

---

### Why This Works
- Only **one pass** over `neighbor_lookup` (vs 5 passes).
- Avoids repeatedly creating millions of small vectors.
- Uses preallocated matrix instead of repeated list binding.
- Cuts complexity roughly by factor â‰ˆ number of vars (5Ã— speedup) and reduces memory thrash.

On ~6.46M rows Ã— 5 vars, this typically reduces runtime from **86+ hours to a few hours** on a 16â€¯GB laptop.

**Preserves original estimand and trained model.**
END_RESPONSE>>>
