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
run_id: OpenAI_F12_T10_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F12_OpenAI_temp_10_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` is the main bottleneck is **incorrect**. The cost of `rbind` for ~6.46M rows and 3 columns (â‰ˆ19M elements) is significant **but not the dominant driver**. The real bottleneck lies in repeated *R-level interpretation overhead and memory churn* from the inner `lapply` closure that:  
- Iterates ~6.46M times across five variables (â‰ˆ32M iterations).  
- Repeatedly allocates small vectors, filters `NA`s, and calls `max/min/mean`â€”all in pure R loops.  

This is orders of magnitude slower than vectorized or compiled alternatives. The neighbor lookup structure amplifies the cost because every row triggers an R function call.  

---

**Correct Optimization Strategy:**  
- **Precompute** a dense neighbor index matrix (with `NA` padding) so neighbors can be accessed without repeated `lapply`.  
- **Vectorize with fast C-level operations**: compute `max`, `min`, `mean` for all rows in a single pass using `matrixStats` or `data.table`.  
- Avoid repeatedly binding listsâ€”write results to a preallocated numeric matrix.  

---

### **Optimized Implementation**

```r
library(matrixStats)

# Step 1: Build a uniform neighbor matrix (R-level, once)
build_neighbor_matrix <- function(n_neighbors, neighbor_lookup) {
  # Pad all neighbor vectors to same length with NA
  res <- matrix(NA_integer_, nrow = length(neighbor_lookup), ncol = n_neighbors)
  for (i in seq_along(neighbor_lookup)) {
    n <- length(neighbor_lookup[[i]])
    if (n > 0) {
      res[i, 1:n] <- neighbor_lookup[[i]]
    }
  }
  res
}

n_lookup <- build_neighbor_matrix(n_neighbors = max(lengths(neighbor_lookup)), neighbor_lookup = neighbor_lookup)

# Step 2: Compute stats efficiently
compute_neighbor_stats_fast <- function(data, n_lookup, var_name) {
  vals <- data[[var_name]]
  # Create matrix of neighbor values for all rows
  neighbor_vals_mat <- matrix(vals[n_lookup], nrow = nrow(n_lookup))
  
  # Ignore NAs in computations
  row_max  <- rowMaxs(neighbor_vals_mat, na.rm = TRUE)
  row_min  <- rowMins(neighbor_vals_mat, na.rm = TRUE)
  row_mean <- rowMeans(neighbor_vals_mat, na.rm = TRUE)
  
  # Identify rows with no valid neighbors
  no_neighbors <- rowSums(!is.na(neighbor_vals_mat)) == 0
  row_max[no_neighbors]  <- NA
  row_min[no_neighbors]  <- NA
  row_mean[no_neighbors] <- NA
  
  cbind(row_max, row_min, row_mean)
}

# Step 3: Loop over variables with minimal overhead
for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, n_lookup, var_name)
  colnames(stats_mat) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

**Why this works:**  
- Converts per-row R calls into **vectorized matrix operations** (C-optimized).  
- Builds neighbor matrix once instead of repeated lookups.  
- Eliminates `do.call(rbind, ...)` entirely and avoids iterative memory growth.  

---

**Expected Speedup:**  
From >86 hours to roughly **minutes** or at worst a few hours on 6.46M rowsâ€”dominated by one-time neighbor matrix build and three vectorized sweeps per variable.  

This preserves:
- Original Random Forest model (unchanged).  
- Original estimands (same formulas).  

**Key libraries**: `matrixStats` and base R for memory efficiency.
END_RESPONSE>>>
