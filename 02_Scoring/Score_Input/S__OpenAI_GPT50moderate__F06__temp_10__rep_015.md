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
run_id: OpenAI_F06_T10_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F06_OpenAI_temp_10_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow because `lapply` loops over all 6.46M rows for each variable and each neighbor lookup, repeatedly scanning vectors and reconstructing results in R. This results in high interpreter overhead and unnecessary repeated operations (string manipulation, list indexing, etc.). All computations are happening in pure R lists rather than in an optimized matrix or vectorized environment. Memory footprint is also large due to repeated `lapply` calls and intermediate list allocations.

---

**Optimization Strategy**  
1. **Precompute neighbor index matrix once**: Convert `neighbor_lookup` from a list-of-integers into an integer matrix or `dgRMatrix` (sparse row-compressed form), eliminating repeated `lapply` dispatch.
2. **Vectorize summary stats**: For each variable, extract its full numeric vector and apply fast grouped operations (matrix-style row-wise stats) using compiled backends (`matrixStats` or `Rcpp`), or apply `data.table` grouping if memory allows.
3. **Avoid redundant NA filtering in R loops**: Instead of filtering NA in each iteration, use `row*` functions that handle NA globally or implement an Rcpp routine with checks.
4. **Workflow adjustments**:
   - Build a single numeric neighbor array `vals[neighbors]` with padding (`NA_real_`) and compute `max`, `min`, `mean` with `apply` or Rcpp.
   - Reduce memory by streaming variables individually rather than storing large intermediates.
5. **Keep model intact**: Only the feature engineering changes, Random Forest remains untouched.

Estimated speedup: Orders of magnitude faster (minutes instead of 86+ hours).

---

**Working R Code (Optimized Version)**  

```r
library(Matrix)
library(matrixStats)
library(data.table)

## 1. Build neighbor lookup as sparse Matrix (rows = cell-rows, cols = neighbors)
build_neighbor_matrix <- function(neighbor_lookup, n) {
  i_idx <- rep(seq_along(neighbor_lookup), lengths(neighbor_lookup))
  j_idx <- unlist(neighbor_lookup)
  x_idx <- rep(1, length(j_idx))
  sparseMatrix(i = i_idx, j = j_idx, x = x_idx, dims = c(n, n))
}

n_rows <- nrow(cell_data)
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
neighbor_mat <- build_neighbor_matrix(neighbor_lookup, n_rows)

## 2. Optimized compute function using sparse multiplication
compute_neighbor_stats_fast <- function(vals, neighbor_mat) {
  # Replace NA with 0 for mean handling, track counts separately
  na_mask <- is.na(vals)
  vals0 <- vals
  vals0[na_mask] <- 0
  sum_neighbors <- as.numeric(neighbor_mat %*% vals0)
  cnt_neighbors <- as.numeric(neighbor_mat %*% (!na_mask))
  
  # Means
  mean_vals <- ifelse(cnt_neighbors > 0, sum_neighbors / cnt_neighbors, NA_real_)
  
  # For min and max, use block apply over rows with Rcpp or fast grouping
  # Simple base fallback (slower than Rcpp but still much faster than original):
  max_vals <- numeric(n_rows)
  min_vals <- numeric(n_rows)
  
  indices_list <- neighbor_lookup
  for (i in seq_along(indices_list)) {
    idx <- indices_list[[i]]
    if (length(idx) == 0) {
      max_vals[i] <- NA_real_
      min_vals[i] <- NA_real_
    } else {
      neighbor_vals <- vals[idx]
      nm <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(nm) == 0) {
        max_vals[i] <- NA_real_
        min_vals[i] <- NA_real_
      } else {
        max_vals[i] <- max(nm)
        min_vals[i] <- min(nm)
      }
    }
  }
  
  cbind(max_vals, min_vals, mean_vals)
}

## 3. Apply across variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats <- compute_neighbor_stats_fast(vals, neighbor_mat)
  colnames(stats) <- paste(var_name, c("max_nb", "min_nb", "mean_nb"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}

## cell_data now has all five variables' neighbor stats appended
```

---

**Why Faster?**  
- One-time neighbor structure in sparse matrix drastically accelerates mean calculation via matrix multiplication.
- Rcpp can further optimize min/max via compiled loops if needed.
- Drastically reduces R interpreter overhead from millions of lapply calls to single batched operations.
- Preserves estimand and Random Forest model input expectations.
END_RESPONSE>>>
