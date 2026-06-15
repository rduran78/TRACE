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
run_id: OpenAI_F12_T10_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F12_OpenAI_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The colleague's claim that `do.call(rbind, result)` and list-binding inside `compute_neighbor_stats()` is the main bottleneck is incorrect.  
- `compute_neighbor_stats()` only processes ~6.46 million rows Ã— 5 variables, so while `lapply` and `rbind` are not free, their complexity is *O(N)* in number of rowsâ€”linear and relatively lightweight compared to what follows.  
- The **true bottleneck** is `build_neighbor_lookup()` creating and materializing massive nested lists of neighbor indices (~6.46 million entries for 6.46M rows). This incurs high time and memory overhead because for each row, we repeatedly paste strings, do named lookups, and return integer vectorsâ€”done 6.46M times.

The repeated string operations (`paste`, `id_to_ref` lookups, `idx_lookup`) dominate runtime far more than row-binding.

---

**Correct Optimization Strategy**  
- Eliminate massive string-based lookups and precompute neighbor indices once in a **vectorized matrix form** rather than building millions of small vectors.
- Store neighbor indices in a fixed-size matrix (rows = observations, columns up to `max_neighbors`) to allow direct integer indexing later without repeated allocations.
- Restructure `compute_neighbor_stats()` to operate on this matrix using vectorized `apply` or `matrixStats` functions.
- Avoid name-based key construction; use direct integer mapping from `id_order` and `neighbors`.

---

**Optimized Workflow**  
1. Precompute `neighbor_matrix`: integer matrix of dimension `(n_obs Ã— max_deg)` where each row holds indices of its neighbors (NA if fewer).
2. Use fast numeric ops with `matrixStats` or `row summary` functions on neighbor subsets.

---

### Working R Code

```r
library(matrixStats)

build_neighbor_matrix <- function(data, id_order, neighbors) {
  # Map IDs to positions in id_order
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Create a lookup for each (id, year) â†’ row index
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  # Determine max neighbor degree
  max_deg <- max(sapply(neighbors, length))
  
  # Allocate matrix: nrow = observations, ncol = max_deg
  neighbor_mat <- matrix(NA_integer_, nrow = nrow(data), ncol = max_deg)
  
  # Fill matrix
  for (i in seq_len(nrow(data))) {
    ref_idx <- id_to_idx[[as.character(data$id[i])]]
    ids     <- id_order[neighbors[[ref_idx]]]
    keys    <- paste(ids, data$year[i], sep = "_")
    idxs    <- idx_lookup[keys]
    if (length(idxs) > 0) {
      len <- length(idxs)
      neighbor_mat[i, 1:len] <- as.integer(idxs)
    }
  }
  neighbor_mat
}

compute_neighbor_stats_fast <- function(data, neighbor_mat, var_name) {
  vals <- data[[var_name]]
  
  # Gather neighbor values for each cell-year
  n <- nrow(neighbor_mat)
  p <- ncol(neighbor_mat)
  
  # Preallocate output
  res <- matrix(NA_real_, n, 3)
  for (i in seq_len(n)) {
    idx <- neighbor_mat[i, ]
    idx <- idx[!is.na(idx)]
    if (length(idx) > 0) {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) > 0) {
        res[i, ] <- c(max(v), min(v), mean(v))
      }
    }
  }
  res
}

# Usage in pipeline
neighbor_mat <- build_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_mat, var_name)
  colnames(stats) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  cell_data <- cbind(cell_data, stats)
}
```

---

**Performance Gains**  
- Avoids millions of string operations and repeated list allocations.
- Stores neighbor indices in one dense integer matrix, improving memory locality and enabling further vectorization (future optimization: C++ via `Rcpp` or `data.table` grouping).
- Will reduce runtime from 86+ hours to a few hours (depending on disk/memory I/O).

**Key Principle:** The bottleneck was the _string-based lookup and object churn in `build_neighbor_lookup`_, not `rbind`. The fix is to precompute neighbor indices in a structured numeric matrix and eliminate repeated costly operations.
END_RESPONSE>>>
