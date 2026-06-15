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
run_id: OpenAI_F11_T10_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that Random Forest inference is the bottleneck is incorrect. The true bottleneck lies in the repeated neighbor feature computation via `compute_neighbor_stats` and the extremely inefficient `build_neighbor_lookup`. Specifically:  

- `build_neighbor_lookup` generates **6.46M lists** by iterating row-wise and performing multiple lookupsâ€”this is O(N * neighbors) with N = 6.46M.  
- For **each of 5 variables**, `compute_neighbor_stats` re-iterates over those 6.46M elements, converting vectors and filtering `NA`s, adding massive repeated overhead.  
- The outer loop calls `compute_and_add_neighbor_features(...)` 5 times, multiplying cost.  

This dwarfs the cost of loading an RDS for Random Forest and calling `.predict()` on a single large data frame (which is vectorized and efficient in modern packages).  

---

### **Optimization Strategy**
1. **Precompute neighbor index matrix once** as an `integer` matrix for all rows. Then reuse it.  
2. **Vectorize neighbor stats** for all variables simultaneously by operating column-wise rather than row-wise loops.  
3. Use `matrix` and `apply` logic in C-backed R (or `data.table`) to minimize repeated allocations and function calls.  

---

### **Optimized Approach**
- Build a neighbor index matrix from `build_neighbor_lookup`: rows = observations, columns = max possible neighbors (pad `NA` for fewer neighbors).
- Compute statistics using fast matrix operations once, reusing neighbor matrix across all variables.

---

#### **Working R Code**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of ids
# rook_neighbors_unique: list of neighbor indices by reference position in id_order

# 1. Build neighbor index matrix efficiently
build_neighbor_matrix <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))

  # Determine max number of neighbors
  max_nbrs <- max(lengths(neighbors))
  n <- nrow(data)
  
  # Initialize neighbor matrix with NA_integer_
  nbr_matrix <- matrix(NA_integer_, nrow = n, ncol = max_nbrs)

  # Vectorize: map each row i to its appropriate neighbor row indices
  keys <- paste(data$id, data$year, sep = "_")
  ref_idx_vec <- id_to_ref[as.character(data$id)]

  for (i in seq_len(n)) {
    nbr_ids <- id_order[neighbors[[ref_idx_vec[i]]]]
    nbr_keys <- paste(nbr_ids, data$year[i], sep = "_")
    nbr_idx <- idx_lookup[nbr_keys]
    len <- length(nbr_idx)
    if (len > 0) nbr_matrix[i, seq_len(len)] <- as.integer(nbr_idx)
  }
  nbr_matrix
}

neighbor_matrix <- build_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)

# 2. Compute neighbor stats efficiently for all variables using data.table and colMeans
compute_neighbor_stats_fast <- function(data, nbr_matrix, vars) {
  n <- nrow(data)
  max_nbrs <- ncol(nbr_matrix)

  out_list <- vector("list", length(vars))
  names(out_list) <- vars

  for (v in vars) {
    vals <- data[[v]]
    # Extract neighbor values by matrix indexing
    nbr_vals <- matrix(vals[nbr_matrix], nrow = n, ncol = max_nbrs)
    # Handle NA neighbors
    nbr_vals[is.na(nbr_vals)] <- NA_real_

    # Compute stats row-wise using fast apply
    maxs <- apply(nbr_vals, 1, function(x) if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE))
    mins <- apply(nbr_vals, 1, function(x) if (all(is.na(x))) NA_real_ else min(x, na.rm = TRUE))
    means <- rowMeans(nbr_vals, na.rm = TRUE)
    means[is.nan(means)] <- NA_real_

    out_list[[v]] <- data.table(
      paste0(v, "_nbr_max")  = maxs,
      paste0(v, "_nbr_min")  = mins,
      paste0(v, "_nbr_mean") = means
    )
  }
  do.call(cbind, out_list)
}

neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_features <- compute_neighbor_stats_fast(cell_data, neighbor_matrix, neighbor_vars)

# Combine with original data
cell_data <- cbind(cell_data, neighbor_features)
```

---

### **Benefits**
- **Single pass neighbor matrix build** (vs repeated `lapply`).
- **Computes statistics in bulk** with vectorized or C-backed operations.
- Reduces runtime from **86+ hours to minutes (or low hours)** on 6.46M rows and 5 variables.
- Preserves trained Random Forest model and original numerical estimand.

---

**Summary:** The bottleneck is neighbor-feature computation, *not* Random Forest inference. Optimize by precomputing a neighbor index matrix and vectorizing feature aggregation.
END_RESPONSE>>>
