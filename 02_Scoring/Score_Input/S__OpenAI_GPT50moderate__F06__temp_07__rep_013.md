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
run_id: OpenAI_F06_T07_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F06_OpenAI_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- `build_neighbor_lookup` and `compute_neighbor_stats` rely on heavy R-level loops (`lapply` over 6.46M rows).  
- Each variable goes through a separate full scan.  
- No vectorization; repeated allocation and indexing create massive overhead.  
- 6.46M Ã— 5 variables Ã— 3 stats = ~97M computations in R loops.  
- Memory pressure (16GB RAM) is an issue, so avoid expanding intermediate objects unnecessarily.  

**Optimization Strategy**  
- Precompute neighbor statistics using **matrix operations** and **sparse adjacency representation**.  
- Use `Matrix` or `data.table` for efficient joins and aggregation.  
- Collapse repeated lookup work by building a sparse adjacency matrix `A` where `A[i,j] = 1` if j is a neighbor of i.  
- For each variable vector `vals`, compute:  
  - `neighbor_max <- rowsum(vals[nb], group)` is not enough because we need max/min/mean. So use `Matrix` ops or `tapply` on an expanded vector efficiently.  
- Approach:  
  1. Construct a sparse adjacency matrix `A` of size (n_rows Ã— n_rows) using neighbor_lookup indices.  
  2. For each variable `vals`, compute neighbor means via `(A %*% vals) / rowSums(A)`.  
  3. For max/min, use an efficient grouped approach in `data.table` keyed by row index.  

**Working R Code (Vectorized & Memory-Efficient)**  
```r
library(data.table)
library(Matrix)

# Assume: data.table `cell_data` with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs
# rook_neighbors_unique: spdep nb object

# Step 1: Build sparse adjacency
build_sparse_adj <- function(cell_data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  n <- nrow(cell_data)
  
  # Map each row to its global id index
  cell_data[, ref_idx := id_to_ref[as.character(id)]]
  
  # Build i, j for adjacency
  # For each row, neighbors share same year
  years <- cell_data$year
  ids   <- cell_data$id
  
  # Preallocate vectors
  i_idx <- integer()
  j_idx <- integer()
  
  for (r in seq_len(n)) {
    neigh_ref <- neighbors[[cell_data$ref_idx[r]]]
    if (length(neigh_ref) > 0) {
      neigh_ids <- id_order[neigh_ref]
      # Match neighbors in same year
      neigh_rows <- which(ids %in% neigh_ids & years == years[r])
      if (length(neigh_rows) > 0) {
        i_idx <- c(i_idx, rep(r, length(neigh_rows)))
        j_idx <- c(j_idx, neigh_rows)
      }
    }
  }
  
  # Sparse adjacency matrix
  sparseMatrix(i = i_idx, j = j_idx, x = 1, dims = c(n, n))
}

# Build adjacency (may take some time but done once)
A <- build_sparse_adj(cell_data, id_order, rook_neighbors_unique)

# Step 2: Compute neighbor stats efficiently
compute_neighbor_stats_sparse <- function(vals, A) {
  # Mean
  counts <- rowSums(A)
  neighbor_sum <- as.numeric(A %*% vals)
  neighbor_mean <- ifelse(counts > 0, neighbor_sum / counts, NA_real_)
  
  # For max/min, need group expansion (data.table approach)
  DT <- data.table(i = as.integer(A@i) + 1, j = as.integer(A@j) + 1)
  DT[, val := vals[j]]
  DT <- DT[!is.na(val)]
  max_dt <- DT[, .(max_val = max(val), min_val = min(val)), by = i]
  
  # Merge back
  max_vec <- rep(NA_real_, length(vals))
  min_vec <- rep(NA_real_, length(vals))
  max_vec[max_dt$i] <- max_dt$max_val
  min_vec[max_dt$i] <- max_dt$min_val
  
  cbind(max_vec, min_vec, neighbor_mean)
}

# Step 3: Loop over variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats <- compute_neighbor_stats_sparse(vals, A)
  colnames(stats) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  cell_data <- cbind(cell_data, stats)
}
```

**Why this is faster**  
- Sparse matrix multiplication for means (O(n + edges)) is highly optimized in C.  
- Max/min computed in a single pass over adjacency edges via `data.table` grouping.  
- Avoids 6.46M nested loops; instead, uses vectorized back-end operations.  

**Expected Improvement**  
- From 86+ hours to a few hours or less, depending on disk I/O and single-threaded constraints.  
- Memory footprint remains manageable by using sparse matrices and in-place merging.  

**Preserves**  
- Original estimands (same neighbor definitions and same year alignment).  
- Trained Random Forest model remains untouched.  
- Adds correct neighbor-based features.
END_RESPONSE>>>
