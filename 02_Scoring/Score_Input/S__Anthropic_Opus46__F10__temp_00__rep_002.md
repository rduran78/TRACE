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
run_id: Anthropic_F10_T00_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a ~6.46M-element character vector â€” this alone is catastrophically slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a pure-R `lapply`**, extracting subsets of a numeric vector and computing `max/min/mean`. The per-element overhead of R function calls, subsetting, and `NA` removal dominates.

3. **The topology is year-invariant but the lookup is rebuilt as if it's year-specific.** Rook neighbors are a spatial property â€” the same adjacency holds for every year. The current code re-resolves neighbor cell IDs into row indices per cell-year, which is redundant.

**Estimated complexity:** ~6.46M Ã— (string operations + named vector lookups + R-level aggregation) â‰ˆ 86+ hours.

## Optimization Strategy

1. **Build the spatial adjacency once as a sparse matrix (CSC format).** The ~344K cells and ~1.37M directed edges become a sparse `dgCMatrix`. This is built once and costs negligible time/memory.

2. **Reshape each variable into a 344,208 Ã— 28 matrix (cells Ã— years).** This allows vectorized column-wise (per-year) sparse matrixâ€“vector operations.

3. **Compute neighbor aggregates via sparse matrix algebra:**
   - **`neighbor_max`**: No direct sparse-matrix shortcut, but we can iterate over years (28 iterations) and use the sparse structure efficiently via `data.table` or direct CSC traversal.
   - **`neighbor_min`**: Same approach.
   - **`neighbor_mean`**: `A %*% X / degree_vector` â€” fully vectorized sparse matrix multiplication.
   - For max and min, we use a grouped operation on the sparse adjacency's `(i, j)` structure with `data.table`, which is extremely fast.

4. **Melt back to long format and join to the original `cell_data`.** This preserves the original row order and numerical equivalence.

**Expected speedup:** From 86+ hours to **~2â€“5 minutes** on 16 GB RAM.

## Working R Code

```r
# =============================================================================
# Optimized Neighbor Feature Engineering
# â€” Sparse graph neighborhood aggregation in R
# â€” Numerically equivalent to the original pipeline
# =============================================================================

library(Matrix)
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars) {
  
  # -------------------------------------------------------------------------
  # 0. Convert cell_data to data.table for fast operations (by reference)
  # -------------------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
    was_df <- TRUE
  } else {
    was_df <- FALSE
  }
  
  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))
  n_years <- length(years)
  
  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(cell_data)))
  
  # -------------------------------------------------------------------------
  # 1. Build sparse adjacency matrix from the nb object (once)
  #    A[i, j] = 1 means j is a rook neighbor of i
  #    (row i aggregates over its column entries)
  # -------------------------------------------------------------------------
  cat("Building sparse adjacency matrix...\n")
  
  # Expand the nb list into (from, to) edge pairs
  from_idx <- rep(seq_along(rook_neighbors_unique),
                  lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique)
  
  # Remove 0-neighbor entries (nb objects use integer(0) for islands)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  # Sparse adjacency: rows = target node, cols = neighbor node
  A <- sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = 1,
    dims = c(n_cells, n_cells),
    repr = "C"   # CSC for efficient column operations; we'll also use triplet
  )
  
  # Degree vector (number of neighbors per node) for mean computation
  degree_vec <- as.numeric(rowSums(A))  # length n_cells
  
  cat(sprintf("Adjacency: %d directed edges\n", length(from_idx)))
  
  # -------------------------------------------------------------------------
  # 2. Create a mapping from (id, year) -> row index in cell_data
  #    and from cell_id -> spatial index (position in id_order)
  # -------------------------------------------------------------------------
  cat("Building index mappings...\n")
  
  # Spatial index: cell_id -> position in id_order (1..n_cells)
  id_to_spatial <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add spatial index to cell_data
  cell_data[, spatial_idx := id_to_spatial[as.character(id)]]
  
  # Year index: year -> position 1..n_years
  year_to_col <- setNames(seq_along(years), as.character(years))
  cell_data[, year_idx := year_to_col[as.character(year)]]
  
  # -------------------------------------------------------------------------
  # 3. For each variable, build a (n_cells x n_years) matrix, compute
  #    neighbor max/min/mean, and write results back
  # -------------------------------------------------------------------------
  
  # Pre-extract the adjacency in triplet form for max/min computation
  A_T <- as(A, "TsparseMatrix")
  edge_from <- A_T@i + 1L  # 1-based row indices (target nodes)
  edge_to   <- A_T@j + 1L  # 1-based col indices (neighbor nodes)
  n_edges   <- length(edge_from)
  
  # data.table for grouped aggregation of max/min
  # We reuse this structure for every variable and year
  edge_dt <- data.table(from = edge_from, to = edge_to)
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))
    t0 <- proc.time()
    
    # --- 3a. Pivot variable into (n_cells x n_years) matrix ----------------
    # Fill with NA by default
    V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    V[cbind(cell_data$spatial_idx, cell_data$year_idx)] <- cell_data[[var_name]]
    
    # --- 3b. Compute neighbor MEAN via sparse matrix multiplication --------
    #   mean_mat = (A %*% V) / degree_vec
    #   A %*% V gives sum of neighbor values; divide by degree for mean
    AV <- as.matrix(A %*% V)  # n_cells x n_years, dense result
    
    mean_mat <- AV / degree_vec  # recycling: degree_vec is length n_cells
    # Where degree is 0, this gives NaN or Inf; set to NA
    mean_mat[degree_vec == 0, ] <- NA_real_
    
    # --- 3c. Compute neighbor MAX and MIN via grouped edge aggregation -----
    #   For each year, look up neighbor values via edge_to, group by edge_from
    max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    for (yy in seq_len(n_years)) {
      # Neighbor values for this year, indexed by edge_to
      nv <- V[edge_to, yy]
      
      # Build a temporary data.table with (target_node, neighbor_value)
      # Filter out NAs before aggregation
      valid_mask <- !is.na(nv)
      if (sum(valid_mask) == 0L) next
      
      tmp <- data.table(
        node = edge_from[valid_mask],
        val  = nv[valid_mask]
      )
      
      # Grouped aggregation â€” extremely fast in data.table
      agg <- tmp[, .(vmax = max(val), vmin = min(val)), by = node]
      
      max_mat[agg$node, yy] <- agg$vmax
      min_mat[agg$node, yy] <- agg$vmin
    }
    
    # --- 3d. Also fix mean where all neighbor values are NA ----------------
    #   AV sum is 0 when all neighbors are NA (since NA is not in sparse mult,
    #   but we used dense V). Actually, Matrix %*% propagates 0 for NA only
    #   if the matrix has 0 there. We need to handle NAs properly.
    #
    #   Correct approach: compute count of non-NA neighbors per cell-year,
    #   and sum of non-NA neighbor values.
    
    # Recompute mean properly handling NAs:
    V_nona <- V
    V_nona[is.na(V_nona)] <- 0
    
    V_notna_indicator <- matrix(as.numeric(!is.na(V)), nrow = n_cells, ncol = n_years)
    
    sum_mat   <- as.matrix(A %*% V_nona)          # sum of non-NA neighbor values
    count_mat <- as.matrix(A %*% V_notna_indicator) # count of non-NA neighbors
    
    mean_mat <- sum_mat / count_mat  # NaN where count is 0
    mean_mat[count_mat == 0] <- NA_real_
    
    # --- 3e. Write results back to cell_data (long format) -----------------
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    cell_data[, (max_col)  := max_mat[cbind(spatial_idx, year_idx)]]
    cell_data[, (min_col)  := min_mat[cbind(spatial_idx, year_idx)]]
    cell_data[, (mean_col) := mean_mat[cbind(spatial_idx, year_idx)]]
    
    elapsed <- (proc.time() - t0)[3]
    cat(sprintf("  Done in %.1f seconds\n", elapsed))
  }
  
  # -------------------------------------------------------------------------
  # 4. Clean up helper columns
  # -------------------------------------------------------------------------
  cell_data[, c("spatial_idx", "year_idx") := NULL]
  
  if (was_df) {
    cell_data <- as.data.frame(cell_data)
  }
  
  return(cell_data)
}


# =============================================================================
# USAGE
# =============================================================================
#
# # Load pre-existing objects
# load("cell_data.RData")          # cell_data data.frame with ~6.46M rows
# load("id_order.RData")           # id_order: vector of 344,208 cell IDs
# load("rook_neighbors.RData")     # rook_neighbors_unique: spdep nb object
# rf_model <- readRDS("rf_model.rds")  # Pre-trained Random Forest â€” NOT retrained
#
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# # Run optimized neighbor feature engineering
# cell_data <- optimize_neighbor_features(
#   cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
# )
#
# # Predict with the pre-trained RF (unchanged)
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Topology construction** | String-paste + named-vector lookup per 6.46M rows | Sparse matrix built once from `nb` object |
| **Mean computation** | R-level `lapply` over 6.46M elements | Sparse matrix multiplication `A %*% V` (BLAS-accelerated) |
| **Max/Min computation** | R-level `lapply` with per-element subsetting | `data.table` grouped aggregation over edge list, 28 year iterations |
| **Memory** | ~6.46M-element list of integer vectors | Sparse matrix (~11 MB) + dense matrices (~77 MB each, 344KÃ—28) â€” fits in 16 GB |
| **Estimated runtime** | 86+ hours | **~2â€“5 minutes** |
| **Numerical equivalence** | Baseline | âœ… Identical `max`, `min`, `mean` with proper NA handling |
| **RF model** | Pre-trained | âœ… **Not retrained** â€” used only for `predict()` |

The key insight is that **rook adjacency is time-invariant**, so the graph topology (sparse matrix) is built once and reused across all 28 years. The mean is computed via a single sparse matrix multiply (O(nnz) per year, ~1.37M operations), while max/min use `data.table`'s optimized grouped aggregation over the same edge structure. This eliminates all per-row string operations and named-vector lookups that dominated the original runtime.
END_RESPONSE>>>
