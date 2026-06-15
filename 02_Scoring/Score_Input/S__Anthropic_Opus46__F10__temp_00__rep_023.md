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
run_id: Anthropic_F10_T00_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each via an anonymous function call with string-pasting, named-vector lookups, and `NA` filtering. This is an O(n) loop over millions of rows with expensive string operations (`paste`, named vector indexing via `idx_lookup[neighbor_keys]`). The string-keyed lookup `idx_lookup` has 6.46M entries, making each hash probe expensive at scale.

2. **`compute_neighbor_stats` iterates over 6.46M list elements** per variable, extracting subsets of a numeric vector, removing NAs, and computing `max/min/mean`. This is called 5 times (once per neighbor source variable), totaling ~32.3 million list-element iterations. Each iteration allocates small vectors and calls three summary functions.

3. **The neighbor lookup is year-aware but redundant**: The spatial topology (which cells neighbor which) is identical across all 28 years. Yet the lookup rebuilds year-specific row indices by pasting cell IDs with years. The topology is static; only the attribute values change by year.

**Root cause**: The implementation treats the problem as a flat 6.46M-row table problem instead of exploiting the separable structure: **fixed spatial graph Ã— repeated yearly attributes**.

---

## Optimization Strategy

### Key Insight: Separate Topology from Temporal Attributes

The rook-neighbor graph has 344,208 nodes and ~1.37M directed edges. This topology is **year-invariant**. We should:

1. **Build a sparse adjacency matrix once** (344,208 Ã— 344,208 CSC matrix with ~1.37M nonzeros) from the `spdep::nb` object.
2. **Reshape each variable into a 344,208 Ã— 28 matrix** (cells Ã— years).
3. **Compute neighbor aggregates via sparse matrix operations** or grouped vectorized operations â€” no R-level loops over millions of elements.

### Specific Techniques

| Technique | Speedup Source |
|---|---|
| Sparse matrix from `spdep::nb` â†’ `Matrix::sparseMatrix` | Built once, O(edges) construction |
| Cell Ã— Year matrix layout | Enables column-wise (year-wise) vectorized operations |
| `dgCMatrix` row-wise iteration via `@p`, `@i`, `@x` slots | Avoids R list overhead |
| `data.table` for fast reshaping and joining | Avoids `paste`-based key lookups |
| Compute max/min/mean per node using sparse row traversal | Replaces 6.46M list iterations with vectorized C-level ops |
| Process all 28 years simultaneously per variable | Eliminates year-loop overhead |

### Expected Speedup

- Neighbor lookup construction: from ~hours to ~seconds (sparse matrix construction).
- Neighbor stats: from ~hours per variable to ~seconds per variable (sparse matrixâ€“dense matrix operations for mean; row-wise sparse traversal for max/min).
- **Total: from 86+ hours to under 5 minutes on 16 GB RAM.**

### Memory Budget

- Sparse adjacency matrix: ~1.37M entries Ã— 12 bytes â‰ˆ 16 MB.
- One cellÃ—year matrix: 344,208 Ã— 28 Ã— 8 bytes â‰ˆ 77 MB.
- Five variables Ã— 3 stats Ã— 77 MB â‰ˆ 1.15 GB for output.
- Comfortable within 16 GB.

---

## Optimized R Code

```r
###############################################################################
# optimized_neighbor_pipeline.R
#
# Computes neighbor max, min, mean for 5 variables across a spatial panel,
# using sparse-matrix graph representation. Numerically equivalent to the
# original build_neighbor_lookup + compute_neighbor_stats pipeline.
#
# Requirements: Matrix, data.table packages
###############################################################################

library(Matrix)
library(data.table)

#' Build a sparse binary adjacency matrix from an spdep::nb object.
#'
#' @param nb_obj   An nb object (list of integer vectors of neighbor indices).
#' @param n        Number of spatial units (length of nb_obj).
#' @return A dgCMatrix of dimension n x n with 1s at neighbor positions.
build_adjacency_matrix <- function(nb_obj, n) {
  # Pre-allocate vectors for COO triplets
  # Count total edges
  edge_counts <- vapply(nb_obj, function(x) {
    # spdep::nb encodes "no neighbors" as a single 0L
    if (length(x) == 1L && x[1L] == 0L) 0L else length(x)
  }, integer(1))
  total_edges <- sum(edge_counts)

  from_idx <- integer(total_edges)
  to_idx   <- integer(total_edges)

  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1L] == 0L) next
    k <- length(nbrs)
    from_idx[pos:(pos + k - 1L)] <- i
    to_idx[pos:(pos + k - 1L)]   <- nbrs
    pos <- pos + k
  }

  sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = rep(1, total_edges),
    dims = c(n, n),
    giveCsparse = TRUE
  )
}

#' Compute neighbor max, min, mean for one variable using sparse adjacency.
#'
#' @param adj       dgCMatrix adjacency matrix (n x n). Entry (i,j)=1 means
#'                  j is a rook neighbor of i.
#' @param val_mat   Dense matrix (n x T) of variable values, cells in rows,
#'                  years in columns. Row order must match adj row/col order.
#' @return A list with three matrices (each n x T): nb_max, nb_min, nb_mean.
compute_neighbor_stats_sparse <- function(adj, val_mat) {
  n <- nrow(val_mat)
  T_ <- ncol(val_mat)

  # --- Neighbor mean via sparse matrix multiplication ---
  # adj %*% val_mat gives sum of neighbor values for each node-year.
  # Divide by neighbor count to get mean.
  # Neighbor count per node (row sums of adj); same for all years.
  nb_count <- as.numeric(rowSums(adj))  # length n

  # For nodes with 0 neighbors, we'll set results to NA.
  has_neighbors <- nb_count > 0

  # Sum of neighbor values: sparse %*% dense (Matrix package handles this)
  # But we need to handle NAs: treat NA as missing (exclude from sum and count).
  # To be numerically equivalent to the original code which does:
  #   neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
  #   mean(neighbor_vals)  -- i.e., mean of non-NA neighbors
  #
  # Strategy for mean with NA handling:
  #   sum_non_na = adj %*% val_mat_zero  (where NAs replaced with 0)
  #   count_non_na = adj %*% notna_mat   (count of non-NA neighbors)
  #   mean = sum_non_na / count_non_na

  notna_mat <- matrix(as.numeric(!is.na(val_mat)), nrow = n, ncol = T_)
  val_mat_zero <- val_mat
  val_mat_zero[is.na(val_mat_zero)] <- 0

  nb_sum   <- as.matrix(adj %*% val_mat_zero)    # n x T
  nb_nonna <- as.matrix(adj %*% notna_mat)        # n x T, count of non-NA neighbors

  # Mean
  nb_mean <- nb_sum / nb_nonna  # produces NaN where nb_nonna == 0
  nb_mean[nb_nonna == 0] <- NA_real_
  nb_mean[!has_neighbors, ] <- NA_real_


  # --- Neighbor max and min via row-wise sparse traversal ---
  # We must iterate over rows of adj and compute max/min of val_mat[neighbors, t]
  # for each year t. We use the CSC structure of adj transposed (= CSR of adj).
  #
  # Convert adj to dgRMatrix (CSR) for efficient row traversal, or equivalently
  # transpose and use CSC column traversal.

  adj_t <- t(adj)  # Now adj_t is CSC; column j of adj_t = row j of adj = neighbors of j

  # Extract CSC slots of adj_t
  p_ptr <- adj_t@p        # length n+1, column pointers
  row_i <- adj_t@i + 1L   # 1-based row indices (= neighbor cell indices)

  nb_max <- matrix(NA_real_, nrow = n, ncol = T_)
  nb_min <- matrix(NA_real_, nrow = n, ncol = T_)

  for (j in seq_len(n)) {
    start <- p_ptr[j] + 1L
    end   <- p_ptr[j + 1L]
    if (end < start) next  # no neighbors

    nbr_indices <- row_i[start:end]

    # Extract sub-matrix: neighbors x years
    sub_mat <- val_mat[nbr_indices, , drop = FALSE]  # k x T

    if (length(nbr_indices) == 1L) {
      # Single neighbor: max = min = value (NA if NA)
      nb_max[j, ] <- sub_mat[1L, ]
      nb_min[j, ] <- sub_mat[1L, ]
    } else {
      # colMins / colMaxs with na.rm = TRUE
      # Use matrixStats if available, otherwise apply
      for (tt in seq_len(T_)) {
        v <- sub_mat[, tt]
        v <- v[!is.na(v)]
        if (length(v) == 0L) next
        nb_max[j, tt] <- max(v)
        nb_min[j, tt] <- min(v)
      }
    }
  }

  list(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}


#' Full optimized pipeline: build graph, compute neighbor features, predict.
#'
#' @param cell_data            data.frame/data.table with columns: id, year,
#'                             and all predictor variables.
#' @param id_order             Integer vector of cell IDs in the order matching
#'                             rook_neighbors_unique (the nb object).
#' @param rook_neighbors_unique  spdep::nb object (list of neighbor index vectors).
#' @param rf_model             Pre-trained Random Forest model object.
#' @param neighbor_source_vars Character vector of variable names to aggregate.
#' @return cell_data with neighbor feature columns appended and predictions added.
run_optimized_pipeline <- function(cell_data,
                                   id_order,
                                   rook_neighbors_unique,
                                   rf_model = NULL,
                                   neighbor_source_vars = c("ntl", "ec",
                                                            "pop_density",
                                                            "def",
                                                            "usd_est_n2")) {

  dt <- as.data.table(cell_data)

  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)

  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(dt)))

  # ---- Step 1: Build sparse adjacency matrix (once) ----
  cat("Building sparse adjacency matrix...\n")
  adj <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
  cat(sprintf("  Adjacency: %d x %d, %d nonzeros\n",
              nrow(adj), ncol(adj), nnzero(adj)))

  # ---- Step 2: Create cell-index and year-index mappings ----
  # Map cell IDs to matrix row indices (1..n_cells)
  id_to_row <- setNames(seq_along(id_order), as.character(id_order))
  # Map years to matrix column indices (1..n_years)
  year_to_col <- setNames(seq_along(years), as.character(years))

  # Add matrix coordinates to dt
  dt[, cell_row := id_to_row[as.character(id)]]
  dt[, year_col := year_to_col[as.character(year)]]

  # ---- Step 3: For each variable, reshape â†’ compute â†’ join back ----
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))

    # 3a. Reshape to n_cells x n_years matrix
    val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    val_mat[cbind(dt$cell_row, dt$year_col)] <- dt[[var_name]]

    # 3b. Compute neighbor stats
    stats <- compute_neighbor_stats_sparse(adj, val_mat)

    # 3c. Extract values back to data.table row order
    idx <- cbind(dt$cell_row, dt$year_col)

    max_col  <- paste0("nb_max_", var_name)
    min_col  <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)

    dt[, (max_col)  := stats$nb_max[idx]]
    dt[, (min_col)  := stats$nb_min[idx]]
    dt[, (mean_col) := stats$nb_mean[idx]]

    cat(sprintf("  Added: %s, %s, %s\n", max_col, min_col, mean_col))
  }

  # Clean up temporary columns
  dt[, cell_row := NULL]
  dt[, year_col := NULL]

  # ---- Step 4: Predict with pre-trained Random Forest (no retraining) ----
  if (!is.null(rf_model)) {
    cat("Generating predictions with pre-trained Random Forest...\n")
    dt[, rf_prediction := predict(rf_model, newdata = dt)]
    cat("  Predictions added.\n")
  }

  cat("Pipeline complete.\n")
  return(dt)
}
```

---

### Further Optimization: Vectorized Max/Min with `matrixStats`

The inner loop over nodes for max/min is the remaining bottleneck. Here is a drop-in replacement using `matrixStats` that eliminates the inner year loop:

```r
compute_neighbor_stats_sparse_fast <- function(adj, val_mat) {
  requireNamespace("matrixStats", quietly = TRUE)

  n  <- nrow(val_mat)
  T_ <- ncol(val_mat)

  # --- Mean (same as before, via sparse matmul) ---
  notna_mat    <- matrix(as.numeric(!is.na(val_mat)), nrow = n, ncol = T_)
  val_mat_zero <- val_mat
  val_mat_zero[is.na(val_mat_zero)] <- 0

  nb_sum   <- as.matrix(adj %*% val_mat_zero)

  nb_nonna <- as.matrix(adj %*% notna_mat)

  nb_mean <- nb_sum / nb_nonna
  nb_mean[nb_nonna == 0] <- NA_real_

  nb_count <- as.numeric(rowSums(adj))
  nb_mean[nb_count == 0, ] <- NA_real_

  # --- Max / Min via CSR row traversal, vectorized over years ---
  adj_t <- t(adj)
  p_ptr <- adj_t@p
  row_i <- adj_t@i + 1L

  nb_max <- matrix(NA_real_, nrow = n, ncol = T_)
  nb_min <- matrix(NA_real_, nrow = n, ncol = T_)

  for (j in seq_len(n)) {
    start <- p_ptr[j] + 1L
    end   <- p_ptr[j + 1L]
    if (end < start) next

    nbr_indices <- row_i[start:end]
    sub_mat <- val_mat[nbr_indices, , drop = FALSE]

    if (length(nbr_indices) == 1L) {
      nb_max[j, ] <- sub_mat[1L, ]
      nb_min[j, ] <- sub_mat[1L, ]
    } else {
      nb_max[j, ] <- matrixStats::colMaxs(sub_mat, na.rm = TRUE)
      nb_min[j, ] <- matrixStats::colMins(sub_mat, na.rm = TRUE)
      # colMaxs/colMins return -Inf/Inf when all NA; fix:
      all_na <- matrixStats::colAlls(is.na(sub_mat))
      nb_max[j, all_na] <- NA_real_
      nb_min[j, all_na] <- NA_real_
    }
  }

  list(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}
```

---

### Even Faster: Rcpp for Max/Min (Optional, Eliminates R Loop Entirely)

If the ~344K-node R loop is still too slow (estimated ~2â€“5 min with `matrixStats`), this Rcpp function reduces it to seconds:

```r
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_neighbor_maxmin(IntegerVector p_ptr,
                            IntegerVector row_i,
                            NumericMatrix val_mat) {
  int n  = val_mat.nrow();
  int T_ = val_mat.ncol();

  NumericMatrix nb_max(n, T_);
  NumericMatrix nb_min(n, T_);

  // Initialize to NA
  std::fill(nb_max.begin(), nb_max.end(), NA_REAL);
  std::fill(nb_min.begin(), nb_min.end(), NA_REAL);

  for (int j = 0; j < n; j++) {
    int start = p_ptr[j];
    int end   = p_ptr[j + 1];
    if (end <= start) continue;

    for (int tt = 0; tt < T_; tt++) {
      double cur_max = R_NegInf;
      double cur_min = R_PosInf;
      int count = 0;

      for (int k = start; k < end; k++) {
        double v = val_mat(row_i[k], tt);  // row_i is 0-based from C
        if (R_IsNA(v)) continue;
        if (v > cur_max) cur_max = v;
        if (v < cur_min) cur_min = v;
        count++;
      }

      if (count > 0) {
        nb_max(j, tt) = cur_max;
        nb_min(j, tt) = cur_min;
      }
    }
  }

  return List::create(Named("nb_max") = nb_max,
                      Named("nb_min") = nb_min);
}
')

compute_neighbor_stats_rcpp <- function(adj, val_mat) {
  n  <- nrow(val_mat)
  T_ <- ncol(val_mat)

  # Mean via sparse matmul (already fast)
  notna_mat    <- matrix(as.numeric(!is.na(val_mat)), nrow = n, ncol = T_)
  val_mat_zero <- val_mat
  val_mat_zero[is.na(val_mat_zero)] <- 0

  nb_sum   <- as.matrix(adj %*% val_mat_zero)
  nb_nonna <- as.matrix(adj %*% notna_mat)

  nb_mean <- nb_sum / nb_nonna
  nb_mean[nb_nonna == 0] <- NA_real_
  nb_count <- as.numeric(rowSums(adj))
  nb_mean[nb_count == 0, ] <- NA_real_

  # Max/Min via Rcpp
  adj_t <- t(adj)
  # adj_t@i is 0-based, which is what the C++ code expects
  maxmin <- sparse_neighbor_maxmin(adj_t@p, adj_t@i, val_mat)

  list(nb_max = maxmin$nb_max, nb_min = maxmin$nb_min, nb_mean = nb_mean)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + hash lookups | Sparse matrix built once from `nb` object |
| **Stats computation** | 6.46M Ã— 5 R list iterations | Sparse matmul (mean) + CSR traversal (max/min) |
| **Year handling** | Redundantly encoded in lookup keys | Implicit via matrix columns |
| **Memory** | ~6.46M R list elements (high overhead) | ~16 MB sparse matrix + ~77 MB per variable matrix |
| **Estimated time** | 86+ hours | **2â€“10 minutes** (R); **< 1 minute** (with Rcpp) |
| **Numerical equivalence** | Baseline | âœ… Identical: same NA handling, same max/min/mean |
| **RF model** | Unchanged | âœ… Preserved, predict-only |
END_RESPONSE>>>
