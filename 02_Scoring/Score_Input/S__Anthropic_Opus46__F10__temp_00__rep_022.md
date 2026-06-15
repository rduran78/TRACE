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
run_id: Anthropic_F10_T00_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query in the worst case, and doing this 6.46M times with string concatenation is extremely slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a serial `lapply`**, extracting subsets of a vector by index, removing NAs, and computing three summary statistics. This is pure R-level looping with no vectorization.

3. **The neighbor lookup is year-coupled but the graph topology is year-invariant.** The rook-neighbor structure is purely spatial (344,208 cells), yet the code rebuilds lookups across all 9.64M cell-year rows, inflating the problem by 28Ã—. The topology should be built once over cells, and statistics computed per year using vectorized sparse-matrix operations.

**Core insight:** The neighbor aggregation (max, min, mean) is equivalent to sparse matrix operations on a 344,208 Ã— 344,208 adjacency matrix, applied independently to each of 28 year-slices for each of 5 variables. This reduces the problem from 6.46M list iterations to 28 Ã— 5 = 140 sparse matrix operations, each on a ~344K-node graph with ~1.37M edges.

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from the `nb` object (344,208 Ã— 344,208, ~1.37M non-zero entries). This is tiny in memory (~16 MB).

2. **Reshape each variable into a 344,208 Ã— 28 matrix** (cells Ã— years), aligning cell order with the adjacency matrix.

3. **Compute neighbor mean** via sparse matrix multiplication: `A %*% X / degree`, where `degree` is the row-sum of A (number of neighbors per cell).

4. **Compute neighbor max and min** using a custom sparse-row-aggregation function that iterates over CSC/CSR structure in C++ via `Rcpp`, or using a chunked R approach with the sparse matrix's slot structure. Alternatively, use `data.table` grouped operations on the edge list for max/min (which `data.table` handles extremely efficiently).

5. **Reassemble** the 15 new columns back into the panel `data.table`, preserving the original row order and numerical values exactly.

**Expected speedup:** From 86+ hours to ~2â€“5 minutes.

## Optimized R Code

```r
# =============================================================================
# Optimized Neighbor Aggregation Pipeline
# Preserves numerical equivalence with original compute_neighbor_stats output.
# =============================================================================

library(data.table)
library(Matrix)

optimize_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                  "def", "usd_est_n2")) {

  # --- Convert to data.table if needed, preserve original row order -----------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  cell_data[, .row_order := .I]

  n_cells <- length(id_order)
  stopifnot(n_cells == length(rook_neighbors_unique))

  # --- Step 1: Build sparse adjacency matrix ONCE (topology is year-invariant)
  # rook_neighbors_unique is an nb object: list of integer vectors of neighbor indices
  # into id_order. We build a sparse matrix A where A[i,j]=1 if j is a rook neighbor of i.

  message("Building sparse adjacency matrix from nb object...")
  edge_from <- integer(0)
  edge_to   <- integer(0)

  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    if (length(nb_i) == 1L && nb_i[1] == 0L) next
    edge_from <- c(edge_from, rep.int(i, length(nb_i)))
    edge_to   <- c(edge_to, nb_i)
  }

  # Pre-allocate more efficiently using vapply to get lengths
  nb_lengths <- vapply(rook_neighbors_unique, function(nb_i) {
    if (length(nb_i) == 1L && nb_i[1] == 0L) 0L else length(nb_i)
  }, integer(1))

  total_edges <- sum(nb_lengths)
  edge_from <- integer(total_edges)
  edge_to   <- integer(total_edges)

  pos <- 1L
  for (i in seq_len(n_cells)) {
    n_nb <- nb_lengths[i]
    if (n_nb == 0L) next
    idx_range <- pos:(pos + n_nb - 1L)
    edge_from[idx_range] <- i
    edge_to[idx_range]   <- rook_neighbors_unique[[i]]
    pos <- pos + n_nb
  }

  # Sparse adjacency matrix (rows = focal cell, cols = neighbor cell)
  A <- sparseMatrix(i = edge_from, j = edge_to, x = 1,
                    dims = c(n_cells, n_cells), repr = "C")  # CSC format

  # Edge list as data.table for max/min computation
  edges_dt <- data.table(from = edge_from, to = edge_to)

  # Degree vector (number of neighbors per cell) for mean computation
  degree <- diff(A@p)  # For CSC of t(A); we need row-sums of A
  # Actually for CSC (column-compressed), column sums = diff(A@p)
  # We need row sums. Use:
  degree_vec <- as.integer(rowSums(A))  # fast for sparse

  # --- Step 2: Create cell-index mapping ----------------------------------------
  # Map id_order to matrix row indices (1:n_cells)
  id_to_matrow <- setNames(seq_len(n_cells), as.character(id_order))

  # Map each row in cell_data to its matrix row index
  cell_data[, .mat_row := id_to_matrow[as.character(id)]]

  # Get sorted unique years
  years <- sort(unique(cell_data$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))
  cell_data[, .year_col := year_to_col[as.character(year)]]

  message(sprintf("Graph: %d nodes, %d directed edges, %d years, %d variables",
                  n_cells, total_edges, n_years, length(neighbor_source_vars)))

  # --- Step 3: For each variable, compute neighbor max, min, mean ---------------

  for (var_name in neighbor_source_vars) {
    message(sprintf("Processing variable: %s", var_name))

    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    # Build cell Ã— year matrix (n_cells rows Ã— n_years cols)
    # Fill with NA
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(cell_data$.mat_row, cell_data$.year_col)] <- cell_data[[var_name]]

    # --- Compute MEAN via sparse matrix multiplication (per year column) --------
    # For each year t: neighbor_mean[i,t] = sum_j A[i,j]*X[j,t] / degree[i]
    # But we must handle NAs: original code drops NAs before computing mean.
    # If all neighbor values are non-NA (common case), sparse matmul is exact.
    # For full correctness with potential NAs, we need:
    #   sum of non-NA neighbor values / count of non-NA neighbor values

    # Create indicator matrix: 1 where X is not NA, 0 otherwise
    X_nona <- X
    X_nona[is.na(X_nona)] <- 0

    X_indicator <- matrix(0, nrow = n_cells, ncol = n_years)
    X_indicator[!is.na(X)] <- 1

    # Sparse matmul: sum of neighbor values (treating NA as 0)
    neighbor_sum   <- as.matrix(A %*% X_nona)        # n_cells Ã— n_years
    neighbor_count <- as.matrix(A %*% X_indicator)    # n_cells Ã— n_years

    # Mean = sum / count; NA where count == 0
    mean_mat <- neighbor_sum / neighbor_count
    mean_mat[neighbor_count == 0] <- NA_real_

    # --- Compute MAX and MIN via edge-list grouped aggregation ------------------
    # This is the most efficient R approach for sparse max/min.
    # For each year, look up neighbor values via the edge list and group-aggregate.

    max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (t in seq_len(n_years)) {
      x_t <- X[, t]  # values for this year, indexed by mat_row

      # Look up neighbor values
      nb_vals <- x_t[edges_dt$to]

      # Build temporary data.table with from-node and neighbor value
      tmp <- data.table(from = edges_dt$from, val = nb_vals)

      # Remove NAs
      tmp <- tmp[!is.na(val)]

      if (nrow(tmp) == 0L) next

      # Grouped max and min
      agg <- tmp[, .(vmax = max(val), vmin = min(val)), by = from]

      max_mat[agg$from, t] <- agg$vmax
      min_mat[agg$from, t] <- agg$vmin
    }

    # --- Step 4: Map results back to cell_data rows -----------------------------
    idx_mat <- cbind(cell_data$.mat_row, cell_data$.year_col)

    set(cell_data, j = col_max,  value = max_mat[idx_mat])
    set(cell_data, j = col_min,  value = min_mat[idx_mat])
    set(cell_data, j = col_mean, value = mean_mat[idx_mat])

    message(sprintf("  -> Added %s, %s, %s", col_max, col_min, col_mean))
  }

  # --- Cleanup temporary columns ------------------------------------------------
  cell_data[, c(".row_order", ".mat_row", ".year_col") := NULL]

  message("Neighbor aggregation complete.")
  return(cell_data)
}


# =============================================================================
# USAGE
# =============================================================================
#
# # Load data (assumed already in memory or loaded from disk)
# # cell_data:                data.frame/data.table with columns: id, year, ntl, ec, ...
# # id_order:                 vector of cell IDs matching rook_neighbors_unique indexing
# # rook_neighbors_unique:    spdep::nb object (list of neighbor index vectors)
# # rf_model:                 pre-trained Random Forest model (DO NOT retrain)
#
# cell_data <- optimize_neighbor_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   neighbor_source_vars   = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
#
# # Apply the pre-trained Random Forest (unchanged)
# predictions <- predict(rf_model, newdata = cell_data)
# =============================================================================
```

## Further Optimization: Rcpp for Max/Min (Optional Drop-In)

The `data.table` grouped aggregation for max/min across 28 years is already fast (~1â€“2 minutes total), but if even that is a concern, here is an Rcpp version that processes all years in one pass:

```r
# Optional: Rcpp-accelerated max/min over sparse neighbors
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_neighbor_maxmin(IntegerVector from, IntegerVector to,
                            NumericMatrix X, int n_cells) {
  // from, to: 0-based edge indices
  // X: n_cells x n_years matrix
  int n_years = X.ncol();
  int n_edges = from.size();

  NumericMatrix max_mat(n_cells, n_years);
  NumericMatrix min_mat(n_cells, n_years);
  LogicalMatrix has_val(n_cells, n_years);

  // Initialize
  std::fill(max_mat.begin(), max_mat.end(), R_NegInf);
  std::fill(min_mat.begin(), min_mat.end(), R_PosInf);
  std::fill(has_val.begin(), has_val.end(), false);

  for (int e = 0; e < n_edges; e++) {
    int i = from[e];  // focal node (0-based)
    int j = to[e];    // neighbor node (0-based)
    for (int t = 0; t < n_years; t++) {
      double v = X(j, t);
      if (!NumericMatrix::is_na(v)) {
        has_val(i, t) = true;
        if (v > max_mat(i, t)) max_mat(i, t) = v;
        if (v < min_mat(i, t)) min_mat(i, t) = v;
      }
    }
  }

  // Set NA where no valid neighbor value existed
  for (int i = 0; i < n_cells; i++) {
    for (int t = 0; t < n_years; t++) {
      if (!has_val(i, t)) {
        max_mat(i, t) = NA_REAL;
        min_mat(i, t) = NA_REAL;
      }
    }
  }

  return List::create(Named("max_mat") = max_mat,
                      Named("min_mat") = min_mat);
}
')

# To use inside the pipeline, replace the per-year max/min loop with:
# res <- sparse_neighbor_maxmin(edges_dt$from - 1L, edges_dt$to - 1L, X, n_cells)
# max_mat <- res$max_mat
# min_mat <- res$min_mat
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Topology construction** | String-paste + named-vector lookup over 6.46M rows | Sparse matrix built once over 344K cells (~1.37M entries) |
| **Mean computation** | R-level `lapply` over 6.46M elements | Sparse matrix multiplication (`A %*% X`), fully vectorized |
| **Max/Min computation** | R-level `lapply` with per-element subsetting | `data.table` grouped aggregation on edge list (or Rcpp) |
| **Redundant work** | Topology re-resolved per cell-year | Topology built once, reused across 28 years Ã— 5 variables |
| **Estimated runtime** | 86+ hours | ~2â€“5 minutes (data.table) or ~30 seconds (Rcpp) |
| **Memory** | ~6.46M-element list of integer vectors | ~16 MB sparse matrix + ~77 MB per variable matrix |
| **Numerical equivalence** | Baseline | Exact: same max, min, mean with identical NA handling |
| **RF model** | Unchanged | Unchanged â€” no retraining |
END_RESPONSE>>>
