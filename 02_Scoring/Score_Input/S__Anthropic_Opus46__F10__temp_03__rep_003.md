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
run_id: Anthropic_F10_T03_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each constructed via string-pasting and named-vector lookups. The `paste(id, year)` key construction and `idx_lookup[neighbor_keys]` named-vector subsetting is O(n) per lookup in the worst case, and doing this 6.46M times is catastrophic. This single function likely accounts for the majority of the 86+ hour runtime.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a pure-R `lapply`**, extracting subsets of a numeric vector and computing `max/min/mean`. While each individual call is cheap, 6.46M Ã— 5 variables = ~32.3 million R-level function invocations with no vectorization.

3. **The neighbor lookup is year-aware but the graph topology is year-invariant.** Rook contiguity is a spatial relationship between cells â€” it doesn't change across years. The current code redundantly encodes the same spatial graph 28 times (once per year per cell), inflating the lookup from ~344K entries to ~6.46M entries.

**Key insight:** The adjacency graph is static. Only the node attributes change per year. We should build the graph topology once over the 344K cells, then use sparse matrixâ€“vector multiplication to compute neighborhood aggregates in vectorized operations.

---

## Optimization Strategy

### Core Idea: Sparse Matrix Aggregation

For each cell `i` with rook neighbors `N(i)`, we need:
- `max(x[N(i)])`, `min(x[N(i)])`, `mean(x[N(i)])`

**Mean** is directly computable via sparse matrixâ€“vector product: if `W` is the row-normalized adjacency matrix, then `W %*% x = mean of neighbors` for each node. If `A` is the binary adjacency matrix and `d` is the degree vector, then `mean_i = (A %*% x)[i] / d[i]`.

**Max and min** cannot be computed via standard matrix multiplication, but we can compute them efficiently year-by-year over the 344K cells using the sparse adjacency structure with compiled C++ code via `Rcpp`.

### Plan

1. **Build a sparse adjacency matrix once** from the `spdep::nb` object (344K Ã— 344K, ~1.37M nonzeros). This is trivial and instant.

2. **Reshape data** so that for each year, we have a vector of length 344K aligned to the cell ordering.

3. **Compute `mean`** via sparse matrixâ€“vector product (`Matrix::sparseMatrix %*% x` then divide by degree). This is fully vectorized C code under the hood.

4. **Compute `max` and `min`** via a small Rcpp function that iterates over the CSR structure of the sparse matrix â€” one pass per variable per year.

5. **Process year-by-year** to keep memory bounded (344K Ã— 5 variables per year â‰ˆ negligible memory).

### Expected Speedup

- **Topology build:** From ~hours to <1 second (one sparse matrix construction).
- **Aggregation:** From ~hours to seconds. 28 years Ã— 5 variables Ã— 344K cells with sparse ops â‰ˆ a few seconds total for mean, and ~10â€“30 seconds total for max/min via Rcpp.
- **Total estimated time:** Under 2 minutes for the entire neighbor feature computation.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original max/min/mean neighbor stats.
# =============================================================================

library(Matrix)
library(data.table)
library(Rcpp)

# ---- Step 0: Rcpp function for sparse max/min (CSC format) ------------------
# We write a single Rcpp function that, given a sparse adjacency matrix in CSC
# format (which is CSR of the transpose), computes max, min, mean per row.
# Since we need row-wise aggregation and Matrix stores in CSC (column-compressed),
# we transpose A so that column j of A^T contains the neighbors of node j,
# then iterate over columns.

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix sparse_neighbor_stats(
    IntegerVector Ap,    // column pointers of CSC matrix (A transposed, so these are "row pointers" of A)
    IntegerVector Ai,    // row indices of CSC matrix
    NumericVector x,     // attribute vector aligned to node order
    int n                // number of nodes
) {
  // Output: n x 3 matrix [max, min, mean]
  NumericMatrix out(n, 3);

  for (int j = 0; j < n; j++) {
    int start = Ap[j];
    int end   = Ap[j + 1];
    int count = 0;
    double vmax = NA_REAL;
    double vmin = NA_REAL;
    double vsum = 0.0;

    for (int k = start; k < end; k++) {
      int neighbor = Ai[k];
      double val = x[neighbor];
      if (!NumericVector::is_na(val)) {
        if (count == 0) {
          vmax = val;
          vmin = val;
          vsum = val;
        } else {
          if (val > vmax) vmax = val;
          if (val < vmin) vmin = val;
          vsum += val;
        }
        count++;
      }
    }

    if (count == 0) {
      out(j, 0) = NA_REAL;
      out(j, 1) = NA_REAL;
      out(j, 2) = NA_REAL;
    } else {
      out(j, 0) = vmax;
      out(j, 1) = vmin;
      out(j, 2) = vsum / count;
    }
  }

  return out;
}
')

# ---- Step 1: Build sparse adjacency matrix from spdep::nb object -----------

build_adjacency_csc <- function(nb_obj) {
  # nb_obj: list of length n, where nb_obj[[i]] is integer vector of neighbor
  # indices (1-based) for node i. 0 means no neighbors (spdep convention).
  n <- length(nb_obj)

  # Build COO triplets (i, j) meaning "j is a neighbor of i"
  from <- integer(0)
  to   <- integer(0)

  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep uses 0L to indicate no neighbors
    nbrs <- nbrs[nbrs != 0L]
    if (length(nbrs) > 0) {
      from <- c(from, rep(i, length(nbrs)))
      to   <- c(to, nbrs)
    }
  }

  # Sparse matrix: A[i,j] = 1 means j is neighbor of i
  # We want row-wise aggregation over columns.
  # Transpose so that column i of At contains neighbors of node i.
  A <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n), giveCsparse = TRUE)
  At <- t(A)  # CSC format: column i has the neighbor indices of node i

  return(list(A = A, At = At, n = n))
}

# ---- Step 2: Main pipeline -------------------------------------------------

run_neighbor_aggregation <- function(cell_data, id_order, rook_neighbors_unique) {

  cat("Building sparse adjacency matrix...\n")
  adj <- build_adjacency_csc(rook_neighbors_unique)
  At  <- adj$At
  n_cells <- adj$n

  # Extract CSC internals of At (0-based indices as used by Matrix package)
  At_p <- At@p        # column pointers (length n_cells + 1)
  At_i <- At@i        # row indices (0-based)

  # Convert cell_data to data.table for fast manipulation
  dt <- as.data.table(cell_data)

  # Build mapping from cell id to position in id_order (1-based node index)
  id_to_node <- setNames(seq_along(id_order), as.character(id_order))

  # Add node index column
  dt[, node_idx := id_to_node[as.character(id)]]

  # Verify alignment
  stopifnot(!anyNA(dt$node_idx))

  # Sort by year and node_idx for efficient processing
  setkey(dt, year, node_idx)

  # Get unique years
  years <- sort(unique(dt$year))

  # Neighbor source variables
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  # Pre-allocate result columns
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
  }

  cat("Computing neighbor statistics year-by-year...\n")

  for (yr in years) {
    cat("  Year:", yr, "\n")

    # Get rows for this year, ordered by node_idx
    yr_mask <- dt$year == yr
    yr_dt   <- dt[yr_mask]

    # Build a full-length vector for each variable (length = n_cells)
    # Cells not present in this year get NA
    for (var_name in neighbor_source_vars) {
      # Initialize full vector with NA
      x_full <- rep(NA_real_, n_cells)
      x_full[yr_dt$node_idx] <- yr_dt[[var_name]]

      # Compute sparse neighbor stats via Rcpp
      stats <- sparse_neighbor_stats(At_p, At_i, x_full, n_cells)
      # stats is n_cells x 3: [max, min, mean]

      # Map back to the year subset using node_idx
      node_indices <- yr_dt$node_idx

      max_col  <- paste0(var_name, "_neighbor_max")
      min_col  <- paste0(var_name, "_neighbor_min")
      mean_col <- paste0(var_name, "_neighbor_mean")

      # Write results back into dt for the rows of this year
      set(dt, which(yr_mask), max_col,  stats[node_indices, 1])
      set(dt, which(yr_mask), min_col,  stats[node_indices, 2])
      set(dt, which(yr_mask), mean_col, stats[node_indices, 3])
    }
  }

  cat("Neighbor aggregation complete.\n")

  # Convert back to data.frame if needed, drop helper column
  dt[, node_idx := NULL]
  return(as.data.frame(dt))
}

# ---- Step 3: Execute and predict --------------------------------------------

# Load pre-existing objects (assumed already in environment or loaded from disk)
# cell_data              : data.frame with ~6.46M rows
# id_order               : vector of 344,208 cell IDs matching rook_neighbors_unique
# rook_neighbors_unique  : spdep::nb object (list of length 344,208)
# rf_model               : pre-trained Random Forest model (DO NOT retrain)

# Run optimized neighbor aggregation
cell_data <- run_neighbor_aggregation(cell_data, id_order, rook_neighbors_unique)

# Predict using the pre-trained Random Forest (unchanged)
cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Verification of Numerical Equivalence

```r
# =============================================================================
# OPTIONAL: Verify numerical equivalence on a subsample
# =============================================================================

verify_equivalence <- function(cell_data_original, cell_data_optimized,
                               neighbor_source_vars, tol = 1e-10) {
  # Sample a subset of rows
  set.seed(42)
  sample_idx <- sample(nrow(cell_data_original), min(10000, nrow(cell_data_original)))

  all_ok <- TRUE
  for (var_name in neighbor_source_vars) {
    for (suffix in c("_neighbor_max", "_neighbor_min", "_neighbor_mean")) {
      col <- paste0(var_name, suffix)
      orig <- cell_data_original[[col]][sample_idx]
      optm <- cell_data_optimized[[col]][sample_idx]

      # Compare, treating NA == NA as TRUE
      matches <- (is.na(orig) & is.na(optm)) | (!is.na(orig) & !is.na(optm) & abs(orig - optm) < tol)
      if (!all(matches)) {
        cat("MISMATCH in", col, ": ", sum(!matches), "differences\n")
        all_ok <- FALSE
      } else {
        cat("OK:", col, "\n")
      }
    }
  }

  if (all_ok) cat("\nAll checks passed. Numerical equivalence confirmed.\n")
  return(all_ok)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Topology build** | ~6.46M string-paste + named-vector lookups | One sparse matrix from `nb` object (<1s) |
| **Aggregation** | 6.46M Ã— 5 R-level `lapply` calls | 28 Ã— 5 sparse CSC traversals in compiled C++ |
| **Memory** | ~6.46M-element list of integer vectors | One 344K Ã— 344K sparse matrix (~11 MB) + one 344K vector per pass |
| **Estimated time** | 86+ hours | **< 2 minutes** |
| **Numerical result** | Original | Identical (same max/min/mean, same NA handling) |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |
END_RESPONSE>>>
