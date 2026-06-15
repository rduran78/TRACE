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
run_id: Anthropic_F10_T00_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query in the worst case, and with 6.46M keys the hash table construction and repeated `paste`/lookup is extremely expensive.

2. **`compute_neighbor_stats` iterates over 6.46 million list elements in an R-level `lapply`**, extracting subsets of a numeric vector and computing `max/min/mean` per element. This is pure interpreted R with no vectorization.

3. **The neighbor topology is year-invariant** (rook neighbors don't change across years), but the lookup is rebuilt monolithically across all cell-years, duplicating the same adjacency structure 28 times. The string-key join (`paste(id, year)`) is the most expensive single operation.

**Estimated complexity**: ~6.46M list elements Ã— 5 variables Ã— 3 stats = ~97M scalar aggregations, but the overhead is dominated by the R-level loop and string operations, not the arithmetic. On a 16 GB laptop, the 86+ hour estimate is credible.

## Optimization Strategy

1. **Separate topology from time.** The rook neighbor graph has 344,208 nodes and ~1.37M directed edges. This is a **static sparse graph**. Build it once as a sparse adjacency structure (CSR format via `dgRMatrix` or integer vectors of row-pointers and column-indices).

2. **Reshape data to a matrix**: 344,208 rows (cells) Ã— 28 columns (years) per variable. Neighbor aggregation then becomes **sparse matrixâ€“dense matrix multiplication** (and analogous operations for max/min), which is massively vectorized.

3. **For `mean`**: `neighbor_mean = (A %*% X) / (A %*% 1-matrix)` where `A` is the binary adjacency matrix. This is a single sparse matrix multiply â€” runs in seconds via the `Matrix` package (CHOLMOD/CSparse backend in C).

4. **For `max` and `min`**: There is no direct sparse-matrix primitive, but we can iterate over the CSR structure in C++ via `Rcpp` to compute row-wise max/min of neighbor values in a single pass. This replaces 6.46M R-level list lookups with a tight C++ loop.

5. **Memory**: The sparse adjacency matrix is ~1.37M non-zeros Ã— 12 bytes â‰ˆ 16 MB. Each variable matrix is 344,208 Ã— 28 Ã— 8 bytes â‰ˆ 77 MB. Total for 5 variables: ~400 MB. Well within 16 GB.

6. **Expected speedup**: From 86+ hours to **minutes** (sparse matrix multiply for mean; Rcpp loop for max/min).

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original max/min/mean statistics.
# Preserves the pre-trained Random Forest model (no retraining).
# =============================================================================

library(Matrix)
library(Rcpp)
library(data.table)

# ---- Step 0: Compile the Rcpp workhorse for row-wise max/min ----

sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// Compute row-wise max, min, mean over neighbor entries of a dense matrix,
// given a CSR (compressed sparse row) adjacency structure.
// p: integer vector of length (n_nodes + 1), row pointers (0-based)
// j: integer vector of length nnz, column indices (0-based cell indices)
// X: numeric matrix of dimension (n_nodes x n_years)
// Returns a list of three matrices: max_mat, min_mat, mean_mat,
// each of dimension (n_nodes x n_years).

// [[Rcpp::export]]
List neighbor_stats_csr(IntegerVector p, IntegerVector j,
                        NumericMatrix X) {
  int n = X.nrow();
  int T = X.ncol();
  NumericMatrix max_mat(n, T);
  NumericMatrix min_mat(n, T);
  NumericMatrix mean_mat(n, T);

  for (int i = 0; i < n; i++) {
    int start = p[i];
    int end   = p[i + 1];
    int degree = end - start;

    if (degree == 0) {
      for (int t = 0; t < T; t++) {
        max_mat(i, t)  = NA_REAL;
        min_mat(i, t)  = NA_REAL;
        mean_mat(i, t) = NA_REAL;
      }
      continue;
    }

    for (int t = 0; t < T; t++) {
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      double vsum = 0.0;
      int    cnt  = 0;

      for (int k = start; k < end; k++) {
        double val = X(j[k], t);
        if (!R_IsNA(val) && !ISNAN(val)) {
          if (val > vmax) vmax = val;
          if (val < vmin) vmin = val;
          vsum += val;
          cnt++;
        }
      }

      if (cnt == 0) {
        max_mat(i, t)  = NA_REAL;
        min_mat(i, t)  = NA_REAL;
        mean_mat(i, t) = NA_REAL;
      } else {
        max_mat(i, t)  = vmax;
        min_mat(i, t)  = vmin;
        mean_mat(i, t) = vsum / (double)cnt;
      }
    }
  }

  return List::create(Named("max") = max_mat,
                      Named("min") = min_mat,
                      Named("mean") = mean_mat);
}
')

# ---- Step 1: Build the sparse adjacency matrix ONCE ----
# rook_neighbors_unique: spdep nb object (list of integer vectors, 1-indexed)
# id_order: vector of cell IDs in the order matching the nb object

build_adjacency_csr <- function(nb_obj) {
  # nb_obj is a list of length n_nodes.
  # nb_obj[[i]] contains integer indices of neighbors of node i (1-based).
  # A zero-element (integer(0) or 0L) means no neighbors.
  n <- length(nb_obj)

  # Build COO then convert to CSR via Matrix package
  from <- integer(0)
  to   <- integer(0)

  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nbrs <- nbrs[nbrs != 0L]
    if (length(nbrs) > 0) {
      from <- c(from, rep(i, length(nbrs)))
      to   <- c(to, nbrs)
    }
  }

  # Create sparse matrix (dgCMatrix is CSC; we need CSR for row-wise ops)
  A <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n), repr = "R")
  # repr = "R" gives dgRMatrix (CSR format)
  return(A)
}

cat("Building sparse adjacency matrix...\n")
A_csr <- build_adjacency_csr(rook_neighbors_unique)
cat("  Nodes:", nrow(A_csr), " Edges:", length(A_csr@j), "\n")

# Extract CSR components for Rcpp (0-based indexing)
csr_p <- A_csr@p          # row pointers, already 0-based (length n+1)
csr_j <- A_csr@j          # column indices, already 0-based

# ---- Step 2: Convert cell_data to data.table for fast reshaping ----

cat("Converting to data.table...\n")
cell_dt <- as.data.table(cell_data)

# Ensure consistent cell ordering matching the nb object
# id_order[k] is the cell ID for the k-th node in the nb object
cell_id_to_node <- setNames(seq_along(id_order), as.character(id_order))

# Add node index and sort
cell_dt[, node_idx := cell_id_to_node[as.character(id)]]

# Get sorted unique years
years_sorted <- sort(unique(cell_dt$year))
n_years <- length(years_sorted)
n_nodes <- length(id_order)

cat("  Cells:", n_nodes, " Years:", n_years,
    " Rows:", nrow(cell_dt), "\n")

# Create a year-to-column-index mapping
year_to_col <- setNames(seq_along(years_sorted), as.character(years_sorted))

# ---- Step 3: For each variable, reshape to matrix, compute stats, merge back ----

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-compute a key for fast assignment back to cell_dt
# We need (node_idx, year) -> row in cell_dt
setkey(cell_dt, node_idx, year)

for (var_name in neighbor_source_vars) {
  cat("Processing variable:", var_name, "...\n")

  # Reshape to n_nodes x n_years matrix
  # Initialize with NA
  X <- matrix(NA_real_, nrow = n_nodes, ncol = n_years)

  # Fill the matrix
  # Use integer indexing for speed
  row_indices <- cell_dt$node_idx
  col_indices <- year_to_col[as.character(cell_dt$year)]
  vals        <- cell_dt[[var_name]]

  X[cbind(row_indices, col_indices)] <- vals

  # Compute neighbor stats via Rcpp CSR traversal
  stats <- neighbor_stats_csr(csr_p, csr_j, X)
  # stats$max, stats$min, stats$mean are each n_nodes x n_years matrices

  # Extract values back to cell_dt row order
  result_max  <- stats$max[cbind(row_indices, col_indices)]
  result_min  <- stats$min[cbind(row_indices, col_indices)]
  result_mean <- stats$mean[cbind(row_indices, col_indices)]

  # Assign new columns
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  cell_dt[, (max_col)  := result_max]
  cell_dt[, (min_col)  := result_min]
  cell_dt[, (mean_col) := result_mean]

  cat("  Done:", var_name, "\n")
}

# ---- Step 4: Clean up and convert back to data.frame for prediction ----

cell_dt[, node_idx := NULL]
cell_data <- as.data.frame(cell_dt)

cat("Neighbor feature engineering complete.\n")
cat("  Columns:", ncol(cell_data), "\n")
cat("  Rows:", nrow(cell_data), "\n")

# ---- Step 5: Apply the pre-trained Random Forest model ----
# The model object (e.g., `rf_model`) is already in memory.
# It was trained on the original feature set and must NOT be retrained.

cat("Generating predictions with pre-trained Random Forest...\n")
cell_data$prediction <- predict(rf_model, newdata = cell_data)
cat("Pipeline complete.\n")
```

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + named-vector lookups | Single sparse matrix build (344K nodes, 1.37M edges) |
| **Aggregation loop** | R-level `lapply` over 6.46M elements Ã— 5 vars | C++ CSR traversal: 5 calls, each ~1.37M edges Ã— 28 years |
| **Data structure** | Repeated `paste(id, year)` string keys | Integer matrix indexing `[node, year]` |
| **Memory** | ~6.46M-element list of integer vectors | ~16 MB sparse matrix + 5 Ã— 77 MB dense matrices â‰ˆ 400 MB |
| **Expected runtime** | 86+ hours | **~2â€“5 minutes** |
| **Numerical equivalence** | Baseline | âœ… Identical `max`, `min`, `mean` with same NA handling |
| **RF model** | Pre-trained | âœ… Unchanged, predict-only |

**Key insight**: The topology is time-invariant. By separating the spatial graph (344K nodes, 1.37M edges) from the temporal dimension (28 years), we convert 6.46 million R-level list operations into a single C++ loop over a CSR structure operating on a compact `(nodes Ã— years)` matrix. The sparse matrix is built once and reused across all 5 variables. The `mean` could alternatively be computed as a sparse matrix multiply (`A %*% X / rowSums(A)`), but the unified Rcpp function handles `max/min/mean` together with proper NA semantics in a single pass, avoiding redundant memory traffic.
END_RESPONSE>>>
