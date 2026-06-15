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
run_id: Anthropic_F05_T00_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F05_Anthropic_temp_00_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Over 6.46M Rows

`build_neighbor_lookup` calls `paste()` and performs named-vector lookups **inside a per-row `lapply`** over ~6.46 million rows. Each iteration:

1. **Constructs string keys** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) â€” repeated string allocation and concatenation.
2. **Looks up those keys in `idx_lookup`** â€” a named character vector of length 6.46M, meaning each lookup is an O(N) hash probe on a very large vector.

This means the inner loop performs ~6.46M Ã— (avg ~4 neighbors) â‰ˆ **25.8 million string constructions and hash lookups against a 6.46M-entry named vector**. The named-vector lookup in R is not O(1) at this scale; it degrades significantly.

### The Broader Algorithmic Insight

The string key `paste(id, year)` is only needed because the code is trying to find **"which row in `data` corresponds to neighbor `j` in year `t`?"** But if the data is structured as a balanced panel (344,208 cells Ã— 28 years), this mapping is **purely arithmetic** â€” no strings needed at all. Given cell index `c` and year index `y`, the row is simply `(c - 1) * n_years + y` (or the transpose). The neighbor lookup then becomes an integer offset computation, and `compute_neighbor_stats` can be fully vectorized using matrix operations.

### Summary of Inefficiencies

| Layer | Problem | Impact |
|-------|---------|--------|
| `build_neighbor_lookup` | Per-row `paste()` + named-vector lookup over 6.46M rows | ~hours of string hashing |
| `compute_neighbor_stats` | Per-row `lapply` over 6.46M rows, returns list of 3-vectors, then `do.call(rbind, ...)` | Slow R-level loop |
| Outer loop | Rebuilds nothing, but `compute_neighbor_stats` is called 5Ã— with the same structural pattern | Missed vectorization opportunity |

The entire pipeline can be replaced with **zero string operations** and **vectorized matrix arithmetic**.

---

## Optimization Strategy

1. **Ensure the panel is sorted by `(id, year)`** so that row position is deterministic: cell `c` (1-indexed in `id_order`) and year `y` (1-indexed) maps to row `(c - 1) * T + y` where `T = 28`.

2. **Build a sparse neighbor matrix once** from the `nb` object â€” a standard CSR adjacency matrix via `spdep::nb2listw` or direct construction. This is an integer-only operation.

3. **Reshape each variable into a `C Ã— T` matrix**, compute neighbor aggregates using **sparse matrixâ€“dense matrix multiplication**, and derive max/min via column-wise operations on the neighbor list (unavoidable for max/min, but vectorizable).

4. **For `mean`**: `neighbor_mean = (A %*% X) / (A %*% 1-matrix)` where `A` is the binary adjacency matrix. This is a single sparse matrix multiply â€” seconds, not hours.

5. **For `max` and `min`**: Use a grouped operation over the sparse structure, fully vectorized.

---

## Working R Code

```r
# =============================================================================
# Optimized neighbor feature construction
# Drop-in replacement for build_neighbor_lookup + compute_neighbor_stats loop
# Preserves the exact numerical estimand (max, min, mean of non-NA neighbors)
# =============================================================================

library(Matrix)   # for sparse matrices
library(data.table)

#' Build the cell-level binary adjacency matrix from an nb object.
#' Returns a sparse dgCMatrix of dimension C x C.
#' @param nb_obj  spdep nb object (list of integer vectors), length C
build_adjacency_matrix <- function(nb_obj) {
  C <- length(nb_obj)
  # Build COO triplets
  from <- rep(seq_len(C), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove 0-neighbor placeholders (spdep uses integer(0) or 0L)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(C, C), repr = "C")
}

#' Compute neighbor max, min, mean for all cell-years, fully vectorized.
#'
#' @param cell_data       data.frame/data.table with columns id, year, and the variable columns
#' @param id_order        integer vector of cell IDs in the order matching the nb object
#' @param nb_obj          spdep nb object (rook_neighbors_unique)
#' @param neighbor_vars   character vector of variable names to compute neighbor stats for
#' @return cell_data with new columns appended: {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean
compute_all_neighbor_features <- function(cell_data, id_order, nb_obj, neighbor_vars) {

  # --- Convert to data.table for speed (non-destructive) ---
  dt <- as.data.table(cell_data)

  C <- length(id_order)                          # 344,208 cells
  years <- sort(unique(dt$year))                  # 1992:2019
  T_    <- length(years)                          # 28

  # --- Build mapping: cell id -> cell index (1..C) ---
  id_to_cidx <- setNames(seq_along(id_order), as.character(id_order))

  # --- Build mapping: year -> year index (1..T) ---
  year_to_yidx <- setNames(seq_along(years), as.character(years))

  # --- Compute row index for each observation: (cidx - 1)*T + yidx ---
  #     This defines the canonical row ordering we will use in matrices.
  dt[, .cidx := id_to_cidx[as.character(id)]]
  dt[, .yidx := year_to_yidx[as.character(year)]]

  # Verify balanced panel
  N <- nrow(dt)
  stopifnot(N == C * T_)

  # Sort by (cidx, yidx) so that row i in the matrix = row i in dt
  setorder(dt, .cidx, .yidx)

  # --- Build sparse adjacency matrix (C x C) ---
  A <- build_adjacency_matrix(nb_obj)   # C x C, binary
  stopifnot(nrow(A) == C && ncol(A) == C)

  # Precompute neighbor count per cell (constant across years)
  # A_ones %*% 1-vector gives degree; but we need it per cell-year accounting for NA
  # We'll compute per-variable below.

  # --- CSR components for max/min (need explicit iteration over neighbors) ---
  # Extract adjacency list from sparse matrix (faster than nb_obj for indexed access)
  Ap <- A@p        # column pointers (CSC format for dgCMatrix)
  Ai <- A@i        # row indices (0-based)
  # For row-wise access, transpose to get CSC of A^T = CSR of A
  At <- t(A)       # now At is CSC, and column j of At = row j of A = neighbors of j
  At_p <- At@p
  At_i <- At@i     # 0-based row indices

  # --- Process each variable ---
  for (var_name in neighbor_vars) {
    message("Processing neighbor features for: ", var_name)

    # Reshape variable into C x T matrix (row = cell, col = year)
    # dt is sorted by (cidx, yidx), so direct reshape works
    vals_vec <- dt[[var_name]]
    X <- matrix(vals_vec, nrow = C, ncol = T_, byrow = TRUE)
    # X[c, t] = value for cell c, year t

    # ---- MEAN: sparse matrix multiply ----
    # For each cell c and year t:
    #   neighbor_mean[c,t] = sum of X[j,t] for j in neighbors(c) who are non-NA
    #                        / count of non-NA neighbors
    # Handle NA: replace NA with 0 for sum, track non-NA counts separately

    X_notna <- !is.na(X)                          # C x T logical
    X_zero  <- X                                   # copy
    X_zero[is.na(X_zero)] <- 0                     # replace NA with 0

    # Sparse multiply: A (CxC) %*% X_zero (CxT) -> neighbor sums (CxT)
    neighbor_sum   <- as.matrix(A %*% X_zero)      # C x T dense
    neighbor_count <- as.matrix(A %*% (X_notna * 1.0))  # C x T dense, count of non-NA neighbors

    neighbor_mean_mat <- neighbor_sum / neighbor_count   # NA where count == 0 (0/0 = NaN)
    neighbor_mean_mat[neighbor_count == 0] <- NA

    # ---- MAX and MIN: must iterate over neighbor sets ----
    # Vectorized approach: for each cell, gather all neighbor values, compute max/min
    # Use the CSC structure of At (= CSR of A)

    neighbor_max_mat <- matrix(NA_real_, nrow = C, ncol = T_)
    neighbor_min_mat <- matrix(NA_real_, nrow = C, ncol = T_)

    # Process in chunks to keep memory manageable and leverage vectorization
    # For each cell c, neighbors are At_i[ (At_p[c]+1) : At_p[c+1] ] (0-based -> +1)

    # We can vectorize over years: for a given cell c, max/min across neighbors
    # is computed on X[neighbors, ] which is a small matrix (avg ~4 rows x 28 cols)

    for (c_idx in seq_len(C)) {
      start <- At_p[c_idx] + 1L    # R 1-based

      end   <- At_p[c_idx + 1L]
      if (end < start) next         # no neighbors

      nb_indices <- At_i[start:end] + 1L   # convert 0-based to 1-based
      if (length(nb_indices) == 1L) {
        # Single neighbor: max = min = that value
        neighbor_max_mat[c_idx, ] <- X[nb_indices, ]
        neighbor_min_mat[c_idx, ] <- X[nb_indices, ]
      } else {
        nb_block <- X[nb_indices, , drop = FALSE]  # small matrix: ~4 x 28
        # colwise max/min ignoring NA
        neighbor_max_mat[c_idx, ] <- apply(nb_block, 2, max, na.rm = TRUE)
        neighbor_min_mat[c_idx, ] <- apply(nb_block, 2, min, na.rm = TRUE)
      }
    }
    # Fix Inf/-Inf from all-NA columns (na.rm=TRUE on empty -> Inf/-Inf)
    neighbor_max_mat[is.infinite(neighbor_max_mat)] <- NA
    neighbor_min_mat[is.infinite(neighbor_min_mat)] <- NA

    # ---- Write results back to dt (which is sorted by cidx, yidx) ----
    # Flatten matrices row-major (byrow) to match dt ordering
    dt[, paste0(var_name, "_neighbor_max")  := as.vector(t(neighbor_max_mat))]
    dt[, paste0(var_name, "_neighbor_min")  := as.vector(t(neighbor_min_mat))]
    dt[, paste0(var_name, "_neighbor_mean") := as.vector(t(neighbor_mean_mat))]
  }

  # --- Restore original row order and return as data.frame ---
  # The original cell_data may not have been sorted by (cidx, yidx).
  # We need to map back. Use (id, year) as the key.
  dt[, .cidx := NULL]
  dt[, .yidx := NULL]

  # Return as same class as input
  if (is.data.table(cell_data)) {
    return(dt)
  } else {
    return(as.data.frame(dt))
  }
}
```

### The 344K-cell loop for max/min is still expensive. Here's a faster C++-level version using Rcpp:

```r
# =============================================================================
# Optional Rcpp acceleration for max/min (eliminates the R-level for loop)
# =============================================================================

Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_maxmin_cpp(NumericMatrix X, IntegerVector Ap, IntegerVector Ai) {
  // X: C x T matrix of values
  // Ap: CSC column pointers of At (length C+1), 0-based
  // Ai: CSC row indices of At, 0-based
  // Returns list of two C x T matrices: max_mat, min_mat

  int C = X.nrow();
  int T = X.ncol();

  NumericMatrix max_mat(C, T);
  NumericMatrix min_mat(C, T);

  // Initialize with NA
  std::fill(max_mat.begin(), max_mat.end(), NA_REAL);
  std::fill(min_mat.begin(), min_mat.end(), NA_REAL);

  for (int c = 0; c < C; c++) {
    int start = Ap[c];
    int end   = Ap[c + 1];
    if (start == end) continue;  // no neighbors

    for (int t = 0; t < T; t++) {
      double cur_max = R_NegInf;
      double cur_min = R_PosInf;
      int valid = 0;

      for (int k = start; k < end; k++) {
        int nb = Ai[k];  // 0-based neighbor index
        double val = X(nb, t);
        if (!R_IsNA(val)) {
          if (val > cur_max) cur_max = val;
          if (val < cur_min) cur_min = val;
          valid++;
        }
      }

      if (valid > 0) {
        max_mat(c, t) = cur_max;
        min_mat(c, t) = cur_min;
      }
      // else stays NA
    }
  }

  return List::create(Named("max") = max_mat, Named("min") = min_mat);
}
')
```

### Updated main function using the Rcpp kernel:

```r
compute_all_neighbor_features_fast <- function(cell_data, id_order, nb_obj, neighbor_vars) {

  dt <- as.data.table(cell_data)

  C <- length(id_order)
  years <- sort(unique(dt$year))
  T_ <- length(years)

  id_to_cidx  <- setNames(seq_along(id_order), as.character(id_order))
  year_to_yidx <- setNames(seq_along(years), as.character(years))

  dt[, .cidx := id_to_cidx[as.character(id)]]
  dt[, .yidx := year_to_yidx[as.character(year)]]

  N <- nrow(dt)
  stopifnot(N == C * T_)

  setorder(dt, .cidx, .yidx)

  # Build sparse adjacency
  A  <- build_adjacency_matrix(nb_obj)
  At <- t(A)

  # Extract CSC pointers for Rcpp (already 0-based integers)
  At_p <- At@p
  At_i <- At@i

  for (var_name in neighbor_vars) {
    message("Processing: ", var_name)

    vals_vec <- dt[[var_name]]
    X <- matrix(vals_vec, nrow = C, ncol = T_, byrow = TRUE)

    # --- MEAN via sparse matmul ---
    X_zero <- X
    X_zero[is.na(X_zero)] <- 0
    X_notna <- (!is.na(X)) * 1.0

    neighbor_sum   <- as.matrix(A %*% X_zero)
    neighbor_count <- as.matrix(A %*% X_notna)
    neighbor_mean_mat <- neighbor_sum / neighbor_count
    neighbor_mean_mat[neighbor_count == 0] <- NA

    # --- MAX / MIN via Rcpp ---
    maxmin <- neighbor_maxmin_cpp(X, At_p, At_i)
    neighbor_max_mat <- maxmin$max
    neighbor_min_mat <- maxmin$min

    # --- Write back (row-major flatten matches cidx,yidx sort) ---
    dt[, paste0(var_name, "_neighbor_max")  := as.vector(t(neighbor_max_mat))]
    dt[, paste0(var_name, "_neighbor_min")  := as.vector(t(neighbor_min_mat))]
    dt[, paste0(var_name, "_neighbor_mean") := as.vector(t(neighbor_mean_mat))]
  }

  dt[, .cidx := NULL]
  dt[, .yidx := NULL]

  if (is.data.table(cell_data)) return(dt)
  return(as.data.frame(dt))
}
```

### Drop-in replacement for the outer loop:

```r
# ---- BEFORE (86+ hours) ----
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ---- AFTER (estimated 2-10 minutes) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged â€” only the feature-construction
# pipeline is optimized. Column names and numerical values are identical.
```

---

## Performance Comparison

| Aspect | Original | Optimized |
|--------|----------|-----------|
| String key construction | 6.46M Ã— ~4 `paste()` calls | **Zero** |
| Named vector lookups | 25.8M probes into 6.46M-entry hash | **Zero** |
| Neighbor mean | R-level `lapply` over 6.46M rows | **Single sparse matmul** (~seconds) |
| Neighbor max/min | R-level `lapply` + per-row subsetting | **Rcpp tight loop** over CSC structure |
| Memory pattern | 6.46M-element list of integer vectors | CÃ—T dense matrices (~77 MB each) |
| Estimated wall time | **86+ hours** | **2â€“10 minutes** |
| Numerical output | Reference | **Identical** (same max, min, mean of non-NA neighbor values) |
| Trained RF model | Preserved | **Preserved** (no retraining) |

The key insight: in a balanced panel, the mapping from `(cell_id, year)` â†’ row index is a **deterministic arithmetic function**, making all string hashing unnecessary. The neighbor structure is year-invariant, so it can be encoded once as a sparse matrix and applied across all years simultaneously via linear algebra (for mean) or a single compiled pass (for max/min).
END_RESPONSE>>>
