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
run_id: Anthropic_F07_T00_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Per-row string pasting and named-vector lookup** â€” `paste()` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called ~6.46 million times inside an `lapply`. Each call does string allocation, hashing, and matching.
2. **The `idx_lookup` named vector has 6.46 million entries** â€” every single neighbor key lookup must hash a string and probe a 6.46M-entry character hash table. With ~1.37M directed neighbor pairs Ã— 28 years â‰ˆ 38.4M lookups, this is extremely expensive in R's named-vector implementation.
3. **`compute_neighbor_stats`** then iterates over the 6.46M-element list again, extracting subsets of a numeric vector. The per-element `lapply` with small vectors defeats vectorization.

**In short:** The algorithm is O(N Ã— k) with enormous constant factors from R-level string operations and interpreted loops, where N â‰ˆ 6.46M and k â‰ˆ average neighbors â‰ˆ 4.

## Optimization Strategy

### Key Insight: Separate Space from Time

Every cell has the **same** neighbors in **every** year. The neighbor topology is time-invariant. Instead of building a 6.46M-element lookup (one per cell-year), build a **344,208-element** spatial lookup (one per cell), then use **vectorized matrix operations** across all years simultaneously.

### Concrete Plan

1. **Restructure data into a matrix**: rows = cells (344,208), columns = years (28). For each variable, this is a 344K Ã— 28 matrix.
2. **Build a sparse adjacency matrix** (344,208 Ã— 344,208) from the `nb` object â€” this is a one-time operation using `spdep::nb2listw` or direct construction via `Matrix::sparseMatrix`.
3. **Compute neighbor stats via sparse matrix multiplication**:
   - **Neighbor mean** = `(A %*% X) / (A %*% (!is.na(X)))` (sparse mat Ã— dense mat â€” highly optimized in the `Matrix` package).
   - **Neighbor max/min** â€” use row-wise sparse iteration (still far faster than 6.46M R-level list lookups).
4. **Flatten back** to the original long panel format.

This replaces ~86 hours of interpreted R loops with a few seconds of optimized sparse linear algebra.

## Working R Code

```r
library(Matrix)
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build sparse binary adjacency matrix from the nb object (one-time)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_adjacency_matrix <- function(nb_obj) {
  # nb_obj: spdep nb object (list of integer vectors of neighbor indices)
  n <- length(nb_obj)
  # Build COO triplets
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove the 0-neighbor sentinel that spdep uses (integer(0) is fine,

  # but some nb objects store 0L for no-neighbor cells)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

A <- build_adjacency_matrix(rook_neighbors_unique)
# A is 344208 x 344208, ~1.37M non-zero entries â€” tiny in memory (~20 MB)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Convert panel to data.table for fast reshaping
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
dt <- as.data.table(cell_data)

# Ensure a consistent cell ordering that matches the nb object
# id_order is the vector of cell IDs in the same order as rook_neighbors_unique
cell_idx <- setNames(seq_along(id_order), as.character(id_order))

# Add spatial index
dt[, sp_idx := cell_idx[as.character(id)]]

# Sort for consistent matrix construction
setorder(dt, sp_idx, year)

years <- sort(unique(dt$year))
n_cells <- length(id_order)
n_years <- length(years)
year_idx <- setNames(seq_along(years), as.character(years))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Neighbor stats via sparse matrix operations
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# For neighbor MEAN:  A %*% X  gives sum of neighbor values per cell.
#                     A %*% V  (V = !is.na(X)) gives count of non-NA neighbors.
#                     mean = sum / count
#
# For neighbor MAX and MIN: we iterate over the sparse structure once,
# which is far cheaper than 6.46M R-level list operations.

compute_neighbor_features_sparse <- function(dt, A, var_name,
                                             id_order, years,
                                             cell_idx, year_idx) {
  n_cells <- length(id_order)
  n_years <- length(years)

  # Build cell Ã— year matrix for this variable
  # Use the already-sorted dt (by sp_idx, year)
  X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  X[cbind(dt$sp_idx, year_idx[as.character(dt$year)])] <- dt[[var_name]]

  # ---- Neighbor MEAN ----
  valid   <- !is.na(X)
  V       <- matrix(as.numeric(valid), nrow = n_cells, ncol = n_years)
  X_zero  <- X
  X_zero[!valid] <- 0

  neighbor_sum   <- as.matrix(A %*% X_zero)   # n_cells x n_years

  neighbor_count <- as.matrix(A %*% V)
  neighbor_mean  <- neighbor_sum / neighbor_count  # NaN where count==0
  neighbor_mean[neighbor_count == 0] <- NA_real_

  # ---- Neighbor MAX and MIN via sparse row iteration ----
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # Extract CSR-like structure from the dgCMatrix (which is CSC)
  # It's easier to work with the row-oriented view.
  # Convert to dgRMatrix or iterate columns of t(A).
  # Efficient approach: use the slot structure of dgCMatrix (CSC of A).
  # In CSC, column j has entries in rows A@i[  (A@p[j]+1) : A@p[j+1]  ]
  # Since A is the adjacency matrix, column j lists all cells i that have

  # j as a neighbor. But we want: for each row i, the set of columns j
  # that are neighbors. That's the same as iterating rows.
  # Transpose to make rows become columns:
  At <- t(A)  # now column i of At = row i of A = neighbors of cell i

  p <- At@p
  row_indices <- At@i + 1L  # 1-based

  for (i in seq_len(n_cells)) {
    start <- p[i] + 1L
    end   <- p[i + 1L]
    if (end < start) next  # no neighbors
    nbrs <- row_indices[start:end]
    nb_vals <- X[nbrs, , drop = FALSE]  # k x n_years matrix
    # Column-wise max/min ignoring NA
    for (t in seq_len(n_years)) {
      v <- nb_vals[, t]
      v <- v[!is.na(v)]
      if (length(v) > 0L) {
        neighbor_max[i, t] <- max(v)
        neighbor_min[i, t] <- min(v)
      }
    }
  }

  # ---- Return as vectors aligned to dt ----
  idx <- cbind(dt$sp_idx, year_idx[as.character(dt$year)])
  list(
    n_max  = neighbor_max[idx],
    n_min  = neighbor_min[idx],
    n_mean = neighbor_mean[idx]
  )
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Even faster MAX/MIN: vectorized C++ via Rcpp (optional but
#    reduces the max/min loop from ~minutes to seconds)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# If Rcpp is available, replace the double for-loop above:

if (requireNamespace("Rcpp", quietly = TRUE)) {
  Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_neighbor_maxmin(IntegerVector p, IntegerVector i,
                            NumericMatrix X) {
  int n = X.nrow();
  int T = X.ncol();
  NumericMatrix nmax(n, T);
  NumericMatrix nmin(n, T);
  std::fill(nmax.begin(), nmax.end(), NA_REAL);
  std::fill(nmin.begin(), nmin.end(), NA_REAL);

  for (int cell = 0; cell < n; cell++) {
    int start = p[cell];
    int end   = p[cell + 1];
    if (start == end) continue;
    for (int t = 0; t < T; t++) {
      double mx = R_NegInf;
      double mn = R_PosInf;
      bool found = false;
      for (int k = start; k < end; k++) {
        double v = X(i[k], t);
        if (!R_IsNA(v)) {
          if (v > mx) mx = v;
          if (v < mn) mn = v;
          found = true;
        }
      }
      if (found) {
        nmax(cell, t) = mx;
        nmin(cell, t) = mn;
      }
    }
  }
  return List::create(Named("nmax") = nmax, Named("nmin") = nmin);
}
')

  compute_neighbor_features_fast <- function(dt, A, var_name,
                                              id_order, years,
                                              cell_idx, year_idx) {
    n_cells <- length(id_order)
    n_years <- length(years)

    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(dt$sp_idx, year_idx[as.character(dt$year)])] <- dt[[var_name]]

    # Mean via sparse matmul
    valid  <- !is.na(X)
    V      <- matrix(as.numeric(valid), nrow = n_cells, ncol = n_years)
    X_zero <- X; X_zero[!valid] <- 0
    neighbor_sum   <- as.matrix(A %*% X_zero)
    neighbor_count <- as.matrix(A %*% V)
    neighbor_mean  <- neighbor_sum / neighbor_count
    neighbor_mean[neighbor_count == 0] <- NA_real_

    # Max/Min via Rcpp
    At <- t(A)
    mm <- sparse_neighbor_maxmin(At@p, At@i, X)

    idx <- cbind(dt$sp_idx, year_idx[as.character(dt$year)])
    list(
      n_max  = mm$nmax[idx],
      n_min  = mm$nmin[idx],
      n_mean = neighbor_mean[idx]
    )
  }
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. Main loop: compute and attach all neighbor features
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_fn <- if (exists("compute_neighbor_features_fast")) {
  compute_neighbor_features_fast
} else {
  compute_neighbor_features_sparse
}

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  res <- compute_fn(dt, A, var_name, id_order, years, cell_idx, year_idx)

  # Use the same column naming convention as the original pipeline
  set(dt, j = paste0(var_name, "_n_max"),  value = res$n_max)
  set(dt, j = paste0(var_name, "_n_min"),  value = res$n_min)
  set(dt, j = paste0(var_name, "_n_mean"), value = res$n_mean)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6. Convert back to data.frame if needed downstream
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
dt[, sp_idx := NULL]  # remove helper column
cell_data <- as.data.frame(dt)

# The trained Random Forest model is untouched â€” only the feature
# engineering (neighbor stats) was accelerated. The numerical values
# are identical to the original implementation.
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup structure** | 6.46M-element list built via string paste + named-vector probe | 344K Ã— 344K sparse matrix (one-time, ~20 MB) |
| **Neighbor mean** | Per-row `lapply` with subsetting | Sparse matrix Ã— dense matrix (`A %*% X`), fully vectorized BLAS |
| **Neighbor max/min** | Per-row `lapply` (6.46M R-level calls) | Single pass over sparse structure via Rcpp, or pure-R column-wise loop over 344K cells |
| **Memory** | ~6.46M-element list of integer vectors + 6.46M-entry named character vector | ~77 MB matrix (344K Ã— 28 Ã— 8 bytes) + 20 MB sparse matrix |
| **Estimated time** | 86+ hours | **~2â€“5 minutes** (with Rcpp), ~15â€“30 min (pure R) |
| **Numerical result** | â€” | **Identical** (same max, min, mean over same neighbor sets) |
| **Trained RF model** | â€” | **Preserved** (no retraining) |
END_RESPONSE>>>
