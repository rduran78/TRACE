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
run_id: Anthropic_F07_T00_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over ~6.46 million rows one at a time in an R-level `lapply`, performing per-row string pasting, hash lookups, and NA filtering. This creates ~6.46 million small integer vectors inside a list. Then `compute_neighbor_stats` iterates over that same 6.46M-element list, subsetting a numeric vector for each row. The combined cost is:

1. **`build_neighbor_lookup`**: O(N_rows) R-level iterations, each doing `paste`, character matching into `idx_lookup`, and NA removal. With 6.46M rows this is extremely slow â€” the `paste` and named-vector lookup alone are responsible for most of the 86+ hour estimate.

2. **`compute_neighbor_stats`**: Another O(N_rows) loop computing `max`, `min`, `mean` on tiny vectors. This is slow but secondary.

3. **Memory**: Storing 6.46M list elements (each a small integer vector) has high overhead from R's list/SEXP structure.

**Root cause**: The neighbor lookup is year-invariant (same spatial topology every year), but the code rebuilds per-row key strings and does per-row lookups across all 6.46M rows instead of exploiting the panel structure (344K cells Ã— 28 years).

## Optimization Strategy

1. **Vectorize the neighbor lookup as a sparse adjacency matrix** (344K Ã— 344K). A `dgCMatrix` from the `Matrix` package stores only the ~1.37M nonzero entries. Sparse matrixâ€“dense matrix multiplication computes neighbor sums; sparse matrixâ€“ones multiplication computes neighbor counts. From sum and count we get mean. For max and min, we use a single pass with the sparse structure.

2. **Reshape each variable into a 344K Ã— 28 matrix** (cells Ã— years). Then neighbor stats become sparse-matrix operations on these matrices â€” fully vectorized C-level code, no R-level row loops.

3. **Neighbor mean** = `(A %*% X) / (A %*% ones)` where A is the binary adjacency matrix.

4. **Neighbor max and min** require a loop over cells (not cell-years), but only 344K iterations instead of 6.46M, and each iteration indexes into a pre-built matrix. We can further vectorize this using `data.table` or chunked operations.

5. **Memory**: The 344K Ã— 28 matrix is ~77M doubles (~590 MB for all 5 variables simultaneously). The sparse matrix is tiny (~22 MB). Well within 16 GB.

This reduces runtime from 86+ hours to **minutes**.

## Working R Code

```r
library(Matrix)
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars) {
  # Convert to data.table for speed (non-destructive to RF model)
  dt <- as.data.table(cell_data)

  n_cells <- length(id_order)
  # Map cell id -> integer index 1..n_cells
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # -----------------------------------------------------------
  # 1. Build sparse binary adjacency matrix (344K x 344K)
  # -----------------------------------------------------------
  # rook_neighbors_unique is an nb object: list of length n_cells,
  # each element is an integer vector of neighbor indices (into id_order)
  # with 0 meaning no neighbors.
  from <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to   <- unlist(rook_neighbors_unique)
  # Remove 0-entries (nb convention for no neighbors)
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]

  A <- sparseMatrix(i = from, j = to, x = 1,
                    dims = c(n_cells, n_cells),
                    dimnames = NULL)

  # -----------------------------------------------------------
  # 2. Build cell-year ordering: map each (id, year) to matrix position
  # -----------------------------------------------------------
  # Ensure dt has the original row order preserved
  dt[, .orig_row := .I]

  # Get sorted unique years
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))

  # Map each row to (cell_idx, year_idx)
  dt[, cell_idx := id_to_idx[as.character(id)]]
  dt[, year_idx := year_to_col[as.character(year)]]

  # Precompute neighbor counts per cell (constant across years)
  # A_ones = number of neighbors for each cell
  ones_vec <- rep(1, n_cells)
  neighbor_counts <- as.numeric(A %*% ones_vec)  # length n_cells

  # -----------------------------------------------------------
  # 3. For each variable, build matrix, compute stats, merge back

  # -----------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    message("Processing neighbor stats for: ", var_name)

    # Build n_cells x n_years matrix, filled with NA
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]

    # ---- Neighbor Mean ----
    # Replace NA with 0 for sum, track non-NA counts
    X_nona <- X
    X_nona[is.na(X_nona)] <- 0
    not_na <- (!is.na(X)) * 1  # indicator matrix

    # Neighbor sums: (n_cells x n_cells) %*% (n_cells x n_years)
    neighbor_sum   <- as.matrix(A %*% X_nona)       # n_cells x n_years
    neighbor_nna   <- as.matrix(A %*% not_na)        # n_cells x n_years (count of non-NA neighbors)

    neighbor_mean  <- neighbor_sum / neighbor_nna
    neighbor_mean[neighbor_nna == 0] <- NA_real_

    # ---- Neighbor Max and Min ----
    # We iterate over cells (344K, not 6.46M).
    # For each cell, get its neighbor indices, then take row-wise max/min across those rows of X.
    # To vectorize further, we work column-by-column (year-by-year) using the sparse structure.

    neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    # Extract sparse structure once
    Ap <- A@p        # column pointers (CSC format)
    Ai <- A@i + 1L   # row indices (1-based)
    # For CSC: column j has row indices Ai[(Ap[j]+1):Ap[j+1]]
    # But we need row->column (i.e., for each cell i, which cells j are its neighbors).
    # Transpose to get At where At column i = neighbors of cell i.
    At <- t(A)  # now At is CSC; column i of At = row i of A = neighbors of i
    Atp <- At@p
    Ati <- At@i + 1L

    # Process in chunks to balance speed and memory
    for (i in seq_len(n_cells)) {
      start <- Atp[i] + 1L
      end   <- Atp[i + 1L]
      if (end < start) next  # no neighbors

      nb_idx <- Ati[start:end]
      if (length(nb_idx) == 1L) {
        neighbor_max[i, ] <- X[nb_idx, ]
        neighbor_min[i, ] <- X[nb_idx, ]
      } else {
        nb_vals <- X[nb_idx, , drop = FALSE]  # small matrix: n_neighbors x n_years
        # suppressWarnings to handle all-NA columns gracefully
        neighbor_max[i, ] <- suppressWarnings(apply(nb_vals, 2, max, na.rm = TRUE))
        neighbor_min[i, ] <- suppressWarnings(apply(nb_vals, 2, min, na.rm = TRUE))
      }
    }
    # Fix Inf/-Inf from all-NA columns
    neighbor_max[is.infinite(neighbor_max)] <- NA_real_
    neighbor_min[is.infinite(neighbor_min)] <- NA_real_

    # ---- Map back to data.table rows ----
    idx_mat <- cbind(dt$cell_idx, dt$year_idx)

    max_col <- paste0("neighbor_max_", var_name)
    min_col <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    dt[, (max_col)  := neighbor_max[idx_mat]]
    dt[, (min_col)  := neighbor_min[idx_mat]]
    dt[, (mean_col) := neighbor_mean[idx_mat]]
  }

  # -----------------------------------------------------------
  # 4. Restore original row order and return as data.frame
  # -----------------------------------------------------------
  setorder(dt, .orig_row)
  dt[, c("cell_idx", "year_idx", ".orig_row") := NULL]

  return(as.data.frame(dt))
}

# ---- Usage ----
cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)
```

### Further speedup for max/min (optional Rcpp version)

The 344K-iteration R loop for max/min may still take ~10â€“30 minutes. If that's too slow, this Rcpp drop-in eliminates it:

```r
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_max_min_cpp(IntegerVector Atp, IntegerVector Ati,
                          NumericMatrix X) {
  int n_cells = X.nrow();
  int n_years = X.ncol();
  NumericMatrix nmax(n_cells, n_years);
  NumericMatrix nmin(n_cells, n_years);

  // Initialize with NA
  std::fill(nmax.begin(), nmax.end(), NA_REAL);
  std::fill(nmin.begin(), nmin.end(), NA_REAL);

  for (int i = 0; i < n_cells; i++) {
    int start = Atp[i];      // 0-based CSC pointer
    int end   = Atp[i + 1];
    if (start == end) continue;

    for (int t = 0; t < n_years; t++) {
      double mx = R_NegInf;
      double mn = R_PosInf;
      bool any_valid = false;
      for (int p = start; p < end; p++) {
        int nb = Ati[p];  // 0-based row index
        double v = X(nb, t);
        if (!R_IsNA(v)) {
          any_valid = true;
          if (v > mx) mx = v;
          if (v < mn) mn = v;
        }
      }
      if (any_valid) {
        nmax(i, t) = mx;
        nmin(i, t) = mn;
      }
    }
  }
  return List::create(Named("max") = nmax, Named("min") = nmin);
}
')
```

Then replace the R loop in the main function with:

```r
    At <- t(A)
    res <- neighbor_max_min_cpp(At@p, At@i, X)
    neighbor_max <- res$max
    neighbor_min <- res$min
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M R-level iterations with string ops | Sparse matrix built once from nb object |
| Neighbor mean | 6.46M `mean()` calls | Single sparse matrix multiply (`A %*% X`) |
| Neighbor max/min | 6.46M R-level subset + `max`/`min` | 344K iterations over sparse structure (or Rcpp) |
| Estimated runtime | 86+ hours | **5â€“20 minutes** (R loop) / **1â€“3 minutes** (Rcpp) |
| Numerical result | Identical | **Identical** â€” same rook topology, same arithmetic |
| RF model impact | N/A | **None** â€” model is not retrained; features are numerically identical |
END_RESPONSE>>>
