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
run_id: Anthropic_F10_T10_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each via an anonymous function with character key lookups (`paste`, `setNames`, named vector indexing). The `paste`/`match` pattern on 6.46M rows is extremely slow due to R's string hashing overhead and per-element list allocation.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in an `lapply`**, calling `max`, `min`, `mean` on small vectors inside a pure-R loop. This is called 5 times (once per variable), totaling ~32.3 million R-level function invocations with per-call overhead.

3. **The neighbor lookup conflates topology and time.** Rook neighbors are a *spatial* propertyâ€”they don't change across years. Yet the lookup is built at the cell-year level, repeating the same spatial adjacency structure 28 times and inflating the lookup from ~344K spatial entries to ~6.46M spatiotemporal entries.

**Key insight:** The adjacency graph is purely spatial (344,208 nodes, ~1.37M directed edges). Year is an attribute dimension, not a topological one. By separating topology from time, we can build a sparse adjacency matrix **once** (344K Ã— 344K) and compute all neighbor aggregations via sparse matrixâ€“dense matrix multiplicationâ€”replacing millions of R-level loops with a handful of compiled linear algebra operations.

## Optimization Strategy

1. **Build a sparse adjacency matrix `A`** (344,208 Ã— 344,208) from `rook_neighbors_unique` once. Also build a binary "has-neighbor" indicator and a degree vector `d` (number of neighbors per cell).

2. **Reshape each variable into a dense matrix `V`** of dimension (344,208 cells Ã— 28 years) indexed by `(cell, year)`.

3. **Compute neighbor sums via sparse matrix multiplication:** `S = A %*% V`. This is a single call into compiled C code (Matrix package) and gives the sum of neighbor values for every cell-year.

4. **Compute neighbor means** as `S / d` (elementwise, broadcasting the degree vector).

5. **For max and min**, use a grouped sparse operation: iterate over the *columns* of `A` (i.e., per-cell neighbor sets, only 344K of them) using compiled `dgCMatrix` slot arithmetic, and compute rowwise max/min. This is the tightest loop (~344K iterations, not 6.46M), and each iteration touches only ~4 neighbors on average (rook adjacency).

6. **Flatten back** to the original cell-year data.frame column order and bind.

**Expected speedup:** From 86+ hours down to **minutes** (sparse matrix multiply is O(nnz Ã— ncol) â‰ˆ 1.37M Ã— 28 â‰ˆ 38.4M multiply-adds per variable, done in compiled code; max/min loops are 344K Ã— 28 Ã— ~4 = ~38.5M comparisons per variable in a tight C++ loop via Rcpp).

## Working R Code

```r
# =============================================================================
# Optimized neighborâ€‘aggregation pipeline
# Preserves numerical equivalence with original max / min / mean statistics.
# Requires: Matrix, data.table, Rcpp (all on CRAN)
# =============================================================================

library(Matrix)
library(data.table)
library(Rcpp)

# ---- 0.  Small Rcpp helper for sparseâ€‘row max and min ----------------------
# This avoids a pure-R loop over 344K cells and is the key to fast max/min.

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_row_maxmin(IntegerVector Ap,       // CSC column pointers (length ncol+1)
                       IntegerVector Ai,       // CSC row indices (0-based)
                       NumericMatrix V,         // dense matrix:  nrow x nyear
                       int nrow_out) {
  // A is ncell x ncell in CSC.  We want, for each row i of A,
  // the max and min of V[j, ] across all j where A[i,j] != 0.
  // CSC stores columns, so A[i,j]!=0 means rowâ€‘index i appears in column j.
  // Strategy: scan every column j, and for each nonzero row i in that column,
  // update running max/min of V[j, year] for row i.


  int ncol_A = Ap.size() - 1;
  int nyear  = V.ncol();

  // Initialise output matrices with NA
  NumericMatrix mx(nrow_out, nyear);
  NumericMatrix mn(nrow_out, nyear);
  std::fill(mx.begin(), mx.end(), NA_REAL);
  std::fill(mn.begin(), mn.end(), NA_REAL);

  for (int j = 0; j < ncol_A; j++) {
    int p_start = Ap[j];
    int p_end   = Ap[j + 1];
    for (int p = p_start; p < p_end; p++) {
      int i = Ai[p];                       // row i is a neighbor of column j
      for (int y = 0; y < nyear; y++) {
        double v = V(j, y);
        if (R_IsNA(v)) continue;
        double cur_mx = mx(i, y);
        double cur_mn = mn(i, y);
        if (R_IsNA(cur_mx)) {
          mx(i, y) = v;
          mn(i, y) = v;
        } else {
          if (v > cur_mx) mx(i, y) = v;
          if (v < cur_mn) mn(i, y) = v;
        }
      }
    }
  }
  return List::create(Named("max") = mx, Named("min") = mn);
}
')

# ---- 1.  Build sparse adjacency matrix ONCE --------------------------------

build_adjacency_matrix <- function(id_order, nb_object) {
  # id_order : integer vector of cell IDs in the order used by spdep::nb
  # nb_object: list of integer vectors (spdep nb), 1-indexed into id_order
  n <- length(id_order)
  from <- integer(0)
  to   <- integer(0)
  for (i in seq_len(n)) {
    nbrs <- nb_object[[i]]
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) == 0L) next
    from <- c(from, rep.int(i, length(nbrs)))
    to   <- c(to,   nbrs)
  }
  # A[i, j] = 1 means j is a neighbor of i  (i.e., j's value feeds into i's aggregation)
  A <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  A
}

# ---- 2.  Main pipeline function --------------------------------------------

add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      rook_neighbors_unique,
                                      neighbor_source_vars) {
  # Convert to data.table for fast manipulation
  dt <- as.data.table(cell_data)

  # --- stable orderings -----------------------------------------------------
  unique_ids   <- as.integer(id_order)       # spatial index (length = ncell)
  unique_years <- sort(unique(dt$year))      # temporal index (length = nyear)
  ncell <- length(unique_ids)
  nyear <- length(unique_years)

  # Map cell id -> spatial index (1..ncell)
  id_to_sidx <- setNames(seq_along(unique_ids), as.character(unique_ids))
  # Map year -> temporal index (1..nyear)
  year_to_tidx <- setNames(seq_along(unique_years), as.character(unique_years))

  # Compute spatial and temporal indices for every row of dt
  dt[, sidx := id_to_sidx[as.character(id)]]
  dt[, tidx := year_to_tidx[as.character(year)]]

  # --- adjacency matrix (built once) ----------------------------------------
  cat("Building sparse adjacency matrix ...\n")
  A <- build_adjacency_matrix(id_order, rook_neighbors_unique)
  stopifnot(nrow(A) == ncell, ncol(A) == ncell)

  # Degree vector (number of neighbors per cell, ignoring missingness in values)
  # We will adjust per-variable for NA handling below.

  # CSC slots for Rcpp max/min
  A_csc <- as(A, "dgCMatrix")
  Ap <- A_csc@p
  Ai <- A_csc@i

  # --- perâ€‘variable aggregation ---------------------------------------------
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing variable: %s\n", var_name))

    # Reshape variable into dense matrix V[cell, year]
    V <- matrix(NA_real_, nrow = ncell, ncol = nyear)
    V[cbind(dt$sidx, dt$tidx)] <- dt[[var_name]]

    # ---- MEAN via sparse matâ€‘mul ------------------------------------------
    # neighbor_sum[i, y] = sum of V[j, y] for j in neighbors(i)
    neighbor_sum   <- A %*% V                        # sparse %*% dense -> dense

    # neighbor_count: count of non-NA neighbor values per cell-year
    V_notna <- matrix(as.numeric(!is.na(V)), nrow = ncell, ncol = nyear)
    neighbor_count <- A %*% V_notna

    # Replace V NAs with 0 for summation, then fix up
    V_zero <- V
    V_zero[is.na(V_zero)] <- 0
    neighbor_sum <- as.matrix(A %*% V_zero)
    neighbor_count <- as.matrix(neighbor_count)

    neighbor_mean <- neighbor_sum / neighbor_count   # NA where count == 0
    neighbor_mean[neighbor_count == 0] <- NA_real_

    # ---- MAX / MIN via Rcpp -----------------------------------------------
    maxmin <- sparse_row_maxmin(Ap, Ai, V, ncell)
    neighbor_max <- maxmin$max     # ncell x nyear matrix
    neighbor_min <- maxmin$min

    # ---- Write back to dt in original row order ----------------------------
    lin_idx <- cbind(dt$sidx, dt$tidx)

    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    dt[, (max_col)  := neighbor_max[lin_idx]]
    dt[, (min_col)  := neighbor_min[lin_idx]]
    dt[, (mean_col) := neighbor_mean[lin_idx]]
  }

  # Clean up helper columns
  dt[, c("sidx", "tidx") := NULL]

  # Return as data.frame (or data.table, depending on downstream needs)
  as.data.frame(dt)
}

# ---- 3.  Execute pipeline and predict with preâ€‘trained RF -------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The pre-trained Random Forest model is preserved as-is.  No retraining.
# Predict using the exact same column names (numerical equivalence guaranteed).
predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Guarantees

| Property | How it is preserved |
|---|---|
| **Numerical equivalence** | `mean = sum/count` matches original `mean()` on non-NA neighbors. `max`/`min` are computed by exhaustive comparison over the identical neighbor sets (CSC traversal mirrors the original `idx` lookup). |
| **Pre-trained RF unchanged** | `rf_model` is never modified; `predict()` is called with identical column names and identical numerical values. |
| **Memory** | Peak â‰ˆ sparse matrix (~22 MB for 1.37M entries) + one dense 344KÃ—28 matrix (~77 MB) Ã— a few temporaries. Well within 16 GB. |
| **Speed** | Adjacency built once (seconds). Per-variable: one sparse Ã— dense multiply (~seconds) + one Rcpp pass (~seconds). Total for 5 variables: **under 5 minutes** on a standard laptop, vs. 86+ hours originally. |
END_RESPONSE>>>
