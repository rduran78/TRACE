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
run_id: Anthropic_F10_T10_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation suffers from three major bottlenecks:

1. **O(nÂ²)-scale list construction in `build_neighbor_lookup`**: For each of the ~6.46 million rows, it performs string-pasting, named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but still carry significant per-call overhead at this scale. The `paste(..., sep="_")` call alone creates millions of temporary string objects.

2. **Row-wise `lapply` in `compute_neighbor_stats`**: Called 5 times (once per variable) Ã— 6.46M rows = ~32.3 million R-level function invocations, each performing subsetting, NA removal, and three summary statistics. R's interpreted loop overhead dominates here.

3. **Redundant topology work per row**: The neighbor lookup is built at the *cell-year* level (~6.46M entries), but the topology is invariant across years. There are only ~344K cells and ~1.37M directed edges. The same adjacency structure is needlessly replicated 28 times.

**Memory pressure**: The 6.46M-element list-of-integer-vectors in `neighbor_lookup`, plus intermediate string vectors, likely causes repeated GC pauses and can approach the 16 GB ceiling.

---

## Optimization Strategy

### Core insight: Separate topology (cell-level) from attributes (cell-year level)

1. **Build a sparse adjacency matrix once** at the cell level (344K Ã— 344K, ~1.37M non-zeros). Use `Matrix::sparseMatrix` in CSC format.

2. **Reshape each variable into a cell Ã— year matrix** (344K Ã— 28). This is compact (~77 MB per variable in double precision).

3. **Compute neighbor statistics via sparse matrixâ€“dense matrix multiplication and analogous sparse operations**:
   - **Mean**: `A_norm %*% X` where `A_norm` is the row-normalized adjacency matrix (each row sums to 1 over its neighbors). This gives exact neighbor means in one sparse matrix multiply.
   - **Max / Min**: Use a CSR traversal. R's `Matrix` package stores CSC; we transpose to get CSR-equivalent access. Then iterate over cells (not cell-years) â€” only 344K iterations, each touching ~4 neighbors on average. This is done column-by-column (28 years) in compiled C++ via `Rcpp`.

4. **Reshape results back** to the long panel format and column-bind to `cell_data`.

5. **Predict** using the pre-trained Random Forest model unchanged.

**Expected speedup**: From ~86+ hours to **minutes**. The sparse matrix multiply for means is essentially free (~seconds). The Rcpp loop for max/min over 344K Ã— 28 Ã— 5 â‰ˆ 48M cell-year-variable computations with ~4 neighbors each is ~200M comparisons â€” trivial for compiled code.

---

## Working R Code

```r
# =============================================================================
# Optimized spatial neighbor feature computation
# =============================================================================
# Prerequisites:
#   cell_data            â€” data.frame/data.table with columns: id, year, ntl, ec,
#                          pop_density, def, usd_est_n2, ... (~6.46M rows)
#   id_order             â€” integer vector of 344,208 cell IDs in the order used
#                          by rook_neighbors_unique
#   rook_neighbors_unique â€” spdep nb object (list of length 344,208)
#   rf_model             â€” pre-trained Random Forest model
# =============================================================================

library(data.table)
library(Matrix)
library(Rcpp)

# ---------- Step 0: Convert cell_data to data.table for speed ----------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ---------- Step 1: Build sparse adjacency matrix (cell-level) ---------------
# One-time cost. rook_neighbors_unique is an nb object: list of integer index
# vectors (indices into id_order).

message("Building sparse adjacency matrix...")
n_cells <- length(id_order)

# Construct COO triplets from nb object
edge_from <- integer(0)
edge_to   <- integer(0)

# Pre-allocate by estimating total edges
total_edges <- sum(vapply(rook_neighbors_unique, length, integer(1)))
edge_from   <- integer(total_edges)
edge_to     <- integer(total_edges)

pos <- 1L
for (i in seq_len(n_cells)) {
  nb_i <- rook_neighbors_unique[[i]]
  # spdep nb objects use 0L for no-neighbor indicator; filter those
  nb_i <- nb_i[nb_i > 0L]
  n_nb <- length(nb_i)
  if (n_nb > 0L) {
    edge_from[pos:(pos + n_nb - 1L)] <- i
    edge_to[pos:(pos + n_nb - 1L)]   <- nb_i
    pos <- pos + n_nb
  }
}
# Trim if over-allocated
edge_from <- edge_from[seq_len(pos - 1L)]
edge_to   <- edge_to[seq_len(pos - 1L)]

# Adjacency matrix A: A[i,j] = 1 means j is a neighbor of i
# (row i aggregates over its neighbors in columns)
A <- sparseMatrix(
  i = edge_from,
  j = edge_to,
  x = rep(1, length(edge_from)),
  dims = c(n_cells, n_cells)
)

# Row-normalized version for computing means
row_counts <- diff(A@p)  # number of non-zeros per column in CSC â€” but we need per row
# For CSR-like row sums:
row_sums <- rowSums(A)
row_sums[row_sums == 0] <- NA_real_  # cells with no neighbors -> NA mean

# Normalized adjacency for mean computation
A_norm <- A
# Divide each row by its count: Diagonal^{-1} %*% A
inv_row_sums <- ifelse(is.na(row_sums), 0, 1 / row_sums)
D_inv <- Diagonal(x = inv_row_sums)
A_norm <- D_inv %*% A

message(sprintf("  Adjacency: %d cells, %d directed edges", n_cells, length(edge_from)))

# ---------- Step 2: Create cell index mapping --------------------------------
# Map cell id -> position (1..n_cells) matching id_order
id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

# Map years to column indices
years <- sort(unique(cell_data$year))
n_years <- length(years)
year_to_col <- setNames(seq_len(n_years), as.character(years))

# Add matrix indices to cell_data
cell_data[, `:=`(
  cell_pos = id_to_pos[as.character(id)],
  year_col = year_to_col[as.character(year)]
)]

# ---------- Step 3: Rcpp function for sparse max/min ------------------------
cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// A is in dgCMatrix (CSC) format. We need row-wise access, so we
// work with the transpose (which gives us CSC of A^T = CSR of A).
// At_p, At_i, At_x come from the CSC representation of t(A).
// X is n_cells x n_years matrix (column-major).
// Returns a list of two matrices: max_mat and min_mat (n_cells x n_years).

// [[Rcpp::export]]
List sparse_row_max_min(IntegerVector At_p, IntegerVector At_i,
                        NumericMatrix X) {
  int n = X.nrow();
  int m = X.ncol();
  NumericMatrix max_mat(n, m);
  NumericMatrix min_mat(n, m);

  // Initialize to NA
  std::fill(max_mat.begin(), max_mat.end(), NA_REAL);
  std::fill(min_mat.begin(), min_mat.end(), NA_REAL);

  for (int i = 0; i < n; i++) {
    int start = At_p[i];
    int end   = At_p[i + 1];
    if (start == end) continue;  // no neighbors, stays NA

    for (int yr = 0; yr < m; yr++) {
      double cur_max = R_NegInf;
      double cur_min = R_PosInf;
      int valid = 0;

      for (int k = start; k < end; k++) {
        int j = At_i[k];  // neighbor index (0-based)
        double val = X(j, yr);
        if (!R_IsNA(val) && !ISNAN(val)) {
          if (val > cur_max) cur_max = val;
          if (val < cur_min) cur_min = val;
          valid++;
        }
      }

      if (valid > 0) {
        max_mat(i, yr) = cur_max;
        min_mat(i, yr) = cur_min;
      }
      // else stays NA
    }
  }

  return List::create(Named("max_mat") = max_mat,
                      Named("min_mat") = min_mat);
}
', depends = "Rcpp")

# Precompute the transpose of A for CSR-like row access in Rcpp
At <- t(A)  # At is CSC of A^T = CSR of A
# Extract slots (0-based indices as used by Matrix package)
At_p <- At@p
At_i <- At@i

# ---------- Step 4: Process each variable ------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor statistics for ", length(neighbor_source_vars), " variables...")

for (var_name in neighbor_source_vars) {
  message(sprintf("  Processing: %s", var_name))

  # --- 4a: Reshape long -> cell x year matrix --------------------------------
  # Use data.table fast indexing
  X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  vals <- cell_data[[var_name]]
  cpos <- cell_data$cell_pos
  ycol <- cell_data$year_col

  # Vectorized assignment
  idx <- cbind(cpos, ycol)
  X[idx] <- vals

  # --- 4b: Neighbor MEAN via sparse matrix multiply --------------------------
  # A_norm %*% X gives (n_cells x n_years) matrix of neighbor means.
  # Cells with no neighbors get 0 from the multiply (since inv_row_sums=0);
  # we need to convert those to NA.
  mean_mat <- as.matrix(A_norm %*% X)
  # Mask out cells with no neighbors
  no_neighbor <- is.na(row_sums)
  if (any(no_neighbor)) {
    mean_mat[no_neighbor, ] <- NA_real_
  }
  # Also, if all neighbor values were NA for a cell-year, A_norm %*% X would
  # produce a value based on 0s (NA * 0 = 0 in sparse multiply). We need to
  # handle this correctly.
  # The sparse multiply treats NA as 0. We must correct for this.

  # More robust mean: compute sum of non-NA neighbors and count of non-NA neighbors.
  # Replace NAs with 0 for sum, and use indicator for count.
  X_zero <- X
  X_zero[is.na(X_zero)] <- 0
  X_ind  <- matrix(1, nrow = n_cells, ncol = n_years)
  X_ind[is.na(X)] <- 0

  sum_mat   <- as.matrix(A %*% X_zero)      # sum of non-NA neighbor values
  count_mat <- as.matrix(A %*% X_ind)        # count of non-NA neighbors

  mean_mat <- ifelse(count_mat > 0, sum_mat / count_mat, NA_real_)

  # --- 4c: Neighbor MAX and MIN via Rcpp sparse traversal --------------------
  maxmin <- sparse_row_max_min(At_p, At_i, X)
  max_mat <- maxmin$max_mat
  min_mat <- maxmin$min_mat

  # --- 4d: Map results back to long format -----------------------------------
  # Use the same (cell_pos, year_col) index to read back
  new_max  <- max_mat[idx]
  new_min  <- min_mat[idx]
  new_mean <- mean_mat[idx]

  # Add columns to cell_data (preserving original naming convention)
  max_col  <- paste0(var_name, "_max")
  min_col  <- paste0(var_name, "_min")
  mean_col <- paste0(var_name, "_mean")

  set(cell_data, j = max_col,  value = new_max)
  set(cell_data, j = min_col,  value = new_min)
  set(cell_data, j = mean_col, value = new_mean)

  message(sprintf("    Added: %s, %s, %s", max_col, min_col, mean_col))
}

# ---------- Step 5: Clean up temporary columns --------------------------------
cell_data[, `:=`(cell_pos = NULL, year_col = NULL)]

message("Neighbor feature computation complete.")

# ---------- Step 6: Predict using pre-trained Random Forest -------------------
# The RF model is unchanged â€” same features, same numerical values.
# Convert back to data.frame if the model expects one.
predictions <- predict(rf_model, newdata = as.data.frame(cell_data))

message("Prediction complete.")
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Topology representation** | 6.46M-element list of integer vectors (replicated across years) | Single 344K Ã— 344K sparse matrix (~1.37M non-zeros, ~33 MB) |
| **Mean computation** | R `lapply` â†’ `mean()` per row (6.46M calls) | Sparse matrix multiply `A %*% X` (milliseconds) |
| **Max/Min computation** | R `lapply` â†’ `max()`/`min()` per row | Rcpp CSR traversal: 344K rows Ã— 28 years Ã— ~4 neighbors (seconds) |
| **Per-variable overhead** | ~17 hours | ~10â€“30 seconds |
| **Total estimated time** | 86+ hours | **2â€“5 minutes** |
| **Peak RAM** | Exceeded 16 GB (string keys, lists) | ~2â€“3 GB (sparse matrix + five 344KÃ—28 dense matrices) |
| **Numerical equivalence** | â€” | âœ… Identical `max`, `min`, `mean` over non-NA rook neighbors per cell-year |
| **RF model** | â€” | âœ… Untouched, used only for `predict()` |

### Why numerical equivalence is preserved

- **Mean**: We compute `sum(non-NA neighbor values) / count(non-NA neighbors)` using two sparse multiplies on the NA-zeroed matrix and an indicator matrix, then divide â€” this is algebraically identical to `mean(neighbor_vals[!is.na(neighbor_vals)])`.
- **Max/Min**: The Rcpp function iterates over exactly the same neighbor set (same sparse adjacency) and applies the same NA-skip logic, yielding identical results to `max(neighbor_vals[!is.na(neighbor_vals)])` and `min(...)`.
- **Edge cases**: Zero-neighbor cells and all-NA-neighbor cell-years both produce `NA`, matching the original `if (length(idx) == 0) return(c(NA, NA, NA))` and `if (length(neighbor_vals) == 0) return(c(NA, NA, NA))` logic.
END_RESPONSE>>>
