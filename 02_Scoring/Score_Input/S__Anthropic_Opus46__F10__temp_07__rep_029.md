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
run_id: Anthropic_F10_T07_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each requiring string-pasting, hash lookups (`idx_lookup[neighbor_keys]`), and NA filtering. This is O(N_rows Ã— avg_neighbors) with R-level loop overhead and repeated string allocation â€” the dominant cost.

2. **`compute_neighbor_stats` iterates over ~6.46 million list elements in R**, extracting subsets of a numeric vector. Each iteration has R interpreter overhead, making this O(N_rows) with a large constant factor.

3. **The neighbor topology is year-invariant** (rook neighbors depend only on spatial cell identity), but the lookup is rebuilt monolithically across all cell-years, entangling spatial structure with temporal indexing. This means ~28Ã— redundant work in the lookup construction.

**Key insight:** The rook-neighbor adjacency is a **fixed sparse spatial graph**. The neighbor aggregation (max, min, mean) for a given variable and year is simply a **sparse matrixâ€“vector operation** on the column of that variable restricted to that year. Alternatively, since every cell has the same neighbors in every year, we can construct a single sparse adjacency matrix over cells (not cell-years) and apply it per-year using fast vectorized/matrix operations.

---

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from `rook_neighbors_unique` (344,208 Ã— 344,208, ~1.37M nonzeros). This is tiny in memory (~20 MB in `dgCMatrix` format).

2. **Reshape each variable into a cell Ã— year matrix** (344,208 Ã— 28). This allows vectorized per-year operations.

3. **Compute neighbor stats using sparse matrix multiplication and parallel min/max:**
   - **Mean:** `A %*% X / degree` (sparse matrix multiply, then divide by number of neighbors per cell).
   - **Max / Min:** Iterate over the sparse matrix structure in C++ via `Rcpp` or use grouped operations on the sparse triplet representation with `data.table`.

4. **Avoid any R-level per-row loops** over 6.46M rows.

5. **Memory:** The sparse matrix is ~20 MB; each cellÃ—year matrix is ~77 MB (344K Ã— 28 Ã— 8 bytes). Five variables Ã— 3 stats Ã— 77 MB â‰ˆ 1.15 GB for results â€” fits in 16 GB RAM.

6. **Preserve numerical equivalence:** The sparse matrix encodes exactly the same neighbor sets. Max, min, mean are computed on exactly the same neighbor values.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# =============================================================================
# Prerequisites: data.table, Matrix, Rcpp packages
# install.packages(c("data.table", "Matrix", "Rcpp"))

library(data.table)
library(Matrix)
library(Rcpp)

# ---- Step 0: Rcpp helper for sparse-matrix-based row-wise max, min, mean ----
# This avoids R-level loops entirely for max/min.

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// Computes row-wise max, min, sum, and count of non-NA neighbor values
// given a CSR-like representation of the adjacency and a dense matrix of values.
// adj_p: row pointers (length n_cells + 1), 0-indexed
// adj_j: column indices (length nnz), 0-indexed
// val_mat: n_cells x n_years matrix of variable values
// Output: list of three matrices (max, min, mean), each n_cells x n_years

List neighbor_stats_sparse(IntegerVector adj_p, IntegerVector adj_j,
                           NumericMatrix val_mat) {
  int n_cells = val_mat.nrow();
  int n_years = val_mat.ncol();

  NumericMatrix out_max(n_cells, n_years);
  NumericMatrix out_min(n_cells, n_years);
  NumericMatrix out_mean(n_cells, n_years);

  for (int i = 0; i < n_cells; i++) {
    int start = adj_p[i];
    int end   = adj_p[i + 1];
    int degree = end - start;

    if (degree == 0) {
      for (int t = 0; t < n_years; t++) {
        out_max(i, t)  = NA_REAL;
        out_min(i, t)  = NA_REAL;
        out_mean(i, t) = NA_REAL;
      }
      continue;
    }

    for (int t = 0; t < n_years; t++) {
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      double vsum = 0.0;
      int    cnt  = 0;

      for (int k = start; k < end; k++) {
        int j = adj_j[k];
        double v = val_mat(j, t);
        if (!R_IsNA(v)) {
          if (v > vmax) vmax = v;
          if (v < vmin) vmin = v;
          vsum += v;
          cnt++;
        }
      }

      if (cnt == 0) {
        out_max(i, t)  = NA_REAL;
        out_min(i, t)  = NA_REAL;
        out_mean(i, t) = NA_REAL;
      } else {
        out_max(i, t)  = vmax;
        out_min(i, t)  = vmin;
        out_mean(i, t) = vsum / (double)cnt;
      }
    }
  }

  return List::create(Named("max")  = out_max,
                      Named("min")  = out_min,
                      Named("mean") = out_mean);
}
')

# =============================================================================
# Step 1: Build sparse adjacency matrix from spdep nb object (ONCE)
# =============================================================================
# id_order: vector of 344,208 cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: spdep nb object (list of integer index vectors)

build_adjacency_csr <- function(id_order, nb_obj) {
  n <- length(id_order)
  stopifnot(length(nb_obj) == n)

  # Build COO representation
  from_list <- vector("list", n)
  to_list   <- vector("list", n)

  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    if (length(nbrs) == 1L && nbrs[1L] == 0L) next
    nbrs <- nbrs[nbrs != 0L]
    if (length(nbrs) == 0L) next
    from_list[[i]] <- rep.int(i, length(nbrs))
    to_list[[i]]   <- nbrs
  }

  from_idx <- unlist(from_list, use.names = FALSE)
  to_idx   <- unlist(to_list, use.names = FALSE)

  # Create sparse matrix in dgCMatrix (CSC) format, then transpose to get CSR-like
  # Or directly create dgRMatrix. Easier: create dgCMatrix and extract CSR via t().
  # sparseMatrix builds dgCMatrix by default (CSC).
  # For row-wise access, we want CSR. We can build column-oriented A^T:
  # A_csc = sparseMatrix(i = to_idx, j = from_idx, x = 1, dims = c(n, n))
  # Then A_csc is A^T in CSC = A in CSR.

  # Actually, we just need the p and j arrays for our Rcpp function.
  # Build A in CSC where A[j,i] = 1 means i->j, so columns of A represent
  # "from" nodes. But we want row-wise iteration over "from" nodes.
  # So build A^T in CSC format: rows of A = columns of A^T.

  # Build A^T in CSC (= A in CSR):
  At_csc <- sparseMatrix(
    i = to_idx,
    j = from_idx,
    x = rep(1, length(from_idx)),
    dims = c(n, n),
    giveCsparse = TRUE
  )
  # At_csc@p = column pointers of A^T = row pointers of A
  # At_csc@i = row indices of A^T = column indices of A (0-indexed)

  # For our Rcpp code, we need row pointers and column indices of A.
  # A in CSR: row_ptr = At_csc@p, col_idx = At_csc@i
  # Wait â€” that's not right. Let me be precise.


  # We want: for each "from" node i, the set of "to" neighbors.
  # Build M in CSC where M[to, from] = 1. Then column `from` of M lists all `to` neighbors.
  # M@p are column pointers, M@i are row indices (0-indexed).
  # But Rcpp code iterates over rows. So we need the transpose.

  # Let's just build A where A[from, to] = 1 in dgRMatrix (CSR):
  # Or equivalently, build A^T in CSC format.

  # A[from, to] = 1 means "from has neighbor to"
  # A in CSC: column j lists all rows i such that A[i,j]=1, i.e., all nodes
  #           that have j as a neighbor.
  # A in CSR: row i lists all columns j such that A[i,j]=1, i.e., all neighbors of i.
  # We want CSR. Build A^T in CSC:

  # A^T[to, from] = 1
  # A^T in CSC: column pointers index "from" nodes, row indices are "to" nodes.
  # So At_csc@p[from] to At_csc@p[from+1]-1 gives indices into At_csc@i,
  # which are the "to" neighbors of "from". Exactly what we want!

  At <- sparseMatrix(
    i = to_idx,      # "to" nodes (row of A^T)
    j = from_idx,    # "from" nodes (column of A^T)
    x = rep(1, length(from_idx)),
    dims = c(n, n),
    giveCsparse = TRUE
  )

  list(
    adj_p = At@p,    # integer, length n+1, 0-indexed column pointers of A^T = row pointers of A
    adj_j = At@i,    # integer, 0-indexed row indices of A^T = column indices of A = neighbor IDs
    n = n,
    id_order = id_order
  )
}

# =============================================================================
# Step 2: Reshape panel data variable into cell Ã— year matrix
# =============================================================================
reshape_to_matrix <- function(dt, id_order, years, var_name) {
  # dt: data.table with columns id, year, <var_name>
  # Returns: n_cells x n_years numeric matrix, rows ordered by id_order, cols by years

  n_cells <- length(id_order)
  n_years <- length(years)

  # Create mapping: id -> row index
  id_to_row <- setNames(seq_along(id_order), as.character(id_order))
  # Create mapping: year -> col index
  year_to_col <- setNames(seq_along(years), as.character(years))

  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  row_idx <- id_to_row[as.character(dt$id)]
  col_idx <- year_to_col[as.character(dt$year)]

  valid <- !is.na(row_idx) & !is.na(col_idx)
  mat[cbind(row_idx[valid], col_idx[valid])] <- dt[[var_name]][valid]

  mat
}

# =============================================================================
# Step 3: Melt stats matrices back into the panel data.table
# =============================================================================
melt_stats_to_dt <- function(dt, id_order, years, stats_list, var_name) {
  # stats_list: list with $max, $min, $mean â€” each n_cells x n_years matrix
  # Adds columns: nb_max_<var>, nb_min_<var>, nb_mean_<var> to dt

  id_to_row  <- setNames(seq_along(id_order), as.character(id_order))
  year_to_col <- setNames(seq_along(years), as.character(years))

  row_idx <- id_to_row[as.character(dt$id)]
  col_idx <- year_to_col[as.character(dt$year)]

  valid <- !is.na(row_idx) & !is.na(col_idx)
  lin_idx <- cbind(row_idx[valid], col_idx[valid])

  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)

  dt[[max_col]]  <- NA_real_
  dt[[min_col]]  <- NA_real_
  dt[[mean_col]] <- NA_real_

  dt[[max_col]][valid]  <- stats_list$max[lin_idx]
  dt[[min_col]][valid]  <- stats_list$min[lin_idx]
  dt[[mean_col]][valid] <- stats_list$mean[lin_idx]

  dt
}

# =============================================================================
# Step 4: Main pipeline
# =============================================================================
run_neighbor_aggregation <- function(cell_data, id_order, rook_neighbors_unique,
                                     neighbor_source_vars, rf_model) {
  # Convert to data.table for efficiency if not already
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  years <- sort(unique(cell_data$year))
  cat("Building sparse adjacency (CSR) from nb object...\n")
  adj <- build_adjacency_csr(id_order, rook_neighbors_unique)
  cat(sprintf("  Adjacency: %d cells, %d directed edges\n", adj$n, length(adj$adj_j)))

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))

    # Reshape to matrix: n_cells x n_years
    t0 <- proc.time()
    val_mat <- reshape_to_matrix(cell_data, id_order, years, var_name)
    cat(sprintf("  Reshape: %.1f sec\n", (proc.time() - t0)[3]))

    # Compute neighbor stats via Rcpp (single pass over sparse structure)
    t0 <- proc.time()
    stats <- neighbor_stats_sparse(adj$adj_p, adj$adj_j, val_mat)
    cat(sprintf("  Neighbor stats (Rcpp): %.1f sec\n", (proc.time() - t0)[3]))

    # Melt back into panel
    t0 <- proc.time()
    cell_data <- melt_stats_to_dt(cell_data, id_order, years, stats, var_name)
    cat(sprintf("  Melt to panel: %.1f sec\n", (proc.time() - t0)[3]))

    # Free intermediate matrices
    rm(val_mat, stats)
    gc(verbose = FALSE)
  }

  # ---- Predict with pre-trained Random Forest (no retraining) ----
  cat("Generating predictions with pre-trained Random Forest...\n")
  cell_data$rf_prediction <- predict(rf_model, newdata = cell_data)

  cell_data
}

# =============================================================================
# USAGE EXAMPLE
# =============================================================================
# # Load your data
# cell_data              <- readRDS("cell_data.rds")
# id_order               <- readRDS("id_order.rds")
# rook_neighbors_unique  <- readRDS("rook_neighbors_unique.rds")
# rf_model               <- readRDS("rf_model.rds")
#
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# result <- run_neighbor_aggregation(
#   cell_data, id_order, rook_neighbors_unique,
#   neighbor_source_vars, rf_model
# )
```

---

## Performance Analysis

| Component | Original (est.) | Optimized (est.) | Speedup |
|---|---|---|---|
| **Neighbor lookup** | ~40â€“50 hrs (string ops on 6.46M rows) | ~5â€“10 sec (sparse matrix construction, once) | ~15,000Ã— |
| **Stats per variable** | ~7â€“8 hrs (R list iteration, 6.46M elements) | ~3â€“5 sec (Rcpp, single pass over CSR + dense matrix) | ~5,000Ã— |
| **5 variables total** | ~86+ hrs | **~2â€“4 minutes** total (including reshape/melt) | ~1,500Ã— |
| **Memory peak** | Potentially high (6.46M-element list of integer vectors) | ~2â€“3 GB (sparse matrix ~20 MB + matrices ~77 MB each + panel data.table) | Fits in 16 GB |

### Why this is numerically equivalent

- The sparse adjacency matrix encodes **exactly** the same directed rook-neighbor relationships as the original `nb` object.
- The Rcpp kernel computes `max`, `min`, and `mean` over **exactly** the same set of non-NA neighbor values as the original `compute_neighbor_stats`, with the same NA-handling semantics (skip NAs; return NA if all neighbors are NA or no neighbors exist).
- The `mean` uses the same `sum / count` arithmetic (IEEE 754 double precision). Since the same values are summed in the same data type, results are identical to floating-point precision. The summation order follows the CSR column-index order (ascending), which is deterministic.
- The pre-trained Random Forest model is loaded and used for `predict()` only â€” **no retraining occurs**.
END_RESPONSE>>>
