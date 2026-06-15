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
run_id: Anthropic_F10_T03_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each constructed via string-pasting and named-vector lookups. The `paste(id, year)` key construction and `idx_lookup[neighbor_keys]` named-vector subsetting is O(n) per lookup in the worst case, and doing this 6.46M times is catastrophic. This single function likely accounts for 60â€“70% of the 86+ hour runtime.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a pure-R `lapply`**, extracting subsets of a numeric vector and computing `max/min/mean`. This is called 5 times (once per variable), so ~32.3 million R-level loop iterations with per-element allocation.

3. **The neighbor lookup is year-aware but the graph topology is year-invariant.** The rook-neighbor adjacency is purely spatial â€” it doesn't change across years. Yet the current code rebuilds year-specific row indices for every single cell-year row, duplicating the same spatial topology 28 times with string operations.

**Key insight:** The spatial adjacency graph has only 344,208 nodes and ~1.37M directed edges. This is a small, sparse graph. The year dimension is orthogonal â€” every year has the same graph. We should build the topology once as a sparse matrix and use vectorized sparse matrixâ€“dense matrix multiplication to compute neighborhood aggregates.

---

## Optimization Strategy

### Core Idea: Sparse Matrix Aggregation

1. **Build a sparse adjacency matrix `A`** (344,208 Ã— 344,208) from the `nb` object once. Entry `A[i,j] = 1` if cell `j` is a rook neighbor of cell `i`.

2. **Build a row-degree vector `D`** where `D[i]` = number of neighbors of cell `i` (i.e., row sums of `A`).

3. **Reshape each variable into a matrix `V`** of dimension (344,208 cells Ã— 28 years). This is the "node attribute matrix."

4. **Compute neighbor mean** as: `mean_matrix = (A %*% V) / D` â€” a single sparse matrix multiplication. This is O(nnz Ã— 28) â‰ˆ 38.4M multiply-adds, done in compiled C code via the `Matrix` package.

5. **Compute neighbor max and min** â€” these cannot be done via matrix multiplication. Instead, use the sparse structure of `A` to iterate in compiled code. We use `data.table` grouped operations or a compiled Rcpp routine over the CSR representation of `A`.

6. **Unroll back** to the long panel format and bind columns.

**Expected speedup:** From 86+ hours to **~2â€“10 minutes** depending on the max/min strategy.

### Why This Preserves Numerical Equivalence

- The sparse matrix `A` encodes exactly the same neighbor relationships as `rook_neighbors_unique`.
- `A %*% V` computes exactly `sum of neighbor values` per cell; dividing by degree gives the mean.
- Max and min are computed over exactly the same neighbor sets.
- No approximation, sampling, or model retraining is involved.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE PIPELINE
# Sparse graph aggregation â€” numerically equivalent to original
# =============================================================================

library(Matrix)
library(data.table)

# ---- Step 1: Build sparse adjacency matrix from nb object (ONCE) ----

build_sparse_adjacency <- function(nb_obj, n) {

  # nb_obj: spdep nb object (list of integer vectors of neighbor indices)
  # n: number of spatial cells (length of nb_obj)
  #
  # Returns: sparse dgCMatrix (n x n), A[i,j]=1 if j is neighbor of i

  # Pre-count total edges for pre-allocation
  edge_counts <- vapply(nb_obj, function(x) {
    nx <- x[x != 0L]  # spdep nb uses 0 for no-neighbor regions
    length(nx)
  }, integer(1))

  total_edges <- sum(edge_counts)

  # Pre-allocate triplet vectors
  from_idx <- integer(total_edges)
  to_idx   <- integer(total_edges)

  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs != 0L]
    k <- length(nbrs)
    if (k > 0L) {
      from_idx[pos:(pos + k - 1L)] <- i
      to_idx[pos:(pos + k - 1L)]   <- nbrs
      pos <- pos + k
    }
  }

  sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = rep(1, total_edges),
    dims = c(n, n),
    repr = "C"   # CSC format, efficient for %*%
  )
}


# ---- Step 2: Reshape long panel to cell Ã— year matrix ----

long_to_wide_matrix <- function(dt, var_name, cell_id_map, year_levels) {
  # dt: data.table with columns id, year, and var_name
  # cell_id_map: named integer vector mapping cell id -> row index (1..N)
  # year_levels: sorted unique years
  #
  # Returns: matrix (N_cells x N_years), with NA where missing

  n_cells <- length(cell_id_map)
  n_years <- length(year_levels)
  year_map <- setNames(seq_along(year_levels), as.character(year_levels))

  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  row_i <- cell_id_map[as.character(dt$id)]
  col_j <- year_map[as.character(dt$year)]
  mat[cbind(row_i, col_j)] <- dt[[var_name]]

  mat
}


# ---- Step 3: Compute neighbor MEAN via sparse matmul ----

compute_neighbor_mean <- function(A, V, degree_vec) {
  # A: sparse adjacency (N x N)
  # V: dense matrix (N x T), may contain NA
  # degree_vec: integer vector of neighbor counts per cell
  #
  # Returns: matrix (N x T) of neighbor means

  # Replace NA with 0 for multiplication, track valid counts
  V_clean <- V
  V_clean[is.na(V_clean)] <- 0

  # Indicator matrix: 1 where V is not NA
  V_valid <- matrix(1, nrow = nrow(V), ncol = ncol(V))
  V_valid[is.na(V)] <- 0

  # Sum of neighbor values (only non-NA)
  neighbor_sum   <- A %*% V_clean       # sparse %*% dense -> dense
  # Count of non-NA neighbors per cell-year
  neighbor_count <- A %*% V_valid

  # Mean = sum / count (where count > 0)
  neighbor_mean <- as.matrix(neighbor_sum) / as.matrix(neighbor_count)
  neighbor_mean[as.matrix(neighbor_count) == 0] <- NA_real_

  neighbor_mean
}


# ---- Step 4: Compute neighbor MAX and MIN via CSR iteration ----
#
# We iterate over the sparse structure. For ~1.37M edges Ã— 28 years,
# this is fast even in R if vectorized per-row.
# For maximum speed, we use an Rcpp implementation.
# Fallback pure-R version provided below.

# --- Pure R version (still fast: ~1-3 min) ---

compute_neighbor_max_min <- function(A, V) {
  # A: sparse dgCMatrix (N x N)
  # V: dense matrix (N x T)
  #
  # Returns: list(max = matrix(N x T), min = matrix(N x T))

  # Convert to dgRMatrix (CSR) for efficient row iteration
  A_csr <- as(A, "RsparseMatrix")

  n <- nrow(V)
  n_years <- ncol(V)

  max_mat <- matrix(NA_real_, nrow = n, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n, ncol = n_years)

  # CSR: A_csr@p[i]+1 to A_csr@p[i+1] are the column indices for row i
  p <- A_csr@p
  j <- A_csr@j + 1L  # 0-based to 1-based

  for (i in seq_len(n)) {
    start <- p[i] + 1L
    end   <- p[i + 1L]
    if (end >= start) {
      nbr_indices <- j[start:end]
      nbr_vals <- V[nbr_indices, , drop = FALSE]  # k x T matrix

      # Columnwise max/min ignoring NA
      max_mat[i, ] <- apply(nbr_vals, 2, max, na.rm = TRUE)
      min_mat[i, ] <- apply(nbr_vals, 2, min, na.rm = TRUE)

      # Fix: if all NA in a column, apply returns -Inf/Inf
      all_na <- apply(is.na(nbr_vals), 2, all)
      max_mat[i, all_na] <- NA_real_
      min_mat[i, all_na] <- NA_real_
    }
  }

  list(max = max_mat, min = min_mat)
}


# --- Rcpp version (recommended: ~10-30 sec) ---

if (requireNamespace("Rcpp", quietly = TRUE)) {
  Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List compute_max_min_csr(IntegerVector p, IntegerVector j,
                         NumericMatrix V) {
  int n = V.nrow();
  int T = V.ncol();
  NumericMatrix max_mat(n, T);
  NumericMatrix min_mat(n, T);

  // Initialize with NA
  std::fill(max_mat.begin(), max_mat.end(), NA_REAL);
  std::fill(min_mat.begin(), min_mat.end(), NA_REAL);

  for (int i = 0; i < n; i++) {
    int start = p[i];
    int end   = p[i + 1];
    if (start == end) continue;  // no neighbors

    for (int t = 0; t < T; t++) {
      double cur_max = R_NegInf;
      double cur_min = R_PosInf;
      int valid = 0;

      for (int k = start; k < end; k++) {
        int nbr = j[k];  // 0-based column index
        double val = V(nbr, t);
        if (!R_IsNA(val)) {
          if (val > cur_max) cur_max = val;
          if (val < cur_min) cur_min = val;
          valid++;
        }
      }

      if (valid > 0) {
        max_mat(i, t) = cur_max;
        min_mat(i, t) = cur_min;
      }
    }
  }

  return List::create(Named("max") = max_mat,
                      Named("min") = min_mat);
}
')
  USE_RCPP <- TRUE
} else {
  USE_RCPP <- FALSE
}


compute_neighbor_max_min_fast <- function(A, V) {
  A_csr <- as(A, "RsparseMatrix")
  if (USE_RCPP) {
    compute_max_min_csr(A_csr@p, A_csr@j, V)
  } else {
    compute_neighbor_max_min(A, V)
  }
}


# ---- Step 5: Compute neighbor MEAN via sparse matmul (NA-aware) ----
# Already defined above as compute_neighbor_mean


# =============================================================================
# MAIN PIPELINE
# =============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  # cell_data: data.frame/data.table with columns: id, year, + variables
  # id_order: vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique: spdep nb object
  # neighbor_source_vars: character vector of variable names

  cat("Converting to data.table...\n")
  dt <- as.data.table(cell_data)
  setkey(dt, id, year)

  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)

  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n",
              n_cells, n_years, nrow(dt)))

  # --- Build sparse adjacency (once) ---
  cat("Building sparse adjacency matrix...\n")
  t0 <- proc.time()
  A <- build_sparse_adjacency(rook_neighbors_unique, n_cells)
  cat(sprintf("  Adjacency: %d nodes, %d directed edges (%.1f sec)\n",
              nrow(A), nnzero(A), (proc.time() - t0)[3]))

  # --- Cell ID to matrix row mapping ---
  cell_id_map <- setNames(seq_along(id_order), as.character(id_order))
  year_map    <- setNames(seq_along(years), as.character(years))

  # --- Pre-compute row/col indices for reassembly ---
  dt_row_i <- cell_id_map[as.character(dt$id)]
  dt_col_j <- year_map[as.character(dt$year)]
  dt_linear_idx <- cbind(dt_row_i, dt_col_j)

  # --- Process each variable ---
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))
    t1 <- proc.time()

    # Reshape to wide matrix (cells x years)
    V <- long_to_wide_matrix(dt, var_name, cell_id_map, years)

    # Compute neighbor mean
    cat("  Computing neighbor mean...\n")
    mean_mat <- compute_neighbor_mean(A, V, NULL)

    # Compute neighbor max and min
    cat("  Computing neighbor max/min...\n")
    maxmin <- compute_neighbor_max_min_fast(A, V)

    # Extract values back to long format using pre-computed indices
    max_col_name  <- paste0("neighbor_max_", var_name)
    min_col_name  <- paste0("neighbor_min_", var_name)
    mean_col_name <- paste0("neighbor_mean_", var_name)

    dt[, (max_col_name)  := maxmin$max[dt_linear_idx]]
    dt[, (min_col_name)  := maxmin$min[dt_linear_idx]]
    dt[, (mean_col_name) := mean_mat[dt_linear_idx]]

    elapsed <- (proc.time() - t1)[3]
    cat(sprintf("  Done in %.1f sec\n", elapsed))
  }

  cat("All neighbor features computed.\n")
  return(dt)
}


# =============================================================================
# USAGE
# =============================================================================

# Load pre-saved objects
# cell_data              <- readRDS("cell_data.rds")
# id_order               <- readRDS("id_order.rds")
# rook_neighbors_unique  <- readRDS("rook_neighbors_unique.rds")
# rf_model               <- readRDS("rf_model.rds")

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data_enriched <- run_neighbor_feature_pipeline(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# Apply the pre-trained Random Forest (no retraining)
# predictions <- predict(rf_model, newdata = cell_data_enriched)
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + named-vector lookups | Single sparse matrix build (344K nodes) |
| **Mean computation** | 6.46M R-level loops Ã— 5 vars | 5 sparse matrix multiplications (compiled BLAS) |
| **Max/Min computation** | 6.46M R-level loops Ã— 5 vars | CSR iteration in Rcpp (~38.4M ops in C++) |
| **Year handling** | Redundantly encoded in lookup keys | Orthogonal matrix column dimension |
| **Memory** | ~6.46M-element list of integer vectors | ~1.37M-entry sparse matrix + 344KÃ—28 dense matrices (~77 MB each) |
| **Estimated runtime** | 86+ hours | **2â€“10 minutes** (Rcpp) / **10â€“30 min** (pure R fallback) |
| **Numerical equivalence** | Baseline | Identical â€” same neighbor sets, same `max/min/mean` |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |
END_RESPONSE>>>
