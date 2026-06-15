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
run_id: Anthropic_F10_T00_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each constructed via string-pasting, named-vector lookups (`idx_lookup[neighbor_keys]`), and per-row `lapply`. Named vector lookup in R is O(n) per query on average due to hashing overhead at scale, and doing this 6.46M times with string concatenation is extremely expensive.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in R-level `lapply`**, extracting subsets of a numeric vector per element. This is called 5 times (once per variable), totaling ~32.3 million R-level list iterations with subsetting.

3. **The neighbor topology is year-invariant** (rook contiguity depends only on spatial position), but the lookup is built at the cell-year level, redundantly encoding the same spatial graph 28 times. This inflates the lookup from ~1.37M edges to ~38.4M edge references, and all the string operations scale with the panel length rather than the cross-section.

**Summary**: The 86+ hour runtime is dominated by R-level interpreted loops over millions of list elements with string-key lookups. The algorithm is correct but the data structure and iteration strategy are catastrophically inefficient for R.

---

## Optimization Strategy

### Core Insight
The neighbor graph is **time-invariant**. A cell's rook neighbors are the same in every year. Therefore:

1. **Build the spatial adjacency once** as a sparse matrix (344,208 Ã— 344,208) with ~1.37M nonzero entries.
2. **Reshape each variable into a matrix** of dimension (344,208 cells Ã— 28 years).
3. **Compute neighbor aggregations via sparse matrix operations**: For `mean`, sparse matrixâ€“dense matrix multiplication (`A %*% X`) divided by the row-degree vector gives the neighbor mean in one shot for all cells and all years simultaneously. For `max` and `min`, use a grouped operation over the CSR representation.
4. **Avoid all string operations, all per-row `lapply`, and all list-of-indices structures.**

This reduces the problem from ~32M interpreted R iterations to a handful of sparse matrix multiplications and vectorized grouped operations, bringing runtime from 86+ hours to **minutes**.

### Memory Budget
- Sparse matrix: ~1.37M entries Ã— 3 integers/doubles â‰ˆ 33 MB
- Each dense matrix: 344,208 Ã— 28 Ã— 8 bytes â‰ˆ 77 MB
- 5 variables Ã— 4 matrices (source + 3 stats) â‰ˆ 1.5 GB
- Comfortable within 16 GB RAM.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE PIPELINE
# Sparse-graph neighborhood aggregation via matrix operations
# Numerically equivalent to the original implementation
# =============================================================================

library(Matrix)   # sparse matrices
library(data.table) # fast reshaping and joining

# ---- Step 0: Ensure cell_data is a data.table ----
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ---- Step 1: Build the sparse adjacency matrix (once) ----
# rook_neighbors_unique is an nb object: a list of length n_cells,
# where each element is an integer vector of neighbor indices (into id_order).
# id_order is the vector of cell IDs corresponding to positions 1..n_cells.

build_sparse_adjacency <- function(nb_obj) {
  # nb_obj: list of integer vectors (neighbor indices), length n

  n <- length(nb_obj)
  
  # Pre-count total edges for pre-allocation
  n_edges <- sum(vapply(nb_obj, function(x) {
    # spdep nb objects use 0L to indicate no neighbors
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  # Build COO representation
  from_idx <- integer(n_edges)
  to_idx   <- integer(n_edges)
  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    k <- length(nbrs)
    from_idx[pos:(pos + k - 1L)] <- i
    to_idx[pos:(pos + k - 1L)]   <- nbrs
    pos <- pos + k
  }
  
  # Sparse binary adjacency matrix (row i has 1s in columns that are i's neighbors)
  sparseMatrix(i = from_idx, j = to_idx, x = 1, dims = c(n, n))
}

cat("Building sparse adjacency matrix...\n")
A <- build_sparse_adjacency(rook_neighbors_unique)
n_cells <- nrow(A)

# Row degree vector (number of non-NA neighbors will be adjusted per variable)
degree_vec <- rowSums(A)  # integer-valued, number of rook neighbors per cell

cat(sprintf("Adjacency matrix: %d x %d, %d nonzero entries\n",
            nrow(A), ncol(A), nnzero(A)))

# ---- Step 2: Build cell-index mapping ----
# Map each cell ID to its position in id_order (row/col index in A)
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Determine the sorted unique years
years <- sort(unique(cell_data$year))
n_years <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))

# Map each row of cell_data to (cell_position, year_column)
cell_data[, .cell_pos := id_to_pos[as.character(id)]]
cell_data[, .year_col := year_to_col[as.character(year)]]

# ---- Step 3: Function to reshape a variable into a cell x year matrix ----
reshape_to_matrix <- function(dt, var_name, n_cells, n_years) {
  # Returns an n_cells x n_years matrix
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  vals <- dt[[var_name]]
  cell_pos <- dt$.cell_pos
  year_col <- dt$.year_col
  
  # Vectorized assignment
  idx <- cbind(cell_pos, year_col)
  mat[idx] <- vals
  mat
}

# ---- Step 4: Compute neighbor max, min, mean for one variable ----
# For mean: handle NAs properly to match original behavior.
# Original: for each node, take neighbor values, remove NAs, compute mean.
# This means mean = sum(non-NA neighbor vals) / count(non-NA neighbor vals).
#
# For max/min with NAs: we need grouped operations over the sparse structure.

compute_neighbor_stats_sparse <- function(A, X) {
  # A: n x n sparse adjacency matrix
  # X: n x T dense matrix (may contain NAs)
  # Returns list with max_mat, min_mat, mean_mat (each n x T)
  
  n <- nrow(A)
  n_years <- ncol(X)
  
  # --- MEAN ---
  # Replace NA with 0 for summation, track non-NA counts
  X_nona <- X
  X_nona[is.na(X_nona)] <- 0
  
  # Indicator matrix: 1 where X is not NA, 0 where NA
  X_valid <- matrix(1, nrow = n, ncol = n_years)
  X_valid[is.na(X)] <- 0
  
  # Sum of neighbor values (NAs treated as 0)
  neighbor_sum <- A %*% X_nona          # sparse %*% dense, very fast
  # Count of non-NA neighbors per cell-year
  neighbor_count <- A %*% X_valid       # sparse %*% dense
  
  # Mean = sum / count; where count == 0, result is NA
  mean_mat <- as.matrix(neighbor_sum / neighbor_count)
  mean_mat[as.matrix(neighbor_count) == 0] <- NA_real_
  
  # --- MAX and MIN ---
  # We must iterate over the sparse structure, but we do it efficiently
  # using the CSR (compressed sparse row) representation.
  # dgCMatrix is CSC; we transpose to get rows as columns, or convert to dgRMatrix.
  
  # Convert A to dgRMatrix (CSR) for efficient row-wise access
  # Actually, we'll work with dgCMatrix of t(A): column j of t(A) = row j of A
  At <- as(t(A), "dgCMatrix")  # column j contains the neighbors of node j
  
  max_mat <- matrix(NA_real_, nrow = n, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n, ncol = n_years)
  
  # At@p: column pointers (0-indexed), length n+1
  # At@i: row indices (0-indexed) of nonzero entries
  p <- At@p
  row_idx <- At@i  # 0-indexed
  
  # Process each cell: get its neighbor indices from CSC of At
  # For cell j, neighbors are at row_idx[(p[j]+1):p[j+1]] (converting to 1-indexed)
  # This loop is over 344K cells, not 6.46M cell-years, so it's ~19x fewer iterations.
  # Inside, we do vectorized column operations on the neighbor submatrix.
  
  for (j in seq_len(n)) {
    start <- p[j] + 1L   # 1-indexed start
    end   <- p[j + 1L]   # 1-indexed end (p is 0-indexed, so p[j+1] is already correct)
    
    if (start > end) next  # no neighbors
    
    nbr_indices <- row_idx[start:end] + 1L  # convert to 1-indexed
    
    if (length(nbr_indices) == 1L) {
      # Single neighbor: max = min = that value (or NA)
      max_mat[j, ] <- X[nbr_indices, ]
      min_mat[j, ] <- X[nbr_indices, ]
    } else {
      # Submatrix of neighbor values: k x T
      nbr_vals <- X[nbr_indices, , drop = FALSE]
      # Column-wise max and min, ignoring NAs
      # matrixStats is fast but we avoid extra dependencies; 
      # apply is fine here since inner dimension (k neighbors, typically 2-4) is tiny
      max_mat[j, ] <- apply(nbr_vals, 2, max, na.rm = TRUE)
      min_mat[j, ] <- apply(nbr_vals, 2, min, na.rm = TRUE)
    }
  }
  
  # apply(x, 2, max, na.rm=TRUE) returns -Inf when all NA; convert to NA
  max_mat[is.infinite(max_mat)] <- NA_real_
  min_mat[is.infinite(min_mat)] <- NA_real_
  
  list(max_mat = max_mat, min_mat = min_mat, mean_mat = mean_mat)
}

# ---- Step 5: Flatten matrix back to cell_data column ----
flatten_matrix_to_column <- function(mat, cell_pos, year_col) {
  mat[cbind(cell_pos, year_col)]
}

# ---- Step 6: Main loop over the 5 neighbor source variables ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
t_start <- Sys.time()

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing variable: %s\n", var_name))
  
  # Reshape to cell x year matrix
  X <- reshape_to_matrix(cell_data, var_name, n_cells, n_years)
  
  # Compute neighbor stats via sparse operations
  stats <- compute_neighbor_stats_sparse(A, X)
  
  # Flatten back and add to cell_data
  cell_data[, paste0(var_name, "_neighbor_max")  := flatten_matrix_to_column(stats$max_mat,  .cell_pos, .year_col)]
  cell_data[, paste0(var_name, "_neighbor_min")  := flatten_matrix_to_column(stats$min_mat,  .cell_pos, .year_col)]
  cell_data[, paste0(var_name, "_neighbor_mean") := flatten_matrix_to_column(stats$mean_mat, .cell_pos, .year_col)]
  
  # Free memory
  rm(X, stats)
  gc()
}

t_end <- Sys.time()
cat(sprintf("Neighbor features computed in %.1f minutes.\n", 
            as.numeric(difftime(t_end, t_start, units = "mins"))))

# ---- Step 7: Clean up temporary columns ----
cell_data[, .cell_pos := NULL]
cell_data[, .year_col := NULL]

# ---- Step 8: Apply the pre-trained Random Forest (unchanged) ----
# The model object (e.g., `rf_model`) is already in memory.
# Predict using the enriched cell_data with all ~110 predictor variables.
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized | Equivalence |
|-----------|----------|-----------|-------------|
| **Mean** | `mean(vals[!is.na(vals)])` = sum/count of non-NA | `(A %*% X_nona) / (A %*% X_valid)` = identical sum/count per cell-year | âœ… Exact |
| **Max** | `max(vals[!is.na(vals)])` | `apply(X[nbrs,], 2, max, na.rm=TRUE)` over same neighbor set | âœ… Exact |
| **Min** | `min(vals[!is.na(vals)])` | `apply(X[nbrs,], 2, min, na.rm=TRUE)` over same neighbor set | âœ… Exact |
| **NA handling** | Returns `c(NA,NA,NA)` when no valid neighbors | All three paths produce `NA` for zero-count / all-NA | âœ… Exact |

The adjacency matrix `A` encodes exactly the same directed edges as `rook_neighbors_unique`, so the neighbor sets are identical.

---

## Performance Analysis

| Aspect | Original | Optimized | Speedup |
|--------|----------|-----------|---------|
| Neighbor lookup construction | ~6.46M string-paste + named lookups | One sparse matrix build (344K rows) | ~100x |
| Mean computation (per var) | 6.46M R list iterations | 2 sparse Ã— dense matmuls (344K Ã— 28) | ~500x |
| Max/Min computation (per var) | 6.46M R list iterations | 344K iterations, each over ~4 neighbors Ã— 28 years vectorized | ~19x |
| Total estimated runtime | 86+ hours | **5â€“15 minutes** | ~350â€“1000x |
| Peak memory | Comparable (list overhead vs. matrix) | ~2â€“3 GB | Fits 16 GB âœ… |

### Optional Further Speedup for Max/Min

If the 344K-cell loop for max/min is still a bottleneck, replace it with `Rcpp`:

```r
# Optional: Rcpp version for max/min (drops the loop from ~5 min to ~10 sec)
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_maxmin_cpp(IntegerVector p, IntegerVector i, 
                         NumericMatrix X) {
  int n = X.nrow();
  int T = X.ncol();
  NumericMatrix max_mat(n, T);
  NumericMatrix min_mat(n, T);
  
  // Initialize to NA
  std::fill(max_mat.begin(), max_mat.end(), NA_REAL);
  std::fill(min_mat.begin(), min_mat.end(), NA_REAL);
  
  for (int j = 0; j < n; j++) {
    int start = p[j];
    int end = p[j + 1];
    if (start == end) continue;  // no neighbors
    
    for (int t = 0; t < T; t++) {
      double mx = R_NegInf;
      double mn = R_PosInf;
      int valid = 0;
      for (int k = start; k < end; k++) {
        double val = X(i[k], t);
        if (!R_IsNA(val)) {
          if (val > mx) mx = val;
          if (val < mn) mn = val;
          valid++;
        }
      }
      if (valid > 0) {
        max_mat(j, t) = mx;
        min_mat(j, t) = mn;
      }
    }
  }
  return List::create(Named("max_mat") = max_mat,
                      Named("min_mat") = min_mat);
}
')

# Usage inside compute_neighbor_stats_sparse:
# At <- as(t(A), "dgCMatrix")
# result <- neighbor_maxmin_cpp(At@p, At@i, X)
```

This brings the entire pipeline to **under 5 minutes** on a 16 GB laptop while preserving exact numerical equivalence with the original output and leaving the pre-trained Random Forest model completely untouched.
END_RESPONSE>>>
