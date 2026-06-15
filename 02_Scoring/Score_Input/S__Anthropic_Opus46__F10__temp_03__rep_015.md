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
run_id: Anthropic_F10_T03_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each constructed via string-pasting, named-vector lookup, and `NA` filtering. This is an O(n) loop over millions of rows with expensive string operations (`paste`, named vector indexing). The named-vector `idx_lookup` with 6.46M entries makes each `[` lookup a hash-table probe, but doing it millions of times inside `lapply` with per-call `paste` is extremely slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements** five times (once per variable). Each iteration extracts a subset of a numeric vector, removes NAs, and computes max/min/mean. The R-level `lapply` loop over 6.46M elements is inherently slow â€” this is called 5 times, totaling ~32.3 million R function calls.

3. **The topology is year-invariant but the lookup is rebuilt as if it's year-specific.** Rook neighbors are a spatial relationship that doesn't change across years. The current code embeds year into the lookup keys, effectively duplicating the same adjacency structure 28 times and doing string matching to resolve it.

**Root cause summary:** The 86+ hour runtime comes from ~6.46M R-level iterations with string operations (build), followed by 5 Ã— 6.46M R-level iterations with subsetting (stats) â€” all in pure interpreted R with no vectorization or compiled-code acceleration.

---

## Optimization Strategy

### Key Insight: Separate Topology from Time

The rook-neighbor graph is purely spatial (344,208 nodes). The panel has 28 years. We can:

1. **Build a sparse adjacency matrix once** from the `spdep::nb` object (344,208 Ã— 344,208 sparse matrix). This is a one-time cost.
2. **Reshape each variable into a 344,208 Ã— 28 matrix** (cells Ã— years).
3. **Use sparse matrixâ€“dense matrix multiplication** to compute neighbor sums and neighbor counts in one vectorized operation per variable. Then derive max/min/mean.

For **mean**: If `A` is the binary adjacency matrix and `X` is the cellÃ—year matrix, then `A %*% X` gives neighbor sums, and `A %*% 1_matrix` (or row-sums of A) gives neighbor counts. Mean = sum / count.

For **max and min**: Sparse matrix multiplication doesn't directly give max/min. We must iterate over neighbors. However, we can do this **at the spatial level only** (344,208 cells, not 6.46M cell-years) using compiled C++ code via `Rcpp`, operating on the 344,208 Ã— 28 matrix. This reduces the loop from 6.46M to 344K iterations, each processing 28 years vectorially.

### Plan

| Step | Method | Complexity |
|------|--------|------------|
| Build sparse adjacency matrix | `spdep::nb2listw` â†’ `sparseMatrix` or direct construction | One-time, fast |
| Compute neighbor **mean** | Sparse matrix Ã— dense matrix (`A %*% X / degree`) | Vectorized, seconds |
| Compute neighbor **max, min** | Rcpp loop over 344K cells, each accessing neighbor rows of the year-matrix | Compiled, ~seconds per variable |
| Reshape results back to panel | Matrix indexing | Vectorized |

This eliminates all string operations, reduces R-level iteration from millions to zero (or near-zero), and leverages BLAS/sparse algebra.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original max, min, mean statistics.
# Preserves the pre-trained Random Forest model (no retraining).
# =============================================================================

library(Matrix)    # sparse matrix operations
library(data.table) # fast data manipulation
library(Rcpp)      # compiled C++ for max/min

# ---- Step 0: Prepare data.table for fast manipulation ----
# Assume cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# Assume rook_neighbors_unique is an spdep::nb object (list of integer vectors)
# Assume id_order is the vector of cell IDs corresponding to indices 1..N in the nb object

dt <- as.data.table(cell_data)

N_cells <- length(id_order)   # 344,208
years   <- sort(unique(dt$year))  # 1992:2019
N_years <- length(years)      # 28

# Create mappings
cell_id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))
year_to_col_idx        <- setNames(seq_along(years), as.character(years))

# Ensure dt is sorted by (id, year) for consistent matrix construction
setkey(dt, id, year)

# ---- Step 1: Build sparse binary adjacency matrix (one-time) ----
# From the nb object, construct a sparse matrix A of dimension N_cells x N_cells

build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj: list of length n, each element is integer vector of neighbor indices
  # Construct COO triplets
  from_idx <- rep(seq_len(n), times = lengths(nb_obj))
  to_idx   <- unlist(nb_obj)
  
  # Remove any 0-length entries (islands) â€” they simply contribute no edges
  sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = rep(1, length(from_idx)),
    dims = c(n, n),
    giveCsparse = TRUE
  )
}

cat("Building sparse adjacency matrix...\n")
A <- build_adjacency_matrix(rook_neighbors_unique, N_cells)
# Degree vector (number of neighbors per cell)
degree_vec <- as.numeric(rowSums(A))  # length N_cells

cat(sprintf("Adjacency matrix: %d x %d, %d non-zeros\n", 
            nrow(A), ncol(A), nnzero(A)))

# ---- Step 2: Function to reshape a variable into N_cells x N_years matrix ----

variable_to_matrix <- function(dt, var_name, cell_id_to_spatial_idx, year_to_col_idx, 
                                N_cells, N_years) {
  # Returns a matrix M where M[i, j] = value of var_name for spatial cell i in year j
  # NA for missing cell-year combinations
  
  row_idx <- cell_id_to_spatial_idx[as.character(dt$id)]
  col_idx <- year_to_col_idx[as.character(dt$year)]
  
  M <- matrix(NA_real_, nrow = N_cells, ncol = N_years)
  M[cbind(row_idx, col_idx)] <- dt[[var_name]]
  M
}

# ---- Step 3: Rcpp function for neighbor max and min ----
# This iterates over N_cells (344K) not N_rows (6.46M), and is compiled.

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_max_min_cpp(IntegerVector Ap, IntegerVector Aj, 
                          NumericMatrix X, int n_cells, int n_years) {
  // Ap: row pointers of CSR sparse matrix (length n_cells + 1), 0-indexed
  // Aj: column indices of CSR sparse matrix, 0-indexed
  // X: n_cells x n_years matrix of variable values
  // Returns list with two matrices: max_mat and min_mat (n_cells x n_years)
  
  NumericMatrix max_mat(n_cells, n_years);
  NumericMatrix min_mat(n_cells, n_years);
  
  // Initialize with NA
  std::fill(max_mat.begin(), max_mat.end(), NA_REAL);
  std::fill(min_mat.begin(), min_mat.end(), NA_REAL);
  
  for (int i = 0; i < n_cells; i++) {
    int start = Ap[i];
    int end   = Ap[i + 1];
    int n_neighbors = end - start;
    
    if (n_neighbors == 0) continue;
    
    for (int t = 0; t < n_years; t++) {
      double cur_max = NA_REAL;
      double cur_min = NA_REAL;
      bool found = false;
      
      for (int k = start; k < end; k++) {
        int j = Aj[k];  // neighbor spatial index (0-indexed)
        double val = X(j, t);
        if (!R_IsNA(val)) {
          if (!found) {
            cur_max = val;
            cur_min = val;
            found = true;
          } else {
            if (val > cur_max) cur_max = val;
            if (val < cur_min) cur_min = val;
          }
        }
      }
      
      if (found) {
        max_mat(i, t) = cur_max;
        min_mat(i, t) = cur_min;
      }
    }
  }
  
  return List::create(Named("max_mat") = max_mat,
                      Named("min_mat") = min_mat);
}
')

# ---- Step 4: Compute neighbor mean via sparse matrix multiplication ----

compute_neighbor_mean_matrix <- function(A, X, degree_vec) {
  # A %*% X gives sum of neighbor values for each cell-year
  # Divide by degree to get mean
  # Handle degree == 0 (islands) -> NA
  
  neighbor_sum <- as.matrix(A %*% X)  # N_cells x N_years dense matrix
  
  # Count non-NA neighbors per cell-year:
  # We need count of non-NA neighbor values, not just degree
  # Because some neighbors may have NA for that variable-year
  not_na_indicator <- ifelse(is.na(X), 0, 1)
  neighbor_count <- as.matrix(A %*% not_na_indicator)
  
  mean_mat <- neighbor_sum / neighbor_count
  mean_mat[neighbor_count == 0] <- NA_real_
  
  # Also fix the sum matrix: if a cell has neighbors but all are NA, 
  # A %*% X will give 0 (since NA was not handled). We need to zero out NAs in X first.
  mean_mat
}

compute_neighbor_mean_matrix_correct <- function(A, X_raw, degree_vec) {
  # Replace NA with 0 for multiplication, track counts separately
  X <- X_raw
  X[is.na(X)] <- 0
  
  neighbor_sum <- as.matrix(A %*% X)
  
  not_na <- ifelse(is.na(X_raw), 0, 1)
  neighbor_count <- as.matrix(A %*% not_na)
  
  mean_mat <- neighbor_sum / neighbor_count
  mean_mat[neighbor_count == 0] <- NA_real_
  mean_mat
}

# ---- Step 5: Extract CSR representation for Rcpp ----
# Matrix package stores dgCMatrix in CSC (compressed sparse column).
# We need CSR (compressed sparse row). Transpose to get CSR from CSC of A^T,
# or convert directly.

cat("Preparing CSR representation for Rcpp...\n")
A_csr <- as(A, "RsparseMatrix")  # dgRMatrix: CSR format
# A_csr@p: row pointers (0-indexed, length N_cells + 1)
# A_csr@j: column indices (0-indexed)

Ap <- A_csr@p   # integer vector, length N_cells + 1
Aj <- A_csr@j   # integer vector, 0-indexed column indices

# ---- Step 6: Main loop over variables ----

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We need to map results back from matrix form to the data.table rows
# Create the row-index and col-index vectors for the dt rows
dt_row_spatial_idx <- cell_id_to_spatial_idx[as.character(dt$id)]
dt_col_year_idx    <- year_to_col_idx[as.character(dt$year)]
dt_matrix_idx      <- cbind(dt_row_spatial_idx, dt_col_year_idx)

cat(sprintf("Processing %d variables across %d cells x %d years (%d rows)...\n",
            length(neighbor_source_vars), N_cells, N_years, nrow(dt)))

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Variable: %s\n", var_name))
  
  # Step 6a: Reshape variable to matrix
  X_raw <- variable_to_matrix(dt, var_name, cell_id_to_spatial_idx, 
                               year_to_col_idx, N_cells, N_years)
  
  # Step 6b: Compute neighbor mean via sparse matmul
  cat("    Computing mean...\n")
  mean_mat <- compute_neighbor_mean_matrix_correct(A, X_raw, degree_vec)
  
  # Step 6c: Compute neighbor max and min via Rcpp
  cat("    Computing max and min...\n")
  maxmin <- neighbor_max_min_cpp(Ap, Aj, X_raw, N_cells, N_years)
  max_mat <- maxmin$max_mat
  min_mat <- maxmin$min_mat
  
  # Step 6d: Map back to data.table rows
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  dt[, (col_max)  := max_mat[dt_matrix_idx]]
  dt[, (col_min)  := min_mat[dt_matrix_idx]]
  dt[, (col_mean) := mean_mat[dt_matrix_idx]]
  
  # Free intermediate matrices
  rm(X_raw, mean_mat, max_mat, min_mat, maxmin)
  gc()
  
  cat(sprintf("    Done: added %s, %s, %s\n", col_max, col_min, col_mean))
}

# ---- Step 7: Convert back to data.frame if needed for predict() ----
cell_data <- as.data.frame(dt)

cat("Neighbor feature engineering complete.\n")
cat(sprintf("Final dataset: %d rows x %d columns\n", nrow(cell_data), ncol(cell_data)))

# ---- Step 8: Apply pre-trained Random Forest (unchanged) ----
# The RF model object is already in memory (e.g., `rf_model`)
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized | Equivalence |
|-----------|----------|-----------|-------------|
| **max** | `max(vals[neighbor_idx])` after NA removal | Rcpp iterates over same neighbors (from same `nb` object), same values, computes max identically | âœ… Exact |
| **min** | `min(vals[neighbor_idx])` after NA removal | Same Rcpp logic | âœ… Exact |
| **mean** | `mean(vals[neighbor_idx])` after NA removal | `(A %*% X_zeroed) / (A %*% not_na_indicator)` = sum of non-NA values / count of non-NA values = arithmetic mean | âœ… Exact (up to floating-point associativity, typically <1e-14 relative error) |

The adjacency matrix `A` is constructed from the identical `rook_neighbors_unique` nb object, so the neighbor sets are identical.

---

## Performance Analysis

| Component | Original | Optimized | Speedup |
|-----------|----------|-----------|---------|
| Build lookup | ~6.46M string ops in R | One sparse matrix build (344K nodes) | ~100â€“500Ã— |
| Mean (per var) | 6.46M R `lapply` calls | One sparse matmul (CHOLMOD/BLAS) | ~1000Ã— |
| Max/Min (per var) | 6.46M R `lapply` calls | 344K Rcpp iterations Ã— 28 years (vectorized inner loop) | ~200â€“500Ã— |
| Total variables | 5 Ã— above | 5 Ã— above | â€” |
| **Estimated total** | **86+ hours** | **~2â€“10 minutes** | **~500â€“2500Ã—** |

**Memory footprint:** The sparse matrix is ~1.37M non-zeros Ã— 12 bytes â‰ˆ 16 MB. Each dense cellÃ—year matrix is 344,208 Ã— 28 Ã— 8 bytes â‰ˆ 77 MB. Peak usage with 3 intermediate matrices per variable â‰ˆ 250 MB + base data (~2â€“3 GB for 6.46M Ã— 110 columns). Well within 16 GB.
END_RESPONSE>>>
