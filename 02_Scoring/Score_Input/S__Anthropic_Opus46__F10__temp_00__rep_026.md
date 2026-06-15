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
run_id: Anthropic_F10_T00_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query due to hashing overhead at scale, and doing this 6.46M times is catastrophic. The string concatenation (`paste(id, year, sep="_")`) for ~6.46M Ã— ~4 neighbors â‰ˆ 25M+ string operations is extremely slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in an `lapply`**, extracting subsets of a numeric vector by index, computing max/min/mean. This is called 5 times (once per variable), so ~32.3M R-level function calls with per-element allocation.

3. **The neighbor lookup is year-redundant.** Rook neighbors are a *spatial* topology â€” they don't change across years. Yet the lookup is built at the cell-year level, duplicating the same spatial adjacency structure 28 times. This inflates the lookup from 344K entries to 6.46M entries unnecessarily.

**Root cause:** The implementation treats the problem as a flat row-level operation instead of exploiting the separable structure: **topology is spatial, attributes are spatiotemporal**. The graph adjacency is invariant across years and should be built once over 344K cells, then applied per-year to a matrix of attributes.

## Optimization Strategy

1. **Build a sparse adjacency matrix once** (344,208 Ã— 344,208 CSC matrix) from the `nb` object. This is the graph topology â€” built once, reused 28 Ã— 5 = 140 times.

2. **Reshape each variable into a cell Ã— year matrix** (344,208 rows Ã— 28 columns). This separates spatial topology from temporal attributes.

3. **Compute neighbor statistics via sparse matrix operations:**
   - **Mean:** `A_norm %*% X` where `A_norm` is the row-normalized adjacency matrix (each row sums to 1/degree). This is a single sparse matrixâ€“dense matrix multiply â€” highly optimized in the `Matrix` package.
   - **Max and Min:** Cannot be done via linear algebra. Use a compiled C++ loop via `Rcpp` that iterates over the CSC sparse structure, or use a chunked vectorized approach.

4. **Avoid all string operations, named lookups, and per-row R function calls.**

5. **Memory:** Sparse matrix with ~1.37M non-zeros â‰ˆ 33 MB. Dense matrices 344K Ã— 28 â‰ˆ 77 MB each. Total for 5 variables â‰ˆ 385 MB for source + ~1.15 GB for 15 output matrices. Well within 16 GB.

6. **Time:** Sparse mat-mul for mean: ~0.5s per variable-year â†’ ~70s total. Max/min via Rcpp: ~2-5 minutes total. Full pipeline: **under 10 minutes** vs. 86+ hours.

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original max, min, mean statistics.
# Preserves the pre-trained Random Forest model (no retraining).
# =============================================================================

library(Matrix)
library(data.table)

# Optional but strongly recommended for max/min:
# install.packages("Rcpp")
library(Rcpp)

# ---- Step 0: Ensure cell_data is a data.table for fast operations -----------
cell_dt <- as.data.table(cell_data)

# ---- Step 1: Build spatial ID mapping ---------------------------------------
# id_order is the vector of cell IDs aligned with rook_neighbors_unique (nb object)
# Each element of rook_neighbors_unique[[i]] gives integer indices into id_order

n_cells <- length(id_order)
cat("Number of spatial cells:", n_cells, "\n")

# Create a fast integer mapping: cell_id -> position in id_order
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# ---- Step 2: Build sparse adjacency matrix from nb object (once) ------------
build_adjacency <- function(nb_obj, n) {
  # nb_obj: list of length n, each element is integer vector of neighbor indices
  # Build COO triplets
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0) {
      from_list[[i]] <- rep.int(i, length(nbrs))
      to_list[[i]]   <- nbrs
    }
  }
  
  from_vec <- unlist(from_list, use.names = FALSE)
  to_vec   <- unlist(to_list, use.names = FALSE)
  
  sparseMatrix(
    i = from_vec,
    j = to_vec,
    x = rep(1, length(from_vec)),
    dims = c(n, n),
    repr = "C"   # CSC format, efficient for column operations
  )
}

cat("Building sparse adjacency matrix...\n")
A <- build_adjacency(rook_neighbors_unique, n_cells)
cat("Adjacency matrix: ", nrow(A), "x", ncol(A), 
    " with", nnzero(A), "non-zeros\n")

# Row-normalized version for mean computation
# degree = number of neighbors per cell
degree <- diff(A@p)  # For CSC, this is column counts; we need row counts
# Convert to CSR for row-wise operations
A_csr <- as(A, "RsparseMatrix")
row_degree <- diff(A_csr@p)
# Avoid division by zero for isolated cells
row_degree_safe <- pmax(row_degree, 1L)
# Build row-normalized matrix for mean
# A_norm[i,j] = 1/degree(i) if (i,j) is an edge
A_norm <- A_csr
A_norm@x <- A_norm@x / rep(row_degree_safe, times = diff(A_csr@p))
A_norm <- as(A_norm, "CsparseMatrix")  # Back to CSC for efficient mat-mul

# ---- Step 3: Build cell Ã— year matrices for each variable -------------------
# Sort cell_dt by (id, year) for consistent ordering
setkey(cell_dt, id, year)

# Get the unique years in sorted order
years <- sort(unique(cell_dt$year))
n_years <- length(years)
cat("Number of years:", n_years, "\n")
cat("Expected rows:", n_cells * n_years, " Actual rows:", nrow(cell_dt), "\n")

# Create mapping from cell_dt rows to (cell_position, year_position)
cell_dt[, cell_pos := id_to_pos[as.character(id)]]
year_to_col <- setNames(seq_along(years), as.character(years))
cell_dt[, year_col := year_to_col[as.character(year)]]

# Function to reshape a variable into cell Ã— year matrix
reshape_to_matrix <- function(dt, var_name, n_cells, n_years) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(dt$cell_pos, dt$year_col)] <- dt[[var_name]]
  mat
}

# ---- Step 4: Rcpp function for sparse neighbor max and min ------------------
# This iterates over the CSR structure directly â€” no R-level per-row overhead.

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_neighbor_maxmin(
    IntegerVector row_ptr,   // CSR row pointers (length n_cells + 1), 0-based
    IntegerVector col_idx,   // CSR column indices, 0-based
    NumericMatrix X          // n_cells x n_years matrix of values
) {
  int n = X.nrow();
  int ny = X.ncol();
  
  NumericMatrix max_mat(n, ny);
  NumericMatrix min_mat(n, ny);
  
  for (int i = 0; i < n; i++) {
    int start = row_ptr[i];
    int end   = row_ptr[i + 1];
    
    if (start == end) {
      // No neighbors
      for (int t = 0; t < ny; t++) {
        max_mat(i, t) = NA_REAL;
        min_mat(i, t) = NA_REAL;
      }
      continue;
    }
    
    for (int t = 0; t < ny; t++) {
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      int count = 0;
      
      for (int k = start; k < end; k++) {
        int j = col_idx[k];
        double val = X(j, t);
        if (!R_IsNA(val)) {
          if (val > vmax) vmax = val;
          if (val < vmin) vmin = val;
          count++;
        }
      }
      
      if (count == 0) {
        max_mat(i, t) = NA_REAL;
        min_mat(i, t) = NA_REAL;
      } else {
        max_mat(i, t) = vmax;
        min_mat(i, t) = vmin;
      }
    }
  }
  
  return List::create(
    Named("max_mat") = max_mat,
    Named("min_mat") = min_mat
  );
}
')

# Extract CSR components (0-based for C++)
A_csr2 <- as(A, "RsparseMatrix")
csr_row_ptr <- as.integer(A_csr2@p)    # Already 0-based
csr_col_idx <- as.integer(A_csr2@j)    # Already 0-based

# ---- Step 5: Compute neighbor features for all 5 variables ------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics...\n")
t_start <- proc.time()

for (var_name in neighbor_source_vars) {
  cat("  Processing:", var_name, "... ")
  t0 <- proc.time()
  
  # Reshape to cell Ã— year matrix
  X <- reshape_to_matrix(cell_dt, var_name, n_cells, n_years)
  
  # --- Mean via sparse matrix multiplication ---
  # For cells with no neighbors, A_norm row is all zeros â†’ result is 0, not NA.
  # We need to set those to NA to match original behavior.
  mean_mat <- as.matrix(A_norm %*% X)
  
  # Handle NA propagation: if all neighbor values are NA for a cell-year,
  # the sparse mat-mul gives 0, but we need NA.
  # Also, the original code ignores NAs (uses only non-NA neighbors).
  # Sparse mat-mul treats NA as 0 in X, which is incorrect.
  # We need a corrected approach for mean when NAs exist in X.
  
  # Corrected mean: sum of non-NA neighbor values / count of non-NA neighbor values
  # Step 1: Replace NA with 0 in X for summation
  X_nona <- X
  X_nona[is.na(X_nona)] <- 0
  
  # Step 2: Create indicator matrix (1 where non-NA, 0 where NA)
  X_ind <- matrix(1, nrow = n_cells, ncol = n_years)
  X_ind[is.na(X)] <- 0
  
  # Step 3: Sum of non-NA neighbor values
  sum_mat <- as.matrix(A %*% X_nona)
  
  # Step 4: Count of non-NA neighbors
  count_mat <- as.matrix(A %*% X_ind)
  
  # Step 5: Mean = sum / count, with NA where count == 0
  mean_mat <- ifelse(count_mat > 0, sum_mat / count_mat, NA_real_)
  
  # Also set NA for cells with no neighbors at all
  no_neighbors <- (row_degree == 0)
  if (any(no_neighbors)) {
    mean_mat[no_neighbors, ] <- NA_real_
  }
  
  # --- Max and Min via Rcpp ---
  maxmin <- sparse_neighbor_maxmin(csr_row_ptr, csr_col_idx, X)
  max_mat <- maxmin$max_mat
  min_mat <- maxmin$min_mat
  
  # --- Write results back to cell_dt ---
  # Flatten matrices back to the row order of cell_dt
  # cell_dt has (cell_pos, year_col) mapping
  idx_matrix <- cbind(cell_dt$cell_pos, cell_dt$year_col)
  
  max_col <- paste0("neighbor_max_", var_name)
  min_col <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  cell_dt[, (max_col)  := max_mat[idx_matrix]]
  cell_dt[, (min_col)  := min_mat[idx_matrix]]
  cell_dt[, (mean_col) := mean_mat[idx_matrix]]
  
  elapsed <- (proc.time() - t0)[3]
  cat(round(elapsed, 1), "seconds\n")
  
  # Free intermediate memory
  rm(X, X_nona, X_ind, sum_mat, count_mat, mean_mat, max_mat, min_mat, maxmin)
}

total_elapsed <- (proc.time() - t_start)[3]
cat("Total neighbor feature computation:", round(total_elapsed / 60, 1), "minutes\n")

# ---- Step 6: Clean up helper columns and convert back if needed -------------
cell_dt[, c("cell_pos", "year_col") := NULL]

# Convert back to data.frame if the RF model expects one
cell_data <- as.data.frame(cell_dt)

# ---- Step 7: Apply the pre-trained Random Forest model ----------------------
# The model is NOT retrained. We only call predict().
# Assumes `rf_model` is already loaded in the environment.

cat("Generating predictions with pre-trained Random Forest...\n")
predictions <- predict(rf_model, newdata = cell_data)

cat("Done.\n")
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + named-vector lookups | Single sparse matrix build (344K nodes) |
| **Mean computation** | 6.46M `lapply` calls with `mean()` | Sparse matrix multiplication (`A %*% X`) |
| **Max/Min computation** | 6.46M R-level `lapply` calls | Single Rcpp pass over CSR structure |
| **Topology reuse** | Rebuilt per cell-year (redundant Ã—28) | Built once, reused for all years and variables |
| **String operations** | ~25M+ `paste()` calls | Zero |
| **Estimated runtime** | 86+ hours | **5â€“10 minutes** |
| **Memory** | Large list of 6.46M entries | Sparse matrix (~33 MB) + dense matrices (~77 MB each) |
| **Numerical equivalence** | Baseline | âœ… Preserved (same max, min, mean with NA handling) |
| **RF model** | Pre-trained | âœ… Unchanged, predict-only |

**Key insight:** The graph topology (rook adjacency) is time-invariant. By encoding it as a sparse matrix and reshaping variables into cellÃ—year matrices, all neighbor aggregations become either sparse linear algebra (mean) or a single compiled C++ pass (max/min), eliminating millions of interpreted R function calls.
END_RESPONSE>>>
