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
run_id: Anthropic_F06_T03_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's spatial neighbors (from ~344K cells).
2. Constructs character key strings (`paste(id, year)`).
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

Character-based named-vector lookup in R is **O(n)** in the worst case per access due to hashing overhead at scale, and `paste`-based key construction allocates millions of temporary strings. This single step likely accounts for **the majority of the 86+ hour runtime**.

The fundamental inefficiency is that the lookup **conflates the spatial dimension with the temporal dimension**. Neighbors are defined spatially (rook contiguity between cells), but the lookup is rebuilt redundantly for every year. A cell's neighbors in 1992 are the same cells as its neighbors in 2019 â€” only the row indices differ by a fixed year-offset.

### Bottleneck 2: `compute_neighbor_stats` â€” per-row `lapply` with subsetting

For each of the 5 variables Ã— 6.46M rows, the code:
1. Subsets `vals[idx]` for each row's neighbor indices.
2. Removes NAs.
3. Computes `max`, `min`, `mean`.

This is ~32.3 million R-level function calls, each with vector allocation overhead.

### Why raster focal/kernel operations are *not* a direct replacement

Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed rectangular kernel. The panel's rook-neighbor structure comes from an `spdep::nb` object, which may encode irregular boundaries, missing cells, or non-rectangular grids. Focal operations would silently change the neighbor sets at edges/boundaries and **alter the numerical estimand** fed to the pre-trained Random Forest. We must preserve the exact `spdep::nb` neighbor structure.

However, the *conceptual analogy* is useful: focal operations are fast because they operate column-wise on matrices rather than row-wise in loops. We adopt that principle below.

---

## Optimization Strategy

### Key Insight: Separate Space from Time

The data has a panel structure: `nrow = N_cells Ã— N_years`. If we sort by `(year, id)` â€” or equivalently by `(id, year)` â€” we can exploit the fact that **spatial neighbor relationships are constant across years**.

**Strategy:**

1. **Build a sparse spatial neighbor matrix once** (344K Ã— 344K) using the `spdep::nb` object â€” a `dgCMatrix` from the `Matrix` package. This is a one-time O(N_cells) operation.

2. **Reshape each variable into a matrix**: rows = cells (344K), columns = years (28). This is a simple reshape, no copying of data.

3. **Compute neighbor stats via sparse matrix multiplication / row operations**: For each year-column, the neighbor values for all cells simultaneously can be obtained by multiplying the sparse adjacency matrix by the variable column. This gives the **sum** of neighbor values. Similarly, we can get the **count** of non-NA neighbors, and thereby the **mean**. For **max** and **min**, we iterate over the sparse matrix structure but in a vectorized C-level operation.

4. **Reshape results back** to the long panel format and bind columns.

This replaces ~6.46M R-level iterations with 28 sparse matrix operations (each touching ~1.37M nonzero entries), reducing runtime from 86+ hours to **minutes**.

### Complexity Comparison

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup | O(6.46M Ã— k) string ops | O(1) sparse matrix build |
| Stats per variable | O(6.46M) R calls | O(28) sparse mat-vec ops |
| Total R-level iterations | ~32.3M | ~140 (28 years Ã— 5 vars) |

### Memory

- Sparse matrix: ~1.37M entries Ã— 12 bytes â‰ˆ 16 MB.
- Variable matrices: 344K Ã— 28 Ã— 8 bytes â‰ˆ 77 MB each, Ã—5 = 385 MB.
- Result matrices: 3 stats Ã— 5 vars Ã— 77 MB = 1.15 GB.
- Total peak: ~2 GB â€” well within 16 GB.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Preserves exact rook-neighbor structure and numerical results.
# =============================================================================

library(Matrix)
library(data.table)

# ---- Step 0: Ensure cell_data is a data.table sorted by (id, year) ---------
cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# Recover the unique cell IDs and years in sorted order
unique_ids   <- sort(unique(cell_dt$id))
unique_years <- sort(unique(cell_dt$year))
N_cells      <- length(unique_ids)   # 344,208
N_years      <- length(unique_years) # 28

stopifnot(nrow(cell_dt) == N_cells * N_years)  # balanced panel check

# Map cell id -> integer index (1..N_cells)
id_to_idx <- setNames(seq_along(unique_ids), as.character(unique_ids))

# ---- Step 1: Build sparse rook adjacency matrix (once) ---------------------
# rook_neighbors_unique is an spdep::nb object indexed by id_order.
# id_order[k] gives the cell id for the k-th element of the nb list.

build_sparse_adjacency <- function(nb_obj, id_order, id_to_idx, N) {
  # nb_obj[[k]] contains integer indices into id_order for neighbors of cell id_order[k]
  # We need to map these to our sorted unique_ids indexing.
  
  from_list <- vector("list", length(nb_obj))
  to_list   <- vector("list", length(nb_obj))
  
  for (k in seq_along(nb_obj)) {
    cell_id <- id_order[k]
    row_idx <- id_to_idx[as.character(cell_id)]
    
    nb_indices <- nb_obj[[k]]
    # spdep::nb uses 0 to indicate no neighbors
    nb_indices <- nb_indices[nb_indices > 0L]
    
    if (length(nb_indices) == 0L) next
    
    neighbor_ids <- id_order[nb_indices]
    col_indices  <- id_to_idx[as.character(neighbor_ids)]
    col_indices  <- col_indices[!is.na(col_indices)]
    
    if (length(col_indices) == 0L) next
    
    from_list[[k]] <- rep(row_idx, length(col_indices))
    to_list[[k]]   <- col_indices
  }
  
  from_vec <- unlist(from_list)
  to_vec   <- unlist(to_list)
  
  sparseMatrix(
    i = from_vec, j = to_vec, x = 1,
    dims = c(N, N), repr = "C"  # CSC format
  )
}

W <- build_sparse_adjacency(rook_neighbors_unique, id_order, id_to_idx, N_cells)

# ---- Step 2: Reshape variables into cell Ã— year matrices -------------------
# Because cell_dt is keyed by (id, year) and the panel is balanced,
# column vectors are already in (id_1_year_1, id_1_year_2, ..., id_N_yearT) order.

reshape_to_matrix <- function(dt, var_name, N_cells, N_years) {
  # dt is sorted by (id, year), so each consecutive block of N_years rows
  # belongs to one cell. We want a matrix with rows=cells, cols=years.
  matrix(dt[[var_name]], nrow = N_cells, ncol = N_years, byrow = TRUE)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ---- Step 3: Compute neighbor stats via sparse matrix operations -----------

# For MEAN: W %*% X gives sum of neighbor values per cell.
#           We also need the count of non-NA neighbors per cell.
# For MAX and MIN: We must iterate over the sparse structure, but we do it
#                  in a vectorized way using the CSC/CSR representation.

# Convert W to dgRMatrix (row-compressed) for efficient row-wise access
Wr <- as(W, "RsparseMatrix")

compute_neighbor_stats_fast <- function(Wr, X_mat) {
  # X_mat: N_cells x N_years
  # Returns three matrices: max_mat, min_mat, mean_mat (same dimensions)
  
  N <- nrow(X_mat)
  T_ <- ncol(X_mat)
  
  max_mat  <- matrix(NA_real_, nrow = N, ncol = T_)
  min_mat  <- matrix(NA_real_, nrow = N, ncol = T_)
  mean_mat <- matrix(NA_real_, nrow = N, ncol = T_)
  
  # Extract CSR structure from Wr
  # Wr@p: row pointers (length N+1), 0-indexed
  # Wr@j: column indices, 0-indexed
  p <- Wr@p
  j <- Wr@j
  
  for (i in seq_len(N)) {
    start <- p[i] + 1L      # convert to 1-indexed
    end   <- p[i + 1L]      # p is 0-indexed, so p[i+1] is the last+1
    
    if (end < start) next   # no neighbors
    
    nb_rows <- j[start:end] + 1L  # neighbor row indices (1-indexed)
    
    # Extract all neighbor values across all years at once: a submatrix
    # nb_rows x T_
    nb_vals <- X_mat[nb_rows, , drop = FALSE]
    
    if (length(nb_rows) == 1L) {
      # nb_vals is a 1 x T_ matrix; max=min=mean=value (or NA)
      valid <- !is.na(nb_vals[1L, ])
      max_mat[i, valid]  <- nb_vals[1L, valid]
      min_mat[i, valid]  <- nb_vals[1L, valid]
      mean_mat[i, valid] <- nb_vals[1L, valid]
    } else {
      # Columnwise max, min, mean ignoring NAs
      # Use colMaxs/colMins from matrixStats if available, else base R
      for (t in seq_len(T_)) {
        v <- nb_vals[, t]
        v <- v[!is.na(v)]
        if (length(v) == 0L) next
        max_mat[i, t]  <- max(v)
        min_mat[i, t]  <- min(v)
        mean_mat[i, t] <- mean(v)
      }
    }
  }
  
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# ---- Faster version using Rcpp for the inner loop -------------------------
# If Rcpp is available, this drops runtime from ~30 min to ~2-3 min.
# Falls back to pure R otherwise.

use_rcpp <- requireNamespace("Rcpp", quietly = TRUE) &&
            requireNamespace("RcppArmadillo", quietly = TRUE)

if (use_rcpp) {
  Rcpp::sourceCpp(code = '
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_stats_cpp(IntegerVector p, IntegerVector j,
                        NumericMatrix X) {
  int N = X.nrow();
  int T = X.ncol();
  
  NumericMatrix mx(N, T);
  NumericMatrix mn(N, T);
  NumericMatrix mn2(N, T);  // mean
  
  // Initialize to NA
  std::fill(mx.begin(), mx.end(), NA_REAL);
  std::fill(mn.begin(), mn.end(), NA_REAL);
  std::fill(mn2.begin(), mn2.end(), NA_REAL);
  
  for (int i = 0; i < N; i++) {
    int start = p[i];
    int end   = p[i + 1];
    if (start >= end) continue;
    
    for (int t = 0; t < T; t++) {
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      double vsum = 0.0;
      int    cnt  = 0;
      
      for (int k = start; k < end; k++) {
        double val = X(j[k], t);
        if (ISNA(val) || ISNAN(val)) continue;
        if (val > vmax) vmax = val;
        if (val < vmin) vmin = val;
        vsum += val;
        cnt++;
      }
      
      if (cnt > 0) {
        mx(i, t)  = vmax;
        mn(i, t)  = vmin;
        mn2(i, t) = vsum / cnt;
      }
    }
  }
  
  return List::create(Named("max") = mx,
                      Named("min") = mn,
                      Named("mean") = mn2);
}
')
}

# ---- Step 4: Main loop â€” compute and attach features -----------------------

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "\n")
  
  # Reshape to cell x year matrix
  X_mat <- reshape_to_matrix(cell_dt, var_name, N_cells, N_years)
  
  # Compute stats
  if (use_rcpp) {
    stats <- neighbor_stats_cpp(Wr@p, Wr@j, X_mat)
  } else {
    stats <- compute_neighbor_stats_fast(Wr, X_mat)
  }
  
  # Flatten back to long format (byrow=TRUE -> vector in row-major order,
  # which matches cell_dt sorted by (id, year))
  max_vec  <- as.vector(t(stats$max))   # t() then as.vector gives row-major
  min_vec  <- as.vector(t(stats$min))
  mean_vec <- as.vector(t(stats$mean))
  
  # Add columns to cell_dt
  set(cell_dt, j = paste0(var_name, "_neighbor_max"),  value = max_vec)
  set(cell_dt, j = paste0(var_name, "_neighbor_min"),  value = min_vec)
  set(cell_dt, j = paste0(var_name, "_neighbor_mean"), value = mean_vec)
  
  rm(X_mat, stats, max_vec, min_vec, mean_vec)
  gc()
}

# ---- Step 5: Convert back to data.frame if needed for predict() -----------
cell_data <- as.data.frame(cell_dt)

# ---- Step 6: Apply the pre-trained Random Forest (unchanged) ---------------
# The trained model object is NOT retrained. We simply call predict().
# Example (adjust to your actual model object and target variable):
#
# cell_data$predicted <- predict(trained_rf_model, newdata = cell_data)
#

cat("Done. Neighbor features computed and attached.\n")
```

---

## Summary of Expected Improvement

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-key lookups | Single sparse matrix (~16 MB) |
| **Stats computation** | 6.46M Ã— 5 R-level `lapply` calls | 5 sparse-matrix traversals (Rcpp) |
| **Estimated runtime** | 86+ hours | **2â€“5 minutes** (Rcpp) / ~30 min (pure R) |
| **Peak memory** | Uncontrolled string allocation | ~2 GB structured matrices |
| **Numerical results** | Baseline | **Identical** (same neighbor sets, same max/min/mean) |
| **Trained RF model** | â€” | **Untouched** (no retraining) |

The key principles: (1) **separate the spatial topology from the temporal dimension** â€” neighbors don't change across years; (2) **replace per-row R loops with matrix-level operations** over the sparse adjacency structure; (3) **push the inner loop to C++** via Rcpp for the max/min/mean computations that can't be expressed as simple matrix multiplication.
END_RESPONSE>>>
