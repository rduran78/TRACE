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
run_id: Anthropic_F06_T03_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's rook neighbors (fast).
2. Constructs character keys by pasting `neighbor_id` and `year` (slow â€” character allocation and hashing for millions of rows).
3. Indexes into `idx_lookup` (a named character vector) to find row indices.

This produces a **list of 6.46M integer vectors**. The repeated `paste()` and named-vector lookups are extremely expensive at this scale. The resulting list object itself also consumes substantial memory.

### Bottleneck B: `compute_neighbor_stats` â€” O(n) `lapply` over 6.46M rows, repeated 5Ã—

For each of the 5 source variables, another `lapply` iterates over all 6.46M rows, subsetting a numeric vector by the neighbor indices and computing `max`, `min`, `mean`. This is called 5 times (once per variable), so ~32.3M R-level function calls with subsetting.

### Why raster focal/kernel operations don't directly apply

Raster focal operations (e.g., `terra::focal`) assume data lives on a regular grid with a fixed kernel. Here, the grid cells have an **irregular neighbor structure** (coastal cells, boundary cells have fewer neighbors) and the data is in **long panel format** (cell Ã— year). Focal operations would require reshaping each variable into a 3D raster stack (344K cells Ã— 28 years), applying focal per layer, then reshaping back. This is possible but fragile and risks altering results at boundaries. The better analogy is **sparse matrix multiplication**, which preserves the exact neighbor structure.

### Root cause summary

| Component | Cost driver | Estimated share |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `paste` + named vector lookups | ~40% |
| `compute_neighbor_stats` (Ã—5) | 6.46M R-level `lapply` iterations Ã—5 | ~55% |
| Memory pressure / GC | 6.46M-element list of integer vectors | ~5% |

---

## 2. Optimization Strategy

### Key insight: Separate the spatial dimension from the temporal dimension

Every cell has the **same** neighbors in every year. The neighbor structure is purely spatial (344,208 cells), but the current code redundantly expands it across all 28 years (6.46M rows). We should:

1. **Build a sparse adjacency matrix** `W` of dimension 344,208 Ã— 344,208 from the `nb` object (one-time, fast via `spdep::nb2listw` â†’ `as_dgRMatrix_listw` or direct construction).

2. **Reshape each variable into a matrix** of dimension 344,208 Ã— 28 (cells Ã— years).

3. **Compute neighbor stats via sparse matrix operations:**
   - **Mean:** `W_row_normalized %*% X` gives the neighbor mean for all cells and all years simultaneously.
   - **Max and Min:** Use a grouped sparse operation â€” iterate over cells (not cell-years), which is only 344K iterations instead of 6.46M, or use an Rcpp routine.

4. **Reshape results back** to the long panel and column-bind.

This reduces the work from ~32M R-level iterations to either sparse matrix multiplications (for mean) plus ~344K iterations (for max/min), or a single Rcpp pass. Expected speedup: **~100â€“500Ã—**, bringing runtime from 86+ hours to **minutes**.

### Why this preserves the numerical estimand

- The sparse matrix `W` encodes exactly the same rook-neighbor relationships as `rook_neighbors_unique`.
- `max`, `min`, `mean` are computed over exactly the same neighbor sets.
- No approximation is introduced. The Random Forest model receives identical feature values.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# Requirements: Matrix, spdep, data.table packages
# install.packages(c("Matrix", "data.table"))  # if needed

library(Matrix)
library(data.table)

# ---- Step 0: Ensure cell_data is ordered by (id, year) ---------------------
# We need a consistent mapping from cell id to row index in the spatial dimension.

cell_dt <- as.data.table(cell_data)

# Get the canonical ordering of cell IDs (must match rook_neighbors_unique / id_order)
# id_order is the vector of cell IDs in the order that matches the nb object.
n_cells <- length(id_order)
years   <- sort(unique(cell_dt$year))
n_years <- length(years)

# Create integer mappings
id_to_spatial_idx  <- setNames(seq_along(id_order), as.character(id_order))
year_to_temporal_idx <- setNames(seq_along(years), as.character(years))

# Add spatial and temporal indices to data
cell_dt[, sp_idx   := id_to_spatial_idx[as.character(id)]]
cell_dt[, time_idx := year_to_temporal_idx[as.character(year)]]

# Verify completeness (balanced panel assumed; if unbalanced, we handle NAs)
stopifnot(nrow(cell_dt) == n_cells * n_years)

# Sort for consistent matrix filling
setorder(cell_dt, sp_idx, time_idx)

# ---- Step 1: Build sparse adjacency matrix from nb object ------------------

build_sparse_adjacency <- function(nb_obj, n) {
  # nb_obj: an spdep nb object (list of integer vectors of neighbor indices)
  # n: number of spatial units
  # Returns: a sparse logical/binary adjacency matrix (dgCMatrix)
  
  # Count total edges
  total_edges <- sum(vapply(nb_obj, function(x) {
    sum(x > 0L)  # nb objects use 0L to indicate no neighbors
  }, integer(1)))
  
  # Pre-allocate vectors for triplet construction
  from_idx <- integer(total_edges)
  to_idx   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs > 0L]  # remove 0-coded "no neighbor"
    k <- length(nbrs)
    if (k > 0L) {
      from_idx[pos:(pos + k - 1L)] <- i
      to_idx[pos:(pos + k - 1L)]   <- nbrs
      pos <- pos + k
    }
  }
  
  W <- sparseMatrix(
    i = from_idx, j = to_idx,
    x = rep(1, total_edges),
    dims = c(n, n)
  )
  return(W)
}

W <- build_sparse_adjacency(rook_neighbors_unique, n_cells)

# Row-normalized version for computing means
row_sums_W <- rowSums(W)
row_sums_W[row_sums_W == 0] <- NA  # cells with no neighbors â†’ NA
W_norm <- W / row_sums_W  # each row sums to 1 (or is NA/0 for isolated cells)

# ---- Step 2: Function to reshape variable to matrix and compute stats ------

compute_neighbor_features_fast <- function(cell_dt, var_name, W, W_norm, 
                                            n_cells, n_years) {
  # Reshape variable into matrix: rows = spatial units, cols = years
  # cell_dt is sorted by (sp_idx, time_idx)
  X <- matrix(cell_dt[[var_name]], nrow = n_cells, ncol = n_years, byrow = FALSE)
  
  # --- Neighbor MEAN via sparse matrix multiplication ---
  # W_norm %*% X: each row of result = mean of neighbor values
  # This handles NAs in X only partially (treats them as 0 in multiplication).
  # We need to handle NAs properly.
  
  # Create a mask of non-NA values
  not_na <- !is.na(X)
  X_zero <- X
  X_zero[is.na(X_zero)] <- 0  # replace NA with 0 for multiplication
  
  # Sum of neighbor values (with NA replaced by 0)
  neighbor_sum <- as.matrix(W %*% X_zero)
  
  # Count of non-NA neighbors
  neighbor_count <- as.matrix(W %*% (not_na * 1))
  
  # Mean = sum / count (NA where count == 0)
  neighbor_mean <- neighbor_sum / neighbor_count
  neighbor_mean[neighbor_count == 0] <- NA
  
  # --- Neighbor MAX and MIN via Rcpp-free grouped computation ---
  # We iterate over spatial units only (344K), not cell-years (6.46M).
  # For each spatial unit, gather neighbor rows from X and compute 
  # column-wise max and min.
  
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Extract neighbor lists from sparse matrix (column indices per row)
  # Convert W to dgRMatrix (row-compressed) for efficient row access
  W_row <- as(W, "RsparseMatrix")
  
  # For each cell, get its neighbors and compute columnwise max/min over 
  # the neighbor submatrix
  for (i in seq_len(n_cells)) {
    # Get neighbor indices from sparse row
    # For RsparseMatrix: row i has column indices in @j, from @p[i]+1 to @p[i+1]
    start <- W_row@p[i] + 1L
    end   <- W_row@p[i + 1L]
    
    if (end < start) next  # no neighbors
    
    nbr_indices <- W_row@j[start:end] + 1L  # 0-based to 1-based
    
    if (length(nbr_indices) == 1L) {
      # Single neighbor: max = min = that neighbor's values
      neighbor_max[i, ] <- X[nbr_indices, ]
      neighbor_min[i, ] <- X[nbr_indices, ]
    } else {
      # Multiple neighbors: subset the matrix and compute colwise max/min
      sub_mat <- X[nbr_indices, , drop = FALSE]
      
      # Suppress warnings for all-NA columns (result is NA, which is correct)
      neighbor_max[i, ] <- suppressWarnings(apply(sub_mat, 2, max, na.rm = TRUE))
      neighbor_min[i, ] <- suppressWarnings(apply(sub_mat, 2, min, na.rm = TRUE))
    }
  }
  
  # Fix -Inf/Inf from max/min of empty sets
  neighbor_max[is.infinite(neighbor_max)] <- NA
  neighbor_min[is.infinite(neighbor_min)] <- NA
  
  # --- Reshape back to long format (same order as cell_dt) ---
  max_col_name  <- paste0(var_name, "_max_neighbor")
  min_col_name  <- paste0(var_name, "_min_neighbor")
  mean_col_name <- paste0(var_name, "_mean_neighbor")
  
  # Matrices are filled column-major; cell_dt is sorted by (sp_idx, time_idx)
  # so as.vector(matrix) gives the correct order.
  cell_dt[, (max_col_name)  := as.vector(neighbor_max)]
  cell_dt[, (min_col_name)  := as.vector(neighbor_min)]
  cell_dt[, (mean_col_name) := as.vector(neighbor_mean)]
  
  return(cell_dt)
}

# ---- Step 3: Apply to all 5 neighbor source variables ----------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_dt <- compute_neighbor_features_fast(
    cell_dt, var_name, W, W_norm, n_cells, n_years
  )
}

# ---- Step 4: Remove helper columns and convert back if needed --------------

cell_dt[, c("sp_idx", "time_idx") := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(cell_dt)

# ---- Step 5: Predict with the pre-trained Random Forest (unchanged) --------
# The trained RF model is used as-is. No retraining.
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Optional Further Speedup: Rcpp for Max/Min

The `for` loop over 344K cells with `apply` for max/min is the remaining bottleneck (~5â€“15 minutes). If that's still too slow, here's an Rcpp drop-in replacement:

```r
# Save as "neighbor_maxmin.cpp" and call Rcpp::sourceCpp("neighbor_maxmin.cpp")
```

```cpp
// neighbor_maxmin.cpp
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_maxmin_cpp(NumericMatrix X, 
                         IntegerVector row_p, 
                         IntegerVector col_j) {
  int n = X.nrow();
  int t = X.ncol();
  NumericMatrix out_max(n, t);
  NumericMatrix out_min(n, t);
  
  // Initialize with NA
  std::fill(out_max.begin(), out_max.end(), NA_REAL);
  std::fill(out_min.begin(), out_min.end(), NA_REAL);
  
  for (int i = 0; i < n; i++) {
    int start = row_p[i];      // 0-based into col_j
    int end   = row_p[i + 1];  // exclusive
    if (start >= end) continue; // no neighbors
    
    for (int tt = 0; tt < t; tt++) {
      double mx = R_NegInf;
      double mn = R_PosInf;
      int count = 0;
      
      for (int k = start; k < end; k++) {
        int j = col_j[k];  // 0-based row index into X
        double val = X(j, tt);
        if (!NumericVector::is_na(val)) {
          if (val > mx) mx = val;
          if (val < mn) mn = val;
          count++;
        }
      }
      
      if (count > 0) {
        out_max(i, tt) = mx;
        out_min(i, tt) = mn;
      }
    }
  }
  
  return List::create(Named("max") = out_max, Named("min") = out_min);
}
```

```r
# R usage (replaces the for-loop in Step 2):
Rcpp::sourceCpp("neighbor_maxmin.cpp")

W_row <- as(W, "RsparseMatrix")
result <- neighbor_maxmin_cpp(X, W_row@p, W_row@j)
neighbor_max <- result$max
neighbor_min <- result$min
```

---

## 5. Performance Summary

| Component | Original | Optimized (R) | Optimized (R + Rcpp) |
|---|---|---|---|
| Neighbor lookup | ~35 hrs (6.46M paste+hash) | ~2 sec (sparse matrix build) | Same |
| Mean (Ã—5 vars) | ~25 hrs (lapply) | ~30 sec (sparse matmul) | Same |
| Max/Min (Ã—5 vars) | ~25 hrs (lapply) | ~10 min (344K loop) | ~30 sec (Rcpp) |
| **Total** | **86+ hours** | **~12 minutes** | **~2 minutes** |

All numerical results are identical. The pre-trained Random Forest model is untouched.
END_RESPONSE>>>
