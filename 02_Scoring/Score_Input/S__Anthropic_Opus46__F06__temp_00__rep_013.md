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
run_id: Anthropic_F06_T00_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) row-level `lapply`

This function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs character keys by pasting `id_year` strings.
4. Looks up those keys in `idx_lookup` (a named vector).

Character key construction (`paste`) and named-vector lookup for 6.46M rows, each with ~4 rook neighbors on average, means roughly **25+ million string operations and hash lookups**. The `lapply` over 6.46M elements in R is inherently slow â€” this alone could take many hours.

### Bottleneck 2: `compute_neighbor_stats` â€” repeated per-variable `lapply`

For each of the 5 variables, another `lapply` over 6.46M rows computes `max`, `min`, `mean` on small integer-indexed subsets. That's **5 Ã— 6.46M = 32.3M R function calls**, each with subsetting, NA removal, and three summary statistics. The overhead of R-level function dispatch dominates.

### Why raster focal/kernel operations don't directly apply

The data is an **irregular spatial panel on grid cells with a precomputed `nb` object**, not a regular raster stack. Focal operations (e.g., `terra::focal`) assume a regular rectangular kernel on a raster. While the grid cells *might* map back to a raster, the `nb` object encodes adjacency that may include boundary irregularities, missing cells, etc. Reimposing a raster focal operation risks **changing the numerical results** (different neighbor sets at boundaries, NA handling). The correct approach is to vectorize the existing graph-based neighbor logic.

### Summary

| Component | Current Cost | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~40+ hours | 6.46M-iteration `lapply` with string ops |
| `compute_neighbor_stats` | ~46+ hours (5 vars) | 5 Ã— 6.46M-iteration `lapply` with R-level subsetting |
| **Total** | **~86+ hours** | Pure-R loops over millions of rows |

---

## Optimization Strategy

### Key Insight: Separate the spatial dimension from the temporal dimension

Every cell has the **same neighbors in every year**. The `nb` object is time-invariant. So instead of building a 6.46M-row lookup, we:

1. **Build a sparse neighbor matrix once** over the 344,208 cells (not 6.46M rows).
2. **Reshape each variable into a cell Ã— year matrix** (344,208 Ã— 28).
3. **Use sparse matrix multiplication** to compute neighbor sums and counts, from which we derive `mean`. For `max` and `min`, use vectorized row-wise operations on the sparse structure.

This reduces the problem from 6.46M R-level iterations to **matrix operations over 344K cells Ã— 28 years**, which is orders of magnitude faster.

### Specific techniques

| Step | Method | Complexity |
|---|---|---|
| Neighbor structure | `Matrix::sparseMatrix` from `nb` object (344K Ã— 344K) | One-time, seconds |
| Neighbor mean | Sparse matrix Ã— dense matrix: `(W %*% X) / (W %*% notNA)` | ~seconds per variable |
| Neighbor max/min | Vectorized C++ via `Rcpp` or chunked R using the sparse structure | ~seconds per variable |
| Reassembly | Map cellÃ—year matrices back to the long data.frame | Vectorized indexing |

**Estimated new runtime: 2â€“10 minutes total** (vs. 86+ hours).

**Numerical equivalence**: The sparse matrix encodes exactly the same rook-neighbor relationships. The same NA handling is applied. The Random Forest model is never retouched â€” we only reproduce the same 15 derived features (5 vars Ã— 3 stats) with identical values.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Replaces: build_neighbor_lookup + compute_neighbor_stats + outer loop
# Preserves: exact numerical results, trained RF model untouched
# =============================================================================

library(Matrix)
library(data.table)

optimize_neighbor_features <- function(cell_data, 
                                        id_order, 
                                        rook_neighbors_unique,
                                        neighbor_source_vars) {
  
  # -------------------------------------------------------------------------
  # 0. Convert to data.table for speed (non-destructive)
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # -------------------------------------------------------------------------
  # 1. Build sparse adjacency matrix W (344,208 x 344,208) from nb object
  #    This is the SAME neighbor structure â€” just in matrix form.
  # -------------------------------------------------------------------------
  n_cells <- length(id_order)
  
  # Construct COO triplets from the nb object
  from_idx <- integer(0)
  to_idx   <- integer(0)
  
  for (i in seq_along(rook_neighbors_unique)) {
    nbrs <- rook_neighbors_unique[[i]]
    # spdep::nb encodes "no neighbors" as 0L; skip those
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0) {
      from_idx <- c(from_idx, rep(i, length(nbrs)))
      to_idx   <- c(to_idx, nbrs)
    }
  }
  
  W <- sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = 1,
    dims = c(n_cells, n_cells)
  )
  
  rm(from_idx, to_idx)
  
  cat("Adjacency matrix built:", nnzero(W), "non-zero entries\n")
  
  # -------------------------------------------------------------------------
  # 2. Create cell-index and year-index mappings
  #    Map each row in dt to (cell_position, year_position)
  # -------------------------------------------------------------------------
  # id_order defines the cell ordering consistent with the nb object
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  years_sorted <- sort(unique(dt$year))
  n_years      <- length(years_sorted)
  year_to_col  <- setNames(seq_along(years_sorted), as.character(years_sorted))
  
  # Row positions in dt -> (cell_pos, year_pos)
  cell_pos <- id_to_pos[as.character(dt$id)]
  year_pos <- year_to_col[as.character(dt$year)]
  
  cat("Panel dimensions:", n_cells, "cells x", n_years, "years =", 
      n_cells * n_years, "potential; actual rows:", nrow(dt), "\n")
  
  # -------------------------------------------------------------------------
  # 3. For each variable, build cell x year matrix, compute stats, write back
  # -------------------------------------------------------------------------
  
  for (var_name in neighbor_source_vars) {
    
    cat("Processing variable:", var_name, "... ")
    t0 <- proc.time()
    
    # --- 3a. Build cell x year matrix (dense, NA-filled) ---
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(cell_pos, year_pos)] <- as.numeric(dt[[var_name]])
    
    # --- 3b. Compute neighbor MEAN via sparse matrix multiplication ---
    # notNA indicator matrix
    notNA <- matrix(0, nrow = n_cells, ncol = n_years)
    notNA[!is.na(X)] <- 1
    
    # Replace NA with 0 for multiplication
    X_zero <- X
    X_zero[is.na(X_zero)] <- 0
    
    # W %*% X_zero  = sum of neighbor values (treating NA as 0)
    # W %*% notNA   = count of non-NA neighbors
    neighbor_sum   <- as.matrix(W %*% X_zero)   # n_cells x n_years
    neighbor_count <- as.matrix(W %*% notNA)     # n_cells x n_years
    
    neighbor_mean <- neighbor_sum / neighbor_count
    # Where count == 0, result is NaN from 0/0; convert to NA
    neighbor_mean[neighbor_count == 0] <- NA_real_
    
    # --- 3c. Compute neighbor MAX and MIN ---
    # Strategy: iterate over cells using the sparse structure of W.
    # We use the CSR representation (row-compressed) for efficient row access.
    
    neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # Convert W to dgRMatrix (CSR) for fast row slicing, or use dgCMatrix column access
    # Actually, we'll extract neighbor lists from W's sparse structure directly.
    # For dgCMatrix (CSC), converting to dgRMatrix or using summary() is efficient.
    
    Wt <- as(W, "TsparseMatrix")  # triplet form: Wt@i (0-based row), Wt@j (0-based col)
    
    # Build a simple neighbor list from the sparse matrix (fast, one-time)
    # Group columns (j) by row (i)
    row_1based <- Wt@i + 1L
    col_1based <- Wt@j + 1L
    
    # Split neighbor indices by cell
    nbr_list <- split(col_1based, row_1based)
    # Ensure all cells are represented (some may have no neighbors)
    all_cells_char <- as.character(seq_len(n_cells))
    
    # Process in chunks for memory efficiency
    # For max and min, we MUST touch each cell's neighbors' actual values.
    # Vectorized approach: for each cell, extract neighbor rows from X and compute.
    # With ~4 neighbors on average, this is manageable.
    
    # Fastest pure-R approach: iterate over unique neighbor sets
    # But 344K iterations (not 6.46M!) is very fast.
    
    for (ci_char in names(nbr_list)) {
      ci   <- as.integer(ci_char)
      nbrs <- nbr_list[[ci_char]]
      
      if (length(nbrs) == 0) next
      
      # Extract neighbor values: matrix subset -> length(nbrs) x n_years
      if (length(nbrs) == 1) {
        nbr_vals <- matrix(X[nbrs, ], nrow = 1)
      } else {
        nbr_vals <- X[nbrs, , drop = FALSE]
      }
      
      # Compute column-wise max and min, ignoring NA
      # suppressWarnings to handle all-NA columns
      suppressWarnings({
        neighbor_max[ci, ] <- apply(nbr_vals, 2, max, na.rm = TRUE)
        neighbor_min[ci, ] <- apply(nbr_vals, 2, min, na.rm = TRUE)
      })
    }
    
    # Fix Inf/-Inf from all-NA columns -> NA
    neighbor_max[is.infinite(neighbor_max)] <- NA_real_
    neighbor_min[is.infinite(neighbor_min)] <- NA_real_
    
    # Also set max/min to NA where count == 0 (no valid neighbors at all)
    neighbor_max[neighbor_count == 0] <- NA_real_
    neighbor_min[neighbor_count == 0] <- NA_real_
    
    # --- 3d. Map results back to the long data.table ---
    idx_matrix <- cbind(cell_pos, year_pos)
    
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    dt[, (max_col)  := neighbor_max[idx_matrix]]
    dt[, (min_col)  := neighbor_min[idx_matrix]]
    dt[, (mean_col) := neighbor_mean[idx_matrix]]
    
    elapsed <- (proc.time() - t0)[3]
    cat("done in", round(elapsed, 1), "seconds\n")
    
    # Clean up per-variable temporaries
    rm(X, X_zero, notNA, neighbor_sum, neighbor_count, 
       neighbor_mean, neighbor_max, neighbor_min, nbr_vals)
  }
  
  rm(Wt, row_1based, col_1based, nbr_list)
  gc()
  
  # Return as data.frame (or data.table, depending on downstream needs)
  return(as.data.frame(dt))
}

# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is NOT modified.
# Proceed directly to prediction:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Further Optimization: Rcpp for Max/Min (Optional)

The loop over 344K cells with `apply(..., max/min)` is already fast (~minutes), but if you want it in seconds, here's an Rcpp version:

```r
# Save as neighbor_maxmin.cpp and use Rcpp::sourceCpp("neighbor_maxmin.cpp")

Rcpp::cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_maxmin_cpp(NumericMatrix X, List nbr_list, IntegerVector cell_ids) {
  int n_cells = X.nrow();
  int n_years = X.ncol();
  
  NumericMatrix out_max(n_cells, n_years);
  NumericMatrix out_min(n_cells, n_years);
  
  // Initialize with NA
  std::fill(out_max.begin(), out_max.end(), NA_REAL);
  std::fill(out_min.begin(), out_min.end(), NA_REAL);
  
  for (int k = 0; k < cell_ids.size(); k++) {
    int ci = cell_ids[k] - 1;  // 0-based
    IntegerVector nbrs = as<IntegerVector>(nbr_list[k]);
    int nn = nbrs.size();
    if (nn == 0) continue;
    
    for (int t = 0; t < n_years; t++) {
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      int valid = 0;
      
      for (int j = 0; j < nn; j++) {
        double val = X(nbrs[j] - 1, t);  // 0-based row
        if (!NumericVector::is_na(val)) {
          if (val > vmax) vmax = val;
          if (val < vmin) vmin = val;
          valid++;
        }
      }
      
      if (valid > 0) {
        out_max(ci, t) = vmax;
        out_min(ci, t) = vmin;
      }
    }
  }
  
  return List::create(Named("max") = out_max, Named("min") = out_min);
}
')
```

Replace the R loop in step 3c with:

```r
cell_ids_with_nbrs <- as.integer(names(nbr_list))
result <- neighbor_maxmin_cpp(X, nbr_list[names(nbr_list)], cell_ids_with_nbrs)
neighbor_max <- result$max
neighbor_min <- result$min
```

---

## Performance Summary

| Component | Original | Optimized (R) | Optimized (R + Rcpp) |
|---|---|---|---|
| Neighbor lookup | ~40 hrs | ~5 sec (sparse matrix) | ~5 sec |
| Stats: mean (Ã—5 vars) | ~15 hrs | ~30 sec (sparse matmul) | ~30 sec |
| Stats: max/min (Ã—5 vars) | ~31 hrs | ~10 min (R loop over 344K) | ~30 sec |
| **Total** | **~86 hrs** | **~12 min** | **~2 min** |

The numerical results are identical because the same neighbor graph and the same NA-handling logic are used. The trained Random Forest model is never modified.
END_RESPONSE>>>
