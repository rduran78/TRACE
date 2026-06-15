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
run_id: Anthropic_F10_T10_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **O(NÂ²)-scale neighbor lookup construction**: `build_neighbor_lookup` creates a lookup key for every row (`paste(id, year)`), then for each of ~6.46M rows, searches neighbor keys in a named vector. Named vector lookups in R are hash-based but still carry overhead when done 6.46M Ã— k times (where k is average neighbor count ~4 for rook adjacency).

2. **Row-wise `lapply` over 6.46M rows in `compute_neighbor_stats`**: For each of the 5 variables, we iterate over 6.46M rows in an R-level loop, extracting, subsetting, and computing `max/min/mean`. This is ~32.3M R-level function calls across all variables.

3. **Redundant topology work**: The neighbor graph is identical across all 28 years. The current code embeds year into the lookup keys, effectively rebuilding the topology per-year implicitly. The spatial adjacency structure (344,208 nodes Ã— ~4 neighbors each â‰ˆ 1.37M directed edges) is time-invariant and should be factored out.

**Memory estimate**: 6.46M rows Ã— 110 columns Ã— 8 bytes â‰ˆ 5.7 GB base. Adding 15 new columns (5 vars Ã— 3 stats) adds ~0.78 GB. Total fits in 16 GB but leaves little headroom, ruling out approaches that duplicate the full dataset.

## Optimization Strategy

1. **Separate spatial topology from temporal indexing**: Build a sparse adjacency matrix (344,208 Ã— 344,208) once from the `nb` object. This is a CSR-format sparse matrix with ~1.37M non-zero entries (~16 MB).

2. **Reshape to year-sliced matrices**: For each variable, construct a dense matrix of dimension (344,208 cells Ã— 28 years). This costs 344,208 Ã— 28 Ã— 8 bytes â‰ˆ 77 MB per variable â€” entirely tractable.

3. **Sparse matrix multiplication for aggregation**:
   - **Mean**: `A %*% X / degree_vector` where A is the binary adjacency matrix and X is (cells Ã— years). One sparse matrix multiply replaces 6.46M R-level loops.
   - **Max/Min**: Use a row-wise sparse sweep. For each row of A, extract column indices (neighbors), then vectorize across years. We group this by the CSR structure of A, which is far cheaper than the original approach.

4. **Vectorize max/min via C++-level operations**: Use the `Matrix` package CSR internals (`i`, `p`, `x` slots) to iterate over neighbor sets. For max/min, since sparse matrix algebra doesn't directly support these, we use a tight `for` loop over 344,208 cells (not 6.46M rows), computing vectorized max/min across 28 years simultaneously. This reduces the loop from 6.46M to 344K iterations, each doing vectorized operations over 28 elements.

5. **Join back** by cell index and year index into the original data.frame.

**Expected speedup**: The mean computation via sparse matrix multiply is O(nnz Ã— T) â‰ˆ 38.4M FLOPs per variable â€” essentially instantaneous. Max/min over 344K cells with vectorized year ops should run in seconds per variable. Total estimated time: **2â€“5 minutes** instead of 86+ hours.

## Working R Code

```r
library(Matrix)
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars) {
  # ---------------------------------------------------------------
  # STEP 1: Build sparse adjacency matrix once (time-invariant topology)
  # ---------------------------------------------------------------
  n_cells <- length(id_order)
  
  # Build COO representation from nb object
  from_idx <- integer(0)
  to_idx   <- integer(0)
  for (i in seq_along(rook_neighbors_unique)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep nb objects use 0L for no-neighbor entries; filter those
    nb_i <- nb_i[nb_i > 0L]
    if (length(nb_i) > 0) {
      from_idx <- c(from_idx, rep(i, length(nb_i)))
      to_idx   <- c(to_idx, nb_i)
    }
  }
  
  # Sparse binary adjacency matrix (row i has 1s at its neighbor columns)
  A <- sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = rep(1, length(from_idx)),
    dims = c(n_cells, n_cells),
    repr = "C"   # CSR format for fast row operations
  )
  
  # Degree vector (number of neighbors per cell, used for mean)
  degree_vec <- diff(A@p)  # CSR row pointer differences = row nnz counts
  
  rm(from_idx, to_idx)
  
  # ---------------------------------------------------------------
  # STEP 2: Convert to data.table for fast indexing
  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # Create cell index: map id -> position in id_order
  id_map <- setNames(seq_along(id_order), as.character(id_order))
  dt[, cell_idx := id_map[as.character(id)]]
  
  # Get sorted unique years and create year index
  years_unique <- sort(unique(dt$year))
  n_years <- length(years_unique)
  year_map <- setNames(seq_along(years_unique), as.character(years_unique))
  dt[, year_idx := year_map[as.character(year)]]
  
  # Key for fast ordered access
  setkey(dt, cell_idx, year_idx)
  
  # ---------------------------------------------------------------
  # STEP 3: For each variable, build cellÃ—year matrix, compute stats

  # ---------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    cat("Processing variable:", var_name, "\n")
    
    # Build dense matrix: rows=cells, cols=years
    # Initialize with NA
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # Fill from data.table (vectorized)
    X[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]
    
    # ------- MEAN via sparse matrix multiplication -------
    # A %*% X gives sum of neighbor values for each cellÃ—year
    # Then divide by degree to get mean
    neighbor_sum <- A %*% X   # result is n_cells Ã— n_years dense matrix
    neighbor_sum <- as.matrix(neighbor_sum)
    
    # Compute mean: divide by number of valid neighbors
    # But we need to handle NAs properly: count non-NA neighbors
    # Create indicator matrix: 1 where X is not NA, 0 otherwise
    X_valid <- matrix(as.numeric(!is.na(X)), nrow = n_cells, ncol = n_years)
    
    # Replace NA with 0 in X for summation purposes
    X_zero <- X
    X_zero[is.na(X_zero)] <- 0
    
    # Recompute sum using NA-safe version
    neighbor_sum <- as.matrix(A %*% X_zero)
    neighbor_count <- as.matrix(A %*% X_valid)
    
    # Mean
    neighbor_mean <- neighbor_sum / neighbor_count
    neighbor_mean[neighbor_count == 0] <- NA_real_
    
    # ------- MAX and MIN via CSR row iteration -------
    # We iterate over 344K cells (not 6.46M rows) â€” each iteration
    # is vectorized across 28 years
    neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # CSR pointers
    p <- A@p   # length n_cells + 1
    j <- A@j   # column indices (0-based)
    
    for (i in seq_len(n_cells)) {
      start <- p[i] + 1L      # R is 1-based; @p is 0-based
      end   <- p[i + 1L]
      if (end < start) next    # no neighbors
      
      nb_indices <- j[start:end] + 1L   # convert to 1-based row indices of X
      
      if (length(nb_indices) == 1L) {
        # Single neighbor: just copy that row
        neighbor_max[i, ] <- X[nb_indices, ]
        neighbor_min[i, ] <- X[nb_indices, ]
      } else {
        # Multiple neighbors: extract submatrix and compute col max/min
        nb_mat <- X[nb_indices, , drop = FALSE]  # k Ã— n_years
        # Use matrixStats for speed if available; otherwise base R
        neighbor_max[i, ] <- apply(nb_mat, 2, max, na.rm = FALSE)
        neighbor_min[i, ] <- apply(nb_mat, 2, min, na.rm = FALSE)
        
        # Handle all-NA columns
        all_na_cols <- colSums(!is.na(nb_mat)) == 0L
        if (any(all_na_cols)) {
          neighbor_max[i, all_na_cols] <- NA_real_
          neighbor_min[i, all_na_cols] <- NA_real_
        } else {
          # Recompute with na.rm = TRUE for columns that have some valid values
          has_any_na <- anyNA(nb_mat)
          if (has_any_na) {
            neighbor_max[i, ] <- apply(nb_mat, 2, function(v) {
              v2 <- v[!is.na(v)]
              if (length(v2) == 0L) NA_real_ else max(v2)
            })
            neighbor_min[i, ] <- apply(nb_mat, 2, function(v) {
              v2 <- v[!is.na(v)]
              if (length(v2) == 0L) NA_real_ else min(v2)
            })
          }
        }
      }
    }
    
    # ------- Write results back into data.table -------
    max_col <- paste0("max_", var_name)
    min_col <- paste0("min_", var_name)
    mean_col <- paste0("mean_", var_name)
    
    dt[, (max_col)  := neighbor_max[cbind(cell_idx, year_idx)]]
    dt[, (min_col)  := neighbor_min[cbind(cell_idx, year_idx)]]
    dt[, (mean_col) := neighbor_mean[cbind(cell_idx, year_idx)]]
    
    # Free memory
    rm(X, X_valid, X_zero, neighbor_sum, neighbor_count,
       neighbor_mean, neighbor_max, neighbor_min)
    gc()
    
    cat("  Done:", var_name, "\n")
  }
  
  # ---------------------------------------------------------------
  # STEP 4: Clean up helper columns, return as data.frame
  # ---------------------------------------------------------------
  dt[, c("cell_idx", "year_idx") := NULL]
  
  as.data.frame(dt)
}
```

However, the inner loop with `apply` over 344K cells is still costly. Here is a **faster version** that replaces the max/min loop with `matrixStats` or a fully vectorized approach:

```r
# ---------------------------------------------------------------
# PRODUCTION VERSION: Fully optimized with matrixStats
# ---------------------------------------------------------------
# install.packages("matrixStats")  # if not available

optimize_neighbor_features_fast <- function(cell_data, id_order, 
                                             rook_neighbors_unique,
                                             neighbor_source_vars) {
  library(Matrix)
  library(data.table)
  
  n_cells <- length(id_order)
  
  # === STEP 1: Build sparse adjacency matrix (CSR) ===
  from_list <- lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_i <- rook_neighbors_unique[[i]]
    nb_i <- nb_i[nb_i > 0L]
    if (length(nb_i) > 0L) {
      data.table(from = i, to = nb_i)
    } else {
      NULL
    }
  })
  edges <- rbindlist(from_list)
  rm(from_list); gc()
  
  A <- sparseMatrix(
    i = edges$from,
    j = edges$to,
    x = rep(1, nrow(edges)),
    dims = c(n_cells, n_cells),
    repr = "C"
  )
  rm(edges); gc()
  
  # === STEP 2: Index the data ===
  dt <- as.data.table(cell_data)
  id_map <- setNames(seq_along(id_order), as.character(id_order))
  dt[, cell_idx := id_map[as.character(id)]]
  
  years_unique <- sort(unique(dt$year))
  n_years <- length(years_unique)
  year_map <- setNames(seq_along(years_unique), as.character(years_unique))
  dt[, year_idx := year_map[as.character(year)]]
  setkey(dt, cell_idx, year_idx)
  
  # CSR structure
  p_ptr <- A@p
  j_idx <- A@j  # 0-based
  
  # === STEP 3: Process each variable ===
  for (var_name in neighbor_source_vars) {
    cat(sprintf("[%s] Processing: %s\n", Sys.time(), var_name))
    
    # Build cell Ã— year matrix
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]
    
    # --- MEAN via sparse matmul (NA-safe) ---
    X_zero <- X;  X_zero[is.na(X_zero)] <- 0
    X_ind  <- 1 * (!is.na(X))
    
    sum_mat   <- as.matrix(A %*% X_zero)
    count_mat <- as.matrix(A %*% X_ind)
    mean_mat  <- sum_mat / count_mat
    mean_mat[count_mat == 0] <- NA_real_
    
    rm(X_zero, X_ind, sum_mat, count_mat); gc()
    
    # --- MAX and MIN via grouped row sweep ---
    # Strategy: iterate over cells, use vectorized column operations.
    # For max degree ~4 (rook), the inner operation is tiny.
    # The key optimization: avoid apply(); use pmax/pmin directly.
    
    max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    for (i in seq_len(n_cells)) {
      start <- p_ptr[i] + 1L
      end   <- p_ptr[i + 1L]
      if (end < start) next
      
      nb <- j_idx[start:end] + 1L
      k  <- length(nb)
      
      if (k == 1L) {
        max_mat[i, ] <- X[nb, ]
        min_mat[i, ] <- X[nb, ]
      } else if (k == 2L) {
        r1 <- X[nb[1L], ]
        r2 <- X[nb[2L], ]
        max_mat[i, ] <- pmax(r1, r2, na.rm = TRUE)
        min_mat[i, ] <- pmin(r1, r2, na.rm = TRUE)
        both_na <- is.na(r1) & is.na(r2)
        if (any(both_na)) {
          max_mat[i, both_na] <- NA_real_
          min_mat[i, both_na] <- NA_real_
        }
      } else if (k == 3L) {
        r1 <- X[nb[1L], ]; r2 <- X[nb[2L], ]; r3 <- X[nb[3L], ]
        max_mat[i, ] <- pmax(r1, r2, r3, na.rm = TRUE)
        min_mat[i, ] <- pmin(r1, r2, r3, na.rm = TRUE)
        all_na <- is.na(r1) & is.na(r2) & is.na(r3)
        if (any(all_na)) {
          max_mat[i, all_na] <- NA_real_
          min_mat[i, all_na] <- NA_real_
        }
      } else if (k == 4L) {
        r1 <- X[nb[1L], ]; r2 <- X[nb[2L], ]
        r3 <- X[nb[3L], ]; r4 <- X[nb[4L], ]
        max_mat[i, ] <- pmax(r1, r2, r3, r4, na.rm = TRUE)
        min_mat[i, ] <- pmin(r1, r2, r3, r4, na.rm = TRUE)
        all_na <- is.na(r1) & is.na(r2) & is.na(r3) & is.na(r4)
        if (any(all_na)) {
          max_mat[i, all_na] <- NA_real_
          min_mat[i, all_na] <- NA_real_
        }
      } else {
        # General case (degree > 4, rare for rook)
        nb_mat <- X[nb, , drop = FALSE]
        max_mat[i, ] <- apply(nb_mat, 2, function(v) {
          v2 <- v[!is.na(v)]; if (length(v2) == 0L) NA_real_ else max(v2)
        })
        min_mat[i, ] <- apply(nb_mat, 2, function(v) {
          v2 <- v[!is.na(v)]; if (length(v2) == 0L) NA_real_ else min(v2)
        })
      }
    }
    
    # --- Write back ---
    dt[, paste0("max_",  var_name) := max_mat[cbind(cell_idx, year_idx)]]
    dt[, paste0("min_",  var_name) := min_mat[cbind(cell_idx, year_idx)]]
    dt[, paste0("mean_", var_name) := mean_mat[cbind(cell_idx, year_idx)]]
    
    rm(X, mean_mat, max_mat, min_mat); gc()
    cat(sprintf("[%s] Completed: %s\n", Sys.time(), var_name))
  }
  
  dt[, c("cell_idx", "year_idx") := NULL]
  as.data.frame(dt)
}
```

Now, an **even faster version** using Rcpp to eliminate the R-level loop entirely:

```r
# === STEP 3 ALTERNATIVE: Rcpp for max/min (eliminates R loop) ===
# Save this as "neighbor_maxmin.cpp"

Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_neighbor_maxmin(NumericMatrix X,
                            IntegerVector p_ptr,
                            IntegerVector j_idx) {
  // X: n_cells x n_years
  // p_ptr: CSR row pointers (0-based, length n_cells+1)
  // j_idx: CSR column indices (0-based)
  
  int n_cells = X.nrow();
  int n_years = X.ncol();
  
  NumericMatrix max_mat(n_cells, n_years);
  NumericMatrix min_mat(n_cells, n_years);
  
  // Initialize to NA
  std::fill(max_mat.begin(), max_mat.end(), NA_REAL);
  std::fill(min_mat.begin(), min_mat.end(), NA_REAL);
  
  for (int i = 0; i < n_cells; i++) {
    int start = p_ptr[i];
    int end   = p_ptr[i + 1];
    if (start >= end) continue;  // no neighbors
    
    for (int t = 0; t < n_years; t++) {
      double cur_max = R_NegInf;
      double cur_min = R_PosInf;
      int valid_count = 0;
      
      for (int e = start; e < end; e++) {
        int nb = j_idx[e];  // 0-based neighbor index
        double val = X(nb, t);
        if (!NumericMatrix::is_na(val)) {
          if (val > cur_max) cur_max = val;
          if (val < cur_min) cur_min = val;
          valid_count++;
        }
      }
      
      if (valid_count > 0) {
        max_mat(i, t) = cur_max;
        min_mat(i, t) = cur_min;
      }
      // else stays NA
    }
  }
  
  return List::create(Named("max_mat") = max_mat,
                      Named("min_mat") = min_mat);
}
')


# === FINAL PRODUCTION FUNCTION (Rcpp-accelerated) ===

optimize_neighbor_features_rcpp <- function(cell_data, id_order, 
                                             rook_neighbors_unique,
                                             neighbor_source_vars) {
  library(Matrix)
  library(data.table)
  library(Rcpp)
  
  n_cells <- length(id_order)
  
  # --- Build sparse adjacency (CSR) ---
  from_list <- lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_i <- rook_neighbors_unique[[i]]
    nb_i <- nb_i[nb_i > 0L]
    if (length(nb_i) > 0L) data.table(from = i, to = nb_i) else NULL
  })
  edges <- rbindlist(from_list)
  rm(from_list)
  
  A <- sparseMatrix(i = edges$from, j = edges$to,
                    x = rep(1, nrow(edges)),
                    dims = c(n_cells, n_cells), repr = "C")
  rm(edges); gc()
  
  p_ptr <- A@p
  j_idx <- A@j
  
  # --- Index the data ---
  dt <- as.data.table(cell_data)
  id_map   <- setNames(seq_along(id_order), as.character(id_order))
  dt[, cell_idx := id_map[as.character(id)]]
  years_unique <- sort(unique(dt$year))
  n_years  <- length(years_unique)
  year_map <- setNames(seq_along(years_unique), as.character(years_unique))
  dt[, year_idx := year_map[as.character(year)]]
  setkey(dt, cell_idx, year_idx)
  
  # --- Process each variable ---
  for (var_name in neighbor_source_vars) {
    cat(sprintf("[%s] %s ...\n", Sys.time(), var_name))
    
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]
    
    # Mean via sparse matmul
    X0 <- X;  X0[is.na(X0)] <- 0
    Xi <- 1 * (!is.na(X))
    sm <- as.matrix(A %*% X0)
    cn <- as.matrix(A %*% Xi)
    mn <- sm / cn
    mn[cn == 0] <- NA_real_
    rm(X0, Xi, sm, cn)
    
    # Max/Min via Rcpp
    mm <- sparse_neighbor_maxmin(X, as.integer(p_ptr), as.integer(j_idx))
    
    dt[, paste0("max_",  var_name) := mm$max_mat[cbind(cell_idx, year_idx)]]
    dt[, paste0("min_",  var_name) := mm$min_mat[cbind(cell_idx, year_idx)]]
    dt[, paste0("mean_", var_name) := mn[cbind(cell_idx, year_idx)]]
    
    rm(X, mn, mm); gc()
    cat(sprintf("[%s] %s done.\n", Sys.time(), var_name))
  }
  
  dt[, c("cell_idx", "year_idx") := NULL]
  as.data.frame(dt)
}


# === USAGE ===
# Load data and pre-trained model
# cell_data              <- readRDS("cell_data.rds")
# id_order               <- readRDS("id_order.rds")
# rook_neighbors_unique  <- readRDS("rook_neighbors_unique.rds")
# rf_model               <- readRDS("rf_model.rds")   # Pre-trained, do NOT retrain

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features_rcpp(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)

# Apply pre-trained Random Forest (unchanged)
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Data structure** | Named vector key lookup | Sparse CSR adjacency matrix + dense cellÃ—year matrices |
| **Mean computation** | R-level lapply over 6.46M rows | Single sparse matrix multiply per variable |
| **Max/Min computation** | R-level lapply over 6.46M rows | Rcpp loop over 344K cells Ã— 28 years (tight C++ inner loop) |
| **Topology reuse** | Implicit per-row key matching | Built once, reused across all variables and years |
| **Estimated runtime** | 86+ hours | ~2â€“5 minutes |
| **Memory** | Fits in 16 GB | Fits in 16 GB (peak ~10 GB with one cellÃ—year matrix active) |
| **Numerical equivalence** | Baseline | Identical (same max/min/mean with same NA handling) |
| **RF model** | Pre-trained | Unchanged, not retrained |
END_RESPONSE>>>
