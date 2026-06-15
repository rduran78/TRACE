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
run_id: Anthropic_F08_T07_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) indices.** It creates a list of 6.46 million entries, each requiring string-pasting (`paste(id, year)`), hash-table lookups (`idx_lookup[neighbor_keys]`), and NA filtering. This is O(rows Ã— avg_neighbors) string operations â€” roughly 6.46M Ã— 4 â‰ˆ 25.8 million string-match lookups.

2. **The neighbor topology is static.** Rook contiguity among 344,208 cells never changes across the 28 years. Yet the current code re-discovers neighbors for every cell-year row, duplicating work 28Ã— unnecessarily.

3. **`compute_neighbor_stats` operates on the full 6.46M-row data frame.** Because the lookup indices point into the stacked cellÃ—year data frame, every variable's neighbor stats require 6.46M list-element iterations with subsetting, NA checks, and summary computation â€” all in pure R loops.

4. **The outer loop repeats this for 5 variables**, compounding the cost: 5 Ã— 6.46M = 32.3 million R-level `lapply` iterations.

### The Key Insight

> **Neighbor relationships are a property of cells (static). Variable values are a property of cell-years (dynamic).**

If we separate these two concerns, we can:
- Build the neighbor lookup **once over 344,208 cells** (not 6.46M rows).
- Compute neighbor stats **per year** using fast vectorized/matrix operations on 344,208-length vectors, not list-element-wise R loops over millions of rows.

---

## Optimization Strategy

### 1. Build a Cell-Level Neighbor Index (Once)

Construct a sparse adjacency structure over the 344,208 cells. This is simply a cleaned version of `rook_neighbors_unique` mapped to integer cell indices. Cost: negligible, done once.

### 2. Build a Sparse Adjacency Matrix (Once)

Convert the neighbor list into a sparse matrix `W` (344,208 Ã— 344,208) using the `Matrix` package. Each row `i` has 1s in columns corresponding to cell `i`'s rook neighbors. This is the static topology encoded as a reusable linear-algebra object.

### 3. Compute Neighbor Stats via Sparse Matrixâ€“Vector Products (Per Year, Per Variable)

For each year and each variable:
- Extract the 344,208-length variable vector `v` for that year.
- **Neighbor sum** = `W %*% v` (sparse matrixâ€“vector multiply, highly optimized in C).
- **Neighbor count** = `W %*% (!is.na(v))` (to handle NAs correctly).
- **Neighbor mean** = sum / count.
- **Neighbor max/min**: Use a custom but vectorized approach with the neighbor list, or use row-wise sparse operations.

For **max and min**, sparse matrix multiplication doesn't directly apply, but we can use a fast vectorized approach over the cell-level neighbor list (344K iterations instead of 6.46M), or use `data.table` grouped operations.

### 4. Merge Back into the Panel

The results for each year are 344,208-length vectors. Map them back into the full data frame by (cell, year) alignment.

### Expected Speedup

| Component | Current | Optimized |
|---|---|---|
| Neighbor lookup construction | 6.46M string ops | 344K integer list (once) |
| Stat computation iterations | 5 vars Ã— 6.46M = 32.3M | 5 vars Ã— 28 years Ã— 344K = 48.2M but vectorized |
| Per-iteration cost | R-level list subset + summary | C-level sparse matmul (mean) + vectorized (max/min) |
| **Estimated total time** | **86+ hours** | **~2â€“10 minutes** |

---

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build the static cell-level sparse adjacency matrix (ONCE)
# ==============================================================================

build_sparse_adjacency <- function(id_order, neighbors_nb) {
  # id_order: vector of 344,208 cell IDs in the order matching the nb object
  # neighbors_nb: spdep nb object (list of integer index vectors)
  #
  # Returns: a sparse logical/numeric adjacency matrix W (n x n)
  #          AND the id_order for alignment
  
  n <- length(id_order)
  stopifnot(length(neighbors_nb) == n)
  
  # Build COO (coordinate) triplets
  # For each cell i, neighbors_nb[[i]] gives integer indices of its neighbors
  from <- rep(seq_len(n), times = lengths(neighbors_nb))
  to   <- unlist(neighbors_nb)
  
  # Remove any 0-length entries (islands with no neighbors are handled naturally)
  valid <- !is.na(to) & to >= 1L & to <= n
  from  <- from[valid]
  to    <- to[valid]
  
  W <- sparseMatrix(
    i    = from,
    j    = to,
    x    = rep(1, length(from)),
    dims = c(n, n)
  )
  
  list(
    W        = W,
    id_order = id_order,
    id_to_idx = setNames(seq_len(n), as.character(id_order))
  )
}

# ==============================================================================
# STEP 2: Compute neighbor max, min, mean for one variable across all years
#          using the static adjacency
# ==============================================================================

compute_neighbor_features_fast <- function(dt, var_name, adj, neighbors_nb) {
  # dt:            data.table with columns: id, year, <var_name>
  # var_name:      character, the variable to compute neighbor stats for
  # adj:           output of build_sparse_adjacency()
  # neighbors_nb:  the raw nb list (for max/min computation)
  #
  # Returns: dt with three new columns added (modifies by reference)
  
  W        <- adj$W
  id_order <- adj$id_order
  id_to_idx <- adj$id_to_idx
  n        <- length(id_order)
  
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate result columns
  dt[, (max_col)  := NA_real_]
  dt[, (min_col)  := NA_real_]
  dt[, (mean_col) := NA_real_]
  
  # Get sorted unique years
  years <- sort(unique(dt$year))
  
  # Ensure dt is keyed for fast subsetting
  setkey(dt, year, id)
  
  for (yr in years) {
    # Extract the rows for this year
    # We need a vector of length n aligned to id_order
    yr_rows <- dt[.(yr)]  # subset by year via key
    
    # Map yr_rows to the cell index order
    # yr_rows$id needs to be mapped to positions in id_order
    yr_idx <- id_to_idx[as.character(yr_rows$id)]
    
    # Build the aligned vector (length n, NA for missing cells)
    v <- rep(NA_real_, n)
    v[yr_idx] <- yr_rows[[var_name]]
    
    # --- Neighbor MEAN via sparse matrix multiplication ---
    # Handle NAs: replace NA with 0 for sum, track non-NA counts
    v_nona     <- v
    v_nona[is.na(v_nona)] <- 0
    not_na     <- as.numeric(!is.na(v))
    
    neighbor_sum   <- as.numeric(W %*% v_nona)       # length n
    neighbor_count <- as.numeric(W %*% not_na)        # length n
    
    neighbor_mean <- ifelse(neighbor_count > 0,
                            neighbor_sum / neighbor_count,
                            NA_real_)
    
    # --- Neighbor MAX and MIN via vectorized cell-level loop ---
    # This iterates over 344K cells (not 6.46M rows), which is fast
    neighbor_max <- rep(NA_real_, n)
    neighbor_min <- rep(NA_real_, n)
    
    for (i in seq_len(n)) {
      nb_idx <- neighbors_nb[[i]]
      if (length(nb_idx) == 0L) next
      nb_vals <- v[nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0L) next
      neighbor_max[i] <- max(nb_vals)
      neighbor_min[i] <- min(nb_vals)
    }
    
    # Map results back: for each row in yr_rows, get the result from its cell index
    result_max  <- neighbor_max[yr_idx]
    result_min  <- neighbor_min[yr_idx]
    result_mean <- neighbor_mean[yr_idx]
    
    # Write back into dt
    # We need the row indices in the original dt for this year
    row_indices <- which(dt$year == yr)
    # But since we used setkey, the order within dt[.(yr)] matches row_indices
    set(dt, i = row_indices, j = max_col,  value = result_max)
    set(dt, i = row_indices, j = min_col,  value = result_min)
    set(dt, i = row_indices, j = mean_col, value = result_mean)
  }
  
  invisible(dt)
}

# ==============================================================================
# STEP 2b: Even faster max/min using Rcpp (optional, drop-in replacement)
#           If Rcpp is available, this replaces the R-level for loop for max/min
# ==============================================================================

# If you want to avoid the 344K R-loop for max/min, use this Rcpp version:
#
# Rcpp::cppFunction('
# #include <Rcpp.h>
# using namespace Rcpp;
#
# // [[Rcpp::export]]
# List neighbor_max_min_cpp(NumericVector v, List neighbors_nb) {
#   int n = neighbors_nb.size();
#   NumericVector nmax(n, NA_REAL);
#   NumericVector nmin(n, NA_REAL);
#   for (int i = 0; i < n; i++) {
#     IntegerVector nb = neighbors_nb[i];
#     if (nb.size() == 0) continue;
#     double cmax = R_NegInf;
#     double cmin = R_PosInf;
#     bool found = false;
#     for (int j = 0; j < nb.size(); j++) {
#       int idx = nb[j] - 1;  // R is 1-indexed
#       if (idx < 0 || idx >= n) continue;
#       double val = v[idx];
#       if (NumericVector::is_na(val)) continue;
#       found = true;
#       if (val > cmax) cmax = val;
#       if (val < cmin) cmin = val;
#     }
#     if (found) {
#       nmax[i] = cmax;
#       nmin[i] = cmin;
#     }
#   }
#   return List::create(Named("max") = nmax, Named("min") = nmin);
# }
# ')

# ==============================================================================
# STEP 3: Main pipeline (replaces the original outer loop)
# ==============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # Convert to data.table for performance (if not already)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # --- STATIC: Build adjacency once ---
  message("Building static sparse adjacency matrix (344,208 cells)...")
  adj <- build_sparse_adjacency(id_order, rook_neighbors_unique)
  message("  Done. Adjacency matrix: ", nrow(adj$W), " x ", ncol(adj$W),
          ", ", length(adj$W@x), " non-zero entries.")
  
  # --- DYNAMIC: Compute neighbor features per variable ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name, " ...")
    t0 <- proc.time()
    compute_neighbor_features_fast(
      dt           = cell_data,
      var_name     = var_name,
      adj          = adj,
      neighbors_nb = rook_neighbors_unique
    )
    elapsed <- (proc.time() - t0)[3]
    message("  Done in ", round(elapsed, 1), " seconds.")
  }
  
  message("All neighbor features computed.")
  return(cell_data)
}

# ==============================================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================================

# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# # Now proceed directly to prediction with the pre-trained Random Forest:
# # predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Concern | Original | Redesigned |
|---|---|---|
| **Topology encoding** | Rebuilt per cell-year row (6.46M string lookups) | Sparse matrix built once over 344K cells |
| **Mean computation** | R `lapply` over 6.46M entries | Sparse matrixâ€“vector product (`W %*% v`), C-optimized |
| **Max/Min computation** | R `lapply` over 6.46M entries | R loop over 344K cells Ã— 28 years (or Rcpp) |
| **Total iterations** | ~32.3M R-level list operations | 5 vars Ã— 28 yrs = 140 sparse matmuls + 140 Ã— 344K vectorized loops |
| **Estimated runtime** | 86+ hours | **~2â€“10 minutes** |
| **Numerical equivalence** | â€” | Identical: same max, min, mean over same neighbor sets |
| **RF model** | â€” | Untouched; used only for prediction after features are computed |
| **Memory** | ~6.46M-element list of integer vectors | Sparse matrix (~1.4M entries â‰ˆ 22 MB) + year-slice vectors |

The redesign cleanly separates the **static spatial topology** (encoded once as a sparse matrix and neighbor list) from the **dynamic yearly variable values** (processed per-year in vectorized operations), eliminating the redundant 28Ã— recomputation of neighbor relationships and replacing millions of interpreted R loop iterations with optimized C-level sparse linear algebra.
END_RESPONSE>>>
