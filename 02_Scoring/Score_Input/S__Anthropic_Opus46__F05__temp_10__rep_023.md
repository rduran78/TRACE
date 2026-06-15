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
run_id: Anthropic_F05_T10_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern, not merely a local inefficiency**. Here's the full picture:

### Local Inefficiency
The `paste(..., sep="_")` call inside `build_neighbor_lookup` constructs ~6.46M string keys for `idx_lookup` once, but then *for each of the 6.46M rows*, it re-pastes neighbor IDs with the current year to probe that lookup. With an average of ~8 rook neighbors per cell (1,373,394 directed pairs / 344,208 cells â‰ˆ 4 per cell, doubled for bidirectional â‰ˆ 8), that's ~51.7 million `paste` + hash-lookup operations inside the `lapply`. String hashing on millions of keys is inherently slow in R.

### Broader Algorithmic Problem
The real issue is that **the neighbor topology is year-invariant** â€” a cell's rook neighbors are the same in every year. The current code re-discovers this for every cell-year row by encoding `(id, year)` pairs as strings. This means the 344,208-cell neighbor structure is redundantly traversed 28 times (once per year), doing string work each time.

Furthermore, `compute_neighbor_stats` is called sequentially for each of 5 variables, each time iterating over all 6.46M rows. This is another layer of repeated work.

### Root Cause Summary

| Layer | Waste | Scale |
|---|---|---|
| String key construction | `paste` + named-vector lookup per row | 6.46M Ã— ~8 neighbors = ~51.7M string ops |
| Year-invariant topology re-traversal | Same neighbor set recomputed 28Ã— | 28 Ã— 344,208 = 9,637,824 redundant lookups |
| Per-variable iteration | Full 6.46M-row pass per variable | 5 Ã— 6.46M = 32.3M row visits |

## Optimization Strategy

1. **Separate topology from time**: Build the neighbor index list once at the cell level (344K entries), not the cell-year level (6.46M entries).

2. **Eliminate all string operations**: Use integer indexing throughout. Map cell IDs to integer positions, and for each year-slice, compute a simple integer offset to jump from cell-level neighbor indices to cell-year-level row indices.

3. **Vectorize across years and variables**: Instead of `lapply` over 6.46M rows, operate on a matrix representation where rows = cells, columns = years. For each cell, the neighbor set (row indices into the matrix) is the same across all years. Use matrix slicing to extract neighbor values, then compute `max/min/mean` in a vectorized or compiled fashion.

4. **Use `data.table` for fast joins** and avoid repeated data frame copies.

5. **Process all 5 variables in one pass** over the neighbor structure.

## Working R Code

```r
library(data.table)

# ===========================================================================
# optimized_neighbor_features.R
#
# Replaces: build_neighbor_lookup, compute_neighbor_stats, and the outer loop.
# Preserves: the trained Random Forest model and the original numerical
#            estimand (max, min, mean of each neighbor source variable).
# ===========================================================================

build_and_apply_neighbor_features <- function(cell_data,
                                               id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  # -----------------------------------------------------------------------
  # 0. Convert to data.table for speed; keep original row order
  # -----------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_order := .I]
  
  # -----------------------------------------------------------------------
  # 1. Build year-invariant, integer-indexed neighbor list  (344K entries)
  #
  #    id_order is the vector of cell IDs in the same order as

  #    rook_neighbors_unique (an spdep nb object): element k of the nb
  #    list gives indices into id_order for the neighbors of id_order[k].
  #
  #    We map every cell ID that appears in dt to its position in id_order.
  # -----------------------------------------------------------------------
  id_order_chr <- as.character(id_order)
  n_cells      <- length(id_order)
  id_to_pos    <- setNames(seq_len(n_cells), id_order_chr)
  
  # spdep nb objects: neighbors[[k]] is an integer vector of positions
  # into id_order (0L means no neighbors â€” spdep convention).
  # Convert to a clean list: for each cell position, the integer positions
  # of its neighbors in id_order.
  cell_neighbors <- lapply(seq_len(n_cells), function(k) {
    nb <- rook_neighbors_unique[[k]]
    nb <- nb[nb != 0L]           # drop the spdep "no-neighbor" sentinel
    as.integer(nb)
  })
  
  # -----------------------------------------------------------------------
  # 2. Reshape each source variable into a matrix:  cells Ã— years
  #
  #    Row i  = cell at position i in id_order
  #    Col j  = j-th year in sorted unique years
  #
  #    This lets us look up neighbor values with pure integer indexing.
  # -----------------------------------------------------------------------
  years_sorted <- sort(unique(dt$year))
  n_years      <- length(years_sorted)
  year_to_col  <- setNames(seq_len(n_years), as.character(years_sorted))
  
  # Map each row in dt to (cell_position, year_column)
  dt[, cell_pos := id_to_pos[as.character(id)]]
  dt[, year_col := year_to_col[as.character(year)]]
  
  # Build a linear index into an n_cells Ã— n_years matrix for fast fill
  # R matrices are column-major: element [i, j] is at position i + (j-1)*n_cells
  lin_idx <- dt$cell_pos + (dt$year_col - 1L) * n_cells
  
  # Pre-allocate matrices for each variable
  var_matrices <- lapply(neighbor_source_vars, function(vn) {
    mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mat[lin_idx] <- dt[[vn]]
    mat
  })
  names(var_matrices) <- neighbor_source_vars
  
  # -----------------------------------------------------------------------
  # 3. Compute neighbor stats:  for every (cell, year), for every variable,
  #    gather the neighbor values, compute max / min / mean.
  #
  #    Strategy: iterate over cells (344K), vectorise across years (28)
  #    and variables (5) for each cell's neighbor set.
  #
  #    For a cell with nb_k neighbors, extracting a nb_k Ã— n_years sub-
  #    matrix and calling colMins / colMaxs / colMeans is fast.
  # -----------------------------------------------------------------------
  
  n_vars <- length(neighbor_source_vars)
  
  # Result matrices: n_cells Ã— n_years for each of 3 stats Ã— n_vars
  # We store them as lists of matrices, then map back to dt.
  res_max  <- lapply(seq_len(n_vars), function(v)
    matrix(NA_real_, nrow = n_cells, ncol = n_years))
  res_min  <- lapply(seq_len(n_vars), function(v)
    matrix(NA_real_, nrow = n_cells, ncol = n_years))
  res_mean <- lapply(seq_len(n_vars), function(v)
    matrix(NA_real_, nrow = n_cells, ncol = n_years))
  
  # Main loop: 344,208 iterations (fast; inner work is vectorised)
  for (k in seq_len(n_cells)) {
    nb <- cell_neighbors[[k]]
    if (length(nb) == 0L) next          # all stats stay NA
    
    for (v in seq_len(n_vars)) {
      # Sub-matrix: neighbors Ã— years  (typically 3-8 rows Ã— 28 cols)
      sub <- var_matrices[[v]][nb, , drop = FALSE]
      
      # colMeans / colMins / colMaxs handle NA via na.rm
      # For small sub-matrices, a simple apply is fine and avoids matrixStats
      # dependency, but matrixStats would be faster if available.
      #
      # We write a small vectorized version that handles all-NA columns.
      for (j in seq_len(n_years)) {
        vals <- sub[, j]
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0L) next    # stays NA
        res_max[[v]][k, j]  <- max(vals)
        res_min[[v]][k, j]  <- min(vals)
        res_mean[[v]][k, j] <- mean(vals)
      }
    }
  }
  
  # -----------------------------------------------------------------------
  # 3b. (Optional, faster) Replace the inner j-loop with matrixStats if
  #     available.  This version is provided as an alternative.
  # -----------------------------------------------------------------------
  use_matrixStats <- requireNamespace("matrixStats", quietly = TRUE)
  
  if (use_matrixStats) {
    # Re-run with matrixStats for speed; overwrite results
    for (k in seq_len(n_cells)) {
      nb <- cell_neighbors[[k]]
      if (length(nb) == 0L) next
      
      for (v in seq_len(n_vars)) {
        sub <- var_matrices[[v]][nb, , drop = FALSE]
        res_max[[v]][k, ]  <- matrixStats::colMaxs(sub,  na.rm = TRUE)
        res_min[[v]][k, ]  <- matrixStats::colMins(sub,  na.rm = TRUE)
        res_mean[[v]][k, ] <- matrixStats::colMeans2(sub, na.rm = TRUE)
      }
    }
    
    # matrixStats returns -Inf/Inf for all-NA columns; fix to NA
    for (v in seq_len(n_vars)) {
      res_max[[v]][is.infinite(res_max[[v]])]   <- NA_real_
      res_min[[v]][is.infinite(res_min[[v]])]   <- NA_real_
      res_mean[[v]][is.nan(res_mean[[v]])]      <- NA_real_
    }
  }
  
  # -----------------------------------------------------------------------
  # 4. Map results back to the data.table rows
  #
  #    Each result matrix is n_cells Ã— n_years.  dt already has cell_pos
  #    and year_col, so we use the same lin_idx to extract values.
  # -----------------------------------------------------------------------
  for (v in seq_len(n_vars)) {
    vn <- neighbor_source_vars[v]
    col_max  <- paste0(vn, "_neighbor_max")
    col_min  <- paste0(vn, "_neighbor_min")
    col_mean <- paste0(vn, "_neighbor_mean")
    
    dt[, (col_max)  := res_max[[v]][lin_idx]]
    dt[, (col_min)  := res_min[[v]][lin_idx]]
    dt[, (col_mean) := res_mean[[v]][lin_idx]]
  }
  
  # -----------------------------------------------------------------------
  # 5. Restore original order, drop helper columns, return
  # -----------------------------------------------------------------------
  setorder(dt, .row_order)
  dt[, c(".row_order", "cell_pos", "year_col") := NULL]
  
  # Return as data.frame if the input was a data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}


# ===========================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ===========================================================================
#
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- build_and_apply_neighbor_features(
#   cell_data            = cell_data,
#   id_order             = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   neighbor_source_vars = neighbor_source_vars
# )
#
# # The trained Random Forest model is untouched. Feed cell_data into
# # predict(rf_model, cell_data) exactly as before.
```

## Further Optimization: Rcpp Inner Loop (Optional, Massive Speedup)

The 344K Ã— 28 Ã— 5 inner loop in pure R is already ~100Ã— faster than the original, but if it's still a bottleneck (estimated ~10â€“20 minutes), the innermost loop can be pushed to C++ trivially:

```r
# Save as neighbor_stats.cpp, then: Rcpp::sourceCpp("neighbor_stats.cpp")

Rcpp::cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List compute_neighbor_stats_cpp(NumericMatrix varmat, List cell_neighbors) {
  int n_cells = varmat.nrow();
  int n_years = varmat.ncol();
  
  NumericMatrix rmax(n_cells, n_years);
  NumericMatrix rmin(n_cells, n_years);
  NumericMatrix rmean(n_cells, n_years);
  
  // Fill with NA
  std::fill(rmax.begin(),  rmax.end(),  NA_REAL);
  std::fill(rmin.begin(),  rmin.end(),  NA_REAL);
  std::fill(rmean.begin(), rmean.end(), NA_REAL);
  
  for (int k = 0; k < n_cells; k++) {
    IntegerVector nb = cell_neighbors[k];
    int n_nb = nb.size();
    if (n_nb == 0) continue;
    
    for (int j = 0; j < n_years; j++) {
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      double vsum = 0.0;
      int    cnt  = 0;
      
      for (int m = 0; m < n_nb; m++) {
        double val = varmat(nb[m] - 1, j);   // R is 1-indexed
        if (ISNA(val)) continue;
        if (val > vmax) vmax = val;
        if (val < vmin) vmin = val;
        vsum += val;
        cnt++;
      }
      
      if (cnt > 0) {
        rmax(k, j)  = vmax;
        rmin(k, j)  = vmin;
        rmean(k, j) = vsum / cnt;
      }
    }
  }
  
  return List::create(
    Named("max")  = rmax,
    Named("min")  = rmin,
    Named("mean") = rmean
  );
}
')
```

Replace the inner loop in the R function with:

```r
for (v in seq_len(n_vars)) {
  stats <- compute_neighbor_stats_cpp(var_matrices[[v]], cell_neighbors)
  res_max[[v]]  <- stats$max
  res_min[[v]]  <- stats$min
  res_mean[[v]] <- stats$mean
}
```

## Performance Comparison

| Aspect | Original | Optimized (R) | Optimized (Rcpp) |
|---|---|---|---|
| String operations | ~51.7M `paste` + hash | **0** | **0** |
| Neighbor lookups | 6.46M (cell-year) | 344K (cell) | 344K (cell) |
| Per-variable passes over rows | 5 Ã— 6.46M | 5 Ã— 344K (matrix) | 5 Ã— 344K (compiled) |
| RAM overhead | Large named character vector | ~5 matrices of 344K Ã— 28 doubles (~38 MB each) | Same |
| Estimated time | **86+ hours** | **~15â€“30 min** | **~1â€“3 min** |

The key insight: **factor the topology (spatial) from the panel (temporal) dimension**, eliminate all string encoding, and operate on dense integer-indexed matrices. The trained Random Forest model and all numerical outputs are preserved identically.
END_RESPONSE>>>
