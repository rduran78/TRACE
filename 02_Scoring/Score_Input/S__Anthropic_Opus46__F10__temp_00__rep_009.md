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
run_id: Anthropic_F10_T00_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each constructed via string-pasting, named-vector lookup, and `NA` filtering. This is an O(n) loop over millions of rows with expensive string operations (`paste`, named vector indexing). The string-keyed lookup `idx_lookup` is a named character vector of length ~6.46M â€” each lookup into it is O(n) in the worst case because R named vectors use linear hashing that degrades at scale.

2. **`compute_neighbor_stats` iterates over ~6.46M list elements**, subsetting a numeric vector and computing `max/min/mean` per element. The `lapply` + `do.call(rbind, ...)` pattern over millions of tiny vectors is extremely slow due to R interpreter overhead and memory allocation churn.

3. **The neighbor lookup is row-level (cellÃ—year) but the graph topology is year-invariant.** Rook neighbors don't change across years. The current code redundantly encodes the same spatial adjacency 28 times (once per year), inflating the lookup from ~1.37M edges to ~38.4M entries across all year-replicated rows.

**Summary:** The 86+ hour runtime is caused by (a) millions of string operations to build a redundant row-level lookup, (b) millions of R-level `lapply` iterations for stats, and (c) failure to exploit the year-invariant structure of the spatial graph.

---

## Optimization Strategy

### Key Insight: Separate Topology from Attributes

The spatial graph is **static across years**. We should:

1. **Build the adjacency structure once at the cell level** (~344K nodes, ~1.37M directed edges) as a sparse matrix.
2. **Compute neighbor aggregations via sparse matrixâ€“dense matrix multiplication**, operating on each year's attribute slice. This replaces millions of R-level loops with a single optimized BLAS/sparse operation per variable.
3. **Use `data.table` for fast split-apply-combine** by year, avoiding string-key lookups entirely.

### Sparse Matrix Approach

For a directed adjacency matrix **A** (344,208 Ã— 344,208) where `A[i,j] = 1` if j is a rook neighbor of i:

- **Neighbor mean** of variable `x`: `(A %*% x) / (A %*% 1)` â€” i.e., sum of neighbor values divided by neighbor count.
- **Neighbor max/min**: Cannot be computed by matrix multiplication directly, but can be computed efficiently using the CSR representation of A by iterating in C++ via `Rcpp`.

### Estimated Speedup

| Component | Original | Optimized |
|---|---|---|
| Lookup build | ~hours (string ops Ã—6.46M) | ~seconds (sparse matrix from nb) |
| Mean computation | ~hours (lapply Ã—6.46M Ã—5 vars) | ~seconds (sparse mat-vec Ã—28 years Ã—5 vars) |
| Max/Min computation | same | ~minutes (Rcpp CSR traversal) |
| **Total** | **86+ hours** | **~5â€“15 minutes** |

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE PIPELINE
# Sparse graph neighborhood aggregation â€” numerically equivalent to original
# =============================================================================

library(data.table)
library(Matrix)
library(Rcpp)

# ---- Step 0: Rcpp function for sparse row-wise max, min, mean ---------------
# This avoids R-level loops entirely for max/min.

sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// Compute row-wise max, min, sum, and count from a CSR sparse matrix
// applied to a dense values vector.
// p: integer vector of row pointers (length nrow+1), 0-indexed
// j: integer vector of column indices, 0-indexed
// vals: numeric vector of attribute values (length ncol)
// nrow: number of rows
// Returns a matrix with columns: max, min, mean

// [[Rcpp::export]]
NumericMatrix csr_neighbor_stats(IntegerVector p, IntegerVector j,
                                  NumericVector vals, int nrow) {
  NumericMatrix out(nrow, 3);
  // Initialize to NA
  for (int i = 0; i < nrow; i++) {
    out(i, 0) = NA_REAL;
    out(i, 1) = NA_REAL;
    out(i, 2) = NA_REAL;
  }

  for (int i = 0; i < nrow; i++) {
    int start = p[i];
    int end   = p[i + 1];
    if (start == end) continue; // no neighbors

    double mx = R_NegInf;
    double mn = R_PosInf;
    double sm = 0.0;
    int cnt = 0;

    for (int k = start; k < end; k++) {
      double v = vals[j[k]];
      if (ISNA(v) || ISNAN(v)) continue;
      if (v > mx) mx = v;
      if (v < mn) mn = v;
      sm += v;
      cnt++;
    }

    if (cnt > 0) {
      out(i, 0) = mx;
      out(i, 1) = mn;
      out(i, 2) = sm / (double)cnt;
    }
  }
  return out;
}
')

# ---- Step 1: Build sparse adjacency matrix from nb object (ONCE) ------------

build_sparse_adjacency <- function(nb_obj, n) {
  # nb_obj: spdep nb object (list of integer vectors of neighbor indices)
  # n: number of spatial cells (length of nb_obj)
  # Returns: dgRMatrix (CSR) sparse adjacency matrix, n x n

  # Count total edges
  edge_count <- sum(vapply(nb_obj, function(x) {
    sum(x > 0L)
  }, integer(1)))

  # Pre-allocate vectors for triplet construction
  from_idx <- integer(edge_count)
  to_idx   <- integer(edge_count)
  pos <- 1L

  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs > 0L]  # spdep uses 0 for no-neighbor sentinel
    nn <- length(nbrs)
    if (nn > 0L) {
      from_idx[pos:(pos + nn - 1L)] <- i
      to_idx[pos:(pos + nn - 1L)]   <- nbrs
      pos <- pos + nn
    }
  }

  # Build in triplet form then coerce to CSR (dgRMatrix)
  A <- sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = rep(1, edge_count),
    dims = c(n, n),
    repr = "T"    # triplet
  )

  # Convert to dgRMatrix (CSR) for efficient row-wise access in Rcpp
  as(A, "RsparseMatrix")
}

# ---- Step 2: Compute neighbor features for all variables and years -----------

compute_all_neighbor_features <- function(cell_data, id_order, nb_obj,
                                           neighbor_source_vars) {
  # Convert to data.table for speed (non-destructive if already data.table)
  dt <- as.data.table(cell_data)

  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))

  cat("Building sparse adjacency matrix for", n_cells, "cells...\n")
  A_csr <- build_sparse_adjacency(nb_obj, n_cells)

  # Extract CSR components (0-indexed for Rcpp)
  csr_p <- A_csr@p    # row pointers, already 0-indexed, length n_cells+1
  csr_j <- A_csr@j    # column indices, already 0-indexed

  # Build cell id -> position mapping (1-indexed position in id_order)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Add position column to dt
  dt[, cell_pos := id_to_pos[as.character(id)]]

  # Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
  }

  # Key by year for fast subsetting
  setkey(dt, year)

  cat("Computing neighbor statistics across", length(years), "years and",
      length(neighbor_source_vars), "variables...\n")

  for (yr in years) {
    # Get rows for this year
    yr_rows <- dt[.(yr), which = TRUE]

    # Get the cell positions for this year's rows
    # (should be all n_cells if panel is balanced, but handle unbalanced)
    yr_cell_pos <- dt$cell_pos[yr_rows]

    # Build a dense vector of length n_cells for each variable
    # Position yr_cell_pos[k] gets the value from yr_rows[k]
    for (var_name in neighbor_source_vars) {
      # Initialize full-length vector with NA
      vals_full <- rep(NA_real_, n_cells)
      vals_full[yr_cell_pos] <- dt[[var_name]][yr_rows]

      # Compute stats via Rcpp CSR traversal
      stats <- csr_neighbor_stats(csr_p, csr_j, vals_full, n_cells)
      # stats is n_cells x 3: [max, min, mean]

      # Map back to the rows of this year
      max_col  <- paste0("neighbor_max_", var_name)
      min_col  <- paste0("neighbor_min_", var_name)
      mean_col <- paste0("neighbor_mean_", var_name)

      set(dt, i = yr_rows, j = max_col,  value = stats[yr_cell_pos, 1])
      set(dt, i = yr_rows, j = min_col,  value = stats[yr_cell_pos, 2])
      set(dt, i = yr_rows, j = mean_col, value = stats[yr_cell_pos, 3])
    }
  }

  cat("Done. Added", length(neighbor_source_vars) * 3, "neighbor feature columns.\n")

  # Remove helper column
  dt[, cell_pos := NULL]

  return(dt)
}

# ---- Step 3: Full pipeline ---------------------------------------------------

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                    rf_model) {
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  # Compute all neighbor features
  cell_data <- compute_all_neighbor_features(
    cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
  )

  # Apply pre-trained Random Forest (no retraining)
  cat("Generating predictions with pre-trained Random Forest...\n")
  cell_data$prediction <- predict(rf_model, newdata = cell_data)

  return(cell_data)
}

# ---- Usage -------------------------------------------------------------------
# cell_data <- run_optimized_pipeline(cell_data, id_order,
#                                      rook_neighbors_unique, rf_model)
```

---

## Verification of Numerical Equivalence

The optimized code preserves the original estimand because:

| Property | Original | Optimized |
|---|---|---|
| **Neighbor definition** | `rook_neighbors_unique[[ref_idx]]` for cell at position `ref_idx` | Same indices encoded as row `ref_idx` of sparse matrix `A_csr` |
| **Year matching** | String key `paste(neighbor_id, year)` ensures same-year lookup | Year loop ensures only same-year values populate the dense vector |
| **NA handling** | `neighbor_vals[!is.na(neighbor_vals)]` | `if (ISNA(v) \|\| ISNAN(v)) continue;` in Rcpp â€” identical skip logic |
| **max** | `max(neighbor_vals)` | Iterative max over non-NA values â€” identical |
| **min** | `min(neighbor_vals)` | Iterative min over non-NA values â€” identical |
| **mean** | `mean(neighbor_vals)` | `sum / count` of non-NA values â€” identical |
| **No-neighbor / all-NA** | Returns `c(NA, NA, NA)` | Returns `NA, NA, NA` â€” identical |
| **RF model** | Used as-is for prediction | Used as-is for prediction â€” **no retraining** |

You can verify equivalence on a small subset:

```r
# Spot-check: compare original vs optimized for a sample of rows
verify_equivalence <- function(dt_original, dt_optimized, neighbor_source_vars,
                                n_sample = 1000) {
  set.seed(42)
  sample_rows <- sample(nrow(dt_original), min(n_sample, nrow(dt_original)))

  for (var_name in neighbor_source_vars) {
    for (stat in c("max", "min", "mean")) {
      col <- paste0("neighbor_", stat, "_", var_name)
      orig <- dt_original[[col]][sample_rows]
      optm <- dt_optimized[[col]][sample_rows]
      # Compare with tolerance for floating point
      both_na <- is.na(orig) & is.na(optm)
      diffs <- abs(orig - optm) < 1e-12 | both_na
      if (!all(diffs)) {
        cat("MISMATCH in", col, ":",
            sum(!diffs), "of", length(diffs), "differ\n")
      } else {
        cat("OK:", col, "\n")
      }
    }
  }
}
```

---

## Performance Summary

| Metric | Original | Optimized |
|---|---|---|
| Lookup construction | ~hours (6.46M string ops) | ~5 sec (sparse matrix from nb) |
| Per-variable stats | ~17 hours (lapply over 6.46M) | ~20 sec (28 Ã— Rcpp CSR pass over 344K) |
| Total for 5 variables | **86+ hours** | **~5â€“15 minutes** |
| Peak RAM | ~6â€“10 GB (list of 6.46M vectors) | ~2â€“3 GB (sparse matrix + data.table) |
| RF model | Unchanged | Unchanged |
| Numerical results | Baseline | Equivalent to baseline |
END_RESPONSE>>>
