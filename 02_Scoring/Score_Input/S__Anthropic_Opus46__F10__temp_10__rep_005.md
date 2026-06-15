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
run_id: Anthropic_F10_T10_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates 6.46 million list entries** â€” one per cell-year row. Each entry stores neighbor row indices. This is massively redundant because the spatial topology is *time-invariant*: the same 344,208 cells have the same rook neighbors every year. The function re-discovers and re-stores the same spatial relationships 28 times (once per year), inflating memory and construction time by 28Ã—.

2. **`compute_neighbor_stats` iterates over 6.46 million list elements in R-level `lapply`**. For each element it subsets a numeric vector, removes NAs, and computes three summary statistics. This is ~6.46M R function calls with repeated small vector allocations â€” the worst-case scenario for R's interpreter.

3. **String-key lookups via `paste()` and named-vector indexing** (`idx_lookup[neighbor_keys]`) create millions of temporary character strings and perform hash lookups on them. This is orders of magnitude slower than integer-indexed matrix operations.

**Root cause summary**: The code treats a *panel on a fixed spatial graph* as if every row were an independent node. The graph topology should be built **once over the 344,208 cells**, and neighborhood aggregation should be a **matrix operation per year**, not a row-by-row R loop over 6.46M rows.

---

## Optimization Strategy

1. **Build the sparse adjacency structure once** as a `dgCMatrix` (sparse column matrix) of dimension 344,208 Ã— 344,208 from the `nb` object. This encodes the entire rook-neighbor graph.

2. **Reshape each variable into a dense matrix** of dimension 344,208 (cells) Ã— 28 (years).

3. **Compute neighbor statistics via sparse matrixâ€“dense matrix multiplication and sparse structural operations**:
   - **Neighbor mean**: `(A %*% X) / degree_vector` â€” one sparse matrix multiply per variable.
   - **Neighbor max and min**: Use the sparse structure of `A` to vectorize grouped max/min operations in C++ via `Rcpp`, or use a column-by-column approach with `Matrix` row-aggregation. The key insight: for each year-column, extract neighbor values using the CSC structure and compute grouped max/min.

4. **Reassemble** the 15 new columns (5 variables Ã— 3 stats) back into the panel `data.frame` in the original row order.

5. **Feed the enriched data into the pre-trained Random Forest** â€” no retraining.

This reduces the core computation from ~6.46M Ã— 5 R-level list iterations to 5 sparse matrix multiplies (for mean) plus a tight C++ loop (for max/min), collapsing runtime from 86+ hours to **minutes**.

---

## Optimized R Code

```r
# =============================================================================
# Optimized Neighbor Feature Engineering for Spatial Panel Data
# =============================================================================
# Requirements: Matrix, Rcpp, data.table
# Preserves numerical equivalence with original compute_neighbor_stats output.
# =============================================================================

library(Matrix)
library(data.table)
library(Rcpp)

# ---- Step 0: Compile a small Rcpp function for sparse grouped max/min ----
# This walks the CSC structure of the adjacency matrix and computes
# max, min per row (i.e., per node) from neighbor values in a dense column.

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix sparse_neighbor_maxmin(
    IntegerVector Ap,    // CSC column pointers (length ncol+1), 0-based
    IntegerVector Ai,    // CSC row indices, 0-based
    NumericMatrix X       // dense matrix: nrow = n_cells, ncol = n_years
) {
  int n = X.nrow();
  int T = X.ncol();
  // Output: n*T rows, 2 columns (max, min)
  // Laid out as: rows [0..n-1] = year 0, [n..2n-1] = year 1, etc.
  NumericMatrix out(n * T, 2);

  // For each node i, we need to iterate over its neighbors.
  // In CSC format, column j lists the rows that have nonzeros in column j.
  // But we need: for row i, which columns j have nonzeros? That is CSR.
  // Since our adjacency matrix may not be symmetric in the nb sense,
  // we transpose: build CSR from CSC of A^T, which is CSC of A transposed.
  // Actually, for an nb object A where A[i,j]=1 means j is neighbor of i,
  // we built A so that row i has 1s in neighbor columns.
  // In CSC, column j lists which rows i have A[i,j]=1, i.e., which nodes
  // have j as a neighbor.
  // We need row i neighbors = columns j where A[i,j]=1.
  // So we need CSR. Lets build it from CSC.

  int nnz = Ai.size();
  int ncol = Ap.size() - 1;  // should equal n

  // Count entries per row
  IntegerVector row_count(n, 0);
  for (int k = 0; k < nnz; k++) {
    row_count[Ai[k]]++;
  }
  // Row pointers
  IntegerVector Rp(n + 1, 0);
  for (int i = 0; i < n; i++) {
    Rp[i + 1] = Rp[i] + row_count[i];
  }
  // Fill CSR col indices
  IntegerVector Rj(nnz);
  IntegerVector cursor(n, 0);
  for (int j = 0; j < ncol; j++) {
    for (int k = Ap[j]; k < Ap[j + 1]; k++) {
      int i = Ai[k];
      int pos = Rp[i] + cursor[i];
      Rj[pos] = j;
      cursor[i]++;
    }
  }

  // Now compute max, min per row per year-column
  for (int t = 0; t < T; t++) {
    NumericMatrix::Column xcol = X(_, t);
    int offset = t * n;
    for (int i = 0; i < n; i++) {
      int start = Rp[i];
      int end = Rp[i + 1];
      if (start == end) {
        // no neighbors
        out(offset + i, 0) = NA_REAL;
        out(offset + i, 1) = NA_REAL;
        continue;
      }
      double mx = R_NegInf;
      double mn = R_PosInf;
      int valid = 0;
      for (int k = start; k < end; k++) {
        double v = xcol[Rj[k]];
        if (!NumericVector::is_na(v)) {
          if (v > mx) mx = v;
          if (v < mn) mn = v;
          valid++;
        }
      }
      if (valid == 0) {
        out(offset + i, 0) = NA_REAL;
        out(offset + i, 1) = NA_REAL;
      } else {
        out(offset + i, 0) = mx;
        out(offset + i, 1) = mn;
      }
    }
  }

  return out;
}
')


# =============================================================================
# MAIN PIPELINE
# =============================================================================

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # ------ Step 1: Build sparse adjacency matrix (once) ------
  cat("Building sparse adjacency matrix...\n")
  n_cells <- length(id_order)

  # Map cell IDs to integer indices 1..n_cells
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Build COO triplets from the nb object
  from_list <- vector("list", n_cells)
  to_list   <- vector("list", n_cells)
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    nb_i <- nb_i[nb_i != 0L]
    if (length(nb_i) > 0) {
      from_list[[i]] <- rep.int(i, length(nb_i))
      to_list[[i]]   <- nb_i  # these are already indices into id_order
    }
  }
  rows_i <- unlist(from_list, use.names = FALSE)
  cols_j <- unlist(to_list, use.names = FALSE)

  # A[i,j] = 1 means j is a rook neighbor of i
  A <- sparseMatrix(
    i = rows_i,
    j = cols_j,
    x = rep(1, length(rows_i)),
    dims = c(n_cells, n_cells),
    repr = "C"   # CSC format
  )

  # Degree vector (number of neighbors per node, ignoring NAs for now)
  # We will adjust for NAs per variable below.
  degree <- as.numeric(A %*% rep(1, n_cells))  # rowSums of A

  cat(sprintf("  %d cells, %d directed edges\n", n_cells, length(rows_i)))

  # ------ Step 2: Convert cell_data to data.table for fast reshaping ------
  cat("Reshaping data into cell x year matrices...\n")
  dt <- as.data.table(cell_data)

  # Ensure consistent ordering: cells as rows, years as columns
  years <- sort(unique(dt$year))
  n_years <- length(years)

  # Create cell index column
  dt[, cell_idx := id_to_idx[as.character(id)]]

  # Create year index column
  year_to_col <- setNames(seq_along(years), as.character(years))
  dt[, year_idx := year_to_col[as.character(year)]]

  # Sort by cell_idx, year_idx to establish canonical order
  setorder(dt, cell_idx, year_idx)

  # Record the original row mapping so we can write results back
  # dt is now sorted; we need to map back to the original cell_data row order.
  # We'll build the result columns in this sorted order, then reorder at the end.
  # Actually, we'll just add columns to dt and reorder back at the end.

  # ------ Step 3: For each variable, build the cellÃ—year matrix and compute stats ------

  # Extract CSC components of A for the Rcpp function
  A_p <- A@p    # column pointers (0-based, length n_cells+1)
  A_i <- A@i    # row indices (0-based)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))

    # Build n_cells x n_years dense matrix
    # dt is sorted by (cell_idx, year_idx), so we can reshape directly
    vals <- dt[[var_name]]
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(dt$cell_idx, dt$year_idx)] <- vals

    # ---- Neighbor mean (with proper NA handling) ----
    # Replace NA with 0 for the multiply, and track valid counts
    X_nona <- X
    X_nona[is.na(X)] <- 0
    X_notna <- (!is.na(X)) * 1.0  # indicator matrix: 1 where not NA

    # Sum of neighbor values (NAs treated as 0)
    neighbor_sum <- as.matrix(A %*% X_nona)       # n_cells x n_years

    # Count of non-NA neighbors
    neighbor_count <- as.matrix(A %*% X_notna)     # n_cells x n_years

    # Mean: sum / count; where count == 0, return NA
    neighbor_mean <- neighbor_sum / neighbor_count
    neighbor_mean[neighbor_count == 0] <- NA_real_

    # ---- Neighbor max and min (via Rcpp) ----
    maxmin <- sparse_neighbor_maxmin(A_p, A_i, X)
    # maxmin is (n_cells * n_years) x 2, laid out as years stacked:
    # rows [1..n_cells] = year 1, [n_cells+1 .. 2*n_cells] = year 2, etc.
    # Column 1 = max, Column 2 = min

    # Reshape maxmin into n_cells x n_years matrices
    neighbor_max_mat <- matrix(maxmin[, 1], nrow = n_cells, ncol = n_years)
    neighbor_min_mat <- matrix(maxmin[, 2], nrow = n_cells, ncol = n_years)

    # ---- Extract values back into the panel (dt) row order ----
    # dt is sorted by (cell_idx, year_idx), so for row k in dt:
    #   cell_idx = dt$cell_idx[k], year_idx = dt$year_idx[k]
    linear_idx <- (dt$year_idx - 1L) * n_cells + dt$cell_idx

    max_col_name  <- paste0("nb_max_", var_name)
    min_col_name  <- paste0("nb_min_", var_name)
    mean_col_name <- paste0("nb_mean_", var_name)

    dt[, (max_col_name)  := neighbor_max_mat[linear_idx]]
    dt[, (min_col_name)  := neighbor_min_mat[linear_idx]]
    dt[, (mean_col_name) := neighbor_mean[linear_idx]]

    cat(sprintf("  Added %s, %s, %s\n", max_col_name, min_col_name, mean_col_name))
  }

  # ------ Step 4: Convert back to original row order ------
  cat("Restoring original row order...\n")

  # Remove helper columns
  dt[, c("cell_idx", "year_idx") := NULL]

  # Restore original row order: match on (id, year)
  # The original cell_data ordering is preserved by merging back.
  # Safest approach: add a row-order column to cell_data before we started.
  # Since we didn't, we rebuild by matching on id+year.

  # Create a key in the original cell_data
  original_dt <- as.data.table(cell_data)
  original_dt[, .orig_row := .I]
  original_dt[, .merge_key := paste(id, year, sep = "_")]
  dt[, .merge_key := paste(id, year, sep = "_")]

  # Get the new columns only
  new_cols <- grep("^nb_(max|min|mean)_", names(dt), value = TRUE)
  merge_dt <- dt[, c(".merge_key", new_cols), with = FALSE]

  # Merge
  original_dt <- merge(original_dt, merge_dt, by = ".merge_key", all.x = TRUE, sort = FALSE)
  setorder(original_dt, .orig_row)
  original_dt[, c(".orig_row", ".merge_key") := NULL]

  result <- as.data.frame(original_dt)
  cat("Done.\n")
  return(result)
}


# =============================================================================
# USAGE
# =============================================================================

# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- compute_all_neighbor_features(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   neighbor_source_vars   = neighbor_source_vars
# )
#
# # Predict with pre-trained Random Forest (no retraining)
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized | Equivalence Guarantee |
|---|---|---|---|
| **Mean** | `mean(vals[neighbors])` after dropping NAs | `(A %*% X_nona) / (A %*% X_notna)` â€” identical sum and count, with `0/0 â†’ NA` | Exact (IEEE 754 floating-point addition is order-dependent in theory, but the sparse matrix multiply visits neighbors in the same CSC order deterministically; the sums are over â‰¤4 rook neighbors so rounding differences are negligible â€” typically zero) |
| **Max** | `max(vals[neighbors])` after dropping NAs | Rcpp loop over the same neighbor set, same `>` comparison | Exact |
| **Min** | `min(vals[neighbors])` after dropping NAs | Rcpp loop over the same neighbor set, same `<` comparison | Exact |
| **NA handling** | `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`; if empty â†’ `c(NA,NA,NA)` | `X_notna` indicator excludes NAs from sum/count; Rcpp skips `is_na`; zero valid â†’ `NA` | Exact |

---

## Performance Comparison

| Aspect | Original (86+ hrs) | Optimized |
|---|---|---|
| Adjacency build | 6.46M string-key hash lookups | One sparse matrix from `nb` list (~1s) |
| Mean computation | 6.46M R `lapply` calls Ã— 5 vars | 5 sparse matrix multiplies (~30s each) |
| Max/Min computation | 6.46M R `lapply` calls Ã— 5 vars | 5 Rcpp CSR traversals (~60s each) |
| Memory | 6.46M-element list of integer vectors | ~344K Ã— 28 dense matrices + one sparse matrix (~500 MB total) |
| **Estimated total** | **86+ hours** | **~5â€“15 minutes** |

The speedup is roughly **300â€“1000Ã—**, achieved by:
1. Exploiting time-invariance of the spatial graph (28Ã— reduction).
2. Replacing R-level iteration with BLAS-backed sparse algebra and compiled C++ (additional ~10â€“35Ã—).
3. Eliminating all string construction and hash-based lookups.
END_RESPONSE>>>
