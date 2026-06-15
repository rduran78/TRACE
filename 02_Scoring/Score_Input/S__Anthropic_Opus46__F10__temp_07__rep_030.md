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
run_id: Anthropic_F10_T07_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation suffers from three compounding bottlenecks:

1. **O(nÂ²)-scale key-hashing in `build_neighbor_lookup`**: For ~6.46 million rows, the function creates a named character vector (`idx_lookup`) of length 6.46M, then for each of the 6.46M rows, performs character-key lookups into it. Named vector lookup in R is O(n) per query in the worst case (linear scan of names), making the total cost potentially O(nÂ²). Even with hashing, creating 6.46M paste-based keys and performing 6.46M Ã— k lookups is extremely slow.

2. **Row-level `lapply` in `compute_neighbor_stats`**: For each of 6.46M rows, an anonymous function allocates a 3-element vector, subsets a numeric vector by indices, removes NAs, and computes three summary statistics. The per-element overhead of `lapply` over 6.46M iterations is enormous â€” each iteration has R interpreter overhead, memory allocation, and no vectorization.

3. **Redundant topology recomputation across years**: The rook-neighbor graph is purely spatial â€” it does not change across years. Yet the current `build_neighbor_lookup` entangles spatial topology with temporal indexing, rebuilding the full lookup for all 6.46M rows at once rather than exploiting the fact that the same adjacency structure applies identically to each of the 28 years.

4. **Memory pressure**: The `neighbor_lookup` list of 6.46M integer vectors, plus intermediate character vectors, can consume several GB on a 16 GB laptop, causing GC thrashing.

**Summary**: The runtime is dominated by (a) character-key construction/lookup at the panel level and (b) row-level R-interpreter loops over millions of rows. The adjacency structure is year-invariant but is never exploited as such.

---

## Optimization Strategy

### Core Insight: Separate Topology from Time

The rook-neighbor adjacency is a **spatial** graph over 344,208 cells. It is identical for every year. Instead of building a 6.46M-row lookup, we build a **sparse adjacency matrix once** over the 344,208 cells, then for each year and each variable, we use sparse matrixâ€“vector multiplication to compute neighbor sums and counts, from which we derive mean, max, and min.

### Specific Techniques

| Technique | Speedup Source |
|---|---|
| **Sparse matrix (CSC) representation** | `Matrix::sparseMatrix` stores the adjacency in compressed form. Matrixâ€“vector multiply for sum and count is O(nnz) â‰ˆ 1.37M operations â€” done in compiled C code. |
| **Year-sliced vectorized operations** | For each of 28 years, extract the ~344K-length variable vector, do sparse mat-vec for sum/count â†’ mean. For max/min, use a grouped C-level operation. |
| **`data.table` for fast group indexing** | Sorting and splitting by year via `data.table` is orders of magnitude faster than `paste`-based key lookups. |
| **Pre-built CSR row-pointer structure for max/min** | Sparse mat-vec gives sum and count (â†’ mean), but max/min require iterating over neighbor values. We use a compiled Rcpp loop or a vectorized "expandâ€“groupâ€“summarize" approach via `data.table`. |
| **Single adjacency build, reused 28 Ã— 5 = 140 times** | The sparse matrix is built once (seconds) and reused for every variable Ã— year combination. |

### Expected Speedup

- `build_neighbor_lookup`: from ~hours to **<1 second** (sparse matrix construction).
- `compute_neighbor_stats` per variable: from ~17 hours to **~10â€“30 seconds** (28 years Ã— sparse operations).
- Total: from 86+ hours to **~3â€“5 minutes**.

### Numerical Equivalence

The sparse matrix encodes exactly the same directed rook-neighbor relationships. The sum/count â†’ mean, and explicit neighbor-value max/min, produce bit-identical results to the original code (IEEE 754 floating point addition is order-dependent, but we preserve the same logical aggregation; for exact equivalence we use the same neighbor sets).

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED SPATIAL NEIGHBOR AGGREGATION PIPELINE
# =============================================================================
# Requirements: data.table, Matrix, Rcpp (for max/min)
# Preserves: trained Random Forest model, original numerical estimand
# =============================================================================

library(data.table)
library(Matrix)

# ---------------------------------------------------------------------------
# STEP 0: Compile a small Rcpp helper for row-wise sparse max/min
#         This avoids R-level loops over 344K cells Ã— 28 years.
# ---------------------------------------------------------------------------
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// Compute row-wise max, min, sum, count over a CSR-like structure.
// p: row pointers (length n+1, 0-indexed)
// j: column indices (0-indexed)
// vals: the attribute vector (length n) â€” we look up vals[j[k]]
// Returns an n x 4 matrix: [max, min, sum, count]
// [[Rcpp::export]]
NumericMatrix sparse_neighbor_stats(IntegerVector p, IntegerVector j,
                                    NumericVector vals) {
  int n = p.size() - 1;
  NumericMatrix out(n, 4);
  // columns: 0=max, 1=min, 2=sum, 3=count

  for (int i = 0; i < n; i++) {
    int start = p[i];
    int end   = p[i + 1];
    double mx  = NA_REAL;
    double mn  = NA_REAL;
    double sm  = 0.0;
    int    cnt = 0;

    for (int k = start; k < end; k++) {
      double v = vals[j[k]];
      if (!NumericVector::is_na(v)) {
        if (cnt == 0) {
          mx = v;
          mn = v;
        } else {
          if (v > mx) mx = v;
          if (v < mn) mn = v;
        }
        sm += v;
        cnt++;
      }
    }
    if (cnt == 0) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
      out(i, 2) = NA_REAL;
    } else {
      out(i, 0) = mx;
      out(i, 1) = mn;
      out(i, 2) = sm / cnt;  // mean
    }
    out(i, 3) = (double)cnt;
  }
  return out;
}
')

# ---------------------------------------------------------------------------
# STEP 1: Build the sparse adjacency matrix ONCE (CSR for Rcpp)
# ---------------------------------------------------------------------------
# Inputs:
#   id_order             : integer vector of 344,208 cell IDs in canonical order
#   rook_neighbors_unique: spdep nb object (list of length 344,208)
#                          each element is an integer vector of neighbor indices
#                          (1-based into id_order), with 0 meaning no neighbors.

build_sparse_adjacency <- function(id_order, neighbors) {
  n <- length(id_order)
  
  # Build COO (coordinate) representation
  # For each cell i, neighbors[[i]] gives indices j of its rook neighbors
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  
  for (i in seq_len(n)) {
    nb <- neighbors[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nb <- nb[nb != 0L]
    if (length(nb) > 0) {
      from_list[[i]] <- rep.int(i, length(nb))
      to_list[[i]]   <- nb
    }
  }
  
  from_idx <- unlist(from_list, use.names = FALSE)
  to_idx   <- unlist(to_list, use.names = FALSE)
  
  # Build a sparse matrix in CSR-like form for Rcpp
  # We need row pointers (p) and column indices (j), both 0-indexed
  # Sort by row
  ord <- order(from_idx)
  from_idx <- from_idx[ord]
  to_idx   <- to_idx[ord]
  
  # Row pointers
  p <- integer(n + 1L)
  if (length(from_idx) > 0) {
    tab <- tabulate(from_idx, nbins = n)
    p <- c(0L, cumsum(tab))
  }
  
  # Column indices (0-indexed for Rcpp)
  j <- to_idx - 1L
  
  list(
    n = n,
    p = as.integer(p),
    j = as.integer(j),
    id_order = id_order,
    nnz = length(j)
  )
}

# ---------------------------------------------------------------------------
# STEP 2: Compute neighbor features for all variables, all years
# ---------------------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, adj, neighbor_source_vars) {
  # Convert to data.table for fast operations (by reference if already DT)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  n_cells <- adj$n
  id_order <- adj$id_order
  
  # Create a mapping from cell ID to canonical index (1-based)
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Get sorted unique years
  years <- sort(unique(cell_data$year))
  cat(sprintf("Processing %d cells Ã— %d years Ã— %d variables\n",
              n_cells, length(years), length(neighbor_source_vars)))
  cat(sprintf("Adjacency: %d directed edges (nnz)\n", adj$nnz))
  
  # Pre-allocate result columns
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("n_max_", var_name)
    col_min  <- paste0("n_min_", var_name)
    col_mean <- paste0("n_mean_", var_name)
    cell_data[, (col_max)  := NA_real_]
    cell_data[, (col_min)  := NA_real_]
    cell_data[, (col_mean) := NA_real_]
  }
  
  # Key the data.table for fast subsetting
  setkey(cell_data, year, id)
  
  # Process year by year
  for (yr in years) {
    # Extract rows for this year, in canonical cell order
    year_rows <- cell_data[.(yr)]  # keyed lookup on year
    
    # Map these rows to canonical cell indices
    year_cell_idx <- id_to_idx[as.character(year_rows$id)]
    
    # Build a dense vector of length n_cells for each variable
    # (cells not present in this year get NA â€” their neighbors will skip them)
    
    for (var_name in neighbor_source_vars) {
      # Dense vector: vals[canonical_index] = value
      vals <- rep(NA_real_, n_cells)
      vals[year_cell_idx] <- year_rows[[var_name]]
      
      # Compute neighbor stats using Rcpp CSR traversal
      stats <- sparse_neighbor_stats(adj$p, adj$j, vals)
      # stats is n_cells Ã— 4: [max, min, mean, count]
      
      # Extract results for cells that exist in this year
      col_max  <- paste0("n_max_", var_name)
      col_min  <- paste0("n_min_", var_name)
      col_mean <- paste0("n_mean_", var_name)
      
      # Write back to the data.table rows for this year
      # year_cell_idx maps year_rows to canonical positions
      result_max  <- stats[year_cell_idx, 1]
      result_min  <- stats[year_cell_idx, 2]
      result_mean <- stats[year_cell_idx, 3]
      
      # Use set() for fast by-reference assignment (no copy)
      row_indices <- which(cell_data$year == yr)
      # But we need the rows in the same order as year_rows
      # Since we keyed on (year, id) and year_rows came from cell_data[.(yr)],
      # row_indices should align. Let's be safe:
      set(cell_data, i = row_indices, j = col_max,  value = result_max)
      set(cell_data, i = row_indices, j = col_min,  value = result_min)
      set(cell_data, i = row_indices, j = col_mean, value = result_mean)
    }
    
    if (yr %% 5 == 0 || yr == years[1] || yr == years[length(years)]) {
      cat(sprintf("  Year %d complete\n", yr))
    }
  }
  
  return(cell_data)
}

# ---------------------------------------------------------------------------
# STEP 3: Main execution pipeline
# ---------------------------------------------------------------------------
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model, neighbor_source_vars = NULL) {
  
  if (is.null(neighbor_source_vars)) {
    neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  }
  
  t0 <- proc.time()
  
  # STEP 1: Build adjacency (once, ~1-2 seconds)
  cat("Building sparse adjacency structure...\n")
  adj <- build_sparse_adjacency(id_order, rook_neighbors_unique)
  cat(sprintf("  Done: %d nodes, %d edges (%.1f sec)\n",
              adj$n, adj$nnz, (proc.time() - t0)[3]))
  
  # STEP 2: Compute all neighbor features (~2-4 minutes)
  cat("Computing neighbor statistics...\n")
  t1 <- proc.time()
  cell_data <- compute_all_neighbor_features(cell_data, adj, neighbor_source_vars)
  cat(sprintf("  Done: %.1f sec\n", (proc.time() - t1)[3]))
  
  # STEP 3: Predict with existing Random Forest (no retraining)
  cat("Generating predictions with pre-trained Random Forest...\n")
  t2 <- proc.time()
  cell_data$prediction <- predict(rf_model, newdata = cell_data)
  cat(sprintf("  Done: %.1f sec\n", (proc.time() - t2)[3]))
  
  total_time <- (proc.time() - t0)[3]
  cat(sprintf("\nTotal pipeline time: %.1f sec (%.1f min)\n",
              total_time, total_time / 60))
  
  return(cell_data)
}

# ---------------------------------------------------------------------------
# USAGE:
# ---------------------------------------------------------------------------
# # Load your data
# cell_data              <- readRDS("cell_data.rds")
# id_order               <- readRDS("id_order.rds")
# rook_neighbors_unique  <- readRDS("rook_neighbors_unique.rds")
# rf_model               <- readRDS("rf_model.rds")
#
# # Run
# result <- run_optimized_pipeline(
#   cell_data, id_order, rook_neighbors_unique, rf_model
# )
```

---

## Correctness / Numerical Equivalence Verification

```r
# ---------------------------------------------------------------------------
# OPTIONAL: Verify numerical equivalence against original implementation
# Run on a small subset (e.g., 1000 cells Ã— 2 years) before full execution
# ---------------------------------------------------------------------------
verify_equivalence <- function(cell_data_original, cell_data_optimized,
                               neighbor_source_vars, tol = 1e-10) {
  all_ok <- TRUE
  for (var_name in neighbor_source_vars) {
    for (stat in c("max", "min", "mean")) {
      col <- paste0("n_", stat, "_", var_name)
      orig <- cell_data_original[[col]]
      optm <- cell_data_optimized[[col]]
      
      # Compare non-NA values
      both_present <- !is.na(orig) & !is.na(optm)
      both_na      <- is.na(orig) & is.na(optm)
      
      if (!all(both_present | both_na)) {
        cat(sprintf("MISMATCH (NA pattern): %s\n", col))
        all_ok <- FALSE
        next
      }
      
      max_diff <- max(abs(orig[both_present] - optm[both_present]))
      if (max_diff > tol) {
        cat(sprintf("MISMATCH (values): %s â€” max diff = %e\n", col, max_diff))
        all_ok <- FALSE
      } else {
        cat(sprintf("OK: %s â€” max diff = %e\n", col, max_diff))
      }
    }
  }
  if (all_ok) cat("\nAll checks passed: numerical equivalence confirmed.\n")
  return(all_ok)
}
```

---

## Performance Summary

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| Adjacency construction | Hours (character hashing over 6.46M rows) | ~1â€“2 sec (integer COO â†’ CSR) | ~3000Ã— |
| Neighbor stats (per variable) | ~17 hours (R `lapply` over 6.46M rows) | ~20â€“40 sec (Rcpp CSR traversal, 28 year-slices) | ~1500Ã— |
| Total (5 variables) | 86+ hours | **3â€“5 minutes** | ~1000â€“1700Ã— |
| Peak memory | ~8â€“12 GB (character keys + list of 6.46M vectors) | ~2â€“3 GB (sparse CSR + dense year-slices) | ~4Ã— less |

The trained Random Forest model is loaded and used as-is via `predict()` â€” no retraining occurs. The neighbor aggregation (max, min, mean over directed rook-neighbor attribute values per cell-year) is numerically identical to the original implementation.
END_RESPONSE>>>
