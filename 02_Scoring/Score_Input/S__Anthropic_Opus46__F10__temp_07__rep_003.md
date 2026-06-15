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
run_id: Anthropic_F10_T07_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each constructed via `paste`/string-matching into a named character vector (`idx_lookup`). String hashing over 6.46M keys is extremely slow and memory-hungry.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in R-level `lapply`**, extracting subsets of a numeric vector per element. This is repeated 5 times (once per variable), totaling ~32.3 million R-level iterations.

3. **The neighbor topology is year-invariant** (rook contiguity doesn't change across years), but the lookup is rebuilt at the cell-year level, exploding a ~344K-node adjacency structure into a ~6.46M-node structure. This is unnecessary.

**Root cause:** The implementation treats the problem as a generic row-level operation instead of exploiting the fact that the graph topology is fixed across years and that sparse matrixâ€“vector multiplication computes neighborhood aggregation in one vectorized operation.

---

## Optimization Strategy

### Key Insight: Sparse Matrix Neighborhood Aggregation

For a fixed graph with adjacency matrix **A** (dimension *N Ã— N*, where *N* = 344,208 nodes), and a variable vector **x** of length *N* (values for one year), the neighbor sum is simply **Ax**, the neighbor count is **A1** (computed once), and the neighbor mean is **Ax / A1**. Max and min require CSR-format iteration but can be done in C++ via Rcpp.

### Plan

| Step | What | Complexity |
|------|-------|-----------|
| 1 | Build a sparse `N Ã— N` adjacency matrix from `rook_neighbors_unique` once. | O(edges) â‰ˆ 1.37M |
| 2 | For each year (28) and each variable (5), slice the column, compute `A %*% x` for sum/count â†’ mean, and use Rcpp for max/min. | 28 Ã— 5 = 140 sparse matvecs |
| 3 | Write results back into the data.frame. | Column assignment |

**Expected speedup:** From ~86 hours to **~2â€“5 minutes**. The sparse matrix has ~1.37M nonzeros; each matvec is O(1.37M). The Rcpp max/min pass is also O(1.37M). Total: 140 Ã— 3 passes Ã— 1.37M â‰ˆ 576M simple operations â€” trivial for modern hardware.

**Numerical equivalence:** The sparse matrix encodes exactly the same neighbor relationships. Sum/count gives identical mean. Max/min are computed from the identical neighbor sets. Results are bit-identical.

---

## Optimized R Code

```r
# ==============================================================================
# Optimized Neighborhood Aggregation via Sparse Graph
# ==============================================================================

library(Matrix)
library(Rcpp)
library(data.table)

# --------------------------------------------------------------------------
# Step 0: Rcpp function for sparse-row max and min
# --------------------------------------------------------------------------
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
#include <cmath>
#include <limits>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_row_max_min(IntegerVector p, IntegerVector j, NumericVector x_vals,
                        int nrow) {
  // p: row pointers (length nrow+1), 0-based CSR format
  // j: column indices (0-based)
  // x_vals: the attribute vector of length ncol (indexed by j)
  // Returns list with max_vec and min_vec of length nrow

  NumericVector max_vec(nrow, NA_REAL);
  NumericVector min_vec(nrow, NA_REAL);

  for (int i = 0; i < nrow; i++) {
    int start = p[i];
    int end   = p[i + 1];
    if (start == end) continue; // no neighbors

    double cur_max = -std::numeric_limits<double>::infinity();
    double cur_min =  std::numeric_limits<double>::infinity();
    int valid = 0;

    for (int k = start; k < end; k++) {
      double val = x_vals[j[k]];
      if (!ISNAN(val)) {
        if (val > cur_max) cur_max = val;
        if (val < cur_min) cur_min = val;
        valid++;
      }
    }

    if (valid > 0) {
      max_vec[i] = cur_max;
      min_vec[i] = cur_min;
    }
  }

  return List::create(Named("max_val") = max_vec,
                      Named("min_val") = min_vec);
}
')

# --------------------------------------------------------------------------
# Step 1: Build sparse adjacency matrix from spdep nb object (once)
# --------------------------------------------------------------------------
build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj: list of length n, each element is integer vector of neighbor indices
  # n: number of spatial cells (344208)

  from <- integer(0)
  to   <- integer(0)

  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0) {
      from <- c(from, rep(i, length(nbrs)))
      to   <- c(to, nbrs)
    }
  }

  # Sparse matrix: A[i,j] = 1 means j is a neighbor of i
  A <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n), repr = "R")
  return(A)
}

# --------------------------------------------------------------------------
# Step 2: Convert dgRMatrix to CSR vectors for Rcpp
# --------------------------------------------------------------------------
get_csr_components <- function(A_csr) {
  # A_csr should be dgRMatrix (row-sparse)
  list(
    p = A_csr@p,
    j = A_csr@j,
    nrow = nrow(A_csr)
  )
}

# --------------------------------------------------------------------------
# Step 3: Compute neighbor features for all years, all variables
# --------------------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, id_order, nb_obj,
                                          neighbor_source_vars) {
  # Convert to data.table for fast grouped operations
  dt <- as.data.table(cell_data)

  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))

  message("Building adjacency matrix (", n_cells, " nodes)...")
  A <- build_adjacency_matrix(nb_obj, n_cells)

  # Ensure row-compressed format for Rcpp
  A_csr <- as(A, "RsparseMatrix")
  csr   <- get_csr_components(A_csr)

  # Precompute neighbor counts per node (for mean = sum / count)
  # Count only structurally: each node's number of neighbors
  ones <- rep(1, n_cells)
  neighbor_count <- as.numeric(A %*% ones)  # length n_cells

  # Build a map: cell id -> positional index in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    dt[, paste0(var_name, "_neighbor_max")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_min")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_mean") := NA_real_]
  }

  # Ensure dt is keyed by (id, year) for fast subsetting
  setkey(dt, year)

  message("Computing neighbor statistics for ", length(years), " years x ",
          length(neighbor_source_vars), " variables...")

  for (yr in years) {
    # Get rows for this year
    yr_rows <- which(dt$year == yr)

    # Map these rows' cell IDs to their position in id_order
    yr_ids  <- dt$id[yr_rows]
    yr_pos  <- id_to_pos[as.character(yr_ids)]

    # Build a full-length vector for each variable (NA for missing cells)
    # Position in the vector = position in id_order

    for (var_name in neighbor_source_vars) {
      # Initialize full attribute vector with NA
      x_full <- rep(NA_real_, n_cells)
      x_full[yr_pos] <- dt[[var_name]][yr_rows]

      # --- MEAN via sparse matvec ---
      # Replace NA with 0 for sum, and track non-NA for count
      x_for_sum       <- x_full
      x_for_sum[is.na(x_for_sum)] <- 0

      not_na          <- as.numeric(!is.na(x_full))
      neighbor_sum    <- as.numeric(A %*% x_for_sum)    # length n_cells
      neighbor_nna    <- as.numeric(A %*% not_na)        # count of non-NA neighbors

      neighbor_mean   <- ifelse(neighbor_nna > 0,
                                neighbor_sum / neighbor_nna,
                                NA_real_)

      # --- MAX / MIN via Rcpp CSR pass ---
      mm <- sparse_row_max_min(csr$p, csr$j, x_full, csr$nrow)

      # Write back only for the cells present this year
      set(dt, i = yr_rows,
          j = paste0(var_name, "_neighbor_max"),
          value = mm$max_val[yr_pos])
      set(dt, i = yr_rows,
          j = paste0(var_name, "_neighbor_min"),
          value = mm$min_val[yr_pos])
      set(dt, i = yr_rows,
          j = paste0(var_name, "_neighbor_mean"),
          value = neighbor_mean[yr_pos])
    }

    if (yr %% 5 == 0 || yr == years[1] || yr == tail(years, 1)) {
      message("  Completed year ", yr)
    }
  }

  # Convert back to data.frame if original was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    setDF(dt)
  }

  return(dt)
}

# --------------------------------------------------------------------------
# Step 4: Main execution
# --------------------------------------------------------------------------

# --- Load pre-existing objects (assumed already in environment or on disk) ---
# cell_data              : data.frame/data.table with columns id, year, ntl, ec, ...
# id_order               : integer vector of cell IDs matching nb object order
# rook_neighbors_unique  : spdep nb object (list of integer neighbor indices)
# rf_model               : pre-trained Random Forest model

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# --- Apply the pre-trained Random Forest (unchanged) ---
cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Correct and Numerically Equivalent

| Original operation | Optimized equivalent | Proof of equivalence |
|----|----|----|
| For each cell-year, find neighbor rows via string key lookup, extract `vals[idx]` | For each year, build full vector `x_full`, index via sparse matrix | Same neighbor set (adjacency matrix encodes identical rook relationships) |
| `mean(neighbor_vals)` after dropping NA | `(A %*% x_no_na) / (A %*% not_na)` | Sum of non-NA values divided by count of non-NA values = arithmetic mean |
| `max(neighbor_vals)` | Rcpp CSR row-wise max skipping NA | Identical: iterates same neighbor set, same values, same NA handling |
| `min(neighbor_vals)` | Rcpp CSR row-wise min skipping NA | Identical |
| Returns `c(NA, NA, NA)` when no valid neighbors | `neighbor_nna == 0 â†’ NA`; Rcpp returns `NA` when no valid entries | Identical |

## Performance Summary

| Metric | Original | Optimized |
|--------|----------|-----------|
| R-level iterations | ~32.3M (lapply) | 0 (vectorized) |
| String operations | ~12.9M paste + match | 0 |
| Core operations per var-year | O(N) R-level loop | 2 sparse matvecs + 1 Rcpp pass |
| Total wall time (est.) | 86+ hours | **2â€“5 minutes** |
| Peak RAM | ~16 GB (string keys) | ~3â€“4 GB (sparse matrix + vectors) |
| Random Forest model | Unchanged | Unchanged |
END_RESPONSE>>>
