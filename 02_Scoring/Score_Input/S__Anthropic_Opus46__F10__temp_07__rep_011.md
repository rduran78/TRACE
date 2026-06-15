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
run_id: Anthropic_F10_T07_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries via `lapply` with string-based key lookups.** For each of the 6.46M rows, it constructs paste-based keys, looks them up in a named vector (`idx_lookup`), and filters NAs. Named vector lookup in R is O(n) hashing per call, and doing this 6.46M times with string construction is extremely slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in R-level `lapply`.** For each row, it subsets a numeric vector by indices, removes NAs, and computes max/min/mean. This is pure R-loop overhead repeated ~6.46M Ã— 5 variables = ~32.3M iterations.

3. **The neighbor topology is year-invariant but the lookup is rebuilt entangling topology with year indexing.** Rook neighbors are a spatial property â€” the same 1,373,394 directed edges apply identically to every year. The current code reconstructs row-level lookups that redundantly replicate this topology across all 28 years.

**Core insight:** The adjacency structure has only ~344K nodes and ~1.37M edges. The panel has 28 years. The current code inflates this to a 6.46M-node problem. Instead, we should:
- Operate on the **344K-node spatial graph once per year-variable combination**.
- Use **sparse matrix multiplication** to compute neighborhood aggregates in vectorized C-level operations.

---

## Optimization Strategy

### Step 1: Build a sparse adjacency matrix once (344,208 Ã— 344,208)

Convert the `nb` object into a sparse `dgCMatrix` (CSC format). Each row `i` has 1s in columns `j` where `j` is a rook neighbor of `i`. This matrix `A` has exactly ~1,373,394 nonzero entries.

### Step 2: For each year and each variable, extract the 344K-length vector and use sparse matrix ops

- **Mean:** `A %*% x / A %*% 1_nonNA` (handling NAs properly via indicator masking)
- **Max and Min:** Use a single pass over the CSC sparse structure in C++ via `Rcpp`, or use `{Matrix}` row-wise operations on the neighbor-value matrix.

For max/min, sparse matrix multiplication doesn't directly apply, but we can use an efficient Rcpp function that iterates over the sparse column pointers â€” this is O(nnz) â‰ˆ 1.37M operations per variable-year, totaling 5 Ã— 28 Ã— 1.37M â‰ˆ 192M simple numeric comparisons, which completes in seconds.

### Step 3: Reassemble into the panel data.frame

Map the per-year 344K-length result vectors back into the 6.46M-row panel by year.

### Complexity comparison

| | Original | Optimized |
|---|---|---|
| Lookup construction | O(6.46M) string ops | O(1.37M) sparse matrix build (once) |
| Stats computation | O(6.46M Ã— 5) R-level loops | O(1.37M Ã— 5 Ã— 28) vectorized C-level |
| Estimated time | 86+ hours | **~2â€“5 minutes** |

---

## Optimized R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE PIPELINE
# Sparse graph neighborhood aggregation via Matrix + Rcpp
# Numerically equivalent to original: max, min, mean of rook-neighbor attributes
# ==============================================================================

library(Matrix)
library(data.table)
library(Rcpp)

# --------------------------------------------------------------------------
# STEP 0: Rcpp function for sparse row-wise max, min, mean with NA handling
# --------------------------------------------------------------------------

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix sparse_neighbor_stats(IntegerVector Ap,       // CSR row pointers (length nrow+1)
                                    IntegerVector Aj,       // CSR column indices
                                    NumericVector x,        // attribute vector (length ncol)
                                    int nrow) {
  // Returns nrow x 3 matrix: [max, min, mean]
  NumericMatrix out(nrow, 3);

  for (int i = 0; i < nrow; i++) {
    int start = Ap[i];
    int end   = Ap[i + 1];

    double vmax   = R_NegInf;
    double vmin   = R_PosInf;
    double vsum   = 0.0;
    int    vcount = 0;

    for (int jj = start; jj < end; jj++) {
      double val = x[ Aj[jj] ];
      if (!R_IsNA(val)) {
        if (val > vmax) vmax = val;
        if (val < vmin) vmin = val;
        vsum += val;
        vcount++;
      }
    }

    if (vcount == 0) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
      out(i, 2) = NA_REAL;
    } else {
      out(i, 0) = vmax;
      out(i, 1) = vmin;
      out(i, 2) = vsum / vcount;
    }
  }

  return out;
}
')

# --------------------------------------------------------------------------
# STEP 1: Build sparse adjacency matrix from nb object (ONCE)
# --------------------------------------------------------------------------

build_sparse_adjacency <- function(nb_obj, n) {
  # nb_obj: spdep nb object (list of integer neighbor vectors, 0 = no neighbors)
  # n: number of spatial cells (344208)
  # Returns: dgRMatrix (CSR) for fast row-wise access

  from <- integer(0)
  to   <- integer(0)

  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1L] == 0L) next
    from <- c(from, rep(i, length(nbrs)))
    to   <- c(to, nbrs)
  }

  # Build in CSC (dgCMatrix), then transpose to get CSR-like access
  # Actually we build row->col as (from, to), so this is already correct
  A <- sparseMatrix(i = from, j = to, x = rep(1, length(from)),
                    dims = c(n, n), giveCsparse = TRUE)
  return(A)
}

# --------------------------------------------------------------------------
# STEP 2: Convert dgCMatrix to CSR arrays for Rcpp
# --------------------------------------------------------------------------

to_csr_arrays <- function(A_csc) {
  # Convert dgCMatrix (CSC) to CSR representation
  A_csr <- as(A_csc, "RsparseMatrix")  # dgRMatrix
  list(
    Ap = A_csr@p,    # row pointers (0-based, length nrow+1)
    Aj = A_csr@j,    # column indices (0-based)
    nrow = nrow(A_csr)
  )
}

# --------------------------------------------------------------------------
# STEP 3: Main pipeline
# --------------------------------------------------------------------------

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  cat("Converting to data.table...\n")
  dt <- as.data.table(cell_data)
  n_cells <- length(id_order)

  # --- Build spatial index: map cell id -> position 1..n_cells ---
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # --- Build sparse adjacency (once) ---
  cat("Building sparse adjacency matrix...\n")
  A <- build_sparse_adjacency(rook_neighbors_unique, n_cells)
  csr <- to_csr_arrays(A)
  cat(sprintf("  Adjacency: %d nodes, %d directed edges\n", n_cells, length(csr$Aj)))

  # --- Ensure cell_data has a position column ---
  dt[, cell_pos := id_to_pos[as.character(id)]]

  # --- Get unique years sorted ---
  years <- sort(unique(dt$year))
  cat(sprintf("  Years: %d (%d to %d)\n", length(years), min(years), max(years)))

  # --- Neighbor source variables ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  # --- Pre-allocate result columns ---
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]
  }

  # --- Process year by year ---
  cat("Computing neighbor statistics...\n")
  setkey(dt, year)

  for (yr in years) {
    yr_idx <- dt[.(yr), which = TRUE]
    dt_yr  <- dt[yr_idx]

    # Build full-length attribute vectors (n_cells) for this year
    # Some cells may be missing for a year; they stay NA
    positions <- dt_yr$cell_pos

    for (var_name in neighbor_source_vars) {
      # Initialize full vector as NA
      x_full <- rep(NA_real_, n_cells)
      x_full[positions] <- dt_yr[[var_name]]

      # Compute sparse neighbor stats via Rcpp: O(nnz) ~ 1.37M ops
      stats_full <- sparse_neighbor_stats(csr$Ap, csr$Aj, x_full, csr$nrow)

      # Extract results for cells present this year
      col_max  <- paste0("neighbor_max_", var_name)
      col_min  <- paste0("neighbor_min_", var_name)
      col_mean <- paste0("neighbor_mean_", var_name)

      set(dt, i = yr_idx, j = col_max,  value = stats_full[positions, 1])
      set(dt, i = yr_idx, j = col_min,  value = stats_full[positions, 2])
      set(dt, i = yr_idx, j = col_mean, value = stats_full[positions, 3])
    }

    if (yr %% 5 == 0 || yr == years[1] || yr == years[length(years)]) {
      cat(sprintf("  Year %d done.\n", yr))
    }
  }

  cat("Neighbor feature computation complete.\n")

  # --- Remove helper column ---
  dt[, cell_pos := NULL]

  return(dt)
}

# --------------------------------------------------------------------------
# STEP 4: Predict with pre-trained Random Forest (NO retraining)
# --------------------------------------------------------------------------

run_prediction <- function(dt, rf_model, predictor_vars) {
  cat("Running Random Forest prediction...\n")
  pred_data <- as.data.frame(dt[, ..predictor_vars])
  dt[, prediction := predict(rf_model, newdata = pred_data)]
  cat("Prediction complete.\n")
  return(dt)
}

# ==========================================================================
# USAGE EXAMPLE
# ==========================================================================

# # Load data
# cell_data              <- readRDS("cell_data.rds")
# id_order               <- readRDS("id_order.rds")
# rook_neighbors_unique  <- readRDS("rook_neighbors_unique.rds")
# rf_model               <- readRDS("rf_model.rds")
#
# # Run optimized pipeline
# dt <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# # Predict (model is NOT retrained)
# dt <- run_prediction(dt, rf_model, predictor_vars)
#
# # Convert back to data.frame if needed
# cell_data <- as.data.frame(dt)
```

---

## Summary of Guarantees

| Property | Status |
|---|---|
| **Numerical equivalence** | âœ… Identical max, min, mean over non-NA rook-neighbor values per cell-year |
| **Pre-trained RF preserved** | âœ… `predict()` only; no `train`/`randomForest` call |
| **Column names preserved** | âœ… Same `neighbor_max_*`, `neighbor_min_*`, `neighbor_mean_*` naming convention |
| **Memory footprint** | âœ… Sparse matrix: ~1.37M entries (~22 MB); data.table in-place `set()` avoids copies; peak ~4â€“6 GB well within 16 GB |
| **Runtime estimate** | âœ… 28 years Ã— 5 vars Ã— ~1.37M sparse ops = ~192M C-level comparisons â†’ **~2â€“5 minutes total** (vs. 86+ hours) |
| **Speedup factor** | **~1,000â€“2,500Ã—** |

The key architectural change is decomposing the **panel-level** O(6.46M) R-loop into a **spatial-level** O(344K nodes, 1.37M edges) Rcpp sparse-CSR traversal repeated over 28 years, with `data.table::set()` for zero-copy column writes.
END_RESPONSE>>>
