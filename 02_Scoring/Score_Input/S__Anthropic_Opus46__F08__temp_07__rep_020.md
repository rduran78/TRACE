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
run_id: Anthropic_F08_T07_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental architectural mismatch: **it rebuilds the neighbor lookup per cell-year row (~6.46M rows) when the neighbor topology is defined only over cells (~344K cells) and is invariant across years.**

Specifically:

1. **`build_neighbor_lookup` operates on the full cell-year panel.** It creates ~6.46M entries in `idx_lookup` and then iterates over all ~6.46M rows, performing string concatenation (`paste`) and hash-table lookups for each. This is O(rows Ã— avg_neighbors), producing ~6.46M list elements, when the neighbor graph itself has only ~344K nodes and ~1.37M directed edges that never change.

2. **String-keyed lookups are expensive.** The function uses `paste(id, year, sep="_")` as keys into a named vector. Named-vector lookup in R is O(n) per access in the worst case and involves repeated string hashing. With ~6.46M keys accessed ~6.46M Ã— avg_neighbors times, this is extremely slow.

3. **`compute_neighbor_stats` iterates over 6.46M list elements.** Each element triggers subsetting, NA removal, and three summary statistics. The `lapply` + `do.call(rbind, ...)` pattern over 6.46M rows is itself slow due to memory allocation churn.

4. **The loop runs 5 times** (once per source variable), multiplying everything by 5.

**Key insight:** The neighbor relationship is a **static spatial graph** â€” cell A's neighbors are always the same cells regardless of year. Only the *values* attached to those cells change by year. Therefore, the neighbor lookup should be built once over the ~344K cell topology, and the neighbor statistics should be computed per year using fast vectorized/matrix operations on that static topology.

## Optimization Strategy

1. **Build the neighbor lookup once, over cells only (~344K), not cell-years (~6.46M).** Use `rook_neighbors_unique` directly â€” it already encodes cell-to-cell adjacency. Map cell IDs to integer indices once.

2. **Reshape the computation: for each year, extract the variable vector over cells, then compute neighbor max/min/mean using the static cell-level adjacency.** This turns the inner loop into 28 iterations (one per year) over ~344K cells instead of one iteration over ~6.46M rows.

3. **Use a sparse adjacency matrix (from `Matrix` package) to vectorize neighbor mean.** For neighbor mean: `A %*% x / A %*% ones` where A is the binary adjacency matrix. For min and max, use a fast row-wise sparse approach or `data.table` grouped operations.

4. **Use `data.table` for the panel join** to write results back efficiently.

5. **All 5 variables Ã— 3 stats = 15 new columns, identical numerical results**, preserving the trained Random Forest model's expectations.

### Complexity Comparison

| | Current | Optimized |
|---|---|---|
| Lookup build | O(6.46M Ã— avg_deg) with string ops | O(1) â€” reuse `nb` object |
| Stats computation per variable | O(6.46M Ã— avg_deg) | O(28 Ã— nnz(A)) â‰ˆ O(28 Ã— 1.37M) |
| Total string operations | ~tens of billions | 0 |
| Estimated time | 86+ hours | **~2â€“5 minutes** |

## Working R Code

```r
library(data.table)
library(Matrix)

#' Redesigned pipeline: static topology, year-varying values
#' Preserves the original numerical estimand (neighbor max, min, mean)
#' and the pre-trained Random Forest model (no retraining).

optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars) {

  # -------------------------------------------------------------------
  # STEP 1: Build the static sparse adjacency matrix ONCE (~344K cells)
  # -------------------------------------------------------------------
  # id_order: vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique: spdep nb object (list of integer index vectors)

  n_cells <- length(id_order)
  stopifnot(length(rook_neighbors_unique) == n_cells)

  # Build sparse binary adjacency matrix (row i has 1s in columns that are
  # neighbors of cell i). This encodes the static rook-neighbor topology.
  # We iterate once over the nb object to extract (i, j) pairs.
  from_idx <- integer(0)
  to_idx   <- integer(0)
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nb_i <- nb_i[nb_i > 0L]
    if (length(nb_i) > 0L) {
      from_idx <- c(from_idx, rep.int(i, length(nb_i)))
      to_idx   <- c(to_idx, nb_i)
    }
  }

  # Sparse adjacency matrix: A[i,j] = 1 if j is a neighbor of i

  A <- sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = 1,
    dims = c(n_cells, n_cells),
    repr = "C"   # CSC format for efficient column ops; dgCMatrix
  )

  # Precompute neighbor counts per cell (static)
  ones_vec     <- rep(1, n_cells)
  neighbor_cnt <- as.numeric(A %*% ones_vec)  # length n_cells

  # -------------------------------------------------------------------
  # STEP 2: Convert to data.table for fast indexed operations
  # -------------------------------------------------------------------
  was_df <- is.data.frame(cell_data) && !is.data.table(cell_data)
  dt <- as.data.table(cell_data)

  # Create a mapping from cell id to the integer index in id_order

  id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))

  # Add the cell index column (static per cell, same across years)
  dt[, cell_idx := id_to_idx[as.character(id)]]

  # Verify all cells are matched
  if (anyNA(dt$cell_idx)) {
    warning("Some cell IDs in data do not appear in id_order. ",
            "Those rows will get NA neighbor stats.")
  }

  # Sort by year and cell_idx for efficient year-group processing
  setkey(dt, year, cell_idx)

  # Get sorted unique years
  years <- sort(unique(dt$year))

  # -------------------------------------------------------------------
  # STEP 3: For each variable, compute neighbor max/min/mean per year
  #         using the static adjacency matrix

  # -------------------------------------------------------------------
  # We need the nb list in a form suitable for fast min/max.
  # For mean: use sparse matrix multiplication.
  # For min/max: iterate over cells using the nb list (vectorized per year).

  # Precompute the neighbor list as a simple list of integer vectors
  # (strip spdep attributes, remove 0-entries)
  nb_list <- lapply(seq_len(n_cells), function(i) {
    nb_i <- rook_neighbors_unique[[i]]
    nb_i[nb_i > 0L]
  })

  for (var_name in neighbor_source_vars) {

    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    # Pre-allocate result columns with NA
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    for (yr in years) {

      # Extract the rows for this year, ordered by cell_idx
      yr_rows <- dt[.(yr)]  # keyed lookup on year

      # Build a full-length vector of values indexed by cell_idx
      # (some cells may be missing in a given year; those stay NA)
      vals_full <- rep(NA_real_, n_cells)
      vals_full[yr_rows$cell_idx] <- yr_rows[[var_name]]

      # --- Neighbor MEAN via sparse matrix multiplication ---
      # numerator = A %*% vals (sum of neighbor values; NA treated as 0
      # but we need to handle NAs properly)
      not_na     <- !is.na(vals_full)
      vals_zero  <- vals_full
      vals_zero[!not_na] <- 0

      neighbor_sum     <- as.numeric(A %*% vals_zero)
      neighbor_not_na  <- as.numeric(A %*% as.numeric(not_na))

      n_mean <- ifelse(neighbor_not_na > 0,
                       neighbor_sum / neighbor_not_na,
                       NA_real_)

      # --- Neighbor MAX and MIN via vectorized nb_list access ---
      # Use vapply over cells; this is ~344K iterations, very fast
      n_max <- rep(NA_real_, n_cells)
      n_min <- rep(NA_real_, n_cells)

      # Vectorized approach: for each cell, grab neighbor values
      # We use a compiled-style vapply
      max_min <- vapply(seq_len(n_cells), function(i) {
        nb_i <- nb_list[[i]]
        if (length(nb_i) == 0L) return(c(NA_real_, NA_real_))
        nv <- vals_full[nb_i]
        nv <- nv[!is.na(nv)]
        if (length(nv) == 0L) return(c(NA_real_, NA_real_))
        c(max(nv), min(nv))
      }, numeric(2))
      # max_min is a 2 x n_cells matrix; row 1 = max, row 2 = min

      n_max <- max_min[1L, ]
      n_min <- max_min[2L, ]

      # --- Write results back into dt for this year ---
      # Match rows: yr_rows$cell_idx gives the cell indices present
      idx_present <- yr_rows$cell_idx

      # Use data.table's set() for fast in-place assignment
      # We need the actual row numbers in dt for this year
      row_nums <- which(dt$year == yr)
      # Since dt is keyed by (year, cell_idx), these rows are in cell_idx order
      # and yr_rows$cell_idx gives the corresponding cell indices

      set(dt, i = row_nums, j = max_col,  value = n_max[idx_present])
      set(dt, i = row_nums, j = min_col,  value = n_min[idx_present])
      set(dt, i = row_nums, j = mean_col, value = n_mean[idx_present])
    }

    message("Done: ", var_name)
  }

  # Clean up helper column
  dt[, cell_idx := NULL]

  # Return in original format
  if (was_df) {
    return(as.data.frame(dt))
  } else {
    return(dt)
  }
}
```

### Outer Call (Drop-in Replacement)

```r
# ---- BEFORE (slow, ~86+ hours) ----
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ---- AFTER (fast, ~2-5 minutes) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# Proceed directly to prediction with the pre-trained Random Forest
# predictions <- predict(rf_model, newdata = cell_data)
```

### Optional Further Speedup: Rcpp for Min/Max

The `vapply` loop for min/max over ~344K cells is already fast (~seconds per year), but if desired, it can be replaced with Rcpp:

```r
# Optional: Rcpp version for neighbor min/max (even faster)
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix neighbor_min_max_cpp(NumericVector vals, List nb_list) {
  int n = nb_list.size();
  NumericMatrix out(n, 2); // col 0 = max, col 1 = min

  for (int i = 0; i < n; i++) {
    IntegerVector nb = nb_list[i];
    int m = nb.size();
    if (m == 0) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
      continue;
    }
    double vmax = R_NegInf;
    double vmin = R_PosInf;
    int valid = 0;
    for (int j = 0; j < m; j++) {
      double v = vals[nb[j] - 1]; // R is 1-indexed
      if (!NumericVector::is_na(v)) {
        if (v > vmax) vmax = v;
        if (v < vmin) vmin = v;
        valid++;
      }
    }
    if (valid == 0) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
    } else {
      out(i, 0) = vmax;
      out(i, 1) = vmin;
    }
  }
  return out;
}
')
```

## Summary

| Aspect | Detail |
|---|---|
| **Root cause** | Neighbor lookup built over 6.46M cell-year rows using string keys; topology is static over 344K cells |
| **Key insight** | Separate static topology (cells) from dynamic values (cell-years) |
| **Adjacency** | Built once as a sparse matrix (~1.37M non-zeros) |
| **Mean** | Sparse matrixâ€“vector multiplication per year: O(nnz) â‰ˆ O(1.37M) |
| **Min/Max** | `vapply` or Rcpp over ~344K cells per year using static `nb_list` |
| **Total iterations** | 5 vars Ã— 28 years Ã— 344K cells â‰ˆ 48.2M (vs. original ~6.46M Ã— 5 with string ops) |
| **Estimated speedup** | ~1,000â€“2,000Ã— (from ~86 hours to ~2â€“5 minutes) |
| **Numerical equivalence** | Identical â€” same max, min, mean over same neighbor values with same NA handling |
| **RF model** | Untouched â€” no retraining, same 15 feature columns produced |
END_RESPONSE>>>
