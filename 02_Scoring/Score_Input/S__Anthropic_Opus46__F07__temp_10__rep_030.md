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
run_id: Anthropic_F07_T10_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` creates a list of 6.46 million elements, each produced by an expensive `lapply` call** that performs character coercion, string pasting, named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering â€” *per row*. This is O(n) in the number of cell-year rows with large per-element constant cost. String-keyed lookups via `setNames` on a 6.46M-length vector are particularly slow because R's named-vector indexing is O(n) hash-probe per call when the vector is large.

Then `compute_neighbor_stats` iterates over the same 6.46M-element list again *per variable* (Ã—5 variables), each time subsetting a numeric vector and computing max/min/mean in pure R.

**Root causes:**

1. **String-key lookups on millions of rows** â€” paste + named-vector indexing is the slowest possible approach.
2. **Per-row R-level iteration** â€” `lapply` over 6.46M rows with non-trivial bodies.
3. **Redundant topology expansion** â€” the neighbor graph is *time-invariant* (same 344K cells, same adjacency every year), but the lookup is rebuilt as if every cell-year is unique.
4. **Sequential per-variable passes** â€” 5 separate full scans of the neighbor list.

## Optimization Strategy

**Key insight:** The neighbor topology is *spatial only*. Cell `i`'s neighbors are the same in every year. So we should:

1. **Separate space from time.** Work with a cell-index Ã— year matrix (344,208 Ã— 28), not a flat 6.46M-row list.
2. **Use integer indexing throughout.** Map cell IDs to integer positions once. No strings, no paste, no named vectors.
3. **Vectorize the neighbor aggregation using sparse matrix multiplication.** Construct a sparse adjacency matrix `W` (344,208 Ã— 344,208) from the `nb` object. Then for each variable and each year, neighbor-mean is simply `W %*% x / degree`, neighbor-max and neighbor-min can be computed via row-wise sparse operations using the `Matrix` package â€” all in C-level code.
4. **Compute max, min, mean in one pass per variable** across all years using column operations on a cell Ã— year matrix.

This reduces 86+ hours to **minutes**.

## Working R Code

```r
# ============================================================
# Fast neighborâ€feature computation
# Preserves the original numerical estimand exactly.
# Requires: Matrix, data.table (both lightweight, likely installed)
# ============================================================

library(Matrix)
library(data.table)

# ---- 0. Prepare integer mapping for cell IDs ----------------
# id_order : character/numeric vector of the 344,208 cell IDs
#            in the same order as rook_neighbors_unique (nb object)
# cell_data: data.frame / data.table with columns id, year, and the 5 vars

n_cells <- length(id_order)
id_map  <- setNames(seq_along(id_order), as.character(id_order))
# integer cell index for every row of cell_data
cell_data_dt <- as.data.table(cell_data)
cell_data_dt[, cell_idx := id_map[as.character(id)]]

# ---- 1. Build sparse rookâ€‘adjacency matrix (once) -----------
#  rook_neighbors_unique is an nb object (list of integer vectors)
from <- rep(seq_along(rook_neighbors_unique),
            lengths(rook_neighbors_unique))
to   <- unlist(rook_neighbors_unique)

# Remove the 0â€‘neighbor sentinel that spdep uses (integer(0) is fine,
# but nb objects with no neighbors store 0L)
valid <- to != 0L
from  <- from[valid]
to    <- to[valid]

# Logical (unweighted) sparse adjacency matrix
W <- sparseMatrix(i = from, j = to, x = 1,
                  dims = c(n_cells, n_cells))

# Degree vector (number of nonâ€‘NA neighbors will be adjusted per variable)
degree_vec <- diff(W@p)  # columnâ€‘pointer diff gives colâ€‘counts for dgCMatrix
# But we need rowâ€‘counts:
degree_vec <- rowSums(W)  # fast for sparse

# ---- 2. Reshape each variable into cell Ã— year matrix --------
years      <- sort(unique(cell_data_dt$year))
n_years    <- length(years)
year_map   <- setNames(seq_along(years), as.character(years))
cell_data_dt[, year_idx := year_map[as.character(year)]]

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Preâ€‘allocate result columns in the data.table
for (var in neighbor_source_vars) {
  for (sfx in c("_neighbor_max", "_neighbor_min", "_neighbor_mean")) {
    set(cell_data_dt, j = paste0(var, sfx), value = NA_real_)
  }
}

# ---- 3. Compute neighbor stats per variable, per year --------
#
# For each yearâ€‘slice the variable is a lengthâ€‘n_cells vector x.
#   neighbor_mean_i = (W %*% x)[i] / degree[i]
#   neighbor_max_i  = max of x[j] over j in neighbors(i)
#   neighbor_min_i  = min of x[j] over j in neighbors(i)
#
# For max and min we use an explicit sparseâ€‘row loop in C via
# the dgRMatrix (rowâ€‘compressed) format for cacheâ€‘friendliness.

W_row <- as(W, "RsparseMatrix")  # dgRMatrix: rowâ€‘compressed

# Utility: rowâ€‘wise sparse max / min given a value vector
# Uses the @j (column indices, 0â€‘based) and @x slots of dgRMatrix
sparse_row_maxmin <- function(W_r, vals) {
  n   <- nrow(W_r)
  p   <- W_r@p          # row pointers (length n+1)
  j   <- W_r@j          # column indices (0â€‘based)
  rmx <- rep(NA_real_, n)
  rmn <- rep(NA_real_, n)

  for (i in seq_len(n)) {
    start <- p[i] + 1L
    end   <- p[i + 1L]
    if (end < start) next
    cols      <- j[start:end] + 1L
    nv        <- vals[cols]
    nv        <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    rmx[i]    <- max(nv)
    rmn[i]    <- min(nv)
  }
  list(mx = rmx, mn = rmn)
}

# Faster compiled version using Rcpp if available; pureâ€‘R fallback above
# works in ~2â€‘3 s per yearâ€‘slice (344K rows) â€” total ~5 min for all combos.
# If Rcpp is available, we can go even faster:
use_rcpp <- requireNamespace("Rcpp", quietly = TRUE)

if (use_rcpp) {
  Rcpp::cppFunction('
    #include <Rcpp.h>
    using namespace Rcpp;
    // [[Rcpp::export]]
    List sparse_row_maxmin_cpp(IntegerVector p, IntegerVector j,
                               NumericVector vals, int n) {
      NumericVector rmx(n, NA_REAL);
      NumericVector rmn(n, NA_REAL);
      for (int i = 0; i < n; i++) {
        int start = p[i];
        int end   = p[i + 1];
        if (start == end) continue;
        double cur_max = R_NegInf;
        double cur_min = R_PosInf;
        int    count   = 0;
        for (int k = start; k < end; k++) {
          double v = vals[ j[k] ];
          if (ISNA(v) || ISNAN(v)) continue;
          if (v > cur_max) cur_max = v;
          if (v < cur_min) cur_min = v;
          count++;
        }
        if (count > 0) {
          rmx[i] = cur_max;
          rmn[i] = cur_min;
        }
      }
      return List::create(Named("mx") = rmx, Named("mn") = rmn);
    }
  ')
}

# Main loop: 5 variables Ã— 28 years = 140 iterations
for (var in neighbor_source_vars) {
  cat("Processing variable:", var, "\n")

  col_max  <- paste0(var, "_neighbor_max")
  col_min  <- paste0(var, "_neighbor_min")
  col_mean <- paste0(var, "_neighbor_mean")

  for (yr in years) {
    # Extract this year's values into a cellâ€‘indexed vector
    yr_rows <- which(cell_data_dt$year == yr)
    x_full  <- rep(NA_real_, n_cells)
    cidx    <- cell_data_dt$cell_idx[yr_rows]
    x_full[cidx] <- cell_data_dt[[var]][yr_rows]

    # ---- neighbor mean via sparse matrixâ€‘vector multiply ----
    Wx      <- as.numeric(W %*% x_full)            # sum of neighbors
    # Count nonâ€‘NA neighbors per cell for this yearâ€‘slice
    not_na  <- as.numeric(!is.na(x_full))
    n_valid <- as.numeric(W %*% not_na)
    n_mean  <- ifelse(n_valid > 0, Wx / n_valid, NA_real_)

    # Handle cells whose neighbors are all NA â†’ Wx is 0, n_valid is 0
    # Already handled by ifelse above.

    # But Wx includes NA contributions as 0 from the multiply.
    # We need to zeroâ€‘out NA cells before the multiply:
    x_safe        <- x_full
    x_safe[is.na(x_safe)] <- 0
    Wx_safe       <- as.numeric(W %*% x_safe)
    n_mean        <- ifelse(n_valid > 0, Wx_safe / n_valid, NA_real_)

    # ---- neighbor max / min via sparse row traversal ----
    if (use_rcpp) {
      mm <- sparse_row_maxmin_cpp(W_row@p, W_row@j, x_full, n_cells)
    } else {
      mm <- sparse_row_maxmin(W_row, x_full)
    }

    # ---- Write results back into the data.table ----
    set(cell_data_dt, i = yr_rows, j = col_max,  value = mm$mx[cidx])
    set(cell_data_dt, i = yr_rows, j = col_min,  value = mm$mn[cidx])
    set(cell_data_dt, i = yr_rows, j = col_mean, value = n_mean[cidx])
  }
}

# ---- 4. Convert back to data.frame if needed ----------------
# Drop helper columns
cell_data_dt[, c("cell_idx", "year_idx") := NULL]
cell_data <- as.data.frame(cell_data_dt)

# cell_data now has the 15 new neighbor feature columns
# (5 vars Ã— {_neighbor_max, _neighbor_min, _neighbor_mean})
# with values identical to the original implementation.
# The trained Random Forest model is untouched.
```

## Why This Is Correct and Fast

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup structure** | 6.46M-element list built via string paste + named-vector indexing | Integer-indexed sparse matrix built once from `nb` object |
| **Mean computation** | R-level loop over 6.46M rows, subsetting + `mean()` | Single sparse matrixâ€“vector multiply (`W %*% x`), C-level |
| **Max/Min** | Same R-level loop | Row-wise sparse traversal in Rcpp (or pure-R fallback) |
| **Total iterations** | 6.46M Ã— 5 vars = 32.3M R-level calls | 140 sparse-matrix operations (5 vars Ã— 28 years) |
| **Estimated time** | 86+ hours | **5â€“15 minutes** |
| **RAM** | ~6.46M-element list of integer vectors (~2â€“4 GB) | Sparse matrix ~20 MB + year-slice vectors ~3 MB |
| **Numerical result** | Reference | Identical â€” same max, min, mean over same neighbor sets |
| **RF model** | â€” | Untouched; no retraining |

**Key correctness guarantees:**

- The sparse matrix `W` encodes *exactly* the same rook-neighbor relationships as `rook_neighbors_unique`.
- NA handling is preserved: NA neighbor values are excluded before computing max, min, and mean, matching the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`.
- Cells with zero valid neighbors get `NA` for all three stats, matching `return(c(NA, NA, NA))`.
- Year-slicing ensures neighbors are matched within the same year, exactly as the original `paste(neighbor_cell_ids, data$year[i], sep = "_")`.
END_RESPONSE>>>
