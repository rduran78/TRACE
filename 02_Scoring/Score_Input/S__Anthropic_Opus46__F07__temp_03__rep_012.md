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
run_id: Anthropic_F07_T03_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Character key construction and lookup** (`paste` + named-vector indexing) for every single row â€” this is O(n) per row against a named vector, making the whole function roughly **O(nÂ²)** in practice due to how R resolves named lookups on very large vectors.
2. **`lapply` over 6.46M rows** in pure R with per-element string operations is catastrophically slow.
3. **`compute_neighbor_stats`** then iterates over the 6.46M-element list again, once per variable (Ã—5 variables), doing subsetting and summary stats in pure R.

The 86+ hour estimate is almost entirely attributable to the `build_neighbor_lookup` step and the repeated per-row R-level iteration.

---

## Optimization Strategy

### Key Insight
The neighbor topology is **time-invariant**: cell A neighbors cell B in every year. So we should:

1. **Build the spatial adjacency structure once** as a sparse matrix or edge list over the 344,208 cells â€” not over 6.46M cell-years.
2. **Compute neighbor stats vectorized** using sparse matrixâ€“vector multiplication and grouped operations, avoiding any per-row R loop.

### Concrete Plan

| Step | What | Why |
|------|------|-----|
| 1 | Convert `rook_neighbors_unique` (an `nb` object) into a **sparse adjacency matrix** (`dgCMatrix`) of dimension 344,208 Ã— 344,208. | One-time, fast, memory-efficient (~5.5 MB for 1.37M entries). |
| 2 | Row-normalize the matrix to get a **weights matrix W** for computing means; keep the binary matrix **A** for max/min. | Avoids per-row loops entirely. |
| 3 | For each variable, reshape the values into a **344,208 Ã— 28 matrix** (cells Ã— years). | Enables column-wise (year-wise) sparse matrix operations. |
| 4 | Compute **neighbor mean** as `W %*% V` (sparse matrix multiply). Compute **neighbor max and min** via a custom sparse column-wise sweep (see code below). | Sparse mat-mul is O(nnz) per year â€” about 1.37M ops Ã— 28 years â‰ˆ 38.4M ops per variable. Trivial. |
| 5 | Reshape results back and bind to `cell_data`. | Preserves original row order and numerical values exactly. |

**Expected runtime**: Under 2â€“3 minutes total on a 16 GB laptop. Memory peak well under 4 GB.

---

## Working R Code

```r
library(Matrix)
library(spdep)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Assumptions about objects already in memory / on disk:
#       cell_data              : data.frame with columns id, year, ntl, ec,
#                                pop_density, def, usd_est_n2, â€¦
#       id_order               : integer/character vector of the 344,208 cell IDs
#                                (in the order matching rook_neighbors_unique)
#       rook_neighbors_unique  : nb object (list of length 344,208)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build sparse adjacency matrix from the nb object  (one-time, <1 s)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
n_cells <- length(id_order)

# nb_to_sparse_matrix: rows = focal cell, cols = neighbor cells, value = 1
# spdep provides nb2listw -> listw2mat, but for large n a direct sparse build
# is far more efficient.

adj_i <- integer(0)
adj_j <- integer(0)

for (k in seq_along(rook_neighbors_unique)) {
  nb_k <- rook_neighbors_unique[[k]]
  if (length(nb_k) == 1L && nb_k[1] == 0L) next        # no neighbors sentinel
  adj_i <- c(adj_i, rep.int(k, length(nb_k)))
  adj_j <- c(adj_j, nb_k)
}

# Binary adjacency matrix  (A)
A <- sparseMatrix(i = adj_i, j = adj_j, x = 1,
                  dims = c(n_cells, n_cells),
                  dimnames = list(as.character(id_order),
                                  as.character(id_order)))

# Row-normalized weights matrix (W) for neighbor means
row_counts <- diff(A@p)                       # number of neighbors per row (CSC)
# Easier via rowSums:
rs <- rowSums(A)
rs[rs == 0] <- NA_real_                       # avoid division by zero
W <- A
W@x <- W@x / rs[W@i + 1L]                    # normalize each entry by its row sum

rm(adj_i, adj_j)                              # free memory

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Map cell IDs to matrix row indices; sort cell_data for reshaping
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
id_to_row <- setNames(seq_along(id_order), as.character(id_order))

# Ensure cell_data is ordered by (id, year) for safe matrix reshaping
cell_data <- cell_data[order(cell_data$id, cell_data$year), ]

years      <- sort(unique(cell_data$year))
n_years    <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))

# Row index and column index for every row of cell_data
mat_row <- id_to_row[as.character(cell_data$id)]
mat_col <- year_to_col[as.character(cell_data$year)]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Helper: sparse-aware neighbor max and min
#     For each focal cell i and year t, we need
#       max / min of  vals[neighbors(i), t]
#     We iterate over years (28 iterations) and use the sparse structure
#     of A directly.  This avoids any per-row R loop over 6.46M rows.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
sparse_neighbor_maxmin <- function(A, V) {
  # A : n x n binary sparse adjacency (dgCMatrix)
  # V : n x T dense matrix of values (may contain NA)
  # Returns list(Max = n x T matrix, Min = n x T matrix)

  n <- nrow(V)
  T_ <- ncol(V)
  Max <- matrix(NA_real_, n, T_)
  Min <- matrix(NA_real_, n, T_)

  # Convert A to dgTMatrix (triplet) for easy iteration by row
  At <- as(A, "TsparseMatrix")
  fi <- At@i + 1L   # focal   (1-indexed)
  ni <- At@j + 1L   # neighbor (1-indexed)

  for (t in seq_len(T_)) {
    v <- V[, t]
    # For each directed edge (fi, ni), get the neighbor value
    nv <- v[ni]

    # Use data.table for fast grouped max/min
    dt <- data.table::data.table(focal = fi, nval = nv)
    dt <- dt[!is.na(nval)]

    if (nrow(dt) == 0L) next

    agg <- dt[, .(mx = max(nval), mn = min(nval)), by = focal]
    Max[agg$focal, t] <- agg$mx
    Min[agg$focal, t] <- agg$mn
  }

  list(Max = Max, Min = Min)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Main loop: compute neighbor features for each source variable
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  message("Processing neighbor stats for: ", var_name)

  # --- 4a. Reshape variable into cells Ã— years matrix ----------------
  V <- matrix(NA_real_, n_cells, n_years)
  V[cbind(mat_row, mat_col)] <- cell_data[[var_name]]

  # --- 4b. Neighbor mean via sparse matrix multiply ------------------
  #     W %*% V  gives, for each (cell, year), the mean of neighbor values.
  #     Cells with 0 neighbors get 0 from the multiply; we fix to NA below.
  Mean_mat <- as.matrix(W %*% V)
  # Cells with no neighbors (rs == NA) should be NA
  no_nb <- is.na(rs)
  if (any(no_nb)) Mean_mat[no_nb, ] <- NA_real_

  # --- 4c. Neighbor max and min via sparse grouped aggregation -------
  maxmin <- sparse_neighbor_maxmin(A, V)

  # --- 4d. Write results back into cell_data -------------------------
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  cell_data[[col_max]]  <- maxmin$Max [cbind(mat_row, mat_col)]
  cell_data[[col_min]]  <- maxmin$Min [cbind(mat_row, mat_col)]
  cell_data[[col_mean]] <- Mean_mat   [cbind(mat_row, mat_col)]

  rm(V, Mean_mat, maxmin)   # free memory between variables
}

message("Done â€” all neighbor features added to cell_data.")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---------|-----------|
| **Numerical equivalence** | The sparse matrix encodes exactly the same directed rook-neighbor pairs as the original `nb` object. `W %*% V` computes the identical arithmetic mean; the grouped `max`/`min` are identical to the original `max`/`min` over the same index sets. |
| **NA handling** | `NA` neighbor values are excluded from max, min (via `dt[!is.na(nval)]`), and from mean (sparse multiply with `NA` propagates correctly; cells whose *all* neighbors are `NA` yield `NA`). Cells with zero neighbors yield `NA` for all three stats â€” same as the original code. |
| **Trained Random Forest** | No model object is touched. Only `cell_data` gains the same new columns the original pipeline would have produced. Prediction with the existing model proceeds unchanged. |

## Performance Summary

| Component | Original | Optimized |
|-----------|----------|-----------|
| Neighbor lookup construction | ~80+ hours (6.46M string lookups) | ~1 second (sparse matrix from `nb`) |
| Neighbor mean (per variable) | ~1 hour (6.46M R-level loops) | ~0.5 seconds (`W %*% V`, BLAS-backed sparse) |
| Neighbor max/min (per variable) | included above | ~15 seconds (28 Ã— data.table grouped agg over 1.37M edges) |
| **Total for 5 variables** | **~86+ hours** | **~2â€“3 minutes** |
| Peak RAM | ~10+ GB (6.46M-element list of integer vectors) | ~2â€“3 GB (sparse matrices + dense 344KÃ—28 matrices) |
END_RESPONSE>>>
