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
run_id: Anthropic_F10_T03_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query in the worst case, and doing this 6.46M times with string concatenation is extremely slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a pure-R `lapply`**, extracting variable values and computing `max/min/mean` one node-year at a time. This is repeated 5 times (once per source variable), totaling ~32.3 million R-level function calls.

3. **The neighbor lookup is year-coupled but the graph topology is year-invariant.** Rook contiguity is a spatial relationship between the 344,208 cells â€” it doesn't change across years. Yet the current code rebuilds row-level neighbor indices that redundantly encode the same spatial adjacency for every year, inflating the lookup from ~1.37M edges to ~38.4M edge-year pairs.

**Summary:** The 86+ hour runtime comes from (a) R-level loops over millions of rows, (b) string-based key lookups instead of integer indexing, and (c) failure to exploit the year-invariance of the spatial graph.

---

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from the `nb` object (344,208 Ã— 344,208 CSC matrix via `spdep::nb2listw` â†’ `Matrix::sparseMatrix`). This encodes the full rook topology in ~1.37M nonzero entries.

2. **Reshape each variable into a dense 344,208 Ã— 28 matrix** (cells Ã— years). This allows vectorized column-wise (i.e., year-wise) sparse matrixâ€“vector operations.

3. **Compute neighbor aggregates via sparse matrix algebra:**
   - **Mean:** `A %*% X / A %*% 1` (where `A` is the binary adjacency matrix, `X` is the value matrix, and `1` is an indicator of non-NA).
   - **Max / Min:** Use a modified sparse matrix where NA cells are masked, then iterate over the CSC structure in C++ (via `Rcpp`) or use a grouped operation on the COO triplet form with `data.table`.

4. **Columnar `data.table` join** to write results back into the panel, keyed by `(id, year)`.

5. **Preserve numerical equivalence:** The sparse-matrix mean is algebraically identical to the R-level `mean(neighbor_vals[!is.na(neighbor_vals)])`. Max and min are computed from the exact same neighbor sets.

This reduces the problem from ~32M R-level iterations to ~5 variables Ã— 3 stats Ã— 28 matrix operations on a 344KÃ—344K sparse matrix â€” completing in minutes, not days.

---

## Optimized R Code

```r
# ==============================================================================
# Optimized spatial neighbor aggregation pipeline
# Preserves numerical equivalence with the original implementation.
# Requires: Matrix, data.table, spdep (for nb2mat or manual construction)
# ==============================================================================

library(Matrix)
library(data.table)

# --------------------------------------------------------------------------
# STEP 0: Ensure cell_data is a data.table keyed by (id, year)
# --------------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}
setkey(cell_data, id, year)

# --------------------------------------------------------------------------
# STEP 1: Build the sparse binary adjacency matrix ONCE (344208 x 344208)
#
#   rook_neighbors_unique : an nb object (list of integer vectors)
#   id_order              : vector of cell IDs in the order matching the nb object
# --------------------------------------------------------------------------
build_adjacency_matrix <- function(nb_obj, n = length(nb_obj)) {
  # Convert nb list to COO triplets
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)

  # Remove the 0-neighbor sentinel that spdep uses (integer(0) is fine, but
  # nb objects sometimes store 0L for islands)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]

  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n), repr = "C")
}

cat("Building sparse adjacency matrix...\n")
A <- build_adjacency_matrix(rook_neighbors_unique)
n_cells <- length(id_order)
stopifnot(nrow(A) == n_cells, ncol(A) == n_cells)

# --------------------------------------------------------------------------
# STEP 2: Create integer mappings
#   cell_idx : named integer vector mapping cell ID -> row/col index in A
#   years    : sorted unique years
# --------------------------------------------------------------------------
cell_idx <- setNames(seq_len(n_cells), as.character(id_order))
years    <- sort(unique(cell_data$year))
n_years  <- length(years)
year_idx <- setNames(seq_len(n_years), as.character(years))

# --------------------------------------------------------------------------
# STEP 3: Reshape a variable from long panel to (n_cells x n_years) matrix
#          Rows correspond to id_order; columns to sorted years.
# --------------------------------------------------------------------------
panel_to_matrix <- function(dt, var_name, cell_idx, years, n_cells) {
  # Extract only needed columns for speed
  sub <- dt[, .(id, year, val = get(var_name))]

  # Map to matrix indices
  ri <- cell_idx[as.character(sub$id)]
  ci <- year_idx[as.character(sub$year)]

  mat <- matrix(NA_real_, nrow = n_cells, ncol = length(years))
  mat[cbind(ri, ci)] <- sub$val
  mat
}

# --------------------------------------------------------------------------
# STEP 4: Compute neighbor max, min, mean for one variable across all years
#
#   For MEAN:  For each cell i and year t,
#       mean_i = sum_j A[i,j]*X[j,t] / sum_j A[i,j]*(!is.na(X[j,t]))
#     This is: (A %*% X_filled) / (A %*% notNA)   where X_filled has 0 for NA.
#
#   For MAX / MIN: We iterate over columns (years) and use the sparse structure
#     of A to gather neighbor values per cell, then compute max/min.
#     With only 28 years and ~1.37M edges, this is fast even in R.
# --------------------------------------------------------------------------
compute_neighbor_features_sparse <- function(A, X_mat) {
  n   <- nrow(X_mat)
  k   <- ncol(X_mat)

  # --- MEAN via sparse matrix multiplication ---
  notNA    <- !is.na(X_mat)                    # logical matrix n x k
  X_zero   <- X_mat
  X_zero[is.na(X_zero)] <- 0                   # replace NA with 0 for summation

  # Sparse %*% dense is efficient in Matrix package
  sum_vals   <- as.matrix(A %*% X_zero)        # n x k: sum of neighbor values
  count_vals <- as.matrix(A %*% (notNA * 1.0)) # n x k: count of non-NA neighbors

  mean_mat <- sum_vals / count_vals             # NA where count == 0 (0/0 = NaN -> NA)
  mean_mat[count_vals == 0] <- NA_real_

  # --- MAX and MIN via sparse column traversal ---
  # Extract CSC structure of A once
  # In dgCMatrix: A@p (column pointers), A@i (row indices, 0-based)
  # For row i, we need the column indices j where A[i,j] = 1.
  # It's easier to work with the transpose: At = t(A), then At's columns
  # give us the neighbors of each row in A.
  At <- t(A)  # now At@i[At@p[i]+1 : At@p[i+1]] gives 0-based col indices = neighbors of cell i

  p_ptr <- At@p          # length n+1
  j_idx <- At@i          # 0-based neighbor indices

  max_mat <- matrix(NA_real_, nrow = n, ncol = k)
  min_mat <- matrix(NA_real_, nrow = n, ncol = k)

  # Process year by year (28 iterations â€” very manageable)
  for (t in seq_len(k)) {
    col_vals <- X_mat[, t]  # length n

    for (i in seq_len(n)) {
      start <- p_ptr[i] + 1L   # R is 1-based; p is 0-based
      end   <- p_ptr[i + 1L]
      if (end < start) next     # no neighbors (island)

      nb_indices <- j_idx[start:end] + 1L  # convert to 1-based
      nb_vals    <- col_vals[nb_indices]
      nb_vals    <- nb_vals[!is.na(nb_vals)]

      if (length(nb_vals) > 0L) {
        max_mat[i, t] <- max(nb_vals)
        min_mat[i, t] <- min(nb_vals)
      }
    }
  }

  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# --------------------------------------------------------------------------
# STEP 4b: Even faster MAX/MIN using data.table grouped operations on COO
#           This avoids the nested R loop entirely.
# --------------------------------------------------------------------------
compute_neighbor_features_sparse_fast <- function(A, X_mat, cell_idx, years) {
  n <- nrow(X_mat)
  k <- ncol(X_mat)

  # --- MEAN (same as above, via sparse matmul) ---
  notNA    <- !is.na(X_mat)
  X_zero   <- X_mat
  X_zero[is.na(X_zero)] <- 0

  sum_vals   <- as.matrix(A %*% X_zero)

  count_vals <- as.matrix(A %*% (notNA * 1.0))

  mean_mat <- sum_vals / count_vals
  mean_mat[count_vals == 0] <- NA_real_

  # --- MAX / MIN via COO expansion + data.table ---
  # Convert A to triplet form
  A_T <- as(A, "TsparseMatrix")  # dgTMatrix: @i, @j are 0-based
  from_cell <- A_T@i + 1L        # source cell (1-based)
  to_cell   <- A_T@j + 1L        # neighbor cell (1-based)
  n_edges   <- length(from_cell)

  # For each year, look up neighbor values, then group by source cell
  max_mat <- matrix(NA_real_, nrow = n, ncol = k)
  min_mat <- matrix(NA_real_, nrow = n, ncol = k)

  # Build edge table once
  edge_dt <- data.table(from = from_cell, to = to_cell)

  for (t in seq_len(k)) {
    col_vals <- X_mat[, t]
    edge_dt[, val := col_vals[to]]

    # Group by source, compute max/min (NA removed by na.rm)
    agg <- edge_dt[!is.na(val), .(nb_max = max(val), nb_min = min(val)), by = from]

    max_mat[agg$from, t] <- agg$nb_max
    min_mat[agg$from, t] <- agg$nb_min
  }

  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# --------------------------------------------------------------------------
# STEP 5: Write matrix results back to the data.table panel
# --------------------------------------------------------------------------
matrix_to_panel <- function(dt, mat, col_name, cell_idx, year_idx) {
  ri <- cell_idx[as.character(dt$id)]
  ci <- year_idx[as.character(dt$year)]
  dt[, (col_name) := mat[cbind(ri, ci)]]
  invisible(dt)
}

# --------------------------------------------------------------------------
# STEP 6: Main loop â€” process each source variable
# --------------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Processing neighbor features for", length(neighbor_source_vars), "variables...\n")

for (var_name in neighbor_source_vars) {
  cat("  Variable:", var_name, "... ")
  t0 <- proc.time()

  # Reshape to matrix
  X_mat <- panel_to_matrix(cell_data, var_name, cell_idx, years, n_cells)

  # Compute neighbor stats
  stats <- compute_neighbor_features_sparse_fast(A, X_mat, cell_idx, years)

  # Write back to panel (use same column naming convention as original)
  max_col  <- paste0(var_name, "_max_neighbor")
  min_col  <- paste0(var_name, "_min_neighbor")
  mean_col <- paste0(var_name, "_mean_neighbor")

  matrix_to_panel(cell_data, stats$max,  max_col,  cell_idx, year_idx)
  matrix_to_panel(cell_data, stats$min,  min_col,  cell_idx, year_idx)
  matrix_to_panel(cell_data, stats$mean, mean_col, cell_idx, year_idx)

  elapsed <- (proc.time() - t0)[3]
  cat(round(elapsed, 1), "seconds\n")
}

cat("Neighbor feature computation complete.\n")

# --------------------------------------------------------------------------
# STEP 7: Predict with the pre-trained Random Forest (NO retraining)
# --------------------------------------------------------------------------
# Assuming the trained model object is called `rf_model` and uses predict():
#
#   cell_data$prediction <- predict(rf_model, newdata = cell_data)
#
# The model is preserved as-is. Only the input features were recomputed
# (with numerical equivalence), so predictions are identical.
# --------------------------------------------------------------------------
```

---

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + named-vector lookups | One sparse matrix construction (~1.37M entries) |
| **Mean computation** | 6.46M Ã— 5 R-level loops | 5 Ã— sparse `A %*% X` (BLAS-accelerated) |
| **Max/Min computation** | 6.46M Ã— 5 R-level loops | 5 Ã— 28 `data.table` grouped aggregations on 1.37M edges |
| **Memory** | ~6.46M-element list of integer vectors | ~344K Ã— 28 dense matrix + sparse CSC (~20 MB) |
| **Estimated time** | 86+ hours | **~5â€“15 minutes** on a 16 GB laptop |

### Why numerical equivalence is preserved

- **Mean:** `(A %*% X_zero)[i,t] / (A %*% notNA)[i,t]` computes exactly `sum(neighbor_vals) / length(neighbor_vals)` where `neighbor_vals` excludes NAs â€” identical to `mean(neighbor_vals[!is.na(neighbor_vals)])`.
- **Max/Min:** The `data.table` grouped aggregation computes `max(val)` and `min(val)` over the exact same set of non-NA neighbor values identified by the adjacency matrix, matching the original `max(neighbor_vals)` and `min(neighbor_vals)`.
- **The Random Forest model is never retrained** â€” only the feature columns are recomputed with identical values, so `predict(rf_model, newdata = cell_data)` produces identical output.
END_RESPONSE>>>
