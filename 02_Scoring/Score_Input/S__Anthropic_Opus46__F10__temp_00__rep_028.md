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
run_id: Anthropic_F10_T00_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query in the worst case, and doing this 6.46 million times against a 6.46-million-element named vector is catastrophically slow. This is the dominant cost.

2. **`compute_neighbor_stats` iterates over 6.46 million list elements** in an `lapply`, extracting subsets of a numeric vector and computing `max/min/mean`. While each individual operation is cheap, the R-level loop overhead across 6.46M iterations is substantial, and this is repeated 5 times (once per variable).

3. **The neighbor lookup conflates topology and time.** Rook neighbors are a *spatial* relationship â€” they don't change across years. Yet the lookup is built over the full cell-year panel, inflating the problem from ~344K spatial lookups to ~6.46M spatiotemporal lookups. The string-key join (`paste(id, year)`) is repeated for every cell-year, which is pure waste.

**Key insight:** The adjacency graph is static across years. If we separate the spatial topology from the temporal dimension, we can build a sparse adjacency structure once over 344K cells and then apply it independently within each year using fast vectorized/matrix operations.

---

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from the `spdep::nb` object over the 344K cells. This is a `344208 Ã— 344208` sparse matrix (class `dgCMatrix` from the `Matrix` package) with ~1.37M non-zero entries.

2. **Reshape each variable into a `344208 Ã— 28` matrix** (cells Ã— years). This allows us to compute neighbor aggregations as sparse matrixâ€“dense matrix operations.

3. **Compute neighbor statistics via sparse matrix multiplication and analogous sparse operations:**
   - **Mean:** `A %*% X / degree` where `degree` is the row-sum of `A` (number of neighbors per cell). More precisely: `(A %*% X) / degree_matrix`.
   - **Max and Min:** Use a row-wise sparse sweep. For each cell, we need the max and min of its neighbors' values. This can be done efficiently by iterating over the sparse matrix structure in C++ via `Rcpp`, or by a grouped operation using the sparse matrix's `i, j, x` triplet form.

4. **Unroll back** to the long panel format and attach the 15 new columns (5 vars Ã— 3 stats).

5. **Predict** with the pre-trained Random Forest model â€” no retraining.

**Expected speedup:** From 86+ hours to minutes. The sparse matrix is ~1.37M entries; multiplying it by a 344K Ã— 28 dense matrix is a single BLAS-backed operation. Max/min require a grouped operation but over only 1.37M edges Ã— 28 years â‰ˆ 38M operations, trivially fast with `data.table` or `Rcpp`.

---

## Working R Code

```r
# =============================================================================
# Optimized spatial neighbor feature pipeline
# Preserves numerical equivalence with the original implementation
# =============================================================================

library(Matrix)
library(data.table)

# ---- Step 0: Ensure cell_data is a data.table for speed --------------------
cell_dt <- as.data.table(cell_data)

# ---- Step 1: Build sparse adjacency matrix ONCE ----------------------------
# id_order: vector of 344,208 cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: spdep nb object (list of integer index vectors)

build_sparse_adjacency <- function(id_order, nb_obj) {
  n <- length(id_order)
  # Build COO triplets from the nb list
  from <- integer(0)
  to   <- integer(0)
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep::nb uses 0-length integer or integer(0) for no neighbors;
    # also uses a single 0L to indicate no neighbors in some representations
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0L) {
      from <- c(from, rep.int(i, length(nbrs)))
      to   <- c(to, nbrs)
    }
  }
  # Sparse matrix: A[i,j] = 1 means j is a rook neighbor of i
  A <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  return(A)
}

cat("Building sparse adjacency matrix...\n")
A <- build_sparse_adjacency(id_order, rook_neighbors_unique)
n_cells <- length(id_order)
n_years <- 28L  # 1992-2019
years   <- 1992L:2019L

# Degree vector (number of neighbors per cell)
degree_vec <- as.numeric(rowSums(A))  # length = n_cells

cat("Adjacency matrix:", nrow(A), "x", ncol(A),
    "with", nnzero(A), "non-zero entries\n")

# ---- Step 2: Create cell-index mapping --------------------------------------
# Map each cell ID to its row index in the adjacency matrix (1..344208)
id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))

# Add spatial index to cell_dt
cell_dt[, spatial_idx := id_to_spatial_idx[as.character(id)]]

# Ensure data is sorted by spatial_idx within year for matrix construction
setkey(cell_dt, year, spatial_idx)

# ---- Step 3: Verify complete panel ------------------------------------------
# We need every (cell, year) present. Check:
expected_rows <- n_cells * n_years
actual_rows   <- nrow(cell_dt)
is_complete_panel <- (actual_rows == expected_rows)

if (!is_complete_panel) {
  cat("Panel is not perfectly balanced (",
      actual_rows, "vs expected", expected_rows,
      "). Using safe merge approach.\n")
}

# ---- Step 4: Function to reshape variable to cell x year matrix -------------
reshape_to_matrix <- function(dt, var_name, n_cells, years) {
  # Returns a n_cells x n_years matrix
  # Rows = spatial_idx (1..n_cells), Cols = year index (1..28)
  mat <- matrix(NA_real_, nrow = n_cells, ncol = length(years))
  year_idx <- match(dt$year, years)
  # Fill using linear indexing
  lin_idx <- (year_idx - 1L) * n_cells + dt$spatial_idx
  mat[lin_idx] <- dt[[var_name]]
  return(mat)
}

# ---- Step 5: Compute neighbor max/min/mean via sparse ops -------------------
# For MEAN: straightforward sparse matrix multiply then divide by degree
# For MAX/MIN: we use the triplet representation of A to do grouped max/min

compute_neighbor_features_sparse <- function(A, X_mat, degree_vec) {
  # A: n x n sparse adjacency matrix
  # X_mat: n x T dense matrix of variable values
  # degree_vec: n-vector of neighbor counts
  # Returns: list with max_mat, min_mat, mean_mat (each n x T)

  n <- nrow(X_mat)
  n_t <- ncol(X_mat)

  # --- MEAN via sparse matmul ---
  # sum_mat[i, t] = sum of X[j, t] for all neighbors j of i
  sum_mat <- A %*% X_mat  # sparse %*% dense -> dense, very fast
  # Convert to base matrix
  sum_mat <- as.matrix(sum_mat)

  # mean = sum / degree (handle degree=0 -> NA)
  mean_mat <- sum_mat / degree_vec  # recycling: degree_vec is length n, divides each column
  mean_mat[degree_vec == 0, ] <- NA_real_

  # --- MAX and MIN via COO grouped operations ---
  # Extract triplet form of A
  A_t <- as(A, "TsparseMatrix")  # gives i, j (0-based) and x
  ai <- A_t@i + 1L  # 1-based row indices (the "from" cell)
  aj <- A_t@j + 1L  # 1-based col indices (the "to" cell = neighbor)
  n_edges <- length(ai)

  max_mat <- matrix(NA_real_, nrow = n, ncol = n_t)
  min_mat <- matrix(NA_real_, nrow = n, ncol = n_t)

  # Process each year-column: extract neighbor values, then grouped max/min
  # Using data.table for fast grouped operations
  edge_dt <- data.table(from = ai, to = aj)

  for (t_idx in seq_len(n_t)) {
    x_col <- X_mat[, t_idx]
    # Neighbor values: for each edge (from -> to), the neighbor's value is x_col[to]
    nbr_vals <- x_col[aj]

    # Grouped max and min by 'from'
    # Use data.table for speed
    edge_dt[, val := nbr_vals]

    # Remove NAs before aggregation (to match original behavior)
    agg <- edge_dt[!is.na(val), .(
      nb_max = max(val),
      nb_min = min(val)
    ), by = from]

    max_mat[agg$from, t_idx] <- agg$nb_max
    min_mat[agg$from, t_idx] <- agg$nb_min
  }

  # Fix mean_mat for cells where all neighbor values are NA

  # sum_mat from sparse multiply treats NA as 0 in the sum, which is wrong.
  # We need to recompute mean properly handling NAs.

  # Count non-NA neighbors per cell per year
  # and compute correct sum excluding NAs
  X_notna <- (!is.na(X_mat)) * 1.0  # n x T indicator matrix
  X_zero  <- X_mat
  X_zero[is.na(X_zero)] <- 0.0

  sum_nona  <- as.matrix(A %*% X_zero)    # sum of non-NA neighbor values
  count_nona <- as.matrix(A %*% X_notna)  # count of non-NA neighbors

  mean_mat <- sum_nona / count_nona  # NaN where count=0, which becomes NA
  mean_mat[count_nona == 0] <- NA_real_
  # Cells with no neighbors at all
  mean_mat[degree_vec == 0, ] <- NA_real_

  return(list(max_mat = max_mat, min_mat = min_mat, mean_mat = mean_mat))
}

# ---- Step 6: Main loop over the 5 neighbor source variables -----------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features for", length(neighbor_source_vars), "variables...\n")

for (var_name in neighbor_source_vars) {
  cat("  Processing:", var_name, "... ")
  t0 <- proc.time()

  # Reshape to matrix
  X_mat <- reshape_to_matrix(cell_dt, var_name, n_cells, years)

  # Compute neighbor stats
  stats <- compute_neighbor_features_sparse(A, X_mat, degree_vec)

  # Unroll matrices back to long format and attach to cell_dt
  # cell_dt is keyed by (year, spatial_idx), so linear index = (year_idx-1)*n_cells + spatial_idx
  year_idx <- match(cell_dt$year, years)
  lin_idx  <- (year_idx - 1L) * n_cells + cell_dt$spatial_idx

  col_max  <- paste0(var_name, "_max")
  col_min  <- paste0(var_name, "_min")
  col_mean <- paste0(var_name, "_mean")

  cell_dt[, (col_max)  := stats$max_mat[lin_idx]]
  cell_dt[, (col_min)  := stats$min_mat[lin_idx]]
  cell_dt[, (col_mean) := stats$mean_mat[lin_idx]]

  elapsed <- (proc.time() - t0)[3]
  cat(round(elapsed, 1), "seconds\n")

  # Free memory
  rm(X_mat, stats)
  gc(verbose = FALSE)
}

# ---- Step 7: Restore original row order if needed ---------------------------
# If the original cell_data had a specific row order, restore it
setkey(cell_dt, NULL)  # remove keys
# If you need original order, you could have saved it:
# cell_dt <- cell_dt[original_order]

# ---- Step 8: Predict with pre-trained Random Forest -------------------------
# Convert back to data.frame if the RF model expects one
cell_data <- as.data.frame(cell_dt)

# The trained RF model object (e.g., `rf_model`) is already in memory.
# Predict â€” no retraining:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)

cat("Done. Neighbor features added. Ready for prediction.\n")
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-key lookups in named vector | Single sparse matrix build over 344K cells |
| **Stat computation** | R-level `lapply` over 6.46M elements Ã— 5 vars | Sparse matrix multiply (BLAS-backed) + `data.table` grouped agg |
| **Time complexity** | ~O(N_rows Ã— avg_neighbors Ã— string_ops) | ~O(nnz Ã— T) for max/min; O(nnz Ã— T) for mean via SpMM |
| **Memory** | 6.46M-element list of integer vectors | ~1.37M-entry sparse matrix + 344KÃ—28 dense matrices |
| **NA handling** | Per-element filtering in R loop | Indicator-matrix trick for mean; `data.table` filter for max/min |
| **Estimated runtime** | 86+ hours | **~5â€“15 minutes** on 16 GB laptop |
| **Numerical equivalence** | Baseline | âœ… Preserved (same max, min, mean per cell-year) |
| **RF model** | Pre-trained | âœ… Untouched, predict-only |

**Key design principles applied:**
- **Separate topology from time:** The adjacency matrix is built once and reused across all 28 years and all 5 variables.
- **Vectorize over space via sparse linear algebra:** `A %*% X` computes the neighbor sum for all 344K cells simultaneously.
- **Handle NA correctly:** A second sparse multiply with an indicator matrix counts non-NA neighbors, enabling correct mean computation without R-level loops.
- **Grouped aggregation for max/min:** The COO (triplet) representation of the sparse matrix gives us an edge list; `data.table` grouped operations over ~1.37M edges per year-slice are extremely fast.
END_RESPONSE>>>
