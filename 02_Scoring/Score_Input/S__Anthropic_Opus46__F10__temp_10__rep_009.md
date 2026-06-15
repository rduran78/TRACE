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
run_id: Anthropic_F10_T10_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The original implementation suffers from three compounding performance bottlenecks:

### 1. **O(NÂ²)-scale key-hashing in `build_neighbor_lookup`**
The function creates a lookup via `paste(id, year)` for all ~6.46M rows, then for *each* of those 6.46M rows, it generates neighbor keys with `paste()` and performs named-vector lookups. Named vector lookups in R are O(n) hash probes per call, and doing this 6.46M times against a 6.46M-length named vector is catastrophically slow.

### 2. **Row-wise `lapply` over 6.46M rows in `compute_neighbor_stats`**
Each call extracts a small vector, subsets out NAs, and computes three summary statistics. The R interpreter overhead per iteration is ~microseconds, but 6.46M Ã— 5 variables Ã— ~5Î¼s â‰ˆ 160+ seconds *just* in loop overhead â€” and the actual subsetting/allocation is far worse.

### 3. **Redundant topology recomputation**
The neighbor graph is purely spatial (rook contiguity), yet the lookup embeds year into every key. The same adjacency structure applies identically to all 28 years. The implementation re-resolves the same spatial neighbors for every year.

### Memory profile
The `neighbor_lookup` list stores ~6.46M integer vectors. With R's per-vector overhead (~128 bytes minimum), this consumes >800MB just in list structure, leaving little headroom on 16GB.

---

## Optimization Strategy

**Core insight:** Separate the spatial topology (344K cells Ã— ~4 neighbors each) from the temporal panel (28 years). Build the sparse adjacency structure *once* over cells, then use vectorized sparse-matrixâ€“dense-matrix multiplication to compute all neighbor statistics in bulk.

### Specific techniques:

1. **Build a sparse adjacency matrix (344K Ã— 344K)** from the `nb` object â€” once, ~1.4M non-zero entries.

2. **Reshape each variable into a 344K Ã— 28 cell-by-year matrix.** This aligns spatial structure with matrix rows.

3. **Compute neighbor sums, counts, min, max via sparse matrix operations:**
   - **Mean:** `A %*% X / A %*% (non-NA indicator)` â€” fully vectorized.
   - **Max/Min:** Use a modified sparse matrix approach: iterate over each cell's (few) neighbors using CSC column pointers, but do it in C++ via `Rcpp` for speed, or use a chunked vectorized approach.

4. **Avoid any `lapply` over 6.46M rows.** Everything operates on matrices or sparse algebra.

5. **Memory:** A 344K Ã— 28 dense matrix of doubles is ~77MB per variable. The sparse matrix is ~22MB. Total working memory stays well under 4GB.

6. **Expected speedup:** From 86+ hours to **~2â€“5 minutes**.

---

## Optimized R Code

```r
###############################################################################
# optimized_neighbor_features.R
#
# Drop-in replacement for the original neighbor-feature pipeline.
# Preserves numerical equivalence of max, min, mean neighbor statistics.
# Does NOT retrain or modify the Random Forest model.
###############################################################################

library(Matrix)   # sparse matrices (ships with R)
library(data.table)

# â”€â”€ Step 1: Build sparse adjacency matrix ONCE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# rook_neighbors_unique : spdep nb object (list of integer vectors), length = N_cells
# id_order              : integer vector of cell IDs in the order matching the nb object
#
build_adjacency_matrix <- function(id_order, rook_neighbors_unique) {
  n <- length(id_order)
  # Pre-allocate edge vectors
  n_edges <- sum(lengths(rook_neighbors_unique))
  from <- integer(n_edges)
  to   <- integer(n_edges)
  pos  <- 1L
  for (i in seq_len(n)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nb_i <- nb_i[nb_i != 0L]
    len  <- length(nb_i)
    if (len > 0L) {
      from[pos:(pos + len - 1L)] <- i
      to[pos:(pos + len - 1L)]   <- nb_i
      pos <- pos + len
    }
  }
  # Trim if any 0-neighbor cells caused over-allocation
  from <- from[seq_len(pos - 1L)]
  to   <- to[seq_len(pos - 1L)]
  # Row i has a 1 in column j means "j is a neighbor of i"
  # So A %*% X gives: for each row i, sum of X[j,] over neighbors j
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

# â”€â”€ Step 2: Reshape panel column into cell Ã— year matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# Uses data.table for speed. Returns a list with:
#   mat        : N_cells Ã— N_years numeric matrix (NA-safe)
#   cell_idx   : named integer vector mapping cell_id -> row position in mat
#   year_idx   : named integer vector mapping year   -> col position in mat
#   row_order  : integer vector to reverse-map (cell_row, year_col) back to
#                the original data.table row positions
#
reshape_to_matrix <- function(dt, id_order, years, var_name) {
  n_cells <- length(id_order)
  n_years <- length(years)

  # Fast lookup: cell_id -> matrix row
  cell_idx <- setNames(seq_along(id_order), as.character(id_order))
  year_idx <- setNames(seq_along(years), as.character(years))

  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  ri <- cell_idx[as.character(dt$id)]
  ci <- year_idx[as.character(dt$year)]

  mat[cbind(ri, ci)] <- dt[[var_name]]

  list(mat = mat, cell_idx = cell_idx, year_idx = year_idx,
       ri = ri, ci = ci)
}

# â”€â”€ Step 3: Vectorized neighbor stats via sparse matrix ops â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# For MEAN:
#   neighbor_sum   = A %*% X          (replace NA with 0 before multiply)
#   neighbor_count = A %*% (!is.na(X)) (count non-NA neighbors per cell-year)
#   neighbor_mean  = neighbor_sum / neighbor_count  (0-count -> NA)
#
# For MAX and MIN:
#   We use an Rcpp-free, fully-vectorized approach:
#     - Extract CSC pointers from t(A) (so columns of t(A) = neighbors of each cell)
#     - Expand neighbor values into a long vector with a grouping vector
#     - Use data.table or base tapply for grouped max/min
#   However, for maximum speed without Rcpp, we use an iterative sparse-layer
#   approach: since rook neighbors number at most 4, we iterate over "neighbor
#   rank" (1st neighbor, 2nd neighbor, ..., up to max_degree) and take running
#   max/min via pmin/pmax.  This is at most 4-8 passes over a 344KÃ—28 matrix.
#
compute_neighbor_features_sparse <- function(A, mat) {
  # mat: N_cells Ã— N_years, may contain NA

  n_cells <- nrow(mat)
  n_years <- ncol(mat)

  # â”€â”€ Mean (fully vectorized sparse matmul) â”€â”€
  mat0 <- mat
  mat0[is.na(mat0)] <- 0
  indicator <- matrix(1, nrow = n_cells, ncol = n_years)
  indicator[is.na(mat)] <- 0

  neighbor_sum   <- as.matrix(A %*% mat0)        # N_cells Ã— N_years
  neighbor_count <- as.matrix(A %*% indicator)    # N_cells Ã— N_years

  neighbor_mean <- neighbor_sum / neighbor_count
  neighbor_mean[neighbor_count == 0] <- NA_real_

  # â”€â”€ Max and Min (iterate over neighbor layers) â”€â”€
  # Extract adjacency list from sparse matrix A (CSC format of A transposed)
  # For each cell i, its neighbors are the row indices of non-zero entries in column i of t(A)
  At <- t(A)  # Now column j of At = neighbors of cell j
  # Determine max degree
  degrees <- diff(At@p)
  max_deg <- max(degrees)

  # Initialize max/min matrices with NA

  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # For each neighbor rank k = 1..max_deg, extract the k-th neighbor's values

  # and update running max/min. This avoids any R-level row loop.
  #
  # Build a matrix: neighbor_indices[i, k] = index of i's k-th neighbor (or NA)
  neighbor_indices <- matrix(NA_integer_, nrow = n_cells, ncol = max_deg)
  for (j in seq_len(n_cells)) {
    start <- At@p[j] + 1L
    end   <- At@p[j + 1L]
    if (end >= start) {
      nb_j <- At@i[start:end] + 1L   # 0-based to 1-based
      neighbor_indices[j, seq_along(nb_j)] <- nb_j
    }
  }

  # Now iterate over neighbor ranks (max ~4 for rook)
  for (k in seq_len(max_deg)) {
    idx_k <- neighbor_indices[, k]   # length N_cells, some NA
    has_nb <- !is.na(idx_k)
    if (!any(has_nb)) next

    # Extract neighbor values: N_cells Ã— N_years matrix
    nb_vals <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_vals[has_nb, ] <- mat[idx_k[has_nb], , drop = FALSE]

    # Update running max
    # pmax with na.rm: take the non-NA value, or NA if both NA
    neighbor_max <- pmax(neighbor_max, nb_vals, na.rm = TRUE)
    neighbor_min <- pmin(neighbor_min, nb_vals, na.rm = TRUE)
  }

  # Where neighbor_count == 0, max and min should be NA (already are from initialization)
  # but ensure consistency:
  neighbor_max[neighbor_count == 0] <- NA_real_
  neighbor_min[neighbor_count == 0] <- NA_real_

  list(max = neighbor_max, min = neighbor_min, mean = neighbor_mean)
}

# â”€â”€ Step 4: Full pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

run_neighbor_feature_pipeline <- function(cell_data, id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  cat("Building sparse adjacency matrix...\n")
  A <- build_adjacency_matrix(id_order, rook_neighbors_unique)
  cat(sprintf("  Adjacency: %d cells, %d directed edges, max degree %d\n",
              nrow(A), nnzero(A), max(diff(t(A)@p))))

  # Determine years from data
  years <- sort(unique(cell_data$year))
  cat(sprintf("  Panel: %d years (%dâ€“%d), %d rows\n",
              length(years), min(years), max(years), nrow(cell_data)))

  # Convert to data.table for fast column operations if not already
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))

    # Reshape to matrix
    info <- reshape_to_matrix(cell_data, id_order, years, var_name)

    # Compute neighbor stats (vectorized)
    stats <- compute_neighbor_features_sparse(A, info$mat)

    # Map results back to original row order using (ri, ci) indices
    idx <- cbind(info$ri, info$ci)

    max_col <- paste0("neighbor_max_", var_name)
    min_col <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    cell_data[, (max_col)  := stats$max[idx]]
    cell_data[, (min_col)  := stats$min[idx]]
    cell_data[, (mean_col) := stats$mean[idx]]

    cat(sprintf("  Added: %s, %s, %s\n", max_col, min_col, mean_col))
  }

  cat("Neighbor feature computation complete.\n")
  cell_data
}

###############################################################################
# USAGE (drop-in replacement for the original outer loop)
###############################################################################
#
# # Load your data and nb object as before
# load("rook_neighbors_unique.rda")   # -> rook_neighbors_unique, id_order
# load("cell_data.rda")               # -> cell_data (data.frame or data.table)
# load("trained_rf_model.rda")        # -> rf_model (DO NOT retrain)
#
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- run_neighbor_feature_pipeline(
#   cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
# )
#
# # Predict with the EXISTING trained model (unchanged)
# predictions <- predict(rf_model, newdata = cell_data)
###############################################################################
```

---

## Step 5 (Optional): Even faster `neighbor_indices` construction

The `for (j in seq_len(n_cells))` loop over 344K cells to build `neighbor_indices` takes ~10â€“20 seconds. If that matters, here is a fully vectorized replacement:

```r
# Fully vectorized neighbor_indices construction (replaces the for-loop)
build_neighbor_index_matrix <- function(At, n_cells, max_deg) {
  # At is the transpose of A in dgCMatrix format
  # Column j of At holds the neighbors of cell j
  degrees <- diff(At@p)
  
  # Expand: for each cell j, create a sequence 1:degree[j]
  cell_ids <- rep(seq_len(n_cells), times = degrees)
  ranks    <- sequence(degrees)  # vectorized 1:deg[1], 1:deg[2], ...
  nb_ids   <- At@i + 1L         # all neighbor indices, 1-based
  
  # Allocate and fill
  neighbor_indices <- matrix(NA_integer_, nrow = n_cells, ncol = max_deg)
  neighbor_indices[cbind(cell_ids, ranks)] <- nb_ids
  neighbor_indices
}
```

---

## Complexity & Performance Summary

| Component | Original | Optimized |
|---|---|---|
| Topology construction | O(R Ã— N) hash lookups, R=6.46M | O(E) sparse matrix, E=1.37M, **once** |
| Per-variable stats | 6.46M `lapply` iterations | 1 sparse matmul + 4 vectorized `pmax`/`pmin` passes |
| Memory for lookup | ~800MB+ (R list overhead) | ~22MB sparse matrix + ~77MB per working matrix |
| Total time (estimated) | 86+ hours | **2â€“5 minutes** |
| Numerical equivalence | Baseline | Identical (same `max`, `min`, `mean` over same non-NA neighbor sets) |
| RF model | Unchanged | Unchanged |

The key transformation is recognizing that **neighbor mean is a sparse matrix multiply**, and **neighbor max/min over â‰¤4 rook neighbors is at most 4 vectorized `pmax`/`pmin` passes** over a dense matrix â€” eliminating all R-level row iteration over 6.46 million rows.
END_RESPONSE>>>
