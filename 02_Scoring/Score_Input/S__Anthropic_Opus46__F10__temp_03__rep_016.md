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
run_id: Anthropic_F10_T03_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each constructed via string-pasting and named-vector lookups. The `paste(id, year)` key construction and `idx_lookup[neighbor_keys]` named-vector subsetting is O(n) per lookup in the worst case, and doing this 6.46M times is catastrophic. This alone likely accounts for the majority of the 86+ hour estimate.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a pure-R `lapply`**, extracting subsets of a vector and computing `max/min/mean`. This is called 5 times (once per source variable), totaling ~32.3 million R-level loop iterations with per-element allocation overhead.

3. **The neighbor lookup is year-aware but the graph topology is year-invariant.** Rook contiguity is a spatial relationship between the 344,208 cells â€” it doesn't change across years. Yet the current code rebuilds neighbor row-indices for every cell-year combination, inflating the problem from 344K nodes to 6.46M nodes unnecessarily.

**Key insight:** The adjacency structure is a fixed sparse graph over 344,208 nodes. Year is just a panel dimension. We should separate the spatial topology from the temporal dimension entirely.

---

## Optimization Strategy

1. **Build the sparse adjacency matrix once** over the 344,208 cells (not 6.46M cell-years). Use a `dgCMatrix` (compressed sparse column) from the `Matrix` package. This is O(E) where E â‰ˆ 1.37M edges.

2. **Reshape each variable into a 344,208 Ã— 28 matrix** (cells Ã— years). This allows vectorized column-wise (per-year) operations.

3. **Compute neighbor aggregates via sparse matrixâ€“dense matrix multiplication** and analogous sparse operations:
   - **Mean:** `A_norm %*% X` where `A_norm` is the row-normalized adjacency matrix (each row sums to 1, or the count of neighbors).
   - **Max and Min:** Use a grouped sparse operation â€” iterate over the 344,208 cells (not 6.46M), extracting neighbor rows from the year-matrix. With `dgCMatrix` column pointers, neighbor extraction is O(degree) per node.

4. **Avoid all string operations, named lookups, and per-cell-year R list elements.** Everything is integer-indexed.

5. **Memory:** The cellÃ—year matrix for one variable is 344,208 Ã— 28 Ã— 8 bytes â‰ˆ 77 MB. We need at most a few of these simultaneously. Total peak memory stays well under 16 GB.

6. **Expected speedup:** From 86+ hours to **minutes**. The sparse matrix multiply for mean is O(E Ã— T) â‰ˆ 38.4M multiply-adds â€” trivial. Max/min require a grouped loop over 344K nodes Ã— 28 years, but with vectorized inner operations this completes in seconds per variable.

---

## Optimized R Code

```r
library(Matrix)
library(data.table)

# =============================================================================
# STEP 0: Ensure cell_data is a data.table, sorted by (id, year)
# =============================================================================
cell_dt <- as.data.table(cell_data)
setkeyv(cell_dt, c("id", "year"))

# Unique cell IDs and years (sorted)
unique_ids   <- sort(unique(cell_dt$id))
unique_years <- sort(unique(cell_dt$year))
n_cells      <- length(unique_ids)  # 344,208
n_years      <- length(unique_years) # 28

# Map cell id -> integer index 1..n_cells
id_to_idx <- setNames(seq_along(unique_ids), as.character(unique_ids))

# Verify the data is complete panel (required for matrix reshape)
stopifnot(nrow(cell_dt) == n_cells * n_years)

# =============================================================================
# STEP 1: Build sparse adjacency matrix ONCE (344,208 x 344,208)
# =============================================================================
# rook_neighbors_unique is an nb object: list of length n_cells,
# where element i contains integer indices of neighbors of cell i
# (in the same order as id_order).
# id_order maps position in the nb object -> cell id.

# Build mapping from id_order position to our sorted unique_ids index
id_order_to_idx <- id_to_idx[as.character(id_order)]

# Construct COO (coordinate) triplets for the sparse matrix
# Pre-calculate total number of edges for pre-allocation
n_edges <- sum(lengths(rook_neighbors_unique))

from_vec <- integer(n_edges)
to_vec   <- integer(n_edges)

pos <- 1L
for (i in seq_along(rook_neighbors_unique)) {
  nb_i <- rook_neighbors_unique[[i]]
  if (length(nb_i) > 0 && !(length(nb_i) == 1 && nb_i[1] == 0L)) {
    n_nb <- length(nb_i)
    from_idx <- id_order_to_idx[i]
    to_indices <- id_order_to_idx[nb_i]
    from_vec[pos:(pos + n_nb - 1L)] <- from_idx
    to_vec[pos:(pos + n_nb - 1L)]   <- to_indices
    pos <- pos + n_nb
  }
}
# Trim if any nb entries were empty/zero
from_vec <- from_vec[1:(pos - 1L)]
to_vec   <- to_vec[1:(pos - 1L)]

# Sparse adjacency matrix (rows = focal cell, cols = neighbor cell)
A <- sparseMatrix(
  i    = from_vec,
  j    = to_vec,
  x    = 1,
  dims = c(n_cells, n_cells),
  repr = "C"   # CSC format
)

# Degree vector (number of neighbors per cell) for mean computation
degree_vec <- rowSums(A)  # numeric vector length n_cells

# Row-normalized adjacency for mean: A_norm[i,j] = 1/degree(i) if j is neighbor
# Handle isolated nodes (degree 0) â€” they'll get NaN, we fix to NA later
inv_degree <- ifelse(degree_vec > 0, 1.0 / degree_vec, 0)
A_norm <- Diagonal(x = inv_degree) %*% A

# Transpose of A in CSC for efficient column-wise access = row access of A
# Actually for max/min we need to iterate over rows of A (neighbors of each node).
# Convert A to dgRMatrix (row-compressed) for efficient row slicing,
# or use the CSC transpose trick.
At <- t(A)  # At is CSC; column j of At = row j of A = neighbors of node j

rm(from_vec, to_vec)
gc()

cat("Adjacency matrix built:", nnzero(A), "non-zeros\n")

# =============================================================================
# STEP 2: Helper â€” reshape a variable from long data.table to cell x year matrix
# =============================================================================
# cell_dt is keyed by (id, year), so rows are in (id, year) order.
# That means for each id block, years are consecutive and sorted.
# Matrix column = year, row = cell (in unique_ids order).

reshape_to_matrix <- function(dt, var_name, n_cells, n_years) {
  # dt is sorted by (id, year). Each cell has exactly n_years rows.
  matrix(dt[[var_name]], nrow = n_cells, ncol = n_years, byrow = TRUE)
}

# =============================================================================
# STEP 3: Compute neighbor stats (max, min, mean) for one variable
# =============================================================================
compute_neighbor_features <- function(At, A_norm, degree_vec, X, n_cells, n_years) {
  # X is n_cells x n_years matrix of the source variable
  #
  # MEAN: sparse matrix multiply (exact equivalent of original mean)
  X_mean <- as.matrix(A_norm %*% X)
  # Fix isolated nodes: A_norm rows are zero, product is 0 â€” should be NA
  isolated <- degree_vec == 0
  if (any(isolated)) {
    X_mean[isolated, ] <- NA_real_
  }
  # Also propagate NA: if all neighbors are NA for a cell-year, mean should be NA.
  # The sparse multiply treats NA as... actually, sparse %*% with NA propagates.
  # But the original code removes NAs before computing mean. We need to match that.
  #
  # For exact numerical equivalence with na.rm=TRUE behavior, we need:
  #   mean_i = sum(non-NA neighbor vals) / count(non-NA neighbor vals)
  # This requires two sparse multiplies:
  #   - sum of non-NA values
  #   - count of non-NA values

  # Create a version of X where NA -> 0 for sum, and an indicator matrix
  X_nona <- X
  is_na_mask <- is.na(X)
  X_nona[is_na_mask] <- 0

  # Indicator: 1 where not NA, 0 where NA
  X_valid <- matrix(1, nrow = n_cells, ncol = n_years)
  X_valid[is_na_mask] <- 0

  # Neighbor sums (ignoring NAs)
  neighbor_sum   <- as.matrix(A %*% X_nona)    # n_cells x n_years
  neighbor_count <- as.matrix(A %*% X_valid)    # n_cells x n_years

  # Mean with NA removal
  X_mean <- neighbor_sum / neighbor_count  # produces NaN where count=0 -> becomes NA
  X_mean[neighbor_count == 0] <- NA_real_

  # MAX and MIN: need per-row grouped operations
  # We iterate over cells (344K), not cell-years (6.46M).
  # For each cell, extract neighbor rows from X, compute columnwise max/min.

  X_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  X_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # Extract CSC structure of At: column j of At = neighbors of node j
  At_p <- At@p    # column pointers (0-based), length n_cells+1
  At_i <- At@i    # row indices (0-based)

  for (j in seq_len(n_cells)) {
    # Column j of At (0-indexed): entries from At_p[j]+1 to At_p[j+1]
    start <- At_p[j] + 1L
    end   <- At_p[j + 1L]
    if (end < start) next  # no neighbors (isolated node)

    nb_indices <- At_i[start:end] + 1L  # convert to 1-based

    if (length(nb_indices) == 1L) {
      # Single neighbor: max = min = that neighbor's values
      X_max[j, ] <- X[nb_indices, ]
      X_min[j, ] <- X[nb_indices, ]
    } else {
      # Multiple neighbors: extract submatrix and compute col max/min
      nb_mat <- X[nb_indices, , drop = FALSE]  # k x n_years
      # Use matrixStats for speed if available, otherwise base R
      # colMaxs/colMins with na.rm=TRUE
      for (t in seq_len(n_years)) {
        col_vals <- nb_mat[, t]
        col_vals <- col_vals[!is.na(col_vals)]
        if (length(col_vals) > 0L) {
          X_max[j, t] <- max(col_vals)
          X_min[j, t] <- min(col_vals)
        }
        # else stays NA
      }
    }
  }

  list(max = X_max, min = X_min, mean = X_mean)
}

# =============================================================================
# STEP 3b: Faster max/min using matrixStats (if available)
# =============================================================================
# The inner loop above (344K iterations Ã— 28 years) can be slow in pure R.
# We can vectorize the max/min using a "scatter" approach with sparse ops.
# However, sparse max/min don't exist natively. Let's use Rcpp or matrixStats.

# Optimized version using matrixStats::colMaxs / colMins:
if (requireNamespace("matrixStats", quietly = TRUE)) {
  compute_neighbor_features <- function(At, A, degree_vec, X, n_cells, n_years) {
    # --- MEAN (NA-safe) ---
    X_nona <- X
    is_na_mask <- is.na(X)
    X_nona[is_na_mask] <- 0
    X_valid <- matrix(1, nrow = n_cells, ncol = n_years)
    X_valid[is_na_mask] <- 0
    neighbor_sum   <- as.matrix(A %*% X_nona)
    neighbor_count <- as.matrix(A %*% X_valid)
    X_mean <- neighbor_sum / neighbor_count
    X_mean[neighbor_count == 0] <- NA_real_

    # --- MAX and MIN ---
    X_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    At_p <- At@p
    At_i <- At@i

    for (j in seq_len(n_cells)) {
      start <- At_p[j] + 1L
      end   <- At_p[j + 1L]
      if (end < start) next

      nb_indices <- At_i[start:end] + 1L
      nb_mat <- X[nb_indices, , drop = FALSE]

      X_max[j, ] <- matrixStats::colMaxs(nb_mat, na.rm = TRUE)
      X_min[j, ] <- matrixStats::colMins(nb_mat, na.rm = TRUE)
    }

    # matrixStats returns -Inf/Inf when all values are NA; fix to NA
    X_max[is.infinite(X_max)] <- NA_real_
    X_min[is.infinite(X_min)] <- NA_real_

    list(max = X_max, min = X_min, mean = X_mean)
  }
}

# =============================================================================
# STEP 4: Run the pipeline for all 5 neighbor source variables
# =============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor features for:", var_name, "\n")
  t0 <- proc.time()

  # Reshape to matrix
  X <- reshape_to_matrix(cell_dt, var_name, n_cells, n_years)

  # Compute neighbor stats
  stats <- compute_neighbor_features(At, A, degree_vec, X, n_cells, n_years)

  # Flatten back to long format (byrow=TRUE matches our reshape convention)
  # cell_dt is sorted by (id, year), so row i*n_years + t maps to cell i, year t
  max_col <- paste0(var_name, "_neighbor_max")
  min_col <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")

  cell_dt[, (max_col)  := as.vector(t(stats$max))]
  cell_dt[, (min_col)  := as.vector(t(stats$min))]
  cell_dt[, (mean_col) := as.vector(t(stats$mean))]

  rm(X, stats)
  gc()

  elapsed <- (proc.time() - t0)[3]
  cat("  Done in", round(elapsed, 1), "seconds\n")
}

# =============================================================================
# STEP 5: Predict with the pre-trained Random Forest (unchanged)
# =============================================================================
# Convert back to data.frame if the RF model expects one
cell_data <- as.data.frame(cell_dt)

# The trained model object (e.g., rf_model) is used as-is â€” no retraining.
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Original | Optimized | Equivalence |
|---|---|---|
| `mean(neighbor_vals[!is.na(...)])` | `(A %*% X_nona) / (A %*% X_valid)` | Identical: sum of non-NA values Ã· count of non-NA values, per cell-year |
| `max(neighbor_vals[!is.na(...)])` | `matrixStats::colMaxs(nb_mat, na.rm=TRUE)` | Identical: same elements, same function |
| `min(neighbor_vals[!is.na(...)])` | `matrixStats::colMins(nb_mat, na.rm=TRUE)` | Identical: same elements, same function |
| Returns `c(NA,NA,NA)` when no valid neighbors | `neighbor_count==0 â†’ NA`; `Inf â†’ NA` | Identical NA propagation |

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M string-paste + named-vector lookups | One sparse matrix build (1.37M entries) |
| Mean computation | 6.46M R list iterations Ã— 5 vars | 5 sparse matrix multiplies (~seconds each) |
| Max/Min computation | 6.46M R list iterations Ã— 5 vars | 344K iterations Ã— 5 vars, vectorized over 28 years |
| Memory pattern | ~6.46M R list elements (heavy GC) | Dense matrices ~77 MB each + one sparse matrix |
| **Estimated time** | **86+ hours** | **~5â€“15 minutes total** |
END_RESPONSE>>>
