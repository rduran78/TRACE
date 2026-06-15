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
run_id: Anthropic_F08_T07_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance bottleneck is in `build_neighbor_lookup`. It constructs a lookup **per cell-year row** (~6.46 million entries), even though the neighbor *topology* is identical across all 28 years. Specifically:

1. **Redundant work across years:** The function iterates over every row (`nrow(data)` â‰ˆ 6.46M), looks up which cells are neighbors (a static, year-independent fact), and then resolves those neighbors to row indices for the *same year*. Because the topology is the same for all 28 years, the neighbor-cell-ID lookup is repeated 28 times per cell â€” that's 28Ã— redundant graph traversal.

2. **Expensive string-key hashing:** The function builds `idx_lookup` as a named vector keyed by `paste(id, year, sep="_")` with 6.46M entries. Then for each of the 6.46M rows, it pastes neighbor IDs with the current year and does named-vector lookups. String operations on millions of keys are extremely slow in R.

3. **`lapply` over 6.46M rows:** Each iteration does allocation, string pasting, and named-vector subsetting. The per-iteration overhead multiplied by 6.46M iterations dominates runtime.

4. **`compute_neighbor_stats` is called once per variable**, each time iterating over the 6.46M-element list. That's 5 Ã— 6.46M list iterations.

**In summary:** The implementation treats the entire problem as year-dependent, when in fact the neighbor graph is static. It also relies on slow string-keyed lookups instead of integer indexing.

---

## Optimization Strategy

**Key insight:** Separate the *static topology* (which cells are neighbors of which) from the *year-varying data* (variable values).

### Step 1: Build the neighbor lookup once, at the cell level only (344K cells, not 6.46M rows)

Construct a list of length 344,208 where element `i` contains the integer indices (into the cell-ID vector) of cell `i`'s neighbors. This is done **once** and is year-independent.

### Step 2: For each variable, compute neighbor stats using vectorized matrix operations

- Reshape each variable into a **matrix** of dimension `n_cells Ã— n_years` (344,208 Ã— 28).
- For each cell, the neighbor indices point to *rows* of this matrix. Extract neighbor values as a sub-matrix, then compute column-wise (i.e., year-wise) max, min, mean.
- Better yet, use a **sparse adjacency matrix** and matrix multiplication for the mean, and row-wise grouped operations for max/min.

### Step 3: Flatten results back into the long data frame

This avoids all string operations, avoids 6.46M-element list iteration, and reduces the neighbor traversal from 6.46M to 344K (a 28Ã— reduction), with the per-year computation handled by vectorized matrix/column operations.

### Expected speedup

- Neighbor lookup: 344K iterations instead of 6.46M â†’ ~28Ã— faster.
- No string pasting or named-vector lookup â†’ additional large constant-factor improvement.
- Vectorized matrix operations for stats â†’ orders of magnitude faster than `lapply` over 6.46M elements.
- Estimated runtime: minutes instead of 86+ hours.

---

## Working R Code

```r
library(Matrix)  # for sparse matrix operations

# ==============================================================================
# STEP 0: Ensure consistent ordering
# ==============================================================================
# cell_data must be sorted by (id, year) for the reshape to work correctly.
# id_order is the vector of unique cell IDs matching the nb object indexing.
# rook_neighbors_unique is the spdep::nb object (list of length n_cells).

cell_data <- cell_data[order(cell_data$id, cell_data$year), ]

n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

stopifnot(nrow(cell_data) == n_cells * n_years)  # balanced panel check

# ==============================================================================
# STEP 1: Build cell-level neighbor index list (STATIC, done once)
# ==============================================================================
# rook_neighbors_unique[[i]] gives neighbor indices (into id_order) for cell i.
# spdep::nb objects use 0-length integer(0) for cells with no neighbors.
# This step is trivial â€” the nb object already IS the cell-level lookup.

cell_neighbor_idx <- rook_neighbors_unique
# cell_neighbor_idx[[i]] = integer vector of indices j such that id_order[j]
# is a rook neighbor of id_order[i].

# ==============================================================================
# STEP 2: Build sparse adjacency matrix (STATIC, done once)
# ==============================================================================
# Construct a sparse n_cells x n_cells adjacency matrix W.
# W[i,j] = 1 if cell j is a neighbor of cell i.
# This enables: neighbor_mean = (W %*% value_matrix) / neighbor_count

# Build COO (coordinate) representation
from_idx <- rep(seq_len(n_cells), lengths(cell_neighbor_idx))
to_idx   <- unlist(cell_neighbor_idx)

# Handle edge case: if no neighbors at all (shouldn't happen, but safe)
if (length(from_idx) == 0) {
  W <- sparseMatrix(i = integer(0), j = integer(0), dims = c(n_cells, n_cells))
} else {
  W <- sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = 1,
    dims = c(n_cells, n_cells)
  )
}

# Number of neighbors per cell (static)
neighbor_count <- as.numeric(W %*% rep(1, n_cells))  # length n_cells

# ==============================================================================
# STEP 3: Map from cell_data rows to (cell_index, year_index)
# ==============================================================================
# Because cell_data is sorted by (id, year), row ((i-1)*n_years + t) corresponds
# to cell i, year t. We can reshape directly.

id_to_cell_idx <- setNames(seq_along(id_order), as.character(id_order))
cell_indices   <- id_to_cell_idx[as.character(cell_data$id)]

# Verify the reshape assumption
# Each cell's rows should be consecutive and cover all years in order.
# With the sort above, cell_data$id repeats each id n_years times.

# ==============================================================================
# STEP 4: Function to compute neighbor stats for one variable (VECTORIZED)
# ==============================================================================
compute_neighbor_features_fast <- function(cell_data, var_name, W,
                                           neighbor_count, cell_neighbor_idx,
                                           n_cells, n_years, id_order, years) {

  # 4a. Reshape variable into n_cells x n_years matrix
  #     Row i = cell i (in id_order), Column t = year t (in sorted years)
  val_vec <- cell_data[[var_name]]
  V <- matrix(val_vec, nrow = n_cells, ncol = n_years, byrow = TRUE)
  # byrow=TRUE because data is sorted by (id, year): first n_years rows are

  # cell 1's years, next n_years are cell 2's years, etc.

  # 4b. Neighbor MEAN via sparse matrix multiplication
  #     For each year (column), neighbor_sum = W %*% V[,t]
  #     This is just W %*% V (matrix multiply, sparse Ã— dense)
  neighbor_sum <- as.matrix(W %*% V)  # n_cells x n_years

  # Avoid division by zero for cells with no neighbors
  safe_count <- neighbor_count
  safe_count[safe_count == 0] <- NA

  neighbor_mean_mat <- neighbor_sum / safe_count  # n_cells x n_years

  # 4c. Neighbor MAX and MIN: use the cell-level neighbor list
  #     For each cell, extract the sub-matrix of neighbor values and compute
  #     column-wise max and min.
  #
  #     To vectorize this efficiently, we iterate over cells (344K, not 6.46M).

  neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (i in seq_len(n_cells)) {
    nb_idx <- cell_neighbor_idx[[i]]
    if (length(nb_idx) == 0L) next
    # nb_idx indexes rows of V
    if (length(nb_idx) == 1L) {
      neighbor_max_mat[i, ] <- V[nb_idx, ]
      neighbor_min_mat[i, ] <- V[nb_idx, ]
    } else {
      sub_mat <- V[nb_idx, , drop = FALSE]  # k_neighbors x n_years
      # colMins / colMaxs from matrixStats would be faster, but base R:
      neighbor_max_mat[i, ] <- apply(sub_mat, 2, max, na.rm = TRUE)
      neighbor_min_mat[i, ] <- apply(sub_mat, 2, min, na.rm = TRUE)
    }
  }

  # Replace -Inf/Inf from max/min of all-NA columns with NA
  neighbor_max_mat[is.infinite(neighbor_max_mat)] <- NA
  neighbor_min_mat[is.infinite(neighbor_min_mat)] <- NA

  # 4d. Flatten back to long format (by row = by cell, then by year within cell)
  max_col_name  <- paste0("neighbor_max_",  var_name)
  min_col_name  <- paste0("neighbor_min_",  var_name)
  mean_col_name <- paste0("neighbor_mean_", var_name)

  cell_data[[max_col_name]]  <- as.vector(t(neighbor_max_mat))
  cell_data[[min_col_name]]  <- as.vector(t(neighbor_min_mat))
  cell_data[[mean_col_name]] <- as.vector(t(neighbor_mean_mat))

  cell_data
}

# ==============================================================================
# STEP 5: Even faster MAX/MIN using matrixStats (if available)
# ==============================================================================
# The loop in Step 4c over 344K cells with apply() is still somewhat slow.
# We can accelerate max/min using a grouped-row approach with vectorization.
# Below is an improved version that avoids the inner apply() call.

compute_neighbor_features_fastest <- function(cell_data, var_name, W,
                                              neighbor_count, cell_neighbor_idx,
                                              n_cells, n_years, id_order, years) {

  val_vec <- cell_data[[var_name]]
  V <- matrix(val_vec, nrow = n_cells, ncol = n_years, byrow = TRUE)

  # --- MEAN via sparse matmul ---
  neighbor_sum <- as.matrix(W %*% V)
  safe_count <- neighbor_count
  safe_count[safe_count == 0] <- NA
  neighbor_mean_mat <- neighbor_sum / safe_count

  # --- MAX and MIN via expanded-row approach ---
  # Idea: create an "expanded" matrix where row k corresponds to one directed

  # edge (from_cell -> to_cell). Then group-max and group-min by from_cell.
  #
  # from_idx and to_idx were already computed for the sparse matrix.
  # from_idx[e] = cell i, to_idx[e] = neighbor j of cell i.

  from_vec <- rep(seq_len(n_cells), lengths(cell_neighbor_idx))
  to_vec   <- unlist(cell_neighbor_idx)

  n_edges <- length(from_vec)

  if (n_edges > 0) {
    # Extract neighbor values for all edges: n_edges x n_years
    neighbor_vals_expanded <- V[to_vec, , drop = FALSE]

    # Now compute group max and min by from_vec (the "owning" cell)
    # Use data.table for fast grouped operations if available, else tapply.

    if (requireNamespace("data.table", quietly = TRUE)) {
      # Efficient grouped max/min with data.table
      dt <- data.table::as.data.table(neighbor_vals_expanded)
      dt[, grp := from_vec]

      max_dt <- dt[, lapply(.SD, function(x) {
        x <- x[!is.na(x)]
        if (length(x) == 0) NA_real_ else max(x)
      }), by = grp]
      min_dt <- dt[, lapply(.SD, function(x) {
        x <- x[!is.na(x)]
        if (length(x) == 0) NA_real_ else min(x)
      }), by = grp]

      # Ensure all cells are represented (some may have 0 neighbors)
      neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
      neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

      grp_max <- max_dt$grp
      grp_min <- min_dt$grp
      neighbor_max_mat[grp_max, ] <- as.matrix(max_dt[, -1, with = FALSE])
      neighbor_min_mat[grp_min, ] <- as.matrix(min_dt[, -1, with = FALSE])

    } else {
      # Fallback: loop over cells (344K iterations, no inner apply needed
      # if we use matrixStats)
      neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
      neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

      if (requireNamespace("matrixStats", quietly = TRUE)) {
        for (i in seq_len(n_cells)) {
          nb_idx <- cell_neighbor_idx[[i]]
          if (length(nb_idx) == 0L) next
          sub_mat <- V[nb_idx, , drop = FALSE]
          neighbor_max_mat[i, ] <- matrixStats::colMaxs(sub_mat, na.rm = TRUE)
          neighbor_min_mat[i, ] <- matrixStats::colMins(sub_mat, na.rm = TRUE)
        }
      } else {
        for (i in seq_len(n_cells)) {
          nb_idx <- cell_neighbor_idx[[i]]
          if (length(nb_idx) == 0L) next
          if (length(nb_idx) == 1L) {
            neighbor_max_mat[i, ] <- V[nb_idx, ]
            neighbor_min_mat[i, ] <- V[nb_idx, ]
          } else {
            sub_mat <- V[nb_idx, , drop = FALSE]
            neighbor_max_mat[i, ] <- apply(sub_mat, 2, max, na.rm = TRUE)
            neighbor_min_mat[i, ] <- apply(sub_mat, 2, min, na.rm = TRUE)
          }
        }
      }

      neighbor_max_mat[is.infinite(neighbor_max_mat)] <- NA
      neighbor_min_mat[is.infinite(neighbor_min_mat)] <- NA
    }
  } else {
    neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  }

  # --- Flatten to long format ---
  max_col_name  <- paste0("neighbor_max_",  var_name)
  min_col_name  <- paste0("neighbor_min_",  var_name)
  mean_col_name <- paste0("neighbor_mean_", var_name)

  # t() transposes to n_years x n_cells, as.vector reads column-major,
  # yielding: cell1_year1, cell1_year2, ..., cell1_yearT, cell2_year1, ...
  # which matches cell_data sorted by (id, year).
  cell_data[[max_col_name]]  <- as.vector(t(neighbor_max_mat))
  cell_data[[min_col_name]]  <- as.vector(t(neighbor_min_mat))
  cell_data[[mean_col_name]] <- as.vector(t(neighbor_mean_mat))

  cell_data
}

# ==============================================================================
# STEP 6: Run the outer loop
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_data <- compute_neighbor_features_fastest(
    cell_data, var_name, W, neighbor_count, cell_neighbor_idx,
    n_cells, n_years, id_order, years
  )
}

# ==============================================================================
# STEP 7: Predict with the pre-trained Random Forest (UNCHANGED)
# ==============================================================================
# The trained RF model object (e.g., `rf_model`) is loaded from disk.
# Prediction uses the same column names as before, so no changes needed.
#
# predictions <- predict(rf_model, newdata = cell_data)
#
# The numerical estimand is preserved because we compute identical
# max, min, and mean statistics â€” just via a different (faster) code path.
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup granularity** | Per cell-year row (6.46M) | Per cell (344K) â€” **static** |
| **Key mechanism** | String-pasted named vector lookup | Integer matrix indexing |
| **Mean computation** | `lapply` over 6.46M, per-element `mean()` | Sparse matrix multiply `W %*% V` (vectorized) |
| **Max/Min computation** | `lapply` over 6.46M, per-element `max()`/`min()` | Loop over 344K cells with matrix slicing, or data.table grouped ops |
| **Memory** | 6.46M-element list of integer vectors | ~344K-element list + dense matrices (344K Ã— 28 â‰ˆ 77MB per variable) â€” fits in 16 GB |
| **Estimated runtime** | 86+ hours | **Minutes** (sparse matmul is near-instant; 344K-iteration loop with small sub-matrices is fast) |
| **Numerical equivalence** | â€” | **Identical** max, min, mean values; same column names |
| **RF model** | â€” | **Untouched**; no retraining |
END_RESPONSE>>>
