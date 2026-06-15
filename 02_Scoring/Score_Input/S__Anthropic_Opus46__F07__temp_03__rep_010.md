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
run_id: Anthropic_F07_T03_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates a list of ~6.46 million elements, each built by character-key lookups inside an `lapply` over every row. Specifically:

1. **`paste` + named-vector lookup per row**: For each of the 6.46M rows, the code converts neighbor cell IDs to character keys (`paste(id, year, sep="_")`), then does named-vector indexing (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is O(n) per query in the worst case and has high constant overhead due to string hashing repeated millions of times.

2. **Row-wise `lapply` over 6.46M rows**: Even if each iteration is fast, the R-level loop overhead for 6.46 million iterations is enormous. The estimated 86+ hours is dominated by this function.

3. **`compute_neighbor_stats` is also row-wise** but is comparatively cheaper since it just indexes a numeric vector. Still, it runs 6.46M Ã— 5 = 32.3M iterations total.

4. **Memory**: Building a 6.46M-element list of integer vectors is memory-heavy but fits in 16 GB.

**Root cause**: The problem is fundamentally a **sparse-matrixâ€“vector multiply** (and element-wise min/max), but it's implemented as a sequential R-level loop with string operations.

---

## Optimization Strategy

### Key Insight
Every cell's neighbors are **the same across all 28 years**. The neighbor topology is purely spatial. So we should:

1. **Build a sparse adjacency matrix `W` once** (344,208 Ã— 344,208, ~1.37M non-zero entries) using the `Matrix` package.
2. **Reshape each variable into a matrix** of dimension (344,208 cells Ã— 28 years).
3. **Compute neighbor stats via sparse matrix operations**:
   - **Neighbor mean**: `W %*% X / degree` (one sparse mat-mul, milliseconds).
   - **Neighbor max / min**: Use a grouped operation over the sparse structure â€” iterate over columns of `W` in C-level code via `Matrix` internals or a small Rcpp function.

This replaces 6.46M Ã— 5 R-level iterations with 5 sparse matrix multiplies (for mean) and 5 vectorized grouped operations (for max/min). Expected runtime: **seconds to low minutes** instead of 86+ hours.

### Why this preserves the estimand
- The sparse matrix `W` encodes **exactly** the same rook-neighbor relationships as `rook_neighbors_unique`.
- The numerical operations (max, min, mean of neighbor values) are identical.
- The trained Random Forest model is untouched â€” we only recompute the same input features faster.

---

## Working R Code

```r
# ==============================================================================
# FAST NEIGHBOR FEATURE COMPUTATION
# Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop
# ==============================================================================

library(Matrix)
library(data.table)

# --------------------------------------------------------------------------
# Step 0: Convert cell_data to data.table for fast manipulation
# --------------------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# Ensure consistent ordering: create a cell index and year index
# id_order is the vector of cell IDs in the same order as rook_neighbors_unique
cell_id_map <- setNames(seq_along(id_order), as.character(id_order))
n_cells     <- length(id_order)

# Identify the unique years and create a year index
years       <- sort(unique(cell_dt$year))
n_years     <- length(years)
year_map    <- setNames(seq_along(years), as.character(years))

# Add integer indices for cell and year
cell_dt[, cell_idx := cell_id_map[as.character(id)]]
cell_dt[, year_idx := year_map[as.character(year)]]

# --------------------------------------------------------------------------
# Step 1: Build sparse adjacency matrix W from rook_neighbors_unique (nb object)
# --------------------------------------------------------------------------
# rook_neighbors_unique is a list of length n_cells;
# rook_neighbors_unique[[i]] is an integer vector of neighbor indices (into id_order)

build_sparse_adjacency <- function(nb_obj, n) {
  # Pre-allocate vectors for triplet representation
  # Count total neighbors
  total_nb <- sum(vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  from_vec <- integer(total_nb)
  to_vec   <- integer(total_nb)
  
  pos <- 1L
  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    # spdep::nb uses 0 to denote no neighbors
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    k <- length(nbrs)
    from_vec[pos:(pos + k - 1L)] <- i
    to_vec[pos:(pos + k - 1L)]   <- nbrs
    pos <- pos + k
  }
  
  sparseMatrix(
    i    = from_vec,
    j    = to_vec,
    x    = rep(1, total_nb),
    dims = c(n, n)
  )
}

W <- build_sparse_adjacency(rook_neighbors_unique, n_cells)

# Degree vector (number of neighbors per cell) â€” used for mean
degree_vec <- as.numeric(rowSums(W))  # length n_cells

# --------------------------------------------------------------------------
# Step 2: For each variable, reshape to (n_cells x n_years) matrix,
#         compute neighbor max, min, mean, and write back
# --------------------------------------------------------------------------

# Ensure cell_dt is keyed for fast assignment
setkey(cell_dt, cell_idx, year_idx)

# We need a complete (cell_idx, year_idx) grid to form the matrix.
# If some cell-years are missing, we handle with NA.

# Create the matrix from cell_dt for a given variable
make_cell_year_matrix <- function(dt, var_name, n_cells, n_years) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]
  mat
}

# --------------------------------------------------------------------------
# Neighbor MEAN via sparse matrix multiply
# --------------------------------------------------------------------------
compute_neighbor_mean_matrix <- function(W, X_mat, degree_vec) {
  # W %*% X_mat gives sum of neighbor values for each cell-year
  sum_mat <- as.matrix(W %*% X_mat)
  # Divide by degree; cells with 0 neighbors get NA
  mean_mat <- sum_mat / degree_vec
  mean_mat[degree_vec == 0, ] <- NA_real_
  mean_mat
}

# --------------------------------------------------------------------------
# Neighbor MAX and MIN via sparse structure
# Uses the column-compressed structure of W to avoid R-level row loops
# --------------------------------------------------------------------------
compute_neighbor_minmax_matrix <- function(W, X_mat) {
  # W is n_cells x n_cells sparse (dgCMatrix, column-compressed)
  # For row-wise operations, convert to dgRMatrix (row-compressed) or
  # use the transpose trick: t(W) is column-compressed where column j
  # holds the neighbors of cell j... but we want row i's neighbors.
  #
  # Strategy: iterate over the sparse structure efficiently.
  # Convert to dgTMatrix (triplet) for simple grouped operations via data.table
  
  Wt <- as(W, "TsparseMatrix")  # i, j are 0-based
  
  from_cell <- Wt@i + 1L  # row indices (1-based) = the focal cell
  to_cell   <- Wt@j + 1L  # col indices (1-based) = the neighbor cell
  
  n_cells <- nrow(X_mat)
  n_years <- ncol(X_mat)
  
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # For each year, compute grouped max and min using data.table
  for (t in seq_len(n_years)) {
    # Get neighbor values for this year
    nb_vals <- X_mat[to_cell, t]
    
    dt_tmp <- data.table(
      focal = from_cell,
      val   = nb_vals
    )
    
    # Remove NA neighbor values before aggregation
    dt_tmp <- dt_tmp[!is.na(val)]
    
    if (nrow(dt_tmp) == 0L) next
    
    agg <- dt_tmp[, .(nb_max = max(val), nb_min = min(val)), by = focal]
    
    max_mat[agg$focal, t] <- agg$nb_max
    min_mat[agg$focal, t] <- agg$nb_min
  }
  
  list(max_mat = max_mat, min_mat = min_mat)
}

# --------------------------------------------------------------------------
# Step 3: Main loop over the 5 neighbor source variables
# --------------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Processing neighbor features for: ", var_name)
  
  # Reshape to cell x year matrix
  X_mat <- make_cell_year_matrix(cell_dt, var_name, n_cells, n_years)
  
  # Compute neighbor mean (sparse mat-mul â€” very fast)
  mean_mat <- compute_neighbor_mean_matrix(W, X_mat, degree_vec)
  
  # Compute neighbor max and min
  minmax   <- compute_neighbor_minmax_matrix(W, X_mat)
  max_mat  <- minmax$max_mat
  min_mat  <- minmax$min_mat
  
  # Write results back to cell_dt using the (cell_idx, year_idx) indices
  idx_matrix <- cbind(cell_dt$cell_idx, cell_dt$year_idx)
  
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  set(cell_dt, j = col_max,  value = max_mat[idx_matrix])
  set(cell_dt, j = col_min,  value = min_mat[idx_matrix])
  set(cell_dt, j = col_mean, value = mean_mat[idx_matrix])
  
  # Free memory
  rm(X_mat, mean_mat, max_mat, min_mat, minmax)
  gc()
}

# --------------------------------------------------------------------------
# Step 4: Clean up helper columns and convert back to data.frame if needed
# --------------------------------------------------------------------------
cell_dt[, c("cell_idx", "year_idx") := NULL]

cell_data <- as.data.frame(cell_dt)

message("Done. All neighbor features computed.")
```

---

## Performance Summary

| Component | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M R-level string lookups (86+ hrs) | One-time sparse matrix build (~seconds) |
| **Neighbor mean** | Row-wise `lapply` Ã— 5 vars | `W %*% X_mat` sparse mat-mul Ã— 5 (~seconds each) |
| **Neighbor max/min** | Row-wise `lapply` Ã— 5 vars | 28 grouped `data.table` aggregations Ã— 5 vars (~minutes total) |
| **Total estimated time** | **86+ hours** | **~2â€“10 minutes** |
| **Peak RAM** | ~6.46M-element list + data | Sparse matrix (1.37M entries) + dense 344KÃ—28 matrices (~77 MB each) â€” fits in 16 GB |

### Why the estimand is preserved
- `W` is constructed directly from `rook_neighbors_unique` â€” identical neighbor sets.
- `max`, `min`, `mean` are computed on exactly the same neighbor value sets, with the same NA handling.
- The Random Forest model is never touched â€” only its input features are recomputed identically but faster.
END_RESPONSE>>>
