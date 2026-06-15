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
run_id: Anthropic_F08_T07_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) neighbor indices.** It creates a list of ~6.46 million entries, each requiring string-pasting (`paste(id, year)`), hash-table lookups (`idx_lookup[neighbor_keys]`), and NA filtering. But the neighbor *relationships* are purely spatial â€” cell A neighbors cell B regardless of year. The year dimension is only needed to locate the correct *row* for value extraction. This means the function does O(rows Ã— avg_neighbors) string operations (~6.46M Ã— ~4 = ~25.8M string-match lookups) when only O(cells Ã— avg_neighbors) = ~1.37M topology lookups are needed.

2. **`compute_neighbor_stats` operates on the flattened cellÃ—year vector.** Because the lookup was built on the flattened row space, it works â€” but it was unnecessary to flatten the topology. The values change by year, but the *which-cells-are-neighbors* structure does not.

3. **Memory pressure from the 6.46M-element list.** Each element is an integer vector of neighbor row indices. This list alone consumes substantial RAM and causes GC thrashing on a 16 GB machine.

4. **The `lapply` over 6.46M rows in `compute_neighbor_stats`** is called 5 times (once per variable), totaling ~32.3M R-level function calls with per-element allocation.

### The Key Insight

> **Neighbor topology is static across years; only the attached variable values change by year.**

This means we should:
- Build the neighbor lookup **once, over 344,208 cells** (not 6.46M rows).
- For each variable and each year, extract values for that year's slice and compute stats using the **cell-level** lookup.

This reduces the lookup construction cost by **~18.8Ã—** (28 years) and enables vectorized, year-sliced computation.

---

## Optimization Strategy

### Step 1: Build a Cell-Level Neighbor Lookup (Once)

Construct a list of length 344,208 where element `i` contains the integer indices of cell `i`'s neighbors in `id_order` space. This is derived directly from `rook_neighbors_unique` (the `nb` object) â€” it already *is* this structure. No string operations needed.

### Step 2: Precompute a Cell-to-Row Index Matrix

Create a matrix of dimension `(n_cells Ã— n_years)` mapping each `(cell_index, year_index)` to its row in `cell_data`. This enables O(1) row lookup for any cell in any year.

### Step 3: Vectorized Neighbor Stats by Year

For each year, extract the variable column for all cells in that year (a vector of length ~344,208). Then for each cell, gather neighbor values using the cell-level lookup and compute max/min/mean. This is done with a fast `vapply` over 344,208 cells (not 6.46M rows), repeated for 28 years â€” totaling ~9.6M iterations per variable instead of ~6.46M, but each iteration is far cheaper (no string ops, direct integer indexing into a short vector).

### Step 4: Further Vectorization via Sparse Matrix

Convert the cell-level neighbor list into a sparse adjacency matrix. Then for each year-slice vector `v`:
- **Neighbor mean** = `(A %*% v) / (A %*% ones)` (sparse matrix-vector multiply)
- **Neighbor max/min** = row-wise max/min over neighbor values (use the sparse structure)

Sparse matrix-vector multiply for mean is O(nnz) â‰ˆ 1.37M operations per year â€” essentially instant. Max and min require a grouped operation but can be accelerated.

### Expected Speedup

| Component | Current | Optimized |
|---|---|---|
| Lookup construction | ~6.46M string ops | ~344K integer ops (from `nb` object directly) |
| Stats per variable | 6.46M `lapply` iterations | 28 sparse mat-vec ops (mean) + 28 Ã— 344K grouped ops (max/min) |
| Total function calls | ~32.3M | ~48.2M integer ops but vectorized |
| Estimated time | ~86+ hours | **~5â€“15 minutes** |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Exploits the static-topology / dynamic-variable distinction
# =============================================================================

library(Matrix)
library(data.table)

# ---- Step 1: Build cell-level sparse adjacency matrix (ONCE) ----------------
# rook_neighbors_unique is an nb object: a list of length n_cells,
# where element i is an integer vector of neighbor indices into id_order.
# This IS the static topology. We convert it to a sparse matrix for speed.

build_cell_adjacency <- function(nb_object, n_cells) {
  # nb_object: list of integer vectors (neighbor indices), length = n_cells
  # Returns: a sparse logical/binary adjacency matrix (n_cells x n_cells)
  
  from <- rep(seq_len(n_cells), times = lengths(nb_object))
  to   <- unlist(nb_object)
  
  # Remove the 0-neighbor sentinel if present (spdep uses 0L for no neighbors)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  sparseMatrix(i = from, j = to, x = 1, dims = c(n_cells, n_cells))
}

# ---- Step 2: Build cell-year indexing structure (ONCE) ----------------------
# We need to map (cell_index, year_index) -> row in cell_data.
# Use data.table for fast setup.

build_cell_year_index <- function(cell_data, id_order) {
  # Returns a list with:
  #   years       : sorted unique years
  #   cell_idx_map: named integer vector mapping cell id -> cell index (1..n_cells)
  #   row_matrix  : matrix (n_cells x n_years), entry = row index in cell_data, or NA
  
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]
  
  years <- sort(unique(dt$year))
  n_cells <- length(id_order)
  n_years <- length(years)
  
  cell_idx_map <- setNames(seq_along(id_order), as.character(id_order))
  year_idx_map <- setNames(seq_along(years), as.character(years))
  
  # Map each row to (cell_index, year_index)
  dt[, cell_idx := cell_idx_map[as.character(id)]]
  dt[, year_idx := year_idx_map[as.character(year)]]
  
  # Build the row matrix
  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_matrix[cbind(dt$cell_idx, dt$year_idx)] <- dt$row_idx
  
  list(
    years        = years,
    n_cells      = n_cells,
    n_years      = n_years,
    cell_idx_map = cell_idx_map,
    year_idx_map = year_idx_map,
    row_matrix   = row_matrix
  )
}

# ---- Step 3: Compute neighbor stats using sparse matrix ops -----------------

compute_neighbor_features_optimized <- function(cell_data, var_name, 
                                                 adj_matrix, cy_index,
                                                 nb_object) {
  # Computes neighbor max, min, mean for one variable across all cell-years.
  # Returns a matrix (nrow(cell_data) x 3): [max, min, mean]
  
  n_rows  <- nrow(cell_data)
  n_cells <- cy_index$n_cells
  n_years <- cy_index$n_years
  row_mat <- cy_index$row_matrix  # (n_cells x n_years)
  
  # Pre-allocate output columns
  out_max  <- rep(NA_real_, n_rows)
  out_min  <- rep(NA_real_, n_rows)
  out_mean <- rep(NA_real_, n_rows)
  
  # Precompute neighbor count per cell (static)
  ones_vec     <- rep(1, n_cells)
  neighbor_cnt <- as.numeric(adj_matrix %*% ones_vec)  # length n_cells
  
  # Process each year independently
  for (yr_idx in seq_len(n_years)) {
    
    # Which rows in cell_data correspond to this year?
    row_indices <- row_mat[, yr_idx]  # length n_cells; NA if cell absent this year
    
    # Build the value vector for all cells this year
    # Cells without data this year get NA
    vals <- rep(NA_real_, n_cells)
    present <- !is.na(row_indices)
    vals[present] <- cell_data[[var_name]][row_indices[present]]
    
    # ---- Neighbor MEAN via sparse matrix-vector multiply ----
    # Replace NA with 0 for sum, and track which are non-NA for correct count
    vals_for_sum <- vals
    vals_for_sum[is.na(vals_for_sum)] <- 0
    
    non_na_indicator <- as.numeric(!is.na(vals))
    
    neighbor_sum     <- as.numeric(adj_matrix %*% vals_for_sum)     # sum of non-NA neighbor vals
    neighbor_non_na  <- as.numeric(adj_matrix %*% non_na_indicator) # count of non-NA neighbors
    
    year_mean <- ifelse(neighbor_non_na > 0, neighbor_sum / neighbor_non_na, NA_real_)
    
    # ---- Neighbor MAX and MIN via grouped operations on nb_object ----
    # This is the part that can't be fully vectorized with standard sparse ops.
    # We use a fast vapply over cells, but only 344K iterations (not 6.46M).
    
    year_max <- rep(NA_real_, n_cells)
    year_min <- rep(NA_real_, n_cells)
    
    # Only compute for cells that are present this year
    cells_to_compute <- which(present)
    
    # Batch approach: iterate over cells using the nb_object directly
    for (ci in cells_to_compute) {
      nb_idx <- nb_object[[ci]]
      if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) next
      nb_vals <- vals[nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0L) next
      year_max[ci] <- max(nb_vals)
      year_min[ci] <- min(nb_vals)
    }
    
    # Write results back to the output vectors at the correct row positions
    rows_this_year <- row_indices[present]
    cells_present  <- which(present)
    
    out_max[rows_this_year]  <- year_max[cells_present]
    out_min[rows_this_year]  <- year_min[cells_present]
    out_mean[rows_this_year] <- year_mean[cells_present]
  }
  
  cbind(out_max, out_min, out_mean)
}

# ---- Step 3b: Even faster max/min using Rcpp-style vectorization -----------
# If the for-loop over 344K cells per year is still too slow, we can use
# a pre-sorted edge list approach. But first, let's try a vapply version:

compute_neighbor_maxmin_fast <- function(vals, nb_object, present_cells) {
  # vals: numeric vector length n_cells (NA for absent)
  # nb_object: list of neighbor index vectors
  # present_cells: integer vector of cell indices to compute
  # Returns: matrix (n_cells x 2) with max, min
  
  n <- length(vals)
  out <- matrix(NA_real_, nrow = n, ncol = 2)
  
  result <- vapply(present_cells, function(ci) {
    nb_idx <- nb_object[[ci]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(c(NA_real_, NA_real_))
    }
    nb_vals <- vals[nb_idx]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) == 0L) return(c(NA_real_, NA_real_))
    c(max(nb_vals), min(nb_vals))
  }, numeric(2))
  
  # result is 2 x length(present_cells)
  out[present_cells, 1] <- result[1, ]
  out[present_cells, 2] <- result[2, ]
  out
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# --- One-time static topology setup ---
n_cells <- length(id_order)
adj_matrix <- build_cell_adjacency(rook_neighbors_unique, n_cells)
cy_index   <- build_cell_year_index(cell_data, id_order)

cat("Static topology setup complete.\n")
cat(sprintf("  Cells: %d | Years: %d | Rows: %d | Non-zero edges: %d\n",
            cy_index$n_cells, cy_index$n_years, nrow(cell_data), nnzero(adj_matrix)))

# --- Compute neighbor features for each source variable ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for '%s'...\n", var_name))
  t0 <- proc.time()
  
  stats <- compute_neighbor_features_optimized(
    cell_data, var_name, adj_matrix, cy_index, rook_neighbors_unique
  )
  
  # Add columns with the same naming convention as the original code
  cell_data[[paste0("neighbor_max_", var_name)]]  <- stats[, 1]
  cell_data[[paste0("neighbor_min_", var_name)]]  <- stats[, 2]
  cell_data[[paste0("neighbor_mean_", var_name)]] <- stats[, 3]
  
  elapsed <- (proc.time() - t0)[3]
  cat(sprintf("  Done in %.1f seconds.\n", elapsed))
}

cat("All neighbor features computed. Ready for Random Forest prediction.\n")

# --- Predict with pre-trained Random Forest (unchanged) ---
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

1. **Identical neighbor sets.** The `nb` object (`rook_neighbors_unique`) defines which cells are neighbors. The original code looked up the same neighbor cell IDs per cell â€” we use the same `nb` object indices directly. No neighbor relationship is added, removed, or reordered.

2. **Identical aggregation functions.** `max`, `min`, and `mean` are computed on exactly the same set of non-NA neighbor values per cell-year as the original code. The sparse matrix multiply for `mean` computes `sum(neighbor_vals) / count(non_na_neighbors)` â€” algebraically identical to `mean(neighbor_vals[!is.na(neighbor_vals)])`.

3. **NA handling is preserved.** Cells with no neighbors or all-NA neighbors produce `NA` for all three statistics, exactly as in the original.

4. **The trained Random Forest model is never modified.** The output column names match the original naming convention, so `predict()` works identically.

### Performance Summary

| Metric | Original | Optimized |
|---|---|---|
| Lookup construction | ~6.46M string-paste + hash lookups | ~1.37M sparse matrix entries (once) |
| Iterations per variable | ~6.46M (R-level lapply) | 28 Ã— sparse mat-vec + 28 Ã— ~344K vapply |
| Total estimated time | ~86+ hours | **~5â€“20 minutes** |
| Peak memory | ~6.46M-element list + copies | ~344K-element nb list + sparse matrix (~11MB) |
END_RESPONSE>>>
