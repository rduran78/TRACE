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
run_id: Anthropic_F12_T07_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized matrix assembly step per variable â€” only 5 calls total. That is not where 86+ hours is spent.

**The true bottleneck is `build_neighbor_lookup()`.**

Specifically, inside the `lapply` over all ~6.46 million rows:

```r
lapply(row_ids, function(i) {
  ref_idx           <- id_to_ref[as.character(data$id[i])]
  neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
  neighbor_keys     <- paste(neighbor_cell_ids, data$year[i], sep = "_")
  result            <- idx_lookup[neighbor_keys]
  as.integer(result[!is.na(result)])
})
```

For **each** of the 6.46 million rows, this function:

1. Converts an integer id to character and performs a named-vector lookup (`as.character` + named indexing): **O(1) amortized but with per-call overhead Ã— 6.46M**.
2. Subsets the neighbor list to get neighbor cell IDs.
3. Calls `paste()` to construct string keys for every neighbor of that row (average ~4 rook neighbors â†’ ~25.8 million `paste` calls).
4. Performs named-vector lookup on `idx_lookup` (a named vector of length 6.46M) for each key â€” **this is a repeated hash-table probe on a massive named vector, millions of times**.

The result is that `build_neighbor_lookup` executes roughly **6.46 million R-level function calls**, each doing string allocation, pasting, and named-vector lookups. This dwarfs the cost of `do.call(rbind, ...)` in `compute_neighbor_stats`, which runs only 5 times.

Furthermore, the neighbor lookup is **row-year-invariant per variable** â€” it is correctly built once â€” but the lookup construction itself is the bottleneck because it is implemented as a scalar R loop with expensive string operations over millions of iterations.

## Optimization Strategy

1. **Eliminate per-row string pasting and named-vector lookups entirely.** Instead of building string keys like `"cellid_year"` and looking them up in a named vector, exploit the panel structure: if data is sorted by `(id, year)` or we can build a fast integer-indexed matrix mapping `(cell_index, year_index) â†’ row_number`, then neighbor row indices can be computed via direct integer arithmetic â€” no strings, no hashing.

2. **Vectorize `compute_neighbor_stats`** using the precomputed sparse neighbor structure. Instead of `lapply` over 6.46M rows, represent the neighbor relationships as a sparse matrix and use matrix operations (sparse matrix Ã— dense column) to compute neighbor means, and row-wise sparse operations for min/max.

3. **Preserve the trained Random Forest model and the original numerical estimand.** The output columns must have identical names and identical numerical values (within floating-point tolerance) to the original pipeline.

## Working R Code

```r
library(data.table)
library(Matrix)

# ==============================================================
# STEP 1: Build a fast integer-indexed lookup structure
#         Eliminates all paste() and named-vector lookups.
# ==============================================================

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert data to data.table for fast operations
  dt <- as.data.table(data)
  dt[, row_idx := .I]
  
  # Create integer mappings
  # Map cell id -> sequential index (1..N_cells)
  cell_ids <- as.character(id_order)
  n_cells <- length(cell_ids)
  id_to_cell_idx <- setNames(seq_len(n_cells), cell_ids)
  
  # Map year -> sequential index (1..N_years)
  unique_years <- sort(unique(dt$year))
  n_years <- length(unique_years)
  year_to_year_idx <- setNames(seq_len(n_years), as.character(unique_years))
  
  # Build a matrix: row_position_matrix[cell_idx, year_idx] = row number in data
  # This replaces the named-vector idx_lookup entirely.
  dt[, cell_idx := id_to_cell_idx[as.character(id)]]
  dt[, year_idx := year_to_year_idx[as.character(year)]]
  
  row_position_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_position_matrix[cbind(dt$cell_idx, dt$year_idx)] <- dt$row_idx
  
  # Now build the neighbor lookup using integer indexing only.
  # For each row i with (cell_idx_i, year_idx_i), the neighbor rows are:
  #   row_position_matrix[ neighbors[[cell_idx_i]], year_idx_i ]
  # 
  # But instead of looping per row, we loop per cell and vectorize across years.
  
  n_rows <- nrow(dt)
  
  # Pre-allocate: store neighbor indices as a list of integer vectors (length n_rows)
  # But we will build this much more efficiently using cell-level iteration.
  
  # Group rows by cell_idx
  setkey(dt, cell_idx, year_idx)
  
  # For each cell, get its neighbor cell indices once, then for each year
  # that cell appears in, look up neighbor rows via the matrix.
  
  neighbor_from <- vector("list", n_rows)
  
  for (ci in seq_len(n_cells)) {
    nb_cell_indices <- neighbors[[ci]]
    if (length(nb_cell_indices) == 0) next
    
    # Which rows belong to this cell?
    cell_rows <- dt[cell_idx == ci]
    if (nrow(cell_rows) == 0) next
    
    for (j in seq_len(nrow(cell_rows))) {
      yi <- cell_rows$year_idx[j]
      ri <- cell_rows$row_idx[j]
      nb_rows <- row_position_matrix[nb_cell_indices, yi]
      nb_rows <- nb_rows[!is.na(nb_rows)]
      neighbor_from[[ri]] <- nb_rows
    }
  }
  
  neighbor_from
}

# ==============================================================
# STEP 2: Even faster â€” fully vectorized sparse-matrix approach
#         Eliminates the per-row loop in compute_neighbor_stats.
# ==============================================================

build_neighbor_sparse_and_lookup <- function(data, id_order, neighbors) {
  dt <- as.data.table(data)
  dt[, row_idx := .I]
  n_rows <- nrow(dt)
  
  # Integer mappings
  cell_ids <- as.character(id_order)
  n_cells <- length(cell_ids)
  id_to_cell_idx <- setNames(seq_len(n_cells), cell_ids)
  
  unique_years <- sort(unique(dt$year))
  n_years <- length(unique_years)
  year_to_year_idx <- setNames(seq_len(n_years), as.character(unique_years))
  
  dt[, cell_idx := id_to_cell_idx[as.character(id)]]
  dt[, year_idx := year_to_year_idx[as.character(year)]]
  
  # Build row_position_matrix[cell_idx, year_idx] -> row in data
  row_position_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_position_matrix[cbind(dt$cell_idx, dt$year_idx)] <- dt$row_idx
  
  # Build sparse adjacency matrix at the ROW level (n_rows x n_rows)
  # Entry (i, j) = 1 means row j is a neighbor of row i (same year, neighbor cell)
  
  # Collect all (from_row, to_row) pairs
  from_list <- vector("list", n_cells)
  to_list   <- vector("list", n_cells)
  
  for (ci in seq_len(n_cells)) {
    nb_cell_indices <- neighbors[[ci]]
    if (length(nb_cell_indices) == 0) next
    
    for (yi in seq_len(n_years)) {
      from_row <- row_position_matrix[ci, yi]
      if (is.na(from_row)) next
      
      to_rows <- row_position_matrix[nb_cell_indices, yi]
      to_rows <- to_rows[!is.na(to_rows)]
      if (length(to_rows) == 0) next
      
      from_list[[length(from_list) + 1L]] <- rep.int(from_row, length(to_rows))
      to_list[[length(to_list) + 1L]]     <- to_rows
    }
  }
  
  from_vec <- unlist(from_list)
  to_vec   <- unlist(to_list)
  
  # Sparse adjacency matrix (row-level)
  W <- sparseMatrix(
    i = from_vec, j = to_vec,
    x = 1, dims = c(n_rows, n_rows)
  )
  
  # Also build a simple list-based lookup for min/max (sparse mat can do mean easily)
  # We'll return both.
  
  # Build neighbor_lookup as list (fast integer method)
  neighbor_lookup <- vector("list", n_rows)
  # Split to_vec by from_vec
  ord <- order(from_vec)
  from_sorted <- from_vec[ord]
  to_sorted   <- to_vec[ord]
  breaks <- which(diff(from_sorted) != 0)
  starts <- c(1L, breaks + 1L)
  ends   <- c(breaks, length(from_sorted))
  unique_froms <- from_sorted[starts]
  
  for (k in seq_along(unique_froms)) {
    neighbor_lookup[[unique_froms[k]]] <- to_sorted[starts[k]:ends[k]]
  }
  
  list(
    neighbor_lookup = neighbor_lookup,
    W = W
  )
}

# ==============================================================
# STEP 3: Optimized compute_neighbor_stats using sparse matrix
#         for mean, and vectorized list ops for min/max.
# ==============================================================

compute_neighbor_stats_fast <- function(data, neighbor_lookup, W, var_name) {
  n <- nrow(data)
  vals <- data[[var_name]]
  
  # --- Neighbor mean via sparse matrix multiplication ---
  # Replace NAs with 0 for multiplication, track valid counts
  not_na <- as.numeric(!is.na(vals))
  vals_zero <- ifelse(is.na(vals), 0, vals)
  
  neighbor_sum   <- as.numeric(W %*% vals_zero)
  neighbor_count <- as.numeric(W %*% not_na)
  
  neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
  
  # --- Neighbor max and min via vectorized list operation ---
  neighbor_max <- rep(NA_real_, n)
  neighbor_min <- rep(NA_real_, n)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (is.null(idx) || length(idx) == 0) next
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0) next
    neighbor_max[i] <- max(nv)
    neighbor_min[i] <- min(nv)
  }
  
  cbind(neighbor_max, neighbor_min, neighbor_mean)
}

# ==============================================================
# STEP 4: Replacement for compute_and_add_neighbor_features
# ==============================================================

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup, W) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, W, var_name)
  
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]
  
  data
}

# ==============================================================
# STEP 5: Optimized outer pipeline (drop-in replacement)
# ==============================================================

# Build the sparse structure and lookup ONCE
message("Building optimized neighbor structures...")
system.time({
  nb_struct <- build_neighbor_sparse_and_lookup(
    cell_data, id_order, rook_neighbors_unique
  )
})

neighbor_lookup <- nb_struct$neighbor_lookup
W               <- nb_struct$W

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor features for: %s", var_name))
  system.time({
    cell_data <- compute_and_add_neighbor_features_fast(
      cell_data, var_name, neighbor_lookup, W
    )
  })
}

# The trained Random Forest model is untouched.
# Column names and numerical values are identical to the original pipeline.
```

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| **Alleged bottleneck** | `do.call(rbind, ...)` and list binding in `compute_neighbor_stats()` | **Wrong.** Only 5 calls, each assembling a 6.46M Ã— 3 matrix â€” seconds at most. |
| **True bottleneck** | â€” | `build_neighbor_lookup()`: 6.46M iterations of `paste()` + named-vector hash probes on a 6.46M-entry named vector. This is **O(N_rows Ã— avg_neighbors)** with heavy per-element string allocation overhead. |
| **Secondary bottleneck** | â€” | `compute_neighbor_stats()` loop itself (6.46M R-level function calls per variable Ã— 5 variables), though less severe than the lookup construction. |

**Key optimizations applied:**

1. **Replaced string-key lookups with integer matrix indexing** (`row_position_matrix[cell_idx, year_idx]`), eliminating all `paste()` and named-vector probes â€” reduces `build_neighbor_lookup` from hours to minutes.
2. **Sparse matrix multiplication for neighbor means** â€” replaces 6.46M R-level iterations with a single `W %*% vals` call per variable, leveraging optimized BLAS/sparse routines.
3. **Single construction, reused across all 5 variables** â€” the sparse matrix `W` and `neighbor_lookup` list are built once and reused, amortizing the setup cost.

Expected runtime reduction: from **86+ hours to under 1 hour** on the same 16 GB laptop.
END_RESPONSE>>>
