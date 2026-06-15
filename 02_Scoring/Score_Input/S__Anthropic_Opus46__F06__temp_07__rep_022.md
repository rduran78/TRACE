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
run_id: Anthropic_F06_T07_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (via `id_to_ref` and the `nb` object).
2. Constructs character keys by pasting `neighbor_id` and `year`.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) using character matching on a 6.46-million-element named vector.

Character-key lookups on a vector of length ~6.46M are **O(n)** per call in the worst case (R's named vector lookup is hash-based, but the sheer volumeâ€”6.46M calls each touching multiple keysâ€”is devastating). The result is a **list of 6.46 million integer vectors**, which is also memory-heavy.

### Bottleneck B: `compute_neighbor_stats` â€” `lapply` over 6.46 million entries

For each of 5 variables, this iterates over 6.46M list elements, extracts neighbor values, computes `max/min/mean`, and returns a 3-column matrix. That's 5 Ã— 6.46M = ~32.3 million R-level function calls with per-element allocation.

### Why raster focal/kernel operations are not directly applicable

Raster focal operations (e.g., `terra::focal`) assume a **regular rectangular grid with uniform cell sizes and complete spatiotemporal coverage**. If the panel is a regular grid, focal operations could work for a single time sliceâ€”but the panel structure (28 years stacked in long format) means you'd need to reshape to raster, apply focal per year, and reshape back. More critically, the neighbor structure is **precomputed as an `nb` object**, which may encode irregular boundaries, missing cells, or coastal/border exclusions. Forcing this into a focal kernel risks **changing the numerical estimand**. So we use the `nb` object directly but with vectorized operations.

---

## 2. Optimization Strategy

### Strategy: Fully vectorized sparse-matrix multiplication

The key insight: computing `max`, `min`, and `mean` of neighbor values is a **sparse matrixâ€“vector operation**. We can:

1. **Build a sparse adjacency matrix `W`** (dimensions: 6.46M Ã— 6.46M) from the `nb` object and the panel structureâ€”but this is too large. Instead, we build a **spatial** sparse adjacency matrix (344,208 Ã— 344,208) and exploit the fact that neighbor relationships are **time-invariant**: cell `i`'s neighbors in year `t` are the same cells in year `t`.

2. **Reshape each variable into a matrix** of dimensions (344,208 cells Ã— 28 years), so each column is one year.

3. **Use sparse matrix operations** on the spatial adjacency matrix (344K Ã— 344K) to compute neighbor means directly: `W %*% X / degree`. For max and min, use grouped operations via `data.table`.

4. This reduces the problem from 6.46M R-level iterations to **28 sparse matrixâ€“vector multiplies** on a 344K Ã— 344K matrix (for mean) and vectorized grouped operations (for max/min).

### Expected speedup

- `build_neighbor_lookup`: **eliminated entirely** (replaced by one sparse matrix construction, ~seconds).
- `compute_neighbor_stats`: replaced by sparse matrix multiply (mean) and vectorized group-by (max/min). Each variable takes seconds instead of hours.
- **Total estimated time: 1â€“5 minutes** instead of 86+ hours.

### Preserving the estimand

- The sparse matrix encodes exactly the same rook-neighbor relationships from `rook_neighbors_unique`.
- `max`, `min`, `mean` are computed over exactly the same neighbor sets.
- The trained Random Forest model is not retrained; we only compute the same input features faster.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# Requirements: data.table, Matrix, spdep (for the nb object)
# 
# Inputs:
#   cell_data              â€” data.frame/data.table with columns: id, year, and
#                            the 5 neighbor source variables
#   id_order               â€” integer/character vector of unique cell IDs in the
#                            order matching rook_neighbors_unique
#   rook_neighbors_unique  â€” spdep::nb object (list of integer index vectors)
#
# This code produces EXACTLY the same numerical output as the original
# build_neighbor_lookup + compute_neighbor_stats pipeline.
# =============================================================================

library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # --- Convert to data.table for speed (non-destructive copy) ---------------
  dt <- as.data.table(cell_data)
  
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)
  
  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(dt)))
  
  # =========================================================================
  # STEP 1: Build spatial sparse adjacency matrix (n_cells x n_cells)
  # =========================================================================
  # This encodes the rook neighbor relationships from the nb object.
  # Entry W[i,j] = 1 means cell j is a rook neighbor of cell i.
  
  cat("Building sparse adjacency matrix...\n")
  
  # Construct COO (coordinate) triplets
  from_idx <- integer(0)
  to_idx   <- integer(0)
  
  for (i in seq_along(rook_neighbors_unique)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    nb_i <- nb_i[nb_i > 0L]
    if (length(nb_i) > 0) {
      from_idx <- c(from_idx, rep(i, length(nb_i)))
      to_idx   <- c(to_idx, nb_i)
    }
  }
  
  W <- sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = 1,
    dims = c(n_cells, n_cells)
  )
  
  # Degree vector (number of neighbors per cell) for computing means
  degree <- rowSums(W)  # dense vector of length n_cells
  
  cat(sprintf("Adjacency matrix: %d x %d, %d non-zero entries\n",
              nrow(W), ncol(W), nnzero(W)))
  
  rm(from_idx, to_idx)
  
  # =========================================================================
  # STEP 2: Create a mapping from (id, year) -> row position in the
  #         cell-by-year matrix
  # =========================================================================
  # We need each cell to have a consistent row index in our matrices.
  
  cat("Building cell-year index maps...\n")
  
  # Map cell id -> spatial index (1..n_cells)
  id_to_sidx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map year -> temporal index (1..n_years)
  year_to_tidx <- setNames(seq_along(years), as.character(years))
  
  # Add spatial and temporal indices to dt
  dt[, sidx := id_to_sidx[as.character(id)]]
  dt[, tidx := year_to_tidx[as.character(year)]]
  
  # Ensure dt is sorted by (sidx, tidx) for consistent matrix filling
  setkey(dt, sidx, tidx)
  
  # =========================================================================
  # STEP 3: For each variable, reshape to matrix, compute neighbor stats
  # =========================================================================
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    
    cat(sprintf("Processing variable: %s\n", var_name))
    t0 <- proc.time()
    
    # --- 3a: Build cell-by-year matrix (n_cells x n_years) ----------------
    # Fill with NA by default (handles missing cell-year combos)
    
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(dt$sidx, dt$tidx)] <- dt[[var_name]]
    
    # --- 3b: Compute neighbor MEAN via sparse matrix multiply -------------
    # For each year t: mean_neighbors[i,t] = (W %*% X[,t])[i] / degree[i]
    # W %*% X computes the sum of neighbor values for each cell, each year.
    # We handle NAs by computing the sum of non-NA neighbors and dividing
    # by the count of non-NA neighbors (not the total degree).
    
    # Indicator of non-NA
    X_notna <- (!is.na(X)) * 1.0  # n_cells x n_years, 1 where not NA, 0 where NA
    
    # Replace NA with 0 for the matrix multiply (NAs don't contribute to sum)
    X_zero <- X
    X_zero[is.na(X_zero)] <- 0
    
    # Sum of neighbor values (NA treated as 0, but we correct with count)
    neighbor_sum   <- as.matrix(W %*% X_zero)   # n_cells x n_years
    
    # Count of non-NA neighbors
    neighbor_count <- as.matrix(W %*% X_notna)   # n_cells x n_years
    
    # Mean = sum / count (where count > 0)
    neighbor_mean <- neighbor_sum / neighbor_count
    neighbor_mean[neighbor_count == 0] <- NA_real_
    
    # --- 3c: Compute neighbor MAX and MIN ---------------------------------
    # Sparse matrix multiply doesn't directly give max/min.
    # Strategy: use the COO representation of W to do a vectorized group-by.
    #
    # For each year, we need: for each cell i, max/min of X[neighbors_of_i, year]
    #
    # We vectorize across all years simultaneously using data.table.
    
    # Extract the (from, to) pairs from W
    W_coo <- summary(W)  # gives (i, j, x) triplets
    # W_coo$i = "from" cell (the cell whose neighbors we want)
    # W_coo$j = "to" cell (the neighbor)
    
    n_edges <- nrow(W_coo)
    
    # For each edge (i -> j), extract X[j, ] for all years
    # This creates a matrix: n_edges x n_years
    neighbor_vals_mat <- X[W_coo$j, , drop = FALSE]  # n_edges x n_years
    
    # We need to group by W_coo$i and compute max/min across each group
    # Convert to long format using data.table for efficient grouped operations
    
    # But n_edges * n_years could be ~38M, which is manageable.
    # Actually, n_edges ~ 1.37M, n_years = 28, so ~38.4M rows in long format.
    # That's fine for 16GB RAM.
    
    # More efficient: loop over years (only 28 iterations)
    neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    edge_from <- W_coo$i  # reuse across years
    
    for (t in seq_len(n_years)) {
      vals_t <- X[W_coo$j, t]  # neighbor values for this year, one per edge
      
      # Remove NAs: create a data.table of (from_cell, value) and group
      valid <- !is.na(vals_t)
      if (sum(valid) == 0) next
      
      edge_dt <- data.table(
        from = edge_from[valid],
        val  = vals_t[valid]
      )
      
      # Grouped max and min
      agg <- edge_dt[, .(nb_max = max(val), nb_min = min(val)), by = from]
      
      neighbor_max[agg$from, t] <- agg$nb_max
      neighbor_min[agg$from, t] <- agg$nb_min
    }
    
    # --- 3d: Map results back to dt rows ----------------------------------
    # dt has (sidx, tidx) for each row, so we index directly.
    
    idx_mat <- cbind(dt$sidx, dt$tidx)
    
    max_col <- paste0("max_nb_", var_name)
    min_col <- paste0("min_nb_", var_name)
    mean_col <- paste0("mean_nb_", var_name)
    
    dt[, (max_col)  := neighbor_max[idx_mat]]
    dt[, (min_col)  := neighbor_min[idx_mat]]
    dt[, (mean_col) := neighbor_mean[idx_mat]]
    
    elapsed <- (proc.time() - t0)["elapsed"]
    cat(sprintf("  Done in %.1f seconds\n", elapsed))
    
    # Free memory
    rm(X, X_notna, X_zero, neighbor_sum, neighbor_count, neighbor_mean,
       neighbor_max, neighbor_min, neighbor_vals_mat, edge_dt, agg)
    gc()
  }
  
  # =========================================================================
  # STEP 4: Clean up helper columns and return
  # =========================================================================
  
  dt[, c("sidx", "tidx") := NULL]
  
  cat("All neighbor features computed.\n")
  return(dt)
}


# =============================================================================
# USAGE
# =============================================================================
# 
# # Load your data
# cell_data              <- readRDS("cell_data.rds")
# id_order               <- readRDS("id_order.rds")
# rook_neighbors_unique  <- readRDS("rook_neighbors_unique.rds")
# rf_model               <- readRDS("trained_rf_model.rds")  # NOT retrained
# 
# # Compute optimized neighbor features
# cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique)
# 
# # Now cell_data has the same columns as the original pipeline:
# #   max_nb_ntl, min_nb_ntl, mean_nb_ntl,
# #   max_nb_ec,  min_nb_ec,  mean_nb_ec,
# #   ... etc. for all 5 variables (15 new columns total)
# 
# # Predict with the existing trained Random Forest (unchanged)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M character-key lookups in a named vector | Single sparse matrix construction (~seconds) |
| **Mean computation** | `lapply` over 6.46M elements Ã— 5 vars | Sparse matrix multiply `W %*% X` (28 columns at once) |
| **Max/Min computation** | `lapply` with per-element `max()`/`min()` | `data.table` grouped aggregation, 28 year-loop iterations |
| **Memory** | 6.46M-element list of integer vectors (~GBs) | 344K Ã— 344K sparse matrix (~5.5 MB) + 344K Ã— 28 dense matrices (~77 MB each) |
| **Estimated runtime** | 86+ hours | **~2â€“5 minutes** |
| **Numerical output** | Baseline | **Identical** â€” same neighbor sets, same `max`/`min`/`mean` |
| **RF model** | Pre-trained | **Untouched** â€” only features are recomputed |

The sparse matrix `W` encodes exactly the same directed rook-neighbor relationships as `rook_neighbors_unique`. The NA-handling logic (skip NAs, return NA when no valid neighbors exist) is preserved. The 15 output columns (`max_nb_*`, `min_nb_*`, `mean_nb_*` for each of the 5 variables) are numerically identical to the original implementation.
END_RESPONSE>>>
