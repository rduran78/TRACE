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
run_id: Anthropic_F05_T10_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's why:

### The Core Problem: Per-Row `paste()`/Lookup Inside `lapply()`

`build_neighbor_lookup` constructs `idx_lookup` once (good), but then inside the `lapply()` over **6.46 million rows**, it:

1. Calls `as.character()` and indexes into `id_to_ref` â€” 6.46M times.
2. Indexes into `neighbors[[ref_idx]]` â€” 6.46M times.
3. Calls `paste()` to build `neighbor_keys` â€” 6.46M times, each creating a character vector proportional to the number of neighbors (~4 for rook).
4. Does named-vector lookup `idx_lookup[neighbor_keys]` â€” 6.46M hash lookups.

This means roughly **25+ million string allocations and hash lookups** just to build the neighbor index. Then `compute_neighbor_stats` is called **5 times** (one per variable), each doing `lapply()` over 6.46M rows with subsetting.

### The Structural Insight

The neighbor relationships are **time-invariant**: cell A is a rook neighbor of cell B in every year. The panel is balanced (every cell appears in every year). Therefore, the neighbor lookup can be reformulated as a **matrix operation**: for each cell, its neighbors are the same set of cells in the same year. If we organize data so that all cells within a year are contiguous and in a fixed order, neighbor indexing becomes **arithmetic** â€” no strings, no hashing.

## Optimization Strategy

1. **Sort data by `(year, id)`** so that within each year-block, cells are in a fixed canonical order.
2. **Map neighbor relationships to integer offsets** within a year-block. Since every year-block has the same cells in the same order, a neighbor for cell `i` in any year is always at the same relative offset within that year's block.
3. **Vectorize the aggregation** using matrix operations: reshape each variable into a `(n_cells Ã— n_years)` matrix, use integer-indexed neighbor lists to pull neighbor values, and compute max/min/mean with vectorized column operations.

This eliminates all `paste()`, all hash lookups, and all per-row `lapply()` iterations.

**Estimated speedup**: from 86+ hours to **minutes**.

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Drop-in replacement â€” preserves numerical output and trained RF model
# =============================================================================

library(data.table)

build_neighbor_features_optimized <- function(cell_data, id_order, rook_neighbors_unique,
                                               neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # 1. Convert to data.table for speed; record original row order
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .original_row_order := .I]
  
  # -------------------------------------------------------------------------
  # 2. Establish canonical cell ordering (same as id_order)
  #    id_order is the ordering used by the nb object.
  # -------------------------------------------------------------------------
  n_cells <- length(id_order)
  n_years <- uniqueN(dt$year)
  years   <- sort(unique(dt$year))
  
  stopifnot(nrow(dt) == n_cells * n_years)  # balanced panel check
  
  # Map each id to its position in id_order (1-based)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # -------------------------------------------------------------------------
  # 3. Sort by (year, canonical cell position) so that within each year-block
  #    row i corresponds to id_order[i]. This makes neighbor indexing arithmetic.
  # -------------------------------------------------------------------------
  dt[, .cell_pos := id_to_pos[as.character(id)]]
  setorder(dt, year, .cell_pos)
  
  # Verify structure: within each year, cell_pos should be 1:n_cells
  stopifnot(all(dt[, .(.cell_pos), by = year]$.cell_pos == rep(1:n_cells, n_years)))
  
  # -------------------------------------------------------------------------
  # 4. Build integer neighbor list (positions within a year-block)
  #    rook_neighbors_unique[[k]] gives the neighbor indices of the k-th
  #    element of id_order, already in terms of positions in id_order.
  # -------------------------------------------------------------------------
  # spdep::nb objects store integer indices directly, so:
  nb_list <- lapply(rook_neighbors_unique, function(x) {
    x <- as.integer(x)
    x[x != 0L]   # spdep uses 0 for "no neighbors" in some representations
  })
  
  # -------------------------------------------------------------------------
  # 5. For each variable, reshape into matrix (n_cells x n_years),
  #    compute neighbor stats vectorized, then write back.
  # -------------------------------------------------------------------------
  
  for (var_name in neighbor_source_vars) {
    
    # Reshape: rows = cell positions (1..n_cells), cols = years
    vals_vec <- dt[[var_name]]
    V <- matrix(vals_vec, nrow = n_cells, ncol = n_years, byrow = FALSE)
    # V[c, y] = value of var_name for cell c in year-index y
    
    # Pre-allocate output matrices (n_cells x n_years)
    M_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    M_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    M_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # Loop over cells (not cell-years!) â€” 344K iterations, not 6.46M
    for (c_idx in seq_len(n_cells)) {
      nb_idx <- nb_list[[c_idx]]
      if (length(nb_idx) == 0L) next
      
      # nb_vals: matrix of dimension (n_neighbors x n_years)
      # Each row = one neighbor's time series
      nb_vals <- V[nb_idx, , drop = FALSE]
      
      # Compute stats across neighbors (column-wise operations)
      # For max/min/mean of neighbors at each year:
      if (length(nb_idx) == 1L) {
        # Single neighbor: all stats are the same
        M_max[c_idx, ]  <- nb_vals[1L, ]
        M_min[c_idx, ]  <- nb_vals[1L, ]
        M_mean[c_idx, ] <- nb_vals[1L, ]
      } else {
        # suppressWarnings handles all-NA columns â†’ returns NA (desired behavior)
        M_max[c_idx, ]  <- suppressWarnings(apply(nb_vals, 2L, max,  na.rm = TRUE))
        M_min[c_idx, ]  <- suppressWarnings(apply(nb_vals, 2L, min,  na.rm = TRUE))
        M_mean[c_idx, ] <- colMeans(nb_vals, na.rm = TRUE)
      }
    }
    
    # Fix -Inf/Inf from max/min on all-NA slices â†’ NA
    M_max[is.infinite(M_max)]   <- NA_real_
    M_min[is.infinite(M_min)]   <- NA_real_
    M_mean[is.nan(M_mean)]      <- NA_real_
    
    # Flatten matrices back into column vectors (column-major = year-blocks)
    max_col_name  <- paste0(var_name, "_neighbor_max")
    min_col_name  <- paste0(var_name, "_neighbor_min")
    mean_col_name <- paste0(var_name, "_neighbor_mean")
    
    dt[, (max_col_name)  := as.vector(M_max)]
    dt[, (min_col_name)  := as.vector(M_min)]
    dt[, (mean_col_name) := as.vector(M_mean)]
  }
  
  # -------------------------------------------------------------------------
  # 6. Restore original row order and clean up helper columns
  # -------------------------------------------------------------------------
  setorder(dt, .original_row_order)
  dt[, c(".original_row_order", ".cell_pos") := NULL]
  
  # Return as data.frame if input was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_optimized(
  cell_data             = cell_data,
  id_order              = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars  = neighbor_source_vars
)

# cell_data now has the 15 new columns (5 vars Ã— {max, min, mean})
# The trained Random Forest model can be used directly for prediction â€” 
# no retraining needed, as the numerical estimand is preserved.
```

## Further Optimization: Eliminate the Cell-Level Loop with Sparse Matrix Multiplication

For maximum speed, replace the 344K-iteration cell loop with sparse matrix algebra:

```r
# =============================================================================
# ULTRA-OPTIMIZED VERSION: Sparse matrix neighbor aggregation
# Computes mean in one shot; max/min via grouped operations
# =============================================================================

library(data.table)
library(Matrix)

build_neighbor_features_sparse <- function(cell_data, id_order, rook_neighbors_unique,
                                            neighbor_source_vars) {
  
  dt <- as.data.table(cell_data)
  dt[, .original_row_order := .I]
  
  n_cells <- length(id_order)
  n_years <- uniqueN(dt$year)
  years   <- sort(unique(dt$year))
  
  stopifnot(nrow(dt) == n_cells * n_years)
  
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  dt[, .cell_pos := id_to_pos[as.character(id)]]
  setorder(dt, year, .cell_pos)
  
  # -------------------------------------------------------------------
  # Build sparse adjacency matrix W (n_cells x n_cells)
  # W[i, j] = 1 if j is a neighbor of i
  # -------------------------------------------------------------------
  nb_list <- lapply(rook_neighbors_unique, function(x) {
    x <- as.integer(x)
    x[x != 0L]
  })
  
  i_idx <- rep(seq_along(nb_list), lengths(nb_list))
  j_idx <- unlist(nb_list, use.names = FALSE)
  
  W <- sparseMatrix(
    i    = i_idx,
    j    = j_idx,
    x    = 1,
    dims = c(n_cells, n_cells)
  )
  
  # Number of non-NA neighbors per cell (for mean): recomputed per variable
  # For mean: W %*% V / (count of non-NA neighbors)
  
  for (var_name in neighbor_source_vars) {
    
    vals_vec <- dt[[var_name]]
    V <- matrix(vals_vec, nrow = n_cells, ncol = n_years, byrow = FALSE)
    
    # ---------- MEAN via sparse matrix multiply ----------
    # Sum of neighbor values
    V_no_na <- V
    V_no_na[is.na(V_no_na)] <- 0
    
    sum_mat   <- as.matrix(W %*% V_no_na)  # n_cells x n_years
    
    # Count of non-NA neighbor values
    notna_mat <- matrix(as.numeric(!is.na(V)), nrow = n_cells, ncol = n_years)
    count_mat <- as.matrix(W %*% notna_mat)
    
    mean_mat <- sum_mat / count_mat  # NaN where count=0
    mean_mat[is.nan(mean_mat)] <- NA_real_
    
    # ---------- MAX and MIN via grouped operations ----------
    # Build a long table: (focal_cell, neighbor_cell) pairs, join year-values
    edge_dt <- data.table(focal = i_idx, neighbor = j_idx)
    
    # Melt V into long form: cell_pos, year_idx, value
    val_long <- data.table(
      cell_pos = rep(1:n_cells, n_years),
      year_idx = rep(1:n_years, each = n_cells),
      val      = as.vector(V)
    )
    
    # Join: for each edge, get the neighbor's value at each year
    # Use cross-join with years for efficiency
    edge_year <- CJ(edge_id = seq_len(nrow(edge_dt)), year_idx = 1:n_years)
    edge_year[, focal    := edge_dt$focal[edge_id]]
    edge_year[, neighbor := edge_dt$neighbor[edge_id]]
    
    # Key for fast join
    setkey(val_long, cell_pos, year_idx)
    edge_year[, val := val_long[.(neighbor, year_idx), val, on = .(cell_pos, year_idx)]]
    
    # Compute max and min grouped by (focal, year_idx)
    agg <- edge_year[!is.na(val), 
                     .(nb_max = max(val), nb_min = min(val)), 
                     by = .(focal, year_idx)]
    
    # Fill into matrices
    max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    max_mat[cbind(agg$focal, agg$year_idx)] <- agg$nb_max
    min_mat[cbind(agg$focal, agg$year_idx)] <- agg$nb_min
    
    # ---------- Write results ----------
    dt[, paste0(var_name, "_neighbor_max")  := as.vector(max_mat)]
    dt[, paste0(var_name, "_neighbor_min")  := as.vector(min_mat)]
    dt[, paste0(var_name, "_neighbor_mean") := as.vector(mean_mat)]
  }
  
  setorder(dt, .original_row_order)
  dt[, c(".original_row_order", ".cell_pos") := NULL]
  
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}
```

## Summary

| Aspect | Original | Optimized (Matrix) | Ultra (Sparse) |
|---|---|---|---|
| **Inner loop iterations** | 6.46M row-level | 344K cell-level | 0 (vectorized) |
| **String `paste()` calls** | ~32M | 0 | 0 |
| **Hash lookups** | ~25M | 0 | 0 |
| **Key insight** | Row-level string matching | Time-invariant neighbors â†’ reshape to cell Ã— year matrix | Sparse `W %*% V` for mean; `data.table` grouped agg for max/min |
| **Estimated time** | 86+ hours | ~5â€“15 minutes | ~2â€“10 minutes |
| **RAM safe at 16 GB** | Yes (slow) | Yes (344K Ã— 28 matrices are small) | Yes (sparse matrix is compact) |

**The trained Random Forest model is untouched** â€” these changes only affect feature construction. The numerical outputs (max, min, mean of neighbor values) are identical to the original.
END_RESPONSE>>>
