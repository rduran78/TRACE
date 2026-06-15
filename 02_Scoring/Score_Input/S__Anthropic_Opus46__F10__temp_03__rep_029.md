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
run_id: Anthropic_F10_T03_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`paste` + `idx_lookup[neighbor_keys]`). This is O(NÃ—k) with enormous constant factors from R's string hashing on millions of keys.

2. **`compute_neighbor_stats` iterates over 6.46 million list elements** in an `lapply`, calling `max`/`min`/`mean` individually per node-year. This is pure R-level looping with no vectorization.

3. **The neighbor lookup conflates topology and time.** Rook neighbors are a *spatial* property â€” they don't change across years. Yet the lookup rebuilds temporal keys for every cell-year row, inflating a 344,208-cell spatial graph into a 6.46M-row problem at the lookup stage.

**Key insight:** The adjacency structure is year-invariant. There are only 344,208 nodes with ~1.37M directed edges. The panel has 28 years. We should build the sparse spatial graph **once** (344K Ã— 344K), then for each year-slice (~231K rows), perform sparse matrixâ€“vector multiplication to get sums and counts, and element-wise operations for max/min â€” all vectorized.

## Optimization Strategy

1. **Build a sparse adjacency matrix (once):** Convert the `nb` object into a `dgCMatrix` (344,208 Ã— 344,208). This is ~1.37M non-zero entries â€” trivially small in memory (~16 MB).

2. **Split data by year (or index by year):** For each of the 28 years, extract the variable column as a vector aligned to the cell ordering.

3. **Compute neighbor stats via sparse matrix operations:**
   - **Mean:** `A %*% x / A %*% 1` (sparse mat-vec, microseconds per variable-year).
   - **Sum for mean:** `neighbor_sum = A %*% x`, `neighbor_count = A %*% (non-NA indicator)`, `mean = sum/count`.
   - **Max/Min:** Use a modified sparse approach â€” replace the `x` slot of the adjacency matrix with neighbor values and compute row-wise max/min via `rowMaxs`/`rowMins` from the `sparseMatrixStats` package, or iterate over the CSC structure in C++ via `Rcpp`.

4. **Handle NA propagation** identically to the original: ignore NAs in neighbor values; if all neighbors are NA (or a cell has no neighbors), return NA.

5. **Write results back** to the same columns the original code would produce, preserving the trained RF model's expected feature names.

**Expected speedup:** From ~86 hours to **~2â€“5 minutes**. The dominant cost becomes 5 variables Ã— 28 years Ã— 3 sparse operations = 420 sparse mat-vec products on a 344KÃ—344K matrix with 1.37M nonzeros â€” each taking milliseconds.

## Working R Code

```r
# ==============================================================================
# Optimized spatial neighbor feature computation
# Preserves numerical equivalence with original pipeline
# ==============================================================================

library(Matrix)
library(data.table)

# --------------------------------------------------------------------------
# Step 1: Build sparse adjacency matrix from nb object (ONCE)
# --------------------------------------------------------------------------
build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj: list of length n, each element is integer vector of neighbor indices
  # Build COO triplets
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # Remove any zero-length / empty entries (islands)
  valid <- !is.na(to) & to > 0
  from  <- from[valid]
  to    <- to[valid]
  
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n), repr = "C")
}

# --------------------------------------------------------------------------
# Step 2: Compute max, min, mean of neighbor values using sparse ops
# --------------------------------------------------------------------------
# For max and min we need to handle the sparse structure carefully:
# - Zeros in a sparse matrix are "structural" (no edge), not value zeros
# - We manipulate the @x slot directly

compute_neighbor_stats_sparse <- function(A, values) {
  # A: dgCMatrix adjacency (n x n), values: numeric vector length n
  # Returns: n x 3 matrix [max, min, mean], NA where no valid neighbors
  
  n <- length(values)
  
  # --- Mean via sparse mat-vec ---
  not_na   <- as.numeric(!is.na(values))
  vals_0   <- values
  vals_0[is.na(vals_0)] <- 0  # zero-fill for summation (NAs contribute 0)
  
  neighbor_sum   <- as.numeric(A %*% vals_0)
  neighbor_count <- as.numeric(A %*% not_na)
  
  neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
  
  # --- Max and Min via direct sparse slot manipulation ---
  # A is CSC: @p (col pointers), @i (row indices), @x (values)
  # We want row-wise max/min of neighbor values.
  # Strategy: transpose to CSR-like (which is CSC of t(A)), then iterate columns
  # But more efficiently: create a copy of A, replace @x with the neighbor values,
  # then compute row max/min.
  
  # For each nonzero entry A[i,j]=1, we want the value values[j].
  # In CSC format, column j contains rows A@i[A@p[j]+1 : A@p[j+1]]
  # So A@x positions corresponding to column j should get values[j].
  
  At <- A  # copy
  
  # Map column index to each nonzero entry
  col_indices <- rep(seq_len(n), diff(At@p))  # column index for each nonzero
  
  # Replace x slot with the actual variable values of the neighbor (column = neighbor)
  neighbor_vals_sparse <- values[col_indices]
  
  # Identify entries where the neighbor value is NA
  na_mask <- is.na(neighbor_vals_sparse)
  
  # For max: set NA entries to -Inf so they don't affect max
  x_for_max <- neighbor_vals_sparse
  x_for_max[na_mask] <- -Inf
  
  # For min: set NA entries to +Inf
  x_for_min <- neighbor_vals_sparse
  x_for_min[na_mask] <- Inf
  
  # Now compute row-wise max and min using the sparse structure
  # Convert to dgTMatrix for easy row-based grouping, or use tapply on row indices
  row_indices <- At@i + 1L  # 0-based to 1-based
  
  # Pre-allocate
  neighbor_max <- rep(NA_real_, n)
  neighbor_min <- rep(NA_real_, n)
  
  # Use data.table for fast grouped aggregation
  dt <- data.table(row = row_indices, val_max = x_for_max, val_min = x_for_min)
  
  agg <- dt[, .(rmax = max(val_max), rmin = min(val_min)), by = row]
  
  # Assign results
  neighbor_max[agg$row] <- agg$rmax
  neighbor_min[agg$row] <- agg$rmin
  
  # Fix cases where all neighbors were NA: max would be -Inf, min would be Inf
  neighbor_max[neighbor_max == -Inf] <- NA_real_

  neighbor_min[neighbor_min ==  Inf] <- NA_real_
  
  # Nodes with no neighbors at all (neighbor_count == 0 from structural zeros)
  no_neighbors <- neighbor_count == 0

  neighbor_max[no_neighbors] <- NA_real_
  neighbor_min[no_neighbors] <- NA_real_
  
  cbind(max = neighbor_max, min = neighbor_min, mean = neighbor_mean)
}

# --------------------------------------------------------------------------
# Step 3: Main pipeline
# --------------------------------------------------------------------------
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  
  cat("Converting to data.table...\n")
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  n_cells <- length(id_order)
  cat(sprintf("Building sparse adjacency matrix: %d cells\n", n_cells))
  
  # Build adjacency matrix ONCE
  A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
  cat(sprintf("Adjacency matrix: %d nonzeros (%.1f MB)\n",
              nnzero(A), object.size(A) / 1e6))
  
  # Create mapping from cell id to matrix row/column index
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Get unique years
  years <- sort(unique(cell_data$year))
  cat(sprintf("Processing %d variables x %d years = %d slices\n",
              length(neighbor_source_vars), length(years),
              length(neighbor_source_vars) * length(years)))
  
  # Pre-create output columns
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)
  }
  
  # Add matrix index column for fast alignment
  cell_data[, mat_idx := id_to_idx[as.character(id)]]
  
  # Process by year
  for (yr in years) {
    yr_rows <- which(cell_data$year == yr)
    yr_ids  <- cell_data$id[yr_rows]
    yr_mat_idx <- cell_data$mat_idx[yr_rows]
    
    for (var_name in neighbor_source_vars) {
      # Build full-length vector aligned to matrix indices
      # (some cells may be missing in a given year)
      full_vec <- rep(NA_real_, n_cells)
      full_vec[yr_mat_idx] <- cell_data[[var_name]][yr_rows]
      
      # Compute neighbor stats via sparse matrix
      stats <- compute_neighbor_stats_sparse(A, full_vec)
      
      # Extract results for cells present this year
      col_max  <- paste0("neighbor_max_", var_name)
      col_min  <- paste0("neighbor_min_", var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      
      set(cell_data, i = yr_rows, j = col_max,  value = stats[yr_mat_idx, 1])
      set(cell_data, i = yr_rows, j = col_min,  value = stats[yr_mat_idx, 2])
      set(cell_data, i = yr_rows, j = col_mean, value = stats[yr_mat_idx, 3])
    }
    
    cat(sprintf("  Year %d done\n", yr))
  }
  
  # Clean up helper column
  cell_data[, mat_idx := NULL]
  
  cat("Neighbor feature computation complete.\n")
  return(cell_data)
}

# --------------------------------------------------------------------------
# Step 4: Execute and predict
# --------------------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- run_neighbor_feature_pipeline(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)

# Predict with the pre-trained Random Forest (unchanged)
# rf_model is already loaded; do NOT retrain
cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M string-key list entries | 344K Ã— 344K sparse matrix (1.37M nnz, ~16 MB) |
| Stats computation | 6.46M R-level `lapply` iterations | 140 sparse mat-vec products + 140 `data.table` grouped aggregations |
| Topology reuse | Rebuilt per cell-year | Built once, reused 140 times (5 vars Ã— 28 years) |
| Estimated runtime | ~86+ hours | **~2â€“5 minutes** |
| RAM peak | Potentially >16 GB (list overhead) | ~2â€“3 GB |
| Numerical equivalence | Reference | âœ… Identical max/min/mean, identical NA handling |
| RF model | Unchanged | Unchanged â€” same feature names, same predict call |

**Key guarantees:**
- The sparse matrix encodes exactly the same directed rook-neighbor edges as `rook_neighbors_unique`.
- `max`, `min`, and `mean` are computed over the identical set of non-NA neighbor values.
- NA propagation logic is preserved: nodes with no neighbors or all-NA neighbors receive `NA`.
- The RF model is never retrained â€” only `predict()` is called with the same feature columns.
END_RESPONSE>>>
