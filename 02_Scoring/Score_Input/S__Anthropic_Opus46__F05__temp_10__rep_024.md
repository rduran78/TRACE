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
run_id: Anthropic_F05_T10_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern, not merely a local inefficiency**. Here's the full picture:

### Local Inefficiency
- `paste(data$id, data$year, sep = "_")` constructs ~6.46 million string keys to build `idx_lookup`. This happens once in `build_neighbor_lookup`, which is acceptable in isolation.
- But inside the `lapply` over all 6.46M rows, `paste(neighbor_cell_ids, data$year[i], sep = "_")` is called per row, generating millions of tiny string vectors and performing millions of named-vector lookups (which are hash-table probes on character keys). This is the **inner hot loop** and is extremely slow.

### Broader Algorithmic Problem
The entire approach is **row-wise R-level iteration over 6.46M rows** with string operations inside the loop. This is the classic R anti-pattern that causes 86+ hour runtimes. The fundamental issue is:

1. **The neighbor topology is year-invariant.** Every cell has the same rook neighbors in every year. The `nb` object encodes cell-to-cell adjacency, not cell-year-to-cell-year adjacency. Yet the code reconstructs year-specific lookups inside a per-row loop.

2. **The neighbor lookup can be built once as integer indexing**, completely eliminating string keys. Since `data` is a cell-year panel (presumably sorted or sortable by `id` and `year`), we can compute a direct integer row-index mapping.

3. **The neighbor statistics (max, min, mean) can be computed in a fully vectorized, column-wise manner** using sparse matrix multiplication and sparse-matrix element-wise operations, eliminating all R-level row iteration.

## Optimization Strategy

**Key Insight:** If `W` is the row-normalized spatial weights matrix (344,208 Ã— 344,208) and `X` is a variable arranged as a matrix (344,208 rows Ã— 28 columns, one per year), then `W %*% X` gives neighbor means. For max and min, we use analogous sparse-matrix tricks or a single vectorized pass.

**Steps:**

1. **Build a sparse adjacency matrix** from `rook_neighbors_unique` once (344K Ã— 344K, ~1.37M non-zeros). This replaces all string-key construction.
2. **Map panel rows to (cell_index, year_index)** using integer arithmetic â€” no strings.
3. **Compute neighbor stats vectorized** using sparse matrix operations:
   - **Mean**: `W_rownorm %*% X` (row-normalized weights Ã— values).
   - **Max/Min**: Vectorized grouped operations using the sparse structure.
4. **Join results** back to the panel by integer index.

This reduces the complexity from O(N_rows Ã— k Ã— string_ops) to O(nnz Ã— n_years) with vectorized C-level operations, cutting runtime from 86+ hours to **minutes**.

## Working R Code

```r
library(Matrix)
library(data.table)

# =============================================================================
# STEP 1: Build sparse adjacency matrix from nb object (once)
# =============================================================================
build_sparse_adj <- function(nb_obj, n) {
  # nb_obj: list of length n, each element is integer vector of neighbor indices
  # Returns: sparse dgCMatrix (n x n) binary adjacency
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove 0-neighbor entries (spdep uses integer(0) for islands)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

# =============================================================================
# STEP 2: Compute neighbor stats vectorized via sparse matrix ops
# =============================================================================
compute_all_neighbor_features <- function(cell_data, id_order, nb_obj, 
                                           neighbor_source_vars) {
  # Convert to data.table for fast manipulation
  dt <- as.data.table(cell_data)
  
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)
  
  # --- Integer mappings (no strings!) ---
  # Map cell id -> integer index 1..n_cells matching nb_obj ordering
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  dt[, cell_idx := id_to_idx[as.character(id)]]
  
  # Map year -> integer index 1..n_years
  year_to_idx <- setNames(seq_along(years), as.character(years))
  dt[, year_idx := year_to_idx[as.character(year)]]
  
  # --- Build sparse adjacency matrix ---
  W <- build_sparse_adj(nb_obj, n_cells)
  
  # Precompute number of neighbors per cell (for mean calculation)
  neighbor_counts <- diff(W@p)  # CSC column pointers give col counts for t(W)
  # Actually for row counts on a dgCMatrix, use:
  neighbor_counts_vec <- tabulate(W@i + 1L, nbins = n_cells)
  # But more reliably: rowSums of binary matrix
  neighbor_counts_vec <- as.numeric(Matrix::rowSums(W))
  
  # Row-normalized weight matrix for means
  # Avoid division by zero for islands
  inv_counts <- ifelse(neighbor_counts_vec == 0, 0, 1 / neighbor_counts_vec)
  D_inv <- Diagonal(x = inv_counts)
  W_norm <- D_inv %*% W  # row-normalized
  
  # --- For each variable, compute max, min, mean across neighbors ---
  # Strategy:
  #   - Reshape variable into matrix: n_cells x n_years
  #   - Mean: W_norm %*% X  (sparse mat-mul, very fast)
  #   - Max/Min: iterate over sparse structure but in vectorized C-level ops
  
  # Ensure dt is keyed for fast lookups
  setkey(dt, cell_idx, year_idx)
  
  for (var_name in neighbor_source_vars) {
    cat("Processing neighbor features for:", var_name, "\n")
    
    # Build n_cells x n_years matrix of values
    # Fill with NA for missing cell-year combos
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]
    
    # ---- MEAN ----
    # Replace NA with 0 for multiplication, but track counts of non-NA neighbors
    X_zero <- X
    X_zero[is.na(X_zero)] <- 0
    
    # Binary non-NA indicator
    X_valid <- matrix(as.numeric(!is.na(X)), nrow = n_cells, ncol = n_years)
    
    # Sum of neighbor values (only non-NA contribute)
    # W is binary adjacency, so W %*% X_zero = sum of neighbor values (NA treated as 0)
    neighbor_sum   <- as.matrix(W %*% X_zero)     # n_cells x n_years
    neighbor_count <- as.matrix(W %*% X_valid)     # n_cells x n_years (count of non-NA neighbors)
    
    neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # ---- MAX and MIN via sparse structure ----
    # We iterate over each cell's neighbors using the sparse matrix structure.
    # This is done in a vectorized way per year column.
    
    neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # Extract sparse triplet form for row-wise iteration
    W_t <- as(W, "TsparseMatrix")  # gives @i (row, 0-based), @j (col, 0-based)
    from_idx <- W_t@i + 1L  # row indices (the focal cell)
    to_idx   <- W_t@j + 1L  # col indices (the neighbor cell)
    
    # For each year, compute grouped max/min using data.table
    for (yr in seq_len(n_years)) {
      x_col <- X[, yr]
      neighbor_vals <- x_col[to_idx]  # value of each neighbor
      
      # Use data.table for fast grouped max/min
      edge_dt <- data.table(
        focal    = from_idx,
        nval     = neighbor_vals
      )
      # Remove edges where neighbor value is NA
      edge_dt <- edge_dt[!is.na(nval)]
      
      if (nrow(edge_dt) > 0) {
        stats <- edge_dt[, .(nmax = max(nval), nmin = min(nval)), by = focal]
        neighbor_max[stats$focal, yr] <- stats$nmax
        neighbor_min[stats$focal, yr] <- stats$nmin
      }
    }
    
    # ---- Map results back to panel rows ----
    row_indices <- cbind(dt$cell_idx, dt$year_idx)
    
    max_col  <- paste0("max_neighbor_", var_name)
    min_col  <- paste0("min_neighbor_", var_name)
    mean_col <- paste0("mean_neighbor_", var_name)
    
    dt[, (max_col)  := neighbor_max[row_indices]]
    dt[, (min_col)  := neighbor_min[row_indices]]
    dt[, (mean_col) := neighbor_mean[row_indices]]
  }
  
  # Remove helper columns
  dt[, c("cell_idx", "year_idx") := NULL]
  
  # Return as data.frame if original was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# =============================================================================
# STEP 3: Drop-in replacement call
# =============================================================================
# Usage (replaces the entire outer loop):

cell_data <- compute_all_neighbor_features(
  cell_data        = cell_data,
  id_order         = id_order,
  nb_obj           = rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# The trained Random Forest model is untouched â€” only the feature construction
# pipeline is replaced. Column names match the original pattern:
#   max_neighbor_{var}, min_neighbor_{var}, mean_neighbor_{var}
# Numerical results are identical (same estimand: max/min/mean of non-NA 
# neighbor values within the same year).
```

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| **Inner loop iterations** | 6.46M R-level `lapply` calls | 0 (fully vectorized) |
| **String operations** | ~6.46M `paste()` calls + hash lookups | 0 (integer indexing only) |
| **Mean computation** | R-level per-row subsetting | Sparse matrix multiply (`W %*% X`): C-level, O(nnz) |
| **Max/Min computation** | R-level per-row subsetting | `data.table` grouped aggregation per year (28 passes) |
| **Memory** | Character hash table (~6.46M keys) | Sparse matrix (~1.37M non-zeros) + n_cells Ã— n_years matrices (~10M doubles each â‰ˆ 77MB) |
| **Estimated runtime** | 86+ hours | **~5â€“15 minutes** |

### Why this is correct (preserves the estimand)

- **Mean**: `W %*% X_zero` sums exactly the non-NA neighbor values; dividing by the count of non-NA neighbors (`W %*% X_valid`) reproduces `mean(neighbor_vals[!is.na(neighbor_vals)])`.
- **Max/Min**: The `data.table` grouped aggregation over the same sparse edge list computes exactly `max(neighbor_vals[!is.na()])` and `min(neighbor_vals[!is.na()])` per focal cell per year.
- **Column names** follow the same `{max|min|mean}_neighbor_{var}` pattern, so the trained Random Forest model's feature references remain valid with no retraining needed.

### Further optimization (if needed)

If even the 28-pass max/min loop is too slow, you can collapse it into a single pass using `data.table` with a year column:

```r
# Alternative: single-pass max/min across all years
edge_dt_full <- data.table(
  focal = rep(from_idx, n_years),
  yr    = rep(seq_len(n_years), each = length(from_idx)),
  nval  = as.vector(X[to_idx, ])  # neighbor values for all year columns
)
edge_dt_full <- edge_dt_full[!is.na(nval)]
stats_full <- edge_dt_full[, .(nmax = max(nval), nmin = min(nval)), 
                            by = .(focal, yr)]
# Then scatter back into matrices
```

This trades memory for fewer passes, which is viable at 16 GB RAM given the dataset size (~1.37M edges Ã— 28 years â‰ˆ 38.5M rows of edge data â‰ˆ 900 MB).
END_RESPONSE>>>
