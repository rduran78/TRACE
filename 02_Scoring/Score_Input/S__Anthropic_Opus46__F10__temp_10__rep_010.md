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
run_id: Anthropic_F10_T10_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each built via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per probe in the worst case â€” with 6.46M keys, each lookup is expensive. This alone can take many hours.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a serial `lapply`**, extracting variable values and computing `max/min/mean` per node. With 5 variables, this is ~32.3M iterations of R-level loops.

3. **The neighbor topology is year-invariant** (rook adjacency on a fixed grid), but the lookup is rebuilt monolithically for every cell-year combination, conflating spatial structure with temporal indexing. This means the same adjacency information is redundantly encoded 28 times (once per year).

**Key insight:** The adjacency graph is purely spatial (344,208 nodes, ~1.37M directed edges). The year dimension is a panel dimension â€” every cell has the same neighbors in every year. The current code expands the spatial graph into the full cell-year space unnecessarily.

---

## Optimization Strategy

1. **Separate spatial topology from temporal indexing.** Build the sparse adjacency structure once over 344,208 cells. For each year, slice the data, and use vectorized sparse-matrix operations to compute neighbor statistics.

2. **Use a sparse adjacency matrix (CSC/CSR format via `Matrix` package).** Convert the `nb` object to a sparse logical/binary matrix `A` of dimension 344,208 Ã— 344,208. Entry `A[i,j] = 1` means cell `j` is a rook neighbor of cell `i`.

3. **Compute neighbor statistics via sparse matrix operations:**
   - **Mean:** `A %*% x / A %*% 1` (sparse matrix-vector multiply is highly optimized in C).
   - **Max and Min:** Use row-wise sparse operations. Replace structural zeros with `NA` or sentinel values and compute row extrema.

4. **Process year-by-year** to keep memory bounded (~344K Ã— 5 variables per year slice).

5. **Preserve numerical equivalence:** The sparse-matrix approach computes identical `max`, `min`, `mean` of the exact same neighbor value sets.

6. **Do not retrain the Random Forest.** Only reconstruct the predictor columns identically.

**Expected speedup:** From 86+ hours to **~5â€“15 minutes**. Sparse matrix-vector products over 344K nodes with ~4 neighbors each are nearly instantaneous. The loop is 28 years Ã— 5 variables = 140 sparse operations for mean/sum, plus 140 row-wise max/min operations.

---

## Optimized R Code

```r
library(Matrix)
library(spdep)
library(data.table)

# ===========================================================================
# STEP 0: Load pre-existing objects
# ===========================================================================
# Assumes these are already in the environment or loaded from disk:
#   cell_data              â€” data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#   id_order               â€” integer vector of cell IDs in canonical order (length 344,208)
#   rook_neighbors_unique  â€” nb object (list of length 344,208)
#   rf_model               â€” pre-trained Random Forest model (DO NOT RETRAIN)

# ===========================================================================
# STEP 1: Convert cell_data to data.table for fast indexing
# ===========================================================================
cell_data <- as.data.table(cell_data)

# Canonical cell ordering: map cell id -> row index in spatial graph
n_cells <- length(id_order)
id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))

# ===========================================================================
# STEP 2: Build sparse adjacency matrix ONCE (344,208 x 344,208)
# ===========================================================================
# Convert nb object to a sparse matrix.
# Each entry A[i,j] = 1 means cell j is a rook neighbor of cell i.

build_adjacency_matrix <- function(nb_obj, n) {
  # Pre-count total edges for pre-allocation
  edge_counts <- vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1))
  total_edges <- sum(edge_counts)
  
  # Pre-allocate vectors
  row_idx <- integer(total_edges)
  col_idx <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb <- nb_obj[[i]]
    if (length(nb) == 1L && nb[1] == 0L) next
    k <- length(nb)
    row_idx[pos:(pos + k - 1L)] <- i
    col_idx[pos:(pos + k - 1L)] <- nb
    pos <- pos + k
  }
  
  sparseMatrix(
    i = row_idx, j = col_idx,
    x = rep(1, total_edges),
    dims = c(n, n),
    giveCsparse = TRUE
  )
}

cat("Building sparse adjacency matrix...\n")
A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
cat(sprintf("Adjacency matrix: %d x %d, %d non-zeros\n", nrow(A), ncol(A), nnzero(A)))

# Pre-compute the number of neighbors per cell (used for mean calculation)
# ones vector
ones_vec <- rep(1, n_cells)
neighbor_count <- as.numeric(A %*% ones_vec)  # length n_cells

# ===========================================================================
# STEP 3: Compute neighbor stats using sparse operations
# ===========================================================================
# For max and min, we cannot use simple matrix multiply. Strategy:
#   - Extract the sparse structure of A
#   - For each row i, gather vals[A[i,]@j] and compute max/min
#   - We do this in a vectorized C-level efficient way using the CSC structure.
#
# Optimized approach: use the dgCMatrix (CSC) structure to iterate by column,
# but for row-wise operations, convert to dgRMatrix (CSR) for row-major access.

A_csr <- as(A, "RsparseMatrix")  # dgRMatrix: row-compressed

# Function: given a numeric vector x aligned to id_order, compute neighbor
# max, min, mean for each cell. Returns a 3-column matrix (max, min, mean).
compute_neighbor_stats_sparse <- function(A_csr, A_csc, neighbor_count, x, n) {
  # --- MEAN via sparse mat-vec ---
  # Replace NA in x with 0 for sum, and count non-NA neighbors
  x_nona <- x
  is_na_x <- is.na(x)
  x_nona[is_na_x] <- 0
  
  # Neighbor sum (treating NA as 0)
  nb_sum <- as.numeric(A_csc %*% x_nona)
  
  # Count non-NA neighbors per cell
  not_na_flag <- as.numeric(!is_na_x)
  nb_count_valid <- as.numeric(A_csc %*% not_na_flag)
  
  nb_mean <- ifelse(nb_count_valid > 0, nb_sum / nb_count_valid, NA_real_)
  
  # --- MAX and MIN via CSR row traversal ---
  # Access the internal slots of dgRMatrix
  # dgRMatrix: @p (row pointers, length n+1), @j (column indices, 0-based), @x (values)
  rp <- A_csr@p    # row pointers (length n+1), 0-based
  cj <- A_csr@j    # column indices (0-based)
  
  nb_max <- rep(NA_real_, n)
  nb_min <- rep(NA_real_, n)
  
  for (i in seq_len(n)) {
    start <- rp[i] + 1L       # convert 0-based to 1-based
    end   <- rp[i + 1L]
    if (end < start) next      # no neighbors
    
    col_indices <- cj[start:end] + 1L  # 1-based column indices
    neighbor_vals <- x[col_indices]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    
    if (length(neighbor_vals) > 0L) {
      nb_max[i] <- max(neighbor_vals)
      nb_min[i] <- min(neighbor_vals)
    }
  }
  
  cbind(nb_max, nb_min, nb_mean)
}

# ===========================================================================
# STEP 3b: Even faster max/min â€” vectorized with Rcpp-like approach in pure R
#     Using tapply on the expanded edge list for max/min
# ===========================================================================
# More efficient: expand all edges, get values, and use grouping operations.

compute_neighbor_stats_fast <- function(A_csr, A_csc, neighbor_count, x, n) {
  # --- MEAN via sparse mat-vec ---
  x_nona <- x
  is_na_x <- is.na(x)
  x_nona[is_na_x] <- 0
  
  nb_sum <- as.numeric(A_csc %*% x_nona)
  not_na_flag <- as.numeric(!is_na_x)
  nb_count_valid <- as.numeric(A_csc %*% not_na_flag)
  nb_mean <- ifelse(nb_count_valid > 0, nb_sum / nb_count_valid, NA_real_)
  
  # --- MAX and MIN via edge expansion + data.table grouping ---
  rp <- A_csr@p
  cj <- A_csr@j
  n_edges <- length(cj)
  
  # Build row-index vector from row pointers
  row_lengths <- diff(rp)  # length n
  row_ids <- rep(seq_len(n), times = row_lengths)  # length = n_edges
  col_ids <- cj + 1L  # 1-based
  
  # Get neighbor values
  edge_vals <- x[col_ids]
  
  # Remove edges where neighbor value is NA
  valid <- !is.na(edge_vals)
  row_ids_v <- row_ids[valid]
  edge_vals_v <- edge_vals[valid]
  
  if (length(row_ids_v) > 0L) {
    # Use data.table for fast grouped max/min
    edge_dt <- data.table(row = row_ids_v, val = edge_vals_v)
    agg <- edge_dt[, .(vmax = max(val), vmin = min(val)), by = row]
    
    nb_max <- rep(NA_real_, n)
    nb_min <- rep(NA_real_, n)
    nb_max[agg$row] <- agg$vmax
    nb_min[agg$row] <- agg$vmin
  } else {
    nb_max <- rep(NA_real_, n)
    nb_min <- rep(NA_real_, n)
  }
  
  cbind(nb_max, nb_min, nb_mean)
}

# ===========================================================================
# STEP 4: Process year-by-year, variable-by-variable
# ===========================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data is keyed for fast subsetting
setkey(cell_data, year, id)

years <- sort(unique(cell_data$year))

# Pre-allocate result columns in cell_data
for (var_name in neighbor_source_vars) {
  set(cell_data, j = paste0(var_name, "_neighbor_max"),  value = NA_real_)
  set(cell_data, j = paste0(var_name, "_neighbor_min"),  value = NA_real_)
  set(cell_data, j = paste0(var_name, "_neighbor_mean"), value = NA_real_)
}

# Pre-compute the CSR row pointers and column indices ONCE for edge expansion
# (these are reused every iteration)
rp_global <- A_csr@p
cj_global <- A_csr@j
row_lengths_global <- diff(rp_global)
row_ids_global <- rep(seq_len(n_cells), times = row_lengths_global)
col_ids_global <- cj_global + 1L  # 1-based

cat("Computing neighbor features...\n")
t0 <- proc.time()

for (yr in years) {
  cat(sprintf("  Year %d...\n", yr))
  
  # Get row indices for this year
  yr_rows <- which(cell_data$year == yr)
  
  # Get the cell IDs for this year's data
  yr_ids <- cell_data$id[yr_rows]
  
  # Map cell IDs to spatial indices in id_order
  yr_spatial_idx <- id_to_idx[as.character(yr_ids)]
  
  # Check if all cells are present and build reverse map:
  # spatial_to_yr_row: for spatial index s, which position in yr_rows has that cell?
  # If a cell is missing from this year, it won't have data.
  spatial_to_yr_pos <- rep(NA_integer_, n_cells)
  spatial_to_yr_pos[yr_spatial_idx] <- seq_along(yr_rows)
  
  for (var_name in neighbor_source_vars) {
    # Build the spatial-aligned variable vector
    # x[s] = value of var_name for spatial cell s in year yr (or NA if missing)
    x <- rep(NA_real_, n_cells)
    x[yr_spatial_idx] <- cell_data[[var_name]][yr_rows]
    
    # --- MEAN via sparse mat-vec ---
    x_nona <- x
    is_na_x <- is.na(x)
    x_nona[is_na_x] <- 0
    
    nb_sum <- as.numeric(A %*% x_nona)
    not_na_flag <- as.numeric(!is_na_x)
    nb_count_valid <- as.numeric(A %*% not_na_flag)
    nb_mean <- ifelse(nb_count_valid > 0, nb_sum / nb_count_valid, NA_real_)
    
    # --- MAX and MIN via edge expansion + data.table grouping ---
    edge_vals <- x[col_ids_global]
    
    valid <- !is.na(edge_vals)
    row_ids_v <- row_ids_global[valid]
    edge_vals_v <- edge_vals[valid]
    
    nb_max <- rep(NA_real_, n_cells)
    nb_min <- rep(NA_real_, n_cells)
    
    if (length(row_ids_v) > 0L) {
      edge_dt <- data.table(row = row_ids_v, val = edge_vals_v)
      agg <- edge_dt[, .(vmax = max(val), vmin = min(val)), by = row]
      nb_max[agg$row] <- agg$vmax
      nb_min[agg$row] <- agg$vmin
    }
    
    # Map spatial results back to cell_data rows
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")
    
    set(cell_data, i = yr_rows, j = max_col,  value = nb_max[yr_spatial_idx])
    set(cell_data, i = yr_rows, j = min_col,  value = nb_min[yr_spatial_idx])
    set(cell_data, i = yr_rows, j = mean_col, value = nb_mean[yr_spatial_idx])
  }
}

elapsed <- proc.time() - t0
cat(sprintf("Neighbor feature computation complete in %.1f seconds.\n", elapsed[3]))

# ===========================================================================
# STEP 5: Predict using the pre-trained Random Forest (DO NOT RETRAIN)
# ===========================================================================
# The predictor columns now match the original schema.
# Convert back to data.frame if the RF model expects it.

cell_data_df <- as.data.frame(cell_data)
predictions <- predict(rf_model, newdata = cell_data_df)
cell_data$predicted <- predictions

cat("Prediction complete. Pipeline finished.\n")
```

---

## Summary of Guarantees

| Concern | Status |
|---|---|
| **Numerical equivalence** | âœ… Identical `max`, `min`, `mean` over the same neighbor sets, same NA-handling logic |
| **RF model preserved** | âœ… `predict()` only â€” no retraining |
| **Column names preserved** | âœ… `{var}_neighbor_max`, `{var}_neighbor_min`, `{var}_neighbor_mean` |
| **Memory** | âœ… Year-by-year processing; peak ~1.5 GB well within 16 GB |
| **Speed** | âœ… ~140 sparse mat-vec products + 140 `data.table` grouped aggregations â‰ˆ **5â€“15 minutes** (vs. 86+ hours) |

### Why this is fast

- **Sparse `A %*% x`** runs in O(nnz) â‰ˆ 1.37M multiply-adds in optimized C (CHOLMOD/SuiteSparse). This replaces 6.46M R-level list iterations.
- **`data.table` grouped `max`/`min`** over ~1.37M edges uses radix-sort grouping â€” orders of magnitude faster than per-row `lapply`.
- **The adjacency matrix, row pointers, column indices, and row-ID expansion vector are all built once** and reused across all 28 Ã— 5 = 140 iterations.
END_RESPONSE>>>
