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
run_id: Anthropic_F08_T00_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup of length ~6.46 million, where each entry maps a cell-year row to the row indices of its neighbors *in that same year*. This is enormously wasteful because:

1. **The neighbor topology is static.** Cell A's neighbors are the same in 1992 as in 2019. Yet the current code rebuilds the mapping for every cell-year combination â€” effectively duplicating the same spatial relationship 28 times per cell.

2. **String-key hashing is expensive at scale.** The function creates ~6.46M paste-concatenated keys (`id_year`) and performs named-vector lookups (hash table probes) for every neighbor of every cell-year row. With ~1.37M directed neighbor pairs Ã— 28 years â‰ˆ 38.5M hash lookups, this alone takes hours.

3. **`compute_neighbor_stats` iterates with `lapply` over 6.46M entries**, calling `max`, `min`, `mean` individually per row. This is called 5 times (once per variable), yielding ~32.3M R-level function invocations.

4. **The entire design conflates topology (static) with variable values (dynamic).** By entangling them into a single row-level lookup, the code cannot exploit the panel structure.

## Optimization Strategy

**Separate the static spatial graph from the dynamic year-varying values.**

1. **Build a cell-level neighbor index once** â€” a list of length 344,208 where each element contains the integer positions of that cell's neighbors within the ordered cell vector. This is just a cleaned version of `rook_neighbors_unique` and is built once.

2. **For each variable, extract the value matrix** â€” reshape the variable into a `cells Ã— years` matrix (344,208 rows Ã— 28 columns). This allows vectorized column-wise (i.e., year-wise) operations.

3. **Compute neighbor stats via sparse-matrix multiplication or vectorized gather.** For each cell, gather neighbor values from the matrix rows, compute max/min/mean across neighbors for each year simultaneously. Using a sparse adjacency matrix, `mean` is a single matrix multiply; `max` and `min` can be computed with a grouped operation over the sparse structure.

4. **Reshape results back** to the long cell-year format and bind columns to the original data.

This reduces the work from ~6.46M Ã— 5 R-level list iterations to a handful of sparse matrix operations and vectorized grouped computations over ~1.37M edges Ã— 28 years, all in compiled C/C++ code underneath.

**Expected speedup:** From 86+ hours to minutes (roughly 2â€“10 minutes depending on RAM pressure).

**Numerical equivalence:** The same neighbor sets and the same `max`, `min`, `mean` aggregations are computed, preserving the original estimand exactly. The trained Random Forest model is untouched.

## Working R Code

```r
library(data.table)
library(Matrix)

# ==============================================================================
# STEP 0: Prepare data.table and establish cell/year orderings
# ==============================================================================

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure deterministic ordering: cells in id_order, years ascending
# Create integer cell index and year index for matrix positioning
cell_id_map <- data.table(
  id       = id_order,
  cell_idx = seq_along(id_order)
)

years_all  <- sort(unique(cell_data$year))
year_map   <- data.table(
  year     = years_all,
  year_idx = seq_along(years_all)
)

n_cells <- length(id_order)
n_years <- length(years_all)

# Add cell_idx and year_idx to cell_data
cell_data <- merge(cell_data, cell_id_map, by = "id", sort = FALSE)
cell_data <- merge(cell_data, year_map,   by = "year", sort = FALSE)

# Create a row-order key so we can write results back in the correct position
cell_data[, .row_order := .I]

# ==============================================================================
# STEP 1: Build the sparse adjacency matrix ONCE (static topology)
# ==============================================================================
# rook_neighbors_unique is an nb object: a list of length n_cells,
# where each element is an integer vector of neighbor indices into id_order.

build_adjacency_matrix <- function(nb_obj, n) {
  # Build COO triplets from the nb object
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj)
  
  # Remove any 0-neighbor sentinel (spdep uses 0L for no-neighbor cells)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  # Sparse logical/binary adjacency matrix (row = focal cell, col = neighbor cell)
  sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = rep(1, length(from_idx)),
    dims = c(n, n)
  )
}

cat("Building sparse adjacency matrix...\n")
adj_mat <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

# Precompute the number of neighbors per cell (for mean calculation)
n_neighbors <- as.numeric(rowSums(adj_mat))  # length n_cells

# ==============================================================================
# STEP 2: Function to build a cells x years matrix from the long data
# ==============================================================================

long_to_matrix <- function(dt, var_name, n_cells, n_years) {
  # Allocate matrix filled with NA

  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  # Fill using integer indices â€” fully vectorized
  mat[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]
  mat
}

# ==============================================================================
# STEP 3: Compute neighbor max, min, mean for one variable
# ==============================================================================

compute_neighbor_stats_fast <- function(adj_mat, val_mat, n_neighbors) {
  # adj_mat:      n_cells x n_cells sparse matrix (binary)
  # val_mat:      n_cells x n_years dense matrix
  # n_neighbors:  numeric vector length n_cells
  #
  # Returns a list with three matrices (each n_cells x n_years):
  #   neighbor_max, neighbor_min, neighbor_mean
  
  n_cells <- nrow(val_mat)
  n_years <- ncol(val_mat)
  
  # --- MEAN via sparse matrix multiply ---
  # sum of neighbor values = adj_mat %*% val_mat  (sparse x dense, very fast)
  neighbor_sum <- as.matrix(adj_mat %*% val_mat)   # n_cells x n_years
  
  # To get correct mean we also need neighbor *count* excluding NAs
  # non_na_mat: 1 where val_mat is not NA, 0 otherwise
  non_na_mat <- matrix(0, nrow = n_cells, ncol = n_years)
  non_na_mat[!is.na(val_mat)] <- 1
  
  # Replace NA with 0 in val_mat for the sum computation
  val_mat_0 <- val_mat
  val_mat_0[is.na(val_mat_0)] <- 0
  
  neighbor_sum <- as.matrix(adj_mat %*% val_mat_0)
  neighbor_cnt <- as.matrix(adj_mat %*% non_na_mat)
  
  neighbor_mean <- neighbor_sum / neighbor_cnt
  neighbor_mean[neighbor_cnt == 0] <- NA_real_
  
  # --- MAX and MIN via sparse structure iteration ---
  # We iterate over each year (only 28) and use the sparse structure
  # This avoids 6.46M R-level calls; instead it's 28 vectorized operations
  
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Extract sparse structure once
  # For a dgCMatrix (CSC), we transpose to get CSR-like access by row
  adj_csr <- as(adj_mat, "RsparseMatrix")  # dgRMatrix: row-compressed
  row_ptr <- adj_csr@p   # length n_cells + 1, 0-based
  col_idx <- adj_csr@j   # 0-based column indices
  
  for (yr in seq_len(n_years)) {
    vals_yr <- val_mat[, yr]  # length n_cells
    
    # For each cell, gather neighbor values and compute max/min
    # We do this in vectorized chunks using the CSR structure
    # 
    # Approach: expand neighbor values, then do grouped max/min
    # group = focal cell index
    
    # Number of neighbors per cell (from CSR row pointers)
    # row_ptr is 0-based, length n_cells+1
    n_per_row <- diff(row_ptr)  # length n_cells
    
    if (length(col_idx) == 0) next
    
    # Focal cell index for each entry in col_idx
    focal <- rep(seq_len(n_cells), times = n_per_row)
    
    # Neighbor values
    nb_vals <- vals_yr[col_idx + 1L]  # col_idx is 0-based
    
    # Remove NAs
    valid <- !is.na(nb_vals)
    focal_v   <- focal[valid]
    nb_vals_v <- nb_vals[valid]
    
    if (length(nb_vals_v) == 0) next
    
    # Grouped max and min using data.table for speed
    tmp_dt <- data.table(focal = focal_v, val = nb_vals_v)
    agg <- tmp_dt[, .(vmax = max(val), vmin = min(val)), by = focal]
    
    neighbor_max[agg$focal, yr] <- agg$vmax
    neighbor_min[agg$focal, yr] <- agg$vmin
  }
  
  list(
    neighbor_max  = neighbor_max,
    neighbor_min  = neighbor_min,
    neighbor_mean = neighbor_mean
  )
}

# ==============================================================================
# STEP 4: Matrix back to long-format columns
# ==============================================================================

matrix_to_long_column <- function(mat, dt) {
  # dt must have cell_idx and year_idx columns
  mat[cbind(dt$cell_idx, dt$year_idx)]
}

# ==============================================================================
# STEP 5: Main loop â€” compute neighbor features for all source variables
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing variable: %s\n", var_name))
  
  # Long -> matrix
  val_mat <- long_to_matrix(cell_data, var_name, n_cells, n_years)
  
  # Compute stats
  stats <- compute_neighbor_stats_fast(adj_mat, val_mat, n_neighbors)
  
  # Write back to cell_data
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  cell_data[, (max_col)  := matrix_to_long_column(stats$neighbor_max,  cell_data)]
  cell_data[, (min_col)  := matrix_to_long_column(stats$neighbor_min,  cell_data)]
  cell_data[, (mean_col) := matrix_to_long_column(stats$neighbor_mean, cell_data)]
  
  # Free memory
  rm(val_mat, stats)
  gc()
}

elapsed <- proc.time() - t0
cat(sprintf("Neighbor stats completed in %.1f seconds.\n", elapsed[3]))

# ==============================================================================
# STEP 6: Clean up helper columns, restore original row order
# ==============================================================================

setorder(cell_data, .row_order)
cell_data[, c("cell_idx", "year_idx", ".row_order") := NULL]

# ==============================================================================
# STEP 7: Predict with the pre-trained Random Forest (UNCHANGED)
# ==============================================================================
# The trained RF model object and prediction code remain exactly as before.
# For example:
#   predictions <- predict(rf_model, newdata = cell_data)
#
# The model is NOT retrained. The feature columns have identical names and
# identical numerical values as the original pipeline would have produced.
```

## Summary of Changes

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-entry list built via string hashing | Sparse adjacency matrix built once (344K Ã— 344K) |
| **Mean computation** | `lapply` over 6.46M rows Ã— 5 vars | Sparse matrix multiply (`adj %*% val_mat`), 5 calls |
| **Max/Min computation** | `lapply` over 6.46M rows Ã— 5 vars | Vectorized grouped aggregation over ~1.37M edges Ã— 28 years via `data.table` |
| **Topology reuse** | Rebuilt implicitly per row | Built once, reused for all variables and all years |
| **Estimated runtime** | 86+ hours | ~2â€“10 minutes |
| **Numerical result** | Original estimand | Identical â€” same neighbor sets, same `max`/`min`/`mean` |
| **RF model** | Pre-trained | Untouched, not retrained |
| **RAM** | ~6.46M-length list of integer vectors | Sparse matrix (~22 MB) + one dense 344KÃ—28 matrix (~77 MB) at a time |
END_RESPONSE>>>
