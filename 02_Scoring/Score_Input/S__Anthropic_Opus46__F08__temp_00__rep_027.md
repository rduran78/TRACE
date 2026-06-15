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
run_id: Anthropic_F08_T00_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor topology is **identical across all 28 years**. This means:

1. **Redundant lookup construction**: The neighbor graph has only 344,208 cells, but the function builds 6.46M entries (344,208 Ã— 28) by re-resolving the same neighbor cell IDs for every year. This is a 28Ã— blowup in both time and memory.

2. **Redundant string hashing**: `paste(id, year, sep="_")` is called millions of times to create keys, and `idx_lookup[neighbor_keys]` performs millions of named-vector lookups (which are O(n) hash lookups on character vectors of length 6.46M).

3. **Row-level R `lapply` over 6.46M rows**: Each iteration does allocation, subsetting, `paste`, and NA checking â€” all in interpreted R. This is the dominant wall-clock cost.

4. **`compute_neighbor_stats` also loops over 6.46M entries**: Even though the neighbor *indices* within a given year are structurally the same for each cell, they are recomputed per row.

**In summary**: The static neighbor topology is entangled with the dynamic year dimension, causing a ~28Ã— blowup in work and memory, compounded by slow interpreted-R loops over millions of rows.

## Optimization Strategy

**Separate the static topology from the dynamic data:**

1. **Build the neighbor lookup once, over cells only (344,208 entries)**, mapping each cell index to its neighbor cell indices. This is year-independent.

2. **For each variable, extract a matrix of values**: rows = cells, columns = years. This allows vectorized column-wise (year-wise) operations.

3. **Compute neighbor max/min/mean using vectorized matrix operations**: For each cell, gather neighbor rows from the matrix, then compute stats across neighbors for all years simultaneously. Better yet, use sparse-matrix or direct C++-level operations.

4. **Use `data.table` for fast reshaping** and avoid `paste`-based key lookups entirely.

5. **Optionally use a sparse adjacency matrix** to compute neighbor means as a matrix multiply, and neighbor max/min via row-wise sparse operations.

This reduces the problem from 6.46M interpreted-R iterations to ~344K iterations (or fully vectorized sparse-matrix operations), and eliminates all string-key hashing.

## Working R Code

```r
library(data.table)
library(Matrix)

# ===========================================================================
# STEP 1: Build the static cell-level neighbor lookup ONCE
#         (344,208 cells, not 6.46M cell-years)
# ===========================================================================

# id_order: vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: spdep nb object (list of integer index vectors)

# We need a mapping from cell ID -> position in id_order
# rook_neighbors_unique[[i]] gives neighbor positions for id_order[i]

# Build a sparse adjacency matrix (344208 x 344208) from the nb object.
# This is the static topology.

build_sparse_adjacency <- function(nb_obj, n) {
  # nb_obj: list of integer vectors (neighbor indices), length n
  # Returns a sparse logical/numeric matrix of dimension n x n
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove 0-neighbor entries (spdep uses integer(0) for islands)
  valid <- !is.na(to) & to > 0
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

n_cells <- length(id_order)
cat("Building sparse adjacency matrix for", n_cells, "cells...\n")
W <- build_sparse_adjacency(rook_neighbors_unique, n_cells)

# ===========================================================================
# STEP 2: Convert cell_data to data.table; create cell index and year index
# ===========================================================================

cell_dt <- as.data.table(cell_data)

# Create a mapping from cell ID to cell index (position in id_order)
id_to_cidx <- setNames(seq_along(id_order), as.character(id_order))
cell_dt[, cidx := id_to_cidx[as.character(id)]]

# Create ordered year vector and year index
years_vec <- sort(unique(cell_dt$year))
n_years   <- length(years_vec)
year_to_yidx <- setNames(seq_along(years_vec), as.character(years_vec))
cell_dt[, yidx := year_to_yidx[as.character(year)]]

# Ensure data is keyed for fast access
setkey(cell_dt, cidx, yidx)

# ===========================================================================
# STEP 3: Function to compute neighbor stats for one variable
#         using sparse matrix operations (fully vectorized)
# ===========================================================================

compute_neighbor_features_sparse <- function(dt, W, var_name, id_order,
                                              n_cells, years_vec, n_years) {
  # Build a cell x year matrix of the variable values
  # dt must have columns: cidx, yidx, and var_name
  val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  val_mat[cbind(dt$cidx, dt$yidx)] <- dt[[var_name]]

  # --- Neighbor MEAN ---
  # W %*% val_mat gives sum of neighbor values for each cell x year

  # We need the count of non-NA neighbors per cell x year to get the mean
  not_na_mat <- matrix(0, nrow = n_cells, ncol = n_years)
  not_na_mat[cbind(dt$cidx, dt$yidx)] <- as.numeric(!is.na(dt[[var_name]]))

  # Replace NA with 0 for the sum computation
  val_mat_zero <- val_mat
  val_mat_zero[is.na(val_mat_zero)] <- 0

  neighbor_sum   <- as.matrix(W %*% val_mat_zero)   # n_cells x n_years
  neighbor_count <- as.matrix(W %*% not_na_mat)      # n_cells x n_years

  neighbor_mean <- neighbor_sum / neighbor_count
  neighbor_mean[neighbor_count == 0] <- NA_real_

  # --- Neighbor MAX and MIN ---
  # These cannot be done with simple matrix multiply.
  # We iterate over cells (344K iterations, not 6.46M).
  # Use the nb object directly for the neighbor list.

  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # Pre-extract the neighbor list from the sparse matrix
  # (or reuse rook_neighbors_unique directly)
  # Using rook_neighbors_unique is fastest since it's already a list.

  for (i in seq_len(n_cells)) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) next
    # Extract neighbor rows from val_mat: each row is a neighbor, columns are years
    nb_vals <- val_mat[nb_idx, , drop = FALSE]  # k_neighbors x n_years
    # Columnwise max and min (suppress warnings for all-NA columns)
    neighbor_max[i, ] <- suppressWarnings(apply(nb_vals, 2, max, na.rm = TRUE))
    neighbor_min[i, ] <- suppressWarnings(apply(nb_vals, 2, min, na.rm = TRUE))
  }
  # Fix Inf/-Inf from all-NA slices
  neighbor_max[is.infinite(neighbor_max)] <- NA_real_
  neighbor_min[is.infinite(neighbor_min)] <- NA_real_

  # --- Write results back to dt ---
  max_col <- paste0("neighbor_max_", var_name)
  min_col <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  dt[, (max_col)  := neighbor_max[cbind(cidx, yidx)]]
  dt[, (min_col)  := neighbor_min[cbind(cidx, yidx)]]
  dt[, (mean_col) := neighbor_mean[cbind(cidx, yidx)]]

  return(dt)
}

# ===========================================================================
# STEP 4: Further optimize MAX/MIN with chunked C-style vectorization
#         (avoid per-cell apply loop using vapply + direct indexing)
# ===========================================================================

# Faster version: pre-compute neighbor pointer arrays and use vectorized ops
compute_neighbor_features_fast <- function(dt, W, var_name,
                                            n_cells, n_years,
                                            nb_list) {
  # Build cell x year matrix
  val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  val_mat[cbind(dt$cidx, dt$yidx)] <- dt[[var_name]]

  # --- MEAN via sparse matrix multiply ---
  val_mat_zero <- val_mat
  val_mat_zero[is.na(val_mat_zero)] <- 0

  not_na <- matrix(0, nrow = n_cells, ncol = n_years)
  not_na[cbind(dt$cidx, dt$yidx)] <- as.numeric(!is.na(dt[[var_name]]))

  neighbor_sum   <- as.matrix(W %*% val_mat_zero)

  neighbor_count <- as.matrix(W %*% not_na)
  neighbor_mean  <- neighbor_sum / neighbor_count
  neighbor_mean[neighbor_count == 0] <- NA_real_

  # --- MAX / MIN: vectorized over years, loop over cells ---
  # Pre-allocate
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # Process in chunks to be cache-friendly
  chunk_size <- 10000L
  n_chunks <- ceiling(n_cells / chunk_size)

  for (ch in seq_len(n_chunks)) {
    i_start <- (ch - 1L) * chunk_size + 1L
    i_end   <- min(ch * chunk_size, n_cells)

    for (i in i_start:i_end) {
      nb_idx <- nb_list[[i]]
      if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) next

      if (length(nb_idx) == 1L) {
        neighbor_max[i, ] <- val_mat[nb_idx, ]
        neighbor_min[i, ] <- val_mat[nb_idx, ]
      } else {
        nb_block <- val_mat[nb_idx, , drop = FALSE]
        neighbor_max[i, ] <- colMaxs_na(nb_block)
        neighbor_min[i, ] <- colMins_na(nb_block)
      }
    }
  }

  # Write back
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  dt[, (max_col)  := neighbor_max[cbind(cidx, yidx)]]
  dt[, (min_col)  := neighbor_min[cbind(cidx, yidx)]]
  dt[, (mean_col) := neighbor_mean[cbind(cidx, yidx)]]

  return(dt)
}

# Helper: column-wise max/min ignoring NAs (faster than apply)
colMaxs_na <- function(mat) {
  # mat: k x n_years
  result <- mat[1L, ]
  if (nrow(mat) > 1L) {
    for (r in 2L:nrow(mat)) {
      row_r <- mat[r, ]
      update <- !is.na(row_r) & (is.na(result) | row_r > result)
      result[update] <- row_r[update]
    }
  }
  result
}

colMins_na <- function(mat) {
  result <- mat[1L, ]
  if (nrow(mat) > 1L) {
    for (r in 2L:nrow(mat)) {
      row_r <- mat[r, ]
      update <- !is.na(row_r) & (is.na(result) | row_r < result)
      result[update] <- row_r[update]
    }
  }
  result
}

# ===========================================================================
# STEP 5: Run the pipeline
# ===========================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features for", length(neighbor_source_vars), "variables...\n")
cat("Cells:", n_cells, " Years:", n_years, " Rows:", nrow(cell_dt), "\n")

for (var_name in neighbor_source_vars) {
  cat("  Processing:", var_name, "...\n")
  t0 <- proc.time()

  cell_dt <- compute_neighbor_features_fast(
    dt       = cell_dt,
    W        = W,
    var_name = var_name,
    n_cells  = n_cells,
    n_years  = n_years,
    nb_list  = rook_neighbors_unique
  )

  elapsed <- (proc.time() - t0)[3]
  cat("    Done in", round(elapsed / 60, 1), "minutes.\n")
}

# ===========================================================================
# STEP 6: Clean up helper columns and convert back if needed
# ===========================================================================

cell_dt[, c("cidx", "yidx") := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(cell_dt)

# ===========================================================================
# STEP 7: Predict with the pre-trained Random Forest (unchanged)
# ===========================================================================
# predictions <- predict(rf_model, newdata = cell_data)
# (The RF model is not retrained; only the feature-engineering step is optimized.)
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup size** | 6.46M entries (cell Ã— year) | 344K entries (cell only) |
| **String key hashing** | ~25M+ `paste` + named-vector lookups | **Zero** â€” integer indexing only |
| **Neighbor mean** | R loop over 6.46M rows | Sparse matrix multiply (`W %*% V`), fully vectorized |
| **Neighbor max/min** | R loop over 6.46M rows | R loop over 344K cells, vectorized across 28 years per cell |
| **Memory** | 6.46M-element list of integer vectors + 6.46M-length character key vector | 344K Ã— 28 dense matrix (~77 MB per variable) + sparse W (~22 MB) |
| **Estimated time** | 86+ hours | **~15â€“45 minutes** (depending on hardware) |
| **Numerical result** | â€” | **Identical** (same max, min, mean computed from same neighbor sets) |
| **RF model** | â€” | **Untouched** â€” no retraining |

The key insight: **the neighbor graph is a static spatial property; only the variable values change by year.** By separating topology (sparse matrix / nb list, built once) from data (cell Ã— year matrices, one per variable), we eliminate the 28Ã— redundancy and unlock vectorized linear algebra for the mean and vectorized column operations for max/min.
END_RESPONSE>>>
