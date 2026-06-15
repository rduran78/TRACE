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
run_id: Anthropic_F08_T07_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row neighbor indices for all ~6.46 million cell-year rows.** It does this by pasting `id_year` keys and looking them up in a named vector. But the neighbor *relationships* are purely spatial â€” cell A is a neighbor of cell B regardless of year. The topology is static across all 28 years. The function needlessly recomputes 6.46M lists when only 344,208 cell-level lists are needed.

2. **`compute_neighbor_stats` indexes into the full 6.46M-row data frame using the bloated per-row lookup.** Because the lookup was built at the cell-year level, every stats computation carries the overhead of the inflated structure.

3. **String-based key construction (`paste(id, year, sep="_")`) and named-vector lookups (`setNames`, `idx_lookup[neighbor_keys]`)** are extremely slow at this scale â€” millions of string allocations, hashing, and named-vector searches.

4. **`lapply` over 6.46M rows** with per-element R function calls creates massive interpreter overhead.

### Summary

| Aspect | Current | Optimal |
|---|---|---|
| Neighbor lookup granularity | 6.46M cell-year rows | 344,208 cells (once) |
| Lookup rebuild per year? | Implicitly yes (embedded) | No â€” static, built once |
| Key mechanism | String paste + named vector | Integer index matrix |
| Stats loop | `lapply` over 6.46M rows | Vectorized matrix ops per year |
| Estimated time | 86+ hours | ~2â€“5 minutes |

---

## Optimization Strategy

**Principle: Separate the static neighbor graph from the dynamic yearly variable values.**

1. **Build a cell-level neighbor lookup once** â€” a list of length 344,208 where each element contains integer indices into the *cell-order* vector (not the data frame). This is topology and never changes.

2. **For each year, extract the variable column as a vector indexed by cell order.** Use the static cell-level neighbor lookup to compute max/min/mean via vectorized operations over that vector.

3. **Use `data.table` for fast split-by-year and column assignment**, avoiding copies.

4. **Pre-build a sparse neighbor matrix (or padded neighbor matrix) to fully vectorize** the neighbor aggregation, eliminating all `lapply` calls over millions of rows.

The trained Random Forest model is never touched. The output columns (`*_neighbor_max`, `*_neighbor_min`, `*_neighbor_mean`) are numerically identical to the original implementation.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build the STATIC cell-level neighbor lookup (done ONCE)
# ==============================================================================
# Inputs:
#   id_order            â€” vector of 344,208 cell IDs in the canonical order
#   rook_neighbors_unique â€” spdep nb object (list of length 344,208)
#
# This maps each cell (by its position in id_order) to the positions of its
# neighbors in id_order. This is pure topology â€” no year dependency.
# ==============================================================================

build_cell_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors is an nb object: neighbors[[i]] gives integer indices into
  # id_order for the neighbors of id_order[i].
  # We just need to ensure 0-neighbor cells return integer(0).
  n <- length(id_order)
  lookup <- vector("list", n)
  for (i in seq_len(n)) {
    nb_idx <- neighbors[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    if (length(nb_idx) == 1L && nb_idx[1] == 0L) {
      lookup[[i]] <- integer(0)
    } else {
      lookup[[i]] <- as.integer(nb_idx)
    }
  }
  lookup
}

# Build it once
cell_neighbor_lookup <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)

# ==============================================================================
# STEP 2: Pre-build a padded neighbor matrix for fully vectorized operations
# ==============================================================================
# Convert the ragged list into a fixed-width integer matrix (n_cells x max_neighbors).
# Cells with fewer neighbors get NA padding.
# This enables vectorized matrix-column extraction instead of lapply.
# ==============================================================================

build_padded_neighbor_matrix <- function(cell_neighbor_lookup) {
  max_nb <- max(vapply(cell_neighbor_lookup, length, integer(1)))
  n <- length(cell_neighbor_lookup)
  mat <- matrix(NA_integer_, nrow = n, ncol = max_nb)
  for (i in seq_len(n)) {
    nb <- cell_neighbor_lookup[[i]]
    if (length(nb) > 0L) {
      mat[i, seq_along(nb)] <- nb
    }
  }
  mat
}

nb_matrix <- build_padded_neighbor_matrix(cell_neighbor_lookup)
# nb_matrix: 344,208 rows x max_neighbors cols (typically 4 for rook)

# ==============================================================================
# STEP 3: Vectorized neighbor stats computation for one variable, one year
# ==============================================================================
# Given a numeric vector of values (length = n_cells, ordered by id_order)
# and the padded neighbor matrix, compute max/min/mean across neighbors
# entirely with vectorized matrix operations.
# ==============================================================================

compute_neighbor_stats_vectorized <- function(vals, nb_matrix) {
  # vals: numeric vector of length n_cells (one value per cell for one year)
  # nb_matrix: integer matrix (n_cells x max_neighbors), indices into vals
  
  n <- length(vals)
  k <- ncol(nb_matrix)
  
  # Build a matrix of neighbor values: n_cells x max_neighbors
  # Use vals[nb_matrix], which vectorizes the lookup.
  # NA indices (padding) will produce NA values â€” correct behavior.
  nb_vals <- matrix(vals[nb_matrix], nrow = n, ncol = k)
  
  # Compute row-wise stats, ignoring NAs
  # For cells with ALL neighbors NA (no neighbors or all neighbor vals NA),
  # these will return appropriate NA/NaN â€” we fix below.
  nb_max  <- apply(nb_vals, 1, max,  na.rm = TRUE)
  nb_min  <- apply(nb_vals, 1, min,  na.rm = TRUE)
  nb_mean <- rowMeans(nb_vals, na.rm = TRUE)  # fast C-level
  
  # Fix Inf/-Inf from max/min on all-NA rows
  nb_max[is.infinite(nb_max)] <- NA_real_
  nb_min[is.infinite(nb_min)] <- NA_real_
  # rowMeans already returns NaN for all-NA rows; convert to NA
  nb_mean[is.nan(nb_mean)] <- NA_real_
  
  data.table(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}

# ==============================================================================
# STEP 4: Optimized alternative using matrixStats for even faster row ops
# ==============================================================================
# If matrixStats is available, use rowMaxs/rowMins for C-level speed.
# Falls back to apply() otherwise.
# ==============================================================================

if (requireNamespace("matrixStats", quietly = TRUE)) {
  compute_neighbor_stats_fast <- function(vals, nb_matrix) {
    n <- length(vals)
    k <- ncol(nb_matrix)
    nb_vals <- matrix(vals[nb_matrix], nrow = n, ncol = k)
    
    nb_max  <- matrixStats::rowMaxs(nb_vals, na.rm = TRUE)
    nb_min  <- matrixStats::rowMins(nb_vals, na.rm = TRUE)
    nb_mean <- rowMeans(nb_vals, na.rm = TRUE)
    
    nb_max[is.infinite(nb_max)] <- NA_real_
    nb_min[is.infinite(nb_min)] <- NA_real_
    nb_mean[is.nan(nb_mean)]    <- NA_real_
    
    data.table(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
  }
} else {
  compute_neighbor_stats_fast <- compute_neighbor_stats_vectorized
}

# ==============================================================================
# STEP 5: Main loop â€” iterate over variables, iterate over years
# ==============================================================================
# Convert cell_data to data.table for fast operations.
# For each variable and each year:
#   1. Extract the variable values in id_order for that year.
#   2. Compute vectorized neighbor stats (on 344K cells, not 6.46M rows).
#   3. Write results back into the corresponding rows of cell_data.
# ==============================================================================

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Build a mapping from cell ID to position in id_order (static)
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Ensure cell_data has a column mapping each row to its cell position in id_order
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# Get sorted unique years
years <- sort(unique(cell_data$year))

# Neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate output columns
for (var_name in neighbor_source_vars) {
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  if (is.null(cell_data[[col_max]]))  set(cell_data, j = col_max,  value = NA_real_)
  if (is.null(cell_data[[col_min]]))  set(cell_data, j = col_min,  value = NA_real_)
  if (is.null(cell_data[[col_mean]])) set(cell_data, j = col_mean, value = NA_real_)
}

# Key the data.table for fast subsetting by year
setkey(cell_data, year)

# Main computation loop
n_cells <- length(id_order)

for (var_name in neighbor_source_vars) {
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))
  t0 <- proc.time()
  
  for (yr in years) {
    # Get row indices for this year (fast via key)
    yr_rows <- cell_data[.(yr), which = TRUE]
    
    # Extract cell positions and variable values for this year
    yr_cell_pos <- cell_data$cell_pos[yr_rows]
    yr_vals_raw <- cell_data[[var_name]][yr_rows]
    
    # Build a full-length vector indexed by cell position
    # (handles case where not all cells appear in every year)
    vals_by_cell <- rep(NA_real_, n_cells)
    vals_by_cell[yr_cell_pos] <- yr_vals_raw
    
    # Compute neighbor stats â€” vectorized over all 344,208 cells
    stats <- compute_neighbor_stats_fast(vals_by_cell, nb_matrix)
    
    # Write results back â€” only for cells present in this year
    set(cell_data, i = yr_rows, j = col_max,  value = stats$nb_max[yr_cell_pos])
    set(cell_data, i = yr_rows, j = col_min,  value = stats$nb_min[yr_cell_pos])
    set(cell_data, i = yr_rows, j = col_mean, value = stats$nb_mean[yr_cell_pos])
  }
  
  elapsed <- (proc.time() - t0)[3]
  cat(sprintf("  Done in %.1f seconds\n", elapsed))
}

# Clean up helper column
cell_data[, cell_pos := NULL]

# ==============================================================================
# STEP 6: Proceed to prediction with the pre-trained Random Forest
# ==============================================================================
# The cell_data now has all *_neighbor_max, *_neighbor_min, *_neighbor_mean
# columns populated with numerically identical values to the original code.
# The trained RF model is unchanged â€” proceed directly to predict().
#
# Example:
#   predictions <- predict(trained_rf_model, newdata = cell_data)
# ==============================================================================

cat("Neighbor feature computation complete. Ready for Random Forest prediction.\n")
```

---

## Performance Analysis

| Metric | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M string-key lists | 344K integer lists (once) + padded matrix |
| Per-variable-year computation | `lapply` over 6.46M rows | Vectorized matrix ops on 344K cells |
| Total iterations (inner) | 6.46M Ã— 5 vars = 32.3M | 344K Ã— 5 vars Ã— 28 yrs = 48.2M cells, but vectorized |
| String operations | ~19.4M `paste()` calls | Zero |
| Estimated wall time | 86+ hours | **2â€“5 minutes** |
| Peak memory | Bloated lookup lists | ~344K Ã— 4 matrix (5.5 MB) + data.table in-place |
| Numerical result | Reference | **Identical** |

### Why it's correct (preserves the estimand)

- For each cell-year row, the neighbor max/min/mean is computed over exactly the same set of rook-neighbor cells for the same year, using the same variable values.
- The padded matrix `nb_matrix` encodes exactly the same adjacency as `rook_neighbors_unique`.
- `vals[nb_matrix]` with NA padding + `na.rm = TRUE` produces identical results to the original's `neighbor_vals[!is.na(neighbor_vals)]` filtering.
- The pre-trained Random Forest model is never modified â€” only `predict()` is called on the enriched data.
END_RESPONSE>>>
