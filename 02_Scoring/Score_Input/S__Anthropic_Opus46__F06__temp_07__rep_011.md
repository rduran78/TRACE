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
run_id: Anthropic_F06_T07_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine â€” O(1) hash lookup).
2. Constructs `paste(neighbor_id, year)` key strings for each neighbor.
3. Looks up those keys in `idx_lookup` (a named character vector â€” hash lookup, but repeated millions of times with string allocation).

This produces a **list of 6.46 million integer vectors**. The string construction (`paste`) and named-vector lookups are executed ~6.46M Ã— ~4 neighbors â‰ˆ 25.8M times. The overhead of creating and storing 6.46M list elements, each with its own allocation, is enormous.

**Key insight:** The neighbor relationships are *spatial* â€” they don't change across years. Every cell has the same neighbors in every year. So the lookup can be factored into (a) a spatial neighbor structure (344K cells) and (b) a year-indexing scheme, rather than constructing 6.46M row-level lookups.

### Bottleneck 2: `compute_neighbor_stats` â€” `lapply` over 6.46M elements

For each of 5 variables, we iterate over the 6.46M-element list, subset a numeric vector, remove NAs, and compute `max/min/mean`. That's 5 Ã— 6.46M = 32.3M R-level function calls with repeated allocation.

**Key insight:** Because the neighbor structure is identical across years, we can compute neighbor stats *within each year slice* using matrix operations. If we reshape the variable into a `cells Ã— years` matrix, then for each cell we simply index its (â‰¤4) spatial neighbors and vectorize across all 28 years simultaneously.

### Why raster focal/kernel operations are a useful analogy but not directly applicable

Raster focal operations (e.g., `terra::focal`) compute neighborhood statistics on regular grids extremely efficiently using compiled C code. The analogy is strong: rook contiguity on a regular grid is exactly a 3Ã—3 cross-shaped kernel. **However**, the data has irregular boundaries (not all 344K cells form a perfect rectangle), cells may be missing in certain years, and the neighbor object is precomputed from `spdep::nb`. Directly using `terra::focal` would require reconstructing the grid, handling NA-masking carefully, and verifying exact numerical equivalence with the original pipeline. The safer and nearly-as-fast approach is sparse-matrix multiplication, which preserves the exact neighbor structure.

### Summary of time sinks

| Step | Calls | Estimated time share |
|---|---|---|
| `build_neighbor_lookup` (string ops, 6.46M rows) | 1 | ~30% |
| `compute_neighbor_stats` (lapply, 6.46M Ã— 5 vars) | 5 | ~65% |
| Feature binding | 5 | ~5% |

---

## Optimization Strategy

### Strategy: Sparse adjacency matrix + split-by-year vectorized computation

1. **Build a sparse adjacency matrix `W`** (344,208 Ã— 344,208) from `rook_neighbors_unique`. This is a one-time operation using the `Matrix` package.

2. **For each variable and each year**, extract the vector of values for all cells, compute `W %*% x` (neighbor sum), and analogous operations for max and min using grouped operations. For **mean**: `neighbor_mean = (W %*% x) / (W %*% 1)` (i.e., sum of neighbor values divided by number of non-NA neighbors). For **max and min**: use a loop over the (at most 4) neighbor columns in a dense neighbor-index matrix.

3. **Avoid creating the 6.46M-element list entirely.** The spatial neighbor structure is encoded once in a matrix; year-slicing is handled by `data.table` grouping.

**Expected speedup:** From 86+ hours to ~2â€“10 minutes.

### Why this preserves the numerical estimand

- The sparse matrix encodes exactly the same rook-neighbor relationships as the original `spdep::nb` object.
- `max`, `min`, and `mean` are computed on exactly the same neighbor sets.
- NA handling is preserved explicitly.
- The trained Random Forest model is not retouched â€” only the feature-engineering step is accelerated.

---

## Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# Replaces: build_neighbor_lookup + compute_neighbor_stats + outer loop
# Preserves: exact numerical results, trained RF model untouched
# ==============================================================================

library(data.table)
library(Matrix)

# --------------------------------------------------------------------------
# Step 0: Ensure cell_data is a data.table with columns: id, year, and the
#         5 neighbor source variables. id_order and rook_neighbors_unique
#         are already loaded (as in the original pipeline).
# --------------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# --------------------------------------------------------------------------
# Step 1: Build a dense neighbor-index matrix from the spdep::nb object.
#
#   rook_neighbors_unique is a list of length n_cells (344,208).
#   rook_neighbors_unique[[i]] contains integer indices (into id_order) of
#   the rook neighbors of cell i.
#
#   We build a matrix `nb_mat` of dimension (n_cells x max_neighbors),
#   where nb_mat[i, j] = the j-th neighbor's index (into id_order), or NA.
#   For rook contiguity on a grid, max_neighbors = 4.
# --------------------------------------------------------------------------

n_cells <- length(id_order)
max_nb  <- max(lengths(rook_neighbors_unique))  # should be 4

# Pre-allocate matrix filled with NA
nb_mat <- matrix(NA_integer_, nrow = n_cells, ncol = max_nb)

for (i in seq_len(n_cells)) {
  nb_i <- rook_neighbors_unique[[i]]
  # spdep::nb uses 0L for cells with no neighbors; filter those out
  nb_i <- nb_i[nb_i > 0L]
  if (length(nb_i) > 0L) {
    nb_mat[i, seq_along(nb_i)] <- nb_i
  }
}

# --------------------------------------------------------------------------
# Step 2: Create a mapping from cell id -> spatial index (position in id_order)
# --------------------------------------------------------------------------
id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))

# --------------------------------------------------------------------------
# Step 3: Add spatial index to cell_data (once)
# --------------------------------------------------------------------------
cell_data[, spatial_idx := id_to_spatial_idx[as.character(id)]]

# --------------------------------------------------------------------------
# Step 4: Sort by year and spatial_idx so that within each year,
#         row positions correspond to a known ordering of cells.
#         We use a "year-group" approach: for each year, we build a
#         values vector aligned to spatial_idx and compute neighbor stats.
# --------------------------------------------------------------------------

# Ensure id_order covers all IDs
stopifnot(all(!is.na(cell_data$spatial_idx)))

# Pre-key for fast grouping
setkey(cell_data, year, spatial_idx)

# Get unique years (sorted)
years <- sort(unique(cell_data$year))

# --------------------------------------------------------------------------
# Step 5: For each variable, compute neighbor max, min, mean across all
#         years using vectorized operations.
# --------------------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# We will collect new columns in a list and cbind at the end (avoids
# repeated modification of the data.table).
new_cols <- list()

for (var_name in neighbor_source_vars) {

  cat("Processing neighbor features for:", var_name, "\n")

  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Pre-allocate result vectors (aligned to cell_data rows after setkey)
  res_max  <- rep(NA_real_, nrow(cell_data))
  res_min  <- rep(NA_real_, nrow(cell_data))
  res_mean <- rep(NA_real_, nrow(cell_data))

  # For each year, operate on the slice
  for (yr in years) {

    # Row indices in cell_data for this year
    yr_rows <- cell_data[.(yr), which = TRUE]

    if (length(yr_rows) == 0L) next

    # Spatial indices for cells present in this year
    sp_idx_yr <- cell_data$spatial_idx[yr_rows]

    # Values for this variable in this year, aligned to sp_idx_yr
    vals_yr <- cell_data[[var_name]][yr_rows]

    # Build a full-length vector (n_cells) so we can index by spatial idx.
    # Cells not present in this year will be NA.
    full_vals <- rep(NA_real_, n_cells)
    full_vals[sp_idx_yr] <- vals_yr

    # For each cell present in this year, gather neighbor values
    # nb_mat[sp_idx_yr, ] gives a (length(yr_rows) x max_nb) matrix of
    # neighbor spatial indices.
    nb_idx_mat <- nb_mat[sp_idx_yr, , drop = FALSE]  # dim: n_yr_rows x max_nb

    # Look up neighbor values: create a matrix of neighbor values
    # full_vals[NA] returns NA, which is correct behavior.
    nb_vals_mat <- matrix(full_vals[nb_idx_mat], nrow = length(yr_rows), ncol = max_nb)

    # Compute row-wise max, min, mean ignoring NAs
    # Use matrixStats for speed if available, otherwise base R

    if (requireNamespace("matrixStats", quietly = TRUE)) {
      r_max  <- matrixStats::rowMaxs(nb_vals_mat,  na.rm = TRUE)
      r_min  <- matrixStats::rowMins(nb_vals_mat,  na.rm = TRUE)
      r_mean <- matrixStats::rowMeans2(nb_vals_mat, na.rm = TRUE)
    } else {
      r_max  <- apply(nb_vals_mat, 1, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0) NA_real_ else max(x)
      })
      r_min  <- apply(nb_vals_mat, 1, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0) NA_real_ else min(x)
      })
      r_mean <- apply(nb_vals_mat, 1, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0) NA_real_ else mean(x)
      })
    }

    # matrixStats returns -Inf/Inf when all values are NA; fix that
    r_max[is.infinite(r_max)] <- NA_real_
    r_min[is.infinite(r_min)] <- NA_real_
    # rowMeans2 returns NaN for all-NA rows
    r_mean[is.nan(r_mean)]    <- NA_real_

    # Store results
    res_max[yr_rows]  <- r_max
    res_min[yr_rows]  <- r_min
    res_mean[yr_rows] <- r_mean
  }

  # Assign to data.table (in-place)
  set(cell_data, j = col_max,  value = res_max)
  set(cell_data, j = col_min,  value = res_min)
  set(cell_data, j = col_mean, value = res_mean)

  cat("  Done:", col_max, col_min, col_mean, "\n")
}

# --------------------------------------------------------------------------
# Step 6: Clean up helper column
# --------------------------------------------------------------------------
cell_data[, spatial_idx := NULL]

# --------------------------------------------------------------------------
# Step 7: Apply the pre-trained Random Forest model (unchanged)
# --------------------------------------------------------------------------
# predictions <- predict(trained_rf_model, newdata = cell_data)
# (This line is left as-is from the original pipeline; the model is not
#  retrained. Only the feature engineering above was optimized.)
# --------------------------------------------------------------------------

cat("Neighbor feature engineering complete.\n")
cat("Columns added:\n")
for (var_name in neighbor_source_vars) {
  cat("  ", paste0("neighbor_max_", var_name), "\n")
  cat("  ", paste0("neighbor_min_", var_name), "\n")
  cat("  ", paste0("neighbor_mean_", var_name), "\n")
}
```

---

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + hash lookups â†’ ~6.46M-element list | One-time 344K Ã— 4 integer matrix (`nb_mat`) |
| **Stats computation** | `lapply` over 6.46M list elements Ã— 5 vars = 32.3M R function calls | 28 year-slices Ã— 5 vars = 140 vectorized matrix operations |
| **Memory** | 6.46M-element list of integer vectors (~500 MB+) | 344K Ã— 4 integer matrix (~5.5 MB) + one full_vals vector (~2.6 MB) |
| **Estimated runtime** | 86+ hours | **2â€“10 minutes** (with `matrixStats`; ~20â€“30 min without) |
| **Numerical equivalence** | Baseline | Identical â€” same neighbor sets, same `max/min/mean` |
| **RF model** | Untouched | Untouched |

### Why this is fast

1. **No per-row R function calls.** The inner loop runs 28 times (years), not 6.46M times (rows).
2. **Matrix indexing is vectorized in C.** `full_vals[nb_idx_mat]` performs ~920K lookups in compiled code per year-slice.
3. **`matrixStats` row operations are compiled C.** `rowMaxs`, `rowMins`, `rowMeans2` avoid R-level loops entirely.
4. **Memory-friendly.** The temporary `full_vals` vector is reused each iteration; no 6.46M-element list is ever created.
END_RESPONSE>>>
