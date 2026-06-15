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
run_id: Anthropic_F05_T07_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The repeated string-key work is **not merely a local inefficiency â€” it is a symptom of a larger repeated-lookup architectural problem** operating at three nested levels:

### Level 1: Per-row string construction inside `build_neighbor_lookup`
For each of the ~6.46 million rows, the code:
- Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` to build string keys.
- Looks those keys up in a named character vector (`idx_lookup`) of length 6.46M.

Named-vector lookup in R is **O(n)** per query (linear scan of names), not O(1). With ~4 neighbors per cell on average, this means roughly **6.46M Ã— 4 = 25.8M** string constructions and linear lookups against a 6.46M-element named vector. This alone is catastrophic.

### Level 2: The lookup is year-redundant
Every neighbor relationship is **time-invariant** â€” cell A is a rook neighbor of cell B in every year. Yet the code re-discovers the same spatial neighbor mapping 28 times (once per year of each cell). The `paste(..., data$year[i], ...)` key encodes year, but the spatial topology doesn't change across years. The entire `build_neighbor_lookup` function does 28Ã— redundant work.

### Level 3: Per-variable re-traversal in `compute_neighbor_stats`
After building the lookup, `compute_neighbor_stats` is called once per variable. Each call iterates over all 6.46M rows and gathers neighbor values with an R-level `lapply`. For 5 variables, that's 5 Ã— 6.46M = 32.3M R-level function calls, each allocating small vectors.

**Summary of the scaling problem:**

| Layer | Current cost | Root cause |
|-------|-------------|------------|
| String key construction | ~25.8M `paste` calls | Year encoded in spatial key |
| Named-vector lookup | O(n) per lookup Ã— 25.8M | Named vector, not hash |
| Year duplication | 28Ã— redundant spatial work | Year-invariant topology re-queried per cell-year |
| Per-variable traversal | 5 Ã— 6.46M R-level iterations | Separate `lapply` per variable |

Estimated combined operations: billions of R-level string ops. This explains the 86+ hour estimate.

---

## 2. Optimization Strategy

### Core insight: Separate the spatial topology (time-invariant) from the panel indexing (time-varying)

1. **Build a spatial-only neighbor index once** over the 344,208 unique cells. This is a simple integer-indexed list â€” no strings, no hashing.

2. **Build a (cell, year) â†’ row-index mapping** using `data.table` or a simple integer matrix, enabling O(1) lookup by integer cell-index and year-offset.

3. **Vectorize the neighbor-stats computation** using matrix operations: reshape each variable into a `cells Ã— years` matrix, then compute neighbor max/min/mean using vectorized column operations over the neighbor list. This eliminates all per-row R-level iteration.

4. **Process all 5 variables** in the same matrix framework.

**Expected speedup:** From ~86 hours to **minutes** (roughly 3â€“5 minutes on a modern laptop).

**Numerical equivalence:** The reformulation computes the identical `max`, `min`, `mean` of the identical neighbor sets, so the trained Random Forest model remains valid with no retraining.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# =============================================================================
# Prerequisites:
#   - cell_data: data.frame/data.table with columns 'id', 'year', and all
#                predictor variables. ~6.46M rows.
#   - id_order: integer vector of 344,208 cell IDs in the order used by the
#               nb object.
#   - rook_neighbors_unique: an nb object (list of integer index vectors)
#               of length 344,208.
#   - neighbor_source_vars: c("ntl","ec","pop_density","def","usd_est_n2")
#
# Output:
#   - cell_data with 15 new columns: {var}_nb_max, {var}_nb_min, {var}_nb_mean
#     for each of the 5 variables. Numerically identical to the original code.
# =============================================================================

library(data.table)

# --- Step 0: Convert to data.table if needed --------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# --- Step 1: Build integer-indexed spatial neighbor list (ONCE) -------------
# Map each cell id to its position in id_order (1-based integer index).
# rook_neighbors_unique[[k]] already contains integer indices into id_order,
# so no remapping is needed for the nb list itself.

n_cells <- length(id_order)                          # 344,208
years    <- sort(unique(cell_data$year))              # 1992:2019
n_years  <- length(years)                             # 28

# Map cell id -> integer cell index (position in id_order)
id_to_cidx <- setNames(seq_len(n_cells), as.character(id_order))

# Map year -> integer year index (1..28)
year_to_yidx <- setNames(seq_len(n_years), as.character(years))

# --- Step 2: Build (cell_index, year_index) -> row mapping ------------------
# We need to go from (cidx, yidx) to row number in cell_data.
# Use an integer matrix of size n_cells x n_years.

cat("Building row-index matrix...\n")

# Compute cell index and year index for every row
cell_data[, c_cidx := id_to_cidx[as.character(id)]]
cell_data[, c_yidx := year_to_yidx[as.character(year)]]

# row_matrix[cidx, yidx] = row number in cell_data (or NA if missing)
row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
row_matrix[cbind(cell_data$c_cidx, cell_data$c_yidx)] <- seq_len(nrow(cell_data))

cat("Row-index matrix built:", n_cells, "x", n_years, "\n")

# --- Step 3: Precompute neighbor row indices for every cell-year ------------
# For each cell-year row, we need the row indices of its spatial neighbors
# in the same year. We can compute this as a list of length nrow(cell_data).
#
# Strategy: iterate over years (28 iterations), and within each year,
# vectorize across all cells using the nb list.

cat("Building neighbor row-index lookup (vectorized over years)...\n")

# Pre-extract the nb list as a plain list of integer vectors for speed
nb_list <- lapply(rook_neighbors_unique, function(x) {
  # spdep nb objects use 0 for no-neighbor; filter those out
  x <- as.integer(x)
  x[x > 0L]
})

# Allocate the output list (one entry per row of cell_data)
neighbor_lookup <- vector("list", nrow(cell_data))

for (yy in seq_len(n_years)) {
  # Which rows in cell_data correspond to this year?
  rows_this_year <- which(cell_data$c_yidx == yy)
  cidx_this_year <- cell_data$c_cidx[rows_this_year]

  # For each cell present this year, look up its neighbors' rows in the
  # same year using row_matrix[, yy]
  year_col <- row_matrix[, yy]  # length n_cells; NA where cell absent

  for (j in seq_along(rows_this_year)) {
    row_j <- rows_this_year[j]
    nb_cidxs <- nb_list[[cidx_this_year[j]]]
    if (length(nb_cidxs) == 0L) {
      neighbor_lookup[[row_j]] <- integer(0)
    } else {
      nb_rows <- year_col[nb_cidxs]
      neighbor_lookup[[row_j]] <- nb_rows[!is.na(nb_rows)]
    }
  }

  if (yy %% 5 == 0) cat("  Year", years[yy], "done\n")
}

cat("Neighbor lookup built.\n")

# --- Step 4: Compute neighbor stats vectorized via data.table ---------------
# For each variable, extract neighbor values and compute max/min/mean.
# We use vapply for type-stable fast iteration.

compute_and_add_neighbor_features_fast <- function(dt, var_name, nb_lookup) {
  vals <- dt[[var_name]]

  stats <- vapply(nb_lookup, function(idx) {
    if (length(idx) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    c(max(nv), min(nv), mean(nv))
  }, numeric(3))
  # stats is 3 x nrow matrix

  col_max  <- paste0(var_name, "_nb_max")
  col_min  <- paste0(var_name, "_nb_min")
  col_mean <- paste0(var_name, "_nb_mean")

  dt[, (col_max)  := stats[1L, ]]
  dt[, (col_min)  := stats[2L, ]]
  dt[, (col_mean) := stats[3L, ]]

  dt
}

cat("Computing neighbor statistics for source variables...\n")

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("  Processing:", var_name, "...\n")
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_lookup
  )
  cat("  Done:", var_name, "\n")
}

# --- Step 5: Clean up temporary columns ------------------------------------
cell_data[, c("c_cidx", "c_yidx") := NULL]

cat("All neighbor features computed.\n")
```

---

## 4. Further Speedup: Fully Vectorized Matrix Approach (Optional)

If the `vapply` loop in Step 4 is still too slow (~6.46M iterations Ã— 5 variables), here is a **fully matrix-vectorized** alternative for Step 4 that avoids all R-level per-row iteration for the statistics computation:

```r
# =============================================================================
# ALTERNATIVE STEP 4: Fully matrix-vectorized neighbor stats
# =============================================================================
# Idea: Reshape each variable into a (n_cells x n_years) matrix.
# For each cell, gather neighbor rows from the matrix and compute stats
# across neighbors (vectorized across years via column operations).
# =============================================================================

compute_all_neighbor_features_matrix <- function(dt, source_vars, nb_list,
                                                  id_to_cidx, years,
                                                  row_matrix, n_cells, n_years) {

  cidx_vec <- id_to_cidx[as.character(dt$id)]
  yidx_vec <- match(dt$year, years)

  for (var_name in source_vars) {
    cat("  Matrix method for:", var_name, "\n")

    # Build cell x year matrix of the variable
    var_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    var_mat[cbind(cidx_vec, yidx_vec)] <- dt[[var_name]]

    # For each cell, compute neighbor stats across all years at once
    # Result matrices: n_cells x n_years
    nb_max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (ci in seq_len(n_cells)) {
      nb_idx <- nb_list[[ci]]
      if (length(nb_idx) == 0L) next

      # nb_vals is a (n_neighbors x n_years) sub-matrix
      nb_vals <- var_mat[nb_idx, , drop = FALSE]

      # Compute column-wise stats (across neighbors, for each year)
      # Using colMeans etc. but need to handle NAs
      n_valid <- colSums(!is.na(nb_vals))
      has_data <- n_valid > 0L

      if (any(has_data)) {
        # suppressWarnings for all-NA columns
        nb_max_mat[ci, has_data]  <- suppressWarnings(
          apply(nb_vals[, has_data, drop = FALSE], 2, max, na.rm = TRUE)
        )
        nb_min_mat[ci, has_data]  <- suppressWarnings(
          apply(nb_vals[, has_data, drop = FALSE], 2, min, na.rm = TRUE)
        )
        nb_mean_mat[ci, has_data] <- colMeans(
          nb_vals[, has_data, drop = FALSE], na.rm = TRUE
        )
      }
    }

    # Map back to row order in dt
    col_max  <- paste0(var_name, "_nb_max")
    col_min  <- paste0(var_name, "_nb_min")
    col_mean <- paste0(var_name, "_nb_mean")

    dt[, (col_max)  := nb_max_mat[cbind(cidx_vec, yidx_vec)]]
    dt[, (col_min)  := nb_min_mat[cbind(cidx_vec, yidx_vec)]]
    dt[, (col_mean) := nb_mean_mat[cbind(cidx_vec, yidx_vec)]]

    cat("    Done:", var_name, "\n")
  }

  dt
}

# Usage:
cell_data <- compute_all_neighbor_features_matrix(
  cell_data, neighbor_source_vars, nb_list,
  id_to_cidx, years, row_matrix, n_cells, n_years
)
```

**Trade-off:** The matrix approach loops over 344,208 cells (not 6.46M rows) and vectorizes across 28 years per cell. Each matrix is ~344K Ã— 28 â‰ˆ 77 MB, well within 16 GB RAM for 5 variables.

---

## 5. Performance Summary

| Aspect | Original | Optimized (Step 4a) | Matrix (Step 4b) |
|--------|----------|-------------------|-----------------|
| Neighbor lookup construction | ~25.8M string ops + O(n) lookup | 28 year-loops Ã— ~344K integer lookups | Same |
| Stats computation | 5 Ã— 6.46M `lapply` with string overhead | 5 Ã— 6.46M `vapply` (integer only) | 5 Ã— 344K cell-loops, vectorized over 28 years |
| Estimated runtime | ~86+ hours | ~15â€“30 minutes | ~3â€“10 minutes |
| Numerical output | Baseline | Identical | Identical |
| RAM overhead | String vector ~1 GB | Integer matrix ~37 MB | +5 Ã— 77 MB matrices |
| RF model impact | â€” | None (same features) | None (same features) |

The trained Random Forest model requires no retraining because the output columns are numerically identical to the original implementation.
END_RESPONSE>>>
