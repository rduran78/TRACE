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
run_id: Anthropic_F12_T10_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it's a single matrix assembly step per variable â€” only 5 calls total. Each inner function returns a fixed-length-3 numeric vector, so there is no repeated list *growth* (no quadratic copy pattern). This is O(n) work per variable and takes seconds, not hours.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **`paste()` + named-vector lookup for 6.46M rows Ã— ~4 neighbors each â‰ˆ 25.8M string key constructions and hash lookups.** The function builds `idx_lookup` â€” a named vector of length 6.46M â€” and then for every single row, constructs string keys (`paste(neighbor_id, year, sep="_")`) and performs named-vector lookup (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query in the worst case because R's named vectors use a linear hashing structure that degrades with size.

2. **`lapply` over 6.46M rows with per-row `as.character()`, `paste()`, and subsetting** is inherently slow â€” the per-element overhead of R's interpreter is massive at this scale.

3. **The lookup is redundant across years.** The neighbor *topology* is purely spatial (rook contiguity), yet the function recomputes string-based lookups for every cell-year row. With 344,208 cells and 28 years, the same spatial neighbor structure is being re-derived 28 times via expensive string operations.

**Estimated cost:** 6.46M iterations Ã— (string construction + hash lookup into a 6.46M-entry named vector + NA filtering) â‰ˆ tens of hours on a laptop. This is the 86+ hour bottleneck, not the `rbind`.

## Optimization Strategy

1. **Eliminate the string-key lookup entirely.** Since the data has a regular panel structure (every cell appears in every year), we can exploit the fact that if the data is sorted by `(id, year)` or `(year, id)`, the row index of any `(cell, year)` combination is arithmetically determinable â€” no hash table needed.

2. **Precompute a spatial-only neighbor index (344K entries, not 6.46M).** Map each cell's position-in-`id_order` to its neighbors' positions-in-`id_order` once. Then for any row, the neighbor rows are computed by simple integer arithmetic.

3. **Vectorize `compute_neighbor_stats()` using matrix indexing** instead of `lapply` over 6.46M elements. We can use a sparse-matrix or direct column-vector approach to compute neighbor max/min/mean in bulk.

4. **Overall complexity reduction:** from O(N Ã— k Ã— string_ops) to O(N Ã— k Ã— integer_arithmetic), where N = 6.46M and k â‰ˆ 4 average neighbors. Expected speedup: ~100â€“500Ã—, bringing runtime from 86+ hours to minutes.

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” preserves trained RF model & original numerical estimand
# =============================================================================

library(data.table)

# ---- Step 0: Convert to data.table for fast ordered operations ---------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure a consistent ordering: by id, then by year
# We need to know the unique IDs in the same order as id_order
# and the unique years sorted.
unique_years <- sort(unique(cell_data$year))
n_years      <- length(unique_years)
n_cells      <- length(id_order)

# Create an integer mapping: id -> position in id_order (1-based)
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Sort cell_data by (id position, year) so that row index is deterministic
cell_data[, id_pos := id_to_pos[as.character(id)]]
cell_data[, year_pos := match(year, unique_years)]
setorder(cell_data, id_pos, year_pos)

# Now row index of (cell at position p, year at position t) = (p - 1) * n_years + t
# Verify dimensions
stopifnot(nrow(cell_data) == n_cells * n_years)

# ---- Step 1: Build spatial-only neighbor lookup (344K entries, not 6.46M) ----
# rook_neighbors_unique is an nb object indexed by id_order position.
# Convert to a simple list of integer vectors (positions in id_order).
# Filter out self-references and 0s (nb convention for no neighbors).

spatial_neighbor_pos <- lapply(seq_along(rook_neighbors_unique), function(i) {

  nb_i <- rook_neighbors_unique[[i]]
  nb_i <- nb_i[nb_i > 0L]  # remove the 0 that signals "no neighbors"
  as.integer(nb_i)
})

# ---- Step 2: Expand spatial neighbors to row-level neighbor indices ----------
# For cell at position p in year-position t, its row is (p-1)*n_years + t.
# Its neighbors are at positions spatial_neighbor_pos[[p]], same year t.
# So neighbor rows are (nb_pos - 1) * n_years + t.
#
# We build this as a two-column matrix (row_index, neighbor_row_index)
# for vectorized computation.

# Pre-compute the number of neighbors per cell for memory allocation
n_neighbors_per_cell <- vapply(spatial_neighbor_pos, length, integer(1))
total_directed_pairs <- sum(n_neighbors_per_cell)  # ~1.37M spatial pairs
total_pairs_all_years <- as.numeric(total_directed_pairs) * n_years  # ~38.5M

message(sprintf(

  "Building neighbor index: %d spatial pairs x %d years = %.1fM row-pairs",
  total_directed_pairs, n_years, total_pairs_all_years / 1e6
))

# Build the expanded edge list efficiently
# For each cell position p with neighbors nb1, nb2, ..., nbk:
#   For each year position t in 1:n_years:
#     focal_row = (p-1)*n_years + t
#     neighbor_rows = (nb_j - 1)*n_years + t  for each nb_j

# We vectorize this construction:
# Repeat each cell position by its number of neighbors
focal_pos_expanded   <- rep(seq_len(n_cells), times = n_neighbors_per_cell)
neighbor_pos_expanded <- unlist(spatial_neighbor_pos, use.names = FALSE)

# Now tile across years
# focal_row   = (focal_pos - 1) * n_years + year_pos
# neighbor_row = (neighbor_pos - 1) * n_years + year_pos

# To avoid a huge 38.5M x 2 matrix, we process variable by variable
# using a grouped aggregation approach.

# ---- Step 3: Compute neighbor stats variable by variable ---------------------

# We'll compute max, min, mean of neighbor values using data.table grouping.
# Strategy:
#   1. Create an edge data.table with (focal_cell_pos, neighbor_cell_pos) â€” 1.37M rows
#   2. For each year and variable, join neighbor values via integer arithmetic.
#
# Even more efficient: use matrix operations.
# Reshape each variable into a (n_cells x n_years) matrix, then for each cell
# gather its neighbor rows from the matrix, and compute stats.

compute_all_neighbor_stats_fast <- function(cell_data, id_order, spatial_neighbor_pos,
                                            neighbor_source_vars, n_cells, n_years) {

  # Extract variable values into matrices: rows = cells (in id_order), cols = years
  var_matrices <- lapply(neighbor_source_vars, function(v) {
    # cell_data is sorted by (id_pos, year_pos), so reshaping is direct
    matrix(cell_data[[v]], nrow = n_years, ncol = n_cells)
    # Note: R fills matrices column-major. Each column = one cell's time series.
    # Row t = year_pos t. Column p = cell at position p.
    # But cell_data is sorted (id_pos, year_pos), so consecutive n_years rows
    # belong to the same cell. That means column-major fill gives us:
    #   matrix[t, p] = value for cell p at year t.  âœ“
  })
  names(var_matrices) <- neighbor_source_vars

  # For each variable, compute neighbor max/min/mean
  # Result: 3 new columns per variable, each of length n_cells * n_years
  new_cols <- list()

  for (v in neighbor_source_vars) {
    mat <- var_matrices[[v]]  # n_years x n_cells

    # Initialize result matrices
    max_mat  <- matrix(NA_real_, nrow = n_years, ncol = n_cells)
    min_mat  <- matrix(NA_real_, nrow = n_years, ncol = n_cells)
    mean_mat <- matrix(NA_real_, nrow = n_years, ncol = n_cells)

    # For each cell, gather neighbor columns and compute stats across neighbors
    # This is the inner loop but only 344K iterations (not 6.46M)
    # and each iteration does vectorized operations across 28 years.

    for (p in seq_len(n_cells)) {
      nb <- spatial_neighbor_pos[[p]]
      if (length(nb) == 0L) next
      # nb_vals: n_years x length(nb) submatrix
      if (length(nb) == 1L) {
        # Single neighbor: no need for apply
        nb_vals <- mat[, nb]  # length n_years vector
        max_mat[, p]  <- nb_vals
        min_mat[, p]  <- nb_vals
        mean_mat[, p] <- nb_vals
      } else {
        nb_submat <- mat[, nb, drop = FALSE]  # n_years x k
        # Compute row-wise (across neighbors) stats
        # Use matrixStats for speed if available, otherwise base R
        max_mat[, p]  <- do.call(pmax, c(as.data.frame(nb_submat), na.rm = TRUE))
        min_mat[, p]  <- do.call(pmin, c(as.data.frame(nb_submat), na.rm = TRUE))
        mean_mat[, p] <- rowMeans(nb_submat, na.rm = TRUE)
      }
    }

    # Flatten back to vector in the same order as cell_data (id_pos, year_pos)
    # matrix[t, p] -> vector order: column-major = (t=1,p=1), (t=2,p=1), ..., (t=T,p=1), (t=1,p=2), ...
    # This matches cell_data's sort order (id_pos, year_pos) âœ“
    new_cols[[paste0("neighbor_max_", v)]]  <- as.vector(max_mat)
    new_cols[[paste0("neighbor_min_", v)]]  <- as.vector(min_mat)
    new_cols[[paste0("neighbor_mean_", v)]] <- as.vector(mean_mat)

    message(sprintf("  Done: %s", v))
  }

  return(new_cols)
}

message("Computing optimized neighbor statistics...")
t0 <- Sys.time()

new_columns <- compute_all_neighbor_stats_fast(
  cell_data, id_order, spatial_neighbor_pos,
  neighbor_source_vars, n_cells, n_years
)

# ---- Step 4: Attach new columns to cell_data --------------------------------
for (col_name in names(new_columns)) {
  set(cell_data, j = col_name, value = new_columns[[col_name]])
}

# Clean up temporary columns
cell_data[, c("id_pos", "year_pos") := NULL]

t1 <- Sys.time()
message(sprintf("Neighbor feature engineering completed in %.1f minutes.", 
                as.numeric(difftime(t1, t0, units = "mins"))))

# ---- The trained Random Forest model is untouched ----------------------------
# Predictions proceed as before with the same column names and numerical values.
# The original numerical estimand is preserved because we compute identical
# max, min, mean statistics â€” just via matrix algebra instead of string hashing.
```

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string-key hash lookups into a 6.46M-entry named vector | Eliminated entirely; integer arithmetic |
| `compute_neighbor_stats` | 6.46M `lapply` iterations per variable Ã— 5 variables | 344K iterations per variable, each vectorized over 28 years |
| String operations | ~25.8M `paste()` + named-vector lookups | Zero |
| Estimated runtime | 86+ hours | ~5â€“15 minutes |
| Numerical output | max/min/mean of neighbor values | Identical max/min/mean of neighbor values |
| Trained RF model | Preserved | Preserved |

**Key insight:** The bottleneck was never the `do.call(rbind, ...)` â€” it was the O(N) string-hashed lookup repeated 6.46M times inside `build_neighbor_lookup()`, which needlessly reconstructed a purely spatial relationship for every cell-year row. By recognizing the regular panel structure and using integer arithmetic to map `(cell_position, year_position) â†’ row_index`, we eliminate all string operations and reduce the iteration count by a factor of 28 (years), with each remaining iteration fully vectorized.
END_RESPONSE>>>
