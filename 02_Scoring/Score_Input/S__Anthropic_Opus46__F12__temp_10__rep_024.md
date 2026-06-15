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
run_id: Anthropic_F12_T10_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Audit of Bottleneck Claim and Optimization

## Diagnosis

The colleague is **partially right but misses the deeper bottleneck**. While `do.call(rbind, result)` on 6.46M rows is suboptimal, it runs only 5 times (once per variable) and `result` is already a list of fixed-length numeric vectors â€” `do.call(rbind, ...)` on such a list is actually reasonably fast in R.

**The true dominant bottleneck is `build_neighbor_lookup()`.**

Here's why: it calls `lapply` over **every one of the ~6.46 million rows**, and inside each iteration it:

1. Performs `as.character()` conversion and a named-vector lookup (`id_to_ref`).
2. Subsets `id_order[neighbors[[ref_idx]]]` to get neighbor cell IDs.
3. Calls `paste(..., sep="_")` to build string keys â€” **6.46M calls to `paste()`**, each producing multiple strings.
4. Performs named-vector lookup on `idx_lookup` with those string keys â€” a **character hash lookup repeated ~1.37 billion times in aggregate** (6.46M rows Ã— avg ~4 rook neighbors Ã— 28 years of potential matches).

This is an O(n Ã— k) character-hashing operation with enormous constant factors. The `paste`/string-matching approach converts what should be a simple integer-arithmetic index calculation into millions of string allocations and hash lookups. On ~6.46M rows with ~4 neighbors each, this dwarfs the cost of 5 calls to `do.call(rbind, ...)`.

`compute_neighbor_stats` is already reasonably efficient â€” it's a simple numeric subsetting loop. The `do.call(rbind, ...)` on a list of length-3 vectors is a minor cost.

**Verdict: Reject the colleague's diagnosis as the *main* bottleneck. The main bottleneck is `build_neighbor_lookup()` and its per-row string construction/hashing.**

## Optimization Strategy

1. **Eliminate all string operations in the lookup.** Replace `paste(id, year, sep="_")` keying with direct integer arithmetic. Since we know every (id, year) pair maps to a row, build a matrix/integer-indexed lookup: `row_matrix[id_index, year_index] â†’ row_number`. This turns the inner lookup into a simple integer matrix subset â€” orders of magnitude faster.

2. **Vectorize `compute_neighbor_stats`** using a flat-vector approach with `vapply` (minor improvement, but cleaner).

3. **Keep the trained Random Forest model and all numerical outputs identical** â€” we are only changing how row indices are discovered, not any values.

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED build_neighbor_lookup â€” eliminates all string/paste operations
# =============================================================================
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Map cell id -> integer index (1..N_cells)
  n_cells <- length(id_order)
  id_to_ref <- integer(max(id_order))
  id_to_ref[id_order] <- seq_along(id_order)
  # If id_order values are too large/sparse for direct indexing, use match:
  # id_to_ref <- match(data$id, id_order)  # but below is faster if feasible

  # Map year -> integer index (1..N_years)
  years_unique <- sort(unique(data$year))
  n_years <- length(years_unique)
  year_min <- min(years_unique)
  # Assuming consecutive years: year_index = year - year_min + 1

  # Build a row-index matrix: row_lookup[cell_index, year_index] = row in data
  # This replaces ALL paste/character hashing
  row_lookup <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)

  cell_indices <- id_to_ref[data$id]
  year_indices <- data$year - year_min + 1L

  for (i in seq_len(nrow(data))) {
    row_lookup[cell_indices[i], year_indices[i]] <- i
  }
  # Vectorized alternative (faster):
  row_lookup[cbind(cell_indices, year_indices)] <- seq_len(nrow(data))

  # Now build the neighbor lookup with pure integer operations
  # Pre-extract neighbor cell indices for each cell
  # neighbors[[ref_idx]] gives indices into id_order for the neighbors of cell ref_idx

  n_rows <- nrow(data)
  neighbor_lookup <- vector("list", n_rows)

  for (i in seq_len(n_rows)) {
    ci <- cell_indices[i]
    yi <- year_indices[i]
    nb_cell_indices <- neighbors[[ci]]  # integer vector of neighbor cell indices
    if (length(nb_cell_indices) == 0L) {
      neighbor_lookup[[i]] <- integer(0)
      next
    }
    # Direct integer matrix lookup â€” no strings, no hashing
    nb_rows <- row_lookup[nb_cell_indices, yi]
    neighbor_lookup[[i]] <- nb_rows[!is.na(nb_rows)]
  }

  neighbor_lookup
}

# =============================================================================
# FURTHER OPTIMIZED: fully vectorized build using data.table
# This avoids the R-level for-loop over 6.46M rows entirely
# =============================================================================
build_neighbor_lookup_dt <- function(data, id_order, neighbors) {
  library(data.table)

  n_cells <- length(id_order)
  years_unique <- sort(unique(data$year))
  year_min <- min(years_unique)
  n_years <- length(years_unique)

  # Map id -> cell index
  id_to_ref <- integer(0)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  cell_indices <- id_to_ref[as.character(data$id)]
  year_indices <- data$year - year_min + 1L

  # Build row_lookup matrix: [cell_index, year_index] -> row number
  row_lookup <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_lookup[cbind(cell_indices, year_indices)] <- seq_len(nrow(data))

  # Expand all neighbor relationships into an edge table
  # neighbors[[ci]] = integer vector of neighbor cell indices for cell ci
  nb_lengths <- lengths(neighbors)
  from_cell <- rep(seq_len(n_cells), times = nb_lengths)
  to_cell   <- unlist(neighbors, use.names = FALSE)

  # For each (from_cell, year) combination, look up the target row
  # We need to cross from_cell with all years that from_cell appears in.
  # Instead, work from the data rows directly.

  # For every data row i, cell_indices[i] = ci, year_indices[i] = yi
  # neighbors of ci are neighbors[[ci]]
  # target rows are row_lookup[neighbors[[ci]], yi]

  # Build using a flat expansion:
  row_nb_lengths <- nb_lengths[cell_indices]  # number of neighbors for each row's cell
  n_total <- sum(row_nb_lengths)

  # row index repeated by its neighbor count
  row_rep <- rep(seq_len(nrow(data)), times = row_nb_lengths)
  # neighbor cell indices for each row
  nb_cell_rep <- unlist(neighbors[cell_indices], use.names = FALSE)
  # year index for each row, repeated
  yi_rep <- rep(year_indices, times = row_nb_lengths)

  # Look up target row numbers (vectorized matrix indexing)
  target_rows <- row_lookup[cbind(nb_cell_rep, yi_rep)]

  # Remove NAs
  valid <- !is.na(target_rows)
  row_rep      <- row_rep[valid]
  target_rows  <- target_rows[valid]

  # Split into list by source row
  neighbor_lookup <- split(target_rows, row_rep)

  # Ensure all rows are represented (fill missing with integer(0))
  full_lookup <- vector("list", nrow(data))
  full_lookup[] <- list(integer(0))
  idx <- as.integer(names(neighbor_lookup))
  full_lookup[idx] <- neighbor_lookup

  # Ensure integer type
  full_lookup <- lapply(full_lookup, as.integer)
  full_lookup
}

# =============================================================================
# OPTIMIZED compute_neighbor_stats â€” replaces do.call(rbind,...) with vapply
# =============================================================================
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- vapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    c(max(nv), min(nv), mean(nv))
  }, numeric(3))
  t(result)  # vapply returns 3 x n, transpose to n x 3
}

# =============================================================================
# Wrapper that adds columns (preserves original interface)
# =============================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]
  data
}

# =============================================================================
# MAIN PIPELINE â€” drop-in replacement
# =============================================================================

# Build the lookup ONCE (this is now the fast version)
neighbor_lookup <- build_neighbor_lookup_dt(cell_data, id_order, rook_neighbors_unique)

# Compute features for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_lookup)
}

# The trained Random Forest model is unchanged â€” use predict() as before.
# All numerical outputs are identical (same max, min, mean computations).
```

## Summary of Estimated Speedup

| Component | Original Cost | Optimized Cost | Reason |
|---|---|---|---|
| `build_neighbor_lookup` | ~80+ hours (6.46M Ã— paste + char hash) | ~2â€“5 minutes | Integer matrix indexing; vectorized expansion via `unlist`/`split` eliminates R-level loop |
| `compute_neighbor_stats` (Ã—5) | ~minutes | ~minutes (slightly faster) | `vapply` avoids `do.call(rbind,...)` allocation; minor gain |
| **Total** | **~86+ hours** | **~10â€“20 minutes** | **~250â€“500Ã— speedup** |

The key insight: the original code's catastrophic runtime came from **~6.46 million calls to `paste()` and character hash lookups inside `build_neighbor_lookup()`**, not from `do.call(rbind, ...)` which ran only 5 times on pre-allocated length-3 vectors. The fix replaces all string operations with a single integer matrix (`row_lookup[cell_index, year_index]`) and uses fully vectorized expansion (`rep`/`unlist`/`split`) to avoid any R-level loop over the 6.46M rows.
END_RESPONSE>>>
