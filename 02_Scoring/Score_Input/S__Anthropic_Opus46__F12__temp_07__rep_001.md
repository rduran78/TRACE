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
run_id: Anthropic_F12_T07_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized operation per variable (only 5 calls total). Each inner function returns a fixed-length-3 numeric vector, so there is no "repeated list binding" â€” it's a clean `lapply` followed by one `rbind`. This is a minor cost.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **`paste()` keying and named-vector lookup over 6.46 million rows.** `idx_lookup` is a named integer vector with ~6.46 million entries. For each of the 6.46 million rows, the function calls `paste()` to construct neighbor keys and then does **character-based name matching** (`idx_lookup[neighbor_keys]`) against this massive named vector. Named vector lookup in R is **O(n)** linear scan per query (not hashed), so this is catastrophically slow at scale.

2. **Per-row `as.character()` and `paste()` calls inside `lapply` over 6.46M iterations.** Each iteration constructs character keys, performs string concatenation, and does name-based subsetting â€” all interpreted R with no vectorization benefit.

3. **Redundant recomputation across years.** The neighbor *structure* is purely spatial (rook contiguity) and identical for all 28 years, yet the function rebuilds per-row neighbor indices by pasting year suffixes and looking them up individually. This means the spatial topology is re-resolved ~28 times for every cell.

The `compute_neighbor_stats()` function, by contrast, does simple integer-indexed subsetting (`vals[idx]`) which is O(1) per element â€” extremely fast. The 5Ã— loop over variables is trivial.

**Quantitative estimate:** ~6.46M rows Ã— ~4 average neighbors Ã— character key lookup in a 6.46M-length named vector â‰ˆ tens of billions of character comparisons. This is where the 86+ hours lives, not in `do.call(rbind, ...)`.

## Optimization Strategy

1. **Replace character-key name lookup with integer arithmetic.** Since the panel is balanced (344,208 cells Ã— 28 years), we can compute the row index of any (cell, year) combination directly: `row = (year_offset * n_cells) + cell_position`. This turns the O(n) name lookup into O(1) integer arithmetic.

2. **Build the lookup once using vectorized operations** instead of row-by-row `lapply`. Expand the spatial neighbor list across all years using `rep()` and integer offsets â€” fully vectorized, no `paste()`, no character matching.

3. **Use `vapply` instead of `lapply` + `do.call(rbind, ...)` in `compute_neighbor_stats`** for a minor additional gain (pre-allocated matrix output).

This reduces the complexity from ~O(NÂ² Â· k) character operations to ~O(N Â· k) integer operations, where N = 6.46M and k = average neighbor count.

## Working R Code

```r
# ==============================================================================
# OPTIMIZED PIPELINE â€” preserves trained RF model and original numerical results
# ==============================================================================

# --------------------------------------------------------------------------
# Step 1: Build neighbor lookup via integer arithmetic (replaces build_neighbor_lookup)
# --------------------------------------------------------------------------
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  n_cells <- length(id_order)
  years   <- sort(unique(data$year))
  n_years <- length(years)

  # Verify balanced panel assumption
  stopifnot(nrow(data) == n_cells * n_years)

  # Create a mapping from cell id -> integer position (1..n_cells)
  # Assumes data is sorted by (year, id) or (id, year). We detect the order.
  # We'll enforce a known order: sort by year, then by id within year.
  data_order <- order(data$year, data$id)
  data <- data[data_order, , drop = FALSE]

  # Now row index for cell i (1-based in id_order) and year t (1-based in years)
  # is: (t - 1) * n_cells + i
  # But we need to map data$id to position in id_order.

  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  cell_pos  <- id_to_pos[as.character(data$id)]  # vectorized, one-time cost

  # For each cell position p, get its spatial neighbor positions
  # neighbors[[p]] gives integer indices into id_order
  # We'll build the full lookup as a list of length nrow(data)

  # Pre-expand spatial neighbors into a flat structure for vectorized ops
  # For each row i in data:
  #   cell_pos[i] = p
  #   year_index  = ((i - 1) %/% n_cells) + 1   (since sorted by year, then id)
  #   year_offset = (year_index - 1) * n_cells
  #   neighbor rows = year_offset + neighbors[[p]]

  year_index  <- rep(seq_len(n_years), each = n_cells)
  year_offset <- (year_index - 1L) * n_cells

  # Build lookup list â€” still a list, but inner computation is pure integer

  # Use the spatial neighbors directly (no paste, no character matching)
  neighbor_lookup <- vector("list", nrow(data))

  for (t in seq_len(n_years)) {
    row_start <- (t - 1L) * n_cells
    for (p in seq_len(n_cells)) {
      row_i <- row_start + p
      nb    <- neighbors[[p]]
      if (length(nb) == 0L) {
        neighbor_lookup[[row_i]] <- integer(0)
      } else {
        neighbor_lookup[[row_i]] <- row_start + nb
      }
    }
  }

  # Return both the reordered data and the lookup
  list(data = data, neighbor_lookup = neighbor_lookup, data_order = data_order)
}

# --------------------------------------------------------------------------
# Even faster: fully vectorized build (avoids nested for-loops entirely)
# --------------------------------------------------------------------------
build_neighbor_lookup_vectorized <- function(data, id_order, neighbors) {
  n_cells <- length(id_order)
  years   <- sort(unique(data$year))
  n_years <- length(years)
  N       <- nrow(data)

  stopifnot(N == n_cells * n_years)

  # Sort data into (year, id_order position) layout
  id_to_pos  <- setNames(seq_along(id_order), as.character(id_order))
  data$`.pos` <- id_to_pos[as.character(data$id)]
  data_order  <- order(data$year, data$`.pos`)
  data        <- data[data_order, , drop = FALSE]
  data$`.pos` <- NULL

  # Now row (t-1)*n_cells + p corresponds to year t, cell position p.

  # Convert spdep nb list to flat representation
  nb_lengths <- lengths(neighbors)                          # length n_cells
  nb_flat    <- unlist(neighbors, use.names = FALSE)        # flat neighbor positions
  nb_from    <- rep(seq_len(n_cells), times = nb_lengths)   # which cell each belongs to

  # Replicate across all years
  total_edges <- length(nb_flat)

  # For each year t, the "from" row is (t-1)*n_cells + nb_from
  #                   the "to"   row is (t-1)*n_cells + nb_flat
  from_rows <- integer(total_edges * n_years)
  to_rows   <- integer(total_edges * n_years)

  for (t in seq_len(n_years)) {
    offset <- (t - 1L) * n_cells
    idx_start <- (t - 1L) * total_edges + 1L
    idx_end   <- t * total_edges
    from_rows[idx_start:idx_end] <- offset + nb_from
    to_rows[idx_start:idx_end]   <- offset + nb_flat
  }

  # Now split to_rows by from_rows to get the lookup list
  neighbor_lookup <- split(to_rows, from_rows)

  # The split gives names as character; we need a list indexed 1..N
  # Cells with zero neighbors won't appear in the split, so fill them in
  full_lookup <- vector("list", N)
  for (i in seq_along(full_lookup)) full_lookup[[i]] <- integer(0)

  idx_names <- as.integer(names(neighbor_lookup))
  for (j in seq_along(idx_names)) {
    full_lookup[[ idx_names[j] ]] <- neighbor_lookup[[j]]
  }

  list(data = data, neighbor_lookup = full_lookup, data_order = data_order)
}

# --------------------------------------------------------------------------
# Step 2: Optimized compute_neighbor_stats (minor improvement with vapply)
# --------------------------------------------------------------------------
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- vapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    c(max(nv), min(nv), mean(nv))
  }, numeric(3))
  t(result)  # transpose: vapply returns 3 x N, we want N x 3
}

# --------------------------------------------------------------------------
# Step 3: Wrapper to add features (preserves original column naming)
# --------------------------------------------------------------------------
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_nb_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_nb_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_nb_mean")]] <- stats[, 3]
  data
}

# --------------------------------------------------------------------------
# Step 4: Full pipeline execution
# --------------------------------------------------------------------------

# Build the fast lookup (one-time cost, ~seconds instead of hours)
fast_build <- build_neighbor_lookup_vectorized(
  cell_data, id_order, rook_neighbors_unique
)
cell_data_sorted  <- fast_build$data
neighbor_lookup   <- fast_build$neighbor_lookup
original_order    <- fast_build$data_order

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data_sorted <- compute_and_add_neighbor_features_fast(
    cell_data_sorted, var_name, neighbor_lookup
  )
}

# Restore original row order so downstream RF prediction is unaffected
cell_data[original_order, ] <- cell_data_sorted

# The trained Random Forest model (rf_model) is untouched.
# Predictions proceed exactly as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| **Alleged bottleneck** | `do.call(rbind, ...)` and list binding in `compute_neighbor_stats()` | Only 5 calls, each on pre-allocated length-3 vectors. Minor cost. |
| **True bottleneck** | â€” | `build_neighbor_lookup()`: character-key construction via `paste()` and **O(n) named-vector lookup** over 6.46M entries, repeated ~25.8M times (6.46M rows Ã— ~4 neighbors). |
| **Root cause** | â€” | R named vectors use linear scan for name matching, not hash tables. 6.46M Ã— 4 Ã— O(6.46M) â‰ˆ catastrophic. |
| **Fix** | â€” | Replace character keys with **integer arithmetic**: row = `(year_offset Ã— n_cells) + cell_position`. Use vectorized `split()` to build the full lookup in one pass. |
| **Expected speedup** | â€” | From ~86+ hours to **minutes** (integer indexing is O(1); vectorized construction via `split()` handles ~38.4M edges in one pass). |
| **RF model** | â€” | Preserved unchanged. Column names and numerical values are identical. |
END_RESPONSE>>>
