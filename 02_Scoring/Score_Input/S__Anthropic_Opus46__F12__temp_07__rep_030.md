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
run_id: Anthropic_F12_T07_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: The Colleague Is Partially Right but Misses the Deeper Bottleneck

The colleague identifies `do.call(rbind, result)` and "repeated list binding" in `compute_neighbor_stats()` as the main bottleneck. Let's audit this against the code:

**`compute_neighbor_stats()`** â€” The `do.call(rbind, result)` call binds ~6.46 million small 3-element vectors into a matrix. This is an O(n) operation on pre-allocated list elements and is actually reasonably efficient in R â€” it's a single call, not iterative `rbind` growth. There is no "repeated list binding" inside the function; `lapply` builds the list in one pass. So the colleague's characterization of "repeated list binding" is factually wrong about this function. The `do.call(rbind, ...)` on 6.46M rows is non-trivial but is **not** the dominant cost.

**The true deep bottleneck is `build_neighbor_lookup()`**. Examine what it does:

```r
lapply(row_ids, function(i) {
    ref_idx           <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys     <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    result            <- idx_lookup[neighbor_keys]
    as.integer(result[!is.na(result)])
})
```

This runs **6.46 million iterations**, and in each iteration it:

1. Converts an integer to character and does a named-vector lookup (`id_to_ref`): O(1) amortized but with overhead per call.
2. Subsets the neighbor list to get ~4 neighbor cell IDs (rook neighbors).
3. **Pastes** each neighbor cell ID with the year to create string keys â€” 6.46M Ã— ~4 = ~25.8 million `paste()` calls with string allocation.
4. **Looks up** each key in `idx_lookup`, a named vector of length 6.46 million â€” named vector lookup in R is **hash-based** but the constant factor is large when done ~25.8 million times individually within an `lapply` over 6.46M rows.

The cost profile:
- `build_neighbor_lookup`: ~6.46M iterations Ã— (character coercion + paste + named-vector hash lookup) = **dominant cost, likely 70â€“80+ hours** of the 86-hour runtime.
- `compute_neighbor_stats`: 5 variables Ã— (6.46M simple integer-subsetting iterations + one `do.call(rbind, ...)`) = **relatively fast**, probably minutes to low single-digit hours total.

**Verdict: Reject the colleague's diagnosis.** The main bottleneck is `build_neighbor_lookup()`, specifically the per-row string construction (`paste`) and per-row named-vector lookups over 6.46 million rows. The `do.call(rbind, ...)` in `compute_neighbor_stats` is a secondary, much smaller cost.

---

## Optimization Strategy

### Key Insight
The neighbor lookup is **invariant across years within the same cell**. There are only 344,208 unique cells, each with ~4 rook neighbors. The spatial adjacency doesn't change year to year. The current code redundantly recomputes the same spatial neighbor mapping for every cell-year row (6.46M times) when it only needs to compute it once per cell (344K times) and then expand by year using vectorized integer arithmetic â€” **no strings needed at all**.

### Strategy
1. **Eliminate all string operations.** Replace the `paste`/named-vector-lookup approach with pure integer indexing.
2. **Compute a cell-level lookup once** (344K entries), then expand to cell-year rows using vectorized arithmetic based on a predictable row ordering (cell Ã— year grid).
3. **Vectorize `compute_neighbor_stats`** to avoid per-row `lapply`; use matrix column indexing or `vapply` with pre-extracted vectors, and replace `do.call(rbind, ...)` with direct matrix construction via `vapply`.
4. **Preserve the trained Random Forest model** â€” we only change feature-engineering preprocessing, not the model.
5. **Preserve the original numerical estimand** â€” the computed neighbor max/min/mean values are identical.

Expected speedup: from ~86 hours to **minutes**.

---

## Working R Code

```r
###############################################################################
# OPTIMIZED PIPELINE â€” replaces build_neighbor_lookup + compute_neighbor_stats
# Preserves: trained RF model, original numerical estimand (neighbor max/min/mean)
###############################################################################

#' Build an integer-only neighbor lookup at the CELL-YEAR ROW level.
#'
#' Assumptions (verified against the original code):
#'
#'   â€¢ cell_data is a data.frame / data.table with columns "id" and "year"
#'     (plus the predictor columns).
#'   â€¢ cell_data is ordered (or will be ordered here) by (id, year) so that
#'     rows for the same cell are contiguous and years are sequential.
#'   â€¢ id_order is the vector of unique cell IDs in the order that matches
#'     the index positions in rook_neighbors_unique (the spdep::nb object).
#'   â€¢ rook_neighbors_unique[[k]] gives integer indices into id_order for the
#'     neighbors of id_order[k].
#'
#' The function returns a list of length nrow(cell_data) where each element
#' is an integer vector of row indices into cell_data â€” exactly the same
#' semantics as the original build_neighbor_lookup().

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {

  # ---- Step 0: ensure data is sorted by (id, year) and record the order ----
  # We need a mapping from (cell_index, year) -> row in data.
  # If data is already sorted by (id, year) this is essentially free.

  n_cells <- length(id_order)
  years   <- sort(unique(data$year))
  n_years <- length(years)

  # Map cell id -> position in id_order (integer -> integer, no strings)
  id_int   <- match(data$id, id_order)          # length = nrow(data)
  year_int <- match(data$year, years)            # length = nrow(data)

  # Build a matrix:  row_matrix[cell_index, year_index] = row number in data
  # This replaces the named-vector idx_lookup entirely.
  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_matrix[cbind(id_int, year_int)] <- seq_len(nrow(data))

  # ---- Step 1: build cell-level neighbor index list (344K entries) ---------
  # neighbors[[k]] already gives integer indices into id_order.
  # We just need to ensure they are clean integer vectors (they usually are).
  cell_neighbor_idx <- lapply(neighbors, as.integer)   # 344K, very fast

  # ---- Step 2: expand to row-level lookup (6.46M entries) ------------------
  # For each row i with cell index c and year index y,
  # the neighbor rows are row_matrix[ neighbors[[c]], y ], dropping NAs.

  n_rows <- nrow(data)
  neighbor_lookup <- vector("list", n_rows)

  # Vectorise over cells: for each cell, handle all its years at once
  for (c_idx in seq_len(n_cells)) {
    nb_cells <- cell_neighbor_idx[[c_idx]]
    if (length(nb_cells) == 0L) {
      # All year-rows for this cell get empty neighbor sets
      rows_of_cell <- which(id_int == c_idx)
      for (r in rows_of_cell) neighbor_lookup[[r]] <- integer(0)
      next
    }
    # For every year, pull the row indices of the neighbor cells in that year
    # nb_rows_matrix: |nb_cells| x n_years â€” each column is one year
    nb_rows_matrix <- row_matrix[nb_cells, , drop = FALSE]  # small matrix

    rows_of_cell <- which(id_int == c_idx)
    yr_indices   <- year_int[rows_of_cell]

    for (j in seq_along(rows_of_cell)) {
      col <- yr_indices[j]
      nb_rows <- nb_rows_matrix[, col]
      neighbor_lookup[[ rows_of_cell[j] ]] <- nb_rows[!is.na(nb_rows)]
    }
  }

  neighbor_lookup
}


#' Compute neighbor summary statistics â€” vectorised, no do.call(rbind, ...).
#'
#' Returns a 3-column matrix (max, min, mean) with nrow(data) rows,
#' numerically identical to the original compute_neighbor_stats().

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals   <- data[[var_name]]
  n_rows <- nrow(data)

  # Pre-allocate output matrix
  out <- matrix(NA_real_, nrow = n_rows, ncol = 3L)

  for (i in seq_len(n_rows)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0L) next
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    out[i, 1L] <- max(nv)
    out[i, 2L] <- min(nv)
    out[i, 3L] <- mean(nv)
  }
  out
}


#' Wrapper that mirrors the original compute_and_add_neighbor_features()
#' signature so the outer loop is unchanged.

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1L]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2L]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3L]
  data
}


# ============================================================================
# OUTER LOOP â€” drop-in replacement
# ============================================================================

# Build the lookup ONCE (minutes instead of tens of hours)
neighbor_lookup <- build_neighbor_lookup_fast(
  cell_data, id_order, rook_neighbors_unique
)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_lookup
  )
}

# The trained Random Forest model is untouched.
# The computed neighbor features (max, min, mean) are numerically identical
# to the original pipeline, preserving the original estimand.
```

---

## Why This Is Fast

| Aspect | Original | Optimized |
|---|---|---|
| **String construction** | ~25.8M `paste()` calls | **Zero** â€” pure integer indexing |
| **Hash lookups** | ~25.8M named-vector lookups on a 6.46M-entry vector | **Zero** â€” replaced by matrix `[row, col]` indexing |
| **Redundant neighbor resolution** | Same spatial neighbors recomputed for every year (Ã—28) | Spatial neighbors resolved once per cell; expanded by year via column index |
| **`do.call(rbind, ...)`** | Binds 6.46M 3-element vectors | **Eliminated** â€” pre-allocated matrix filled in-place |
| **Expected runtime** | ~86+ hours | **~5â€“20 minutes** on the same laptop |

### Summary

The colleague's diagnosis is **incorrect**. The `do.call(rbind, ...)` in `compute_neighbor_stats()` is a minor cost. There is no "repeated list binding" in that function. The true bottleneck is `build_neighbor_lookup()`, which performs ~6.46 million iterations of string construction and hash-based named-vector lookups. The fix is to eliminate all string operations and replace them with integer matrix indexing, exploiting the fact that spatial adjacency is year-invariant.
END_RESPONSE>>>
