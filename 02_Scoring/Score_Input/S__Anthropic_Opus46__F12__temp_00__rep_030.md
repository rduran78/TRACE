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
run_id: Anthropic_F12_T00_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized matrix assembly step per variable â€” only 5 calls total. Each inner function returns a fixed-length-3 numeric vector, so there is no "repeated list binding" happening inside `compute_neighbor_stats()` at all; it's a straightforward `lapply` returning pre-allocated 3-element vectors.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Massive character key construction and named-vector lookup:** For each of the ~6.46 million rows, the function calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` to build character keys, then does named-vector indexing into `idx_lookup` (itself a named vector of length ~6.46 million). Named-vector lookup in R is **O(n)** linear scan per query (R's named vectors use linear search, not hash tables). With an average of ~4 rook neighbors per cell, that's ~25.8 million character key lookups, each scanning a 6.46-million-element named vector. This is catastrophically slow â€” **O(nÂ²)** in aggregate.

2. **Repeated `as.character()` and `paste()` inside the per-row `lapply`:** These allocate millions of small character vectors inside a loop.

3. **The lookup is called once but dominates total runtime.** `compute_neighbor_stats()` is called 5 times and is comparatively cheap because it only does integer indexing into a numeric vector â€” an O(1) operation per element.

**Conclusion:** The bottleneck is the O(nÂ²) character-key lookup strategy in `build_neighbor_lookup()`. The fix is to replace the named-vector lookup with a **hash-table lookup** (R `environment` or `data.table` keyed join) and, better yet, eliminate character key construction entirely by using direct integer arithmetic.

---

## Optimization Strategy

1. **Replace character-key named-vector lookup with integer arithmetic.** Since years are contiguous (1992â€“2019, 28 years) and cell IDs can be mapped to integers, we can compute a row index directly: `row = (cell_index - 1) * n_years + year_offset`. This turns the entire lookup into O(1) integer math â€” no strings, no hashing, no searching.

2. **Vectorize `build_neighbor_lookup`** by pre-grouping rows by year-offset and using vectorized integer indexing.

3. **Replace `do.call(rbind, ...)` with direct matrix pre-allocation** in `compute_neighbor_stats()` (a minor but easy improvement).

4. **Preserve the trained Random Forest model** â€” we only change feature engineering, producing numerically identical columns.

5. **Preserve the original numerical estimand** â€” the optimized code computes identical `max`, `min`, `mean` values.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED build_neighbor_lookup
# =============================================================================
# Assumptions validated against dataset facts:
#   - data is sorted by (id, year) or at minimum every id appears with every year
#   - years are contiguous integers 1992:2019
#   - id_order gives the canonical ordering of cell IDs matching the nb object
#
# Strategy: avoid ALL character operations. Use integer arithmetic for O(1) lookup.

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {

  n_cells <- length(id_order)
  years   <- sort(unique(data$year))
  n_years <- length(years)

  # --- Step 1: Map cell id -> integer index (1..n_cells) via hash (environment)
  id_to_ref <- new.env(hash = TRUE, parent = emptyenv(), size = n_cells)
  for (j in seq_along(id_order)) {
    id_to_ref[[as.character(id_order[j])]] <- j
  }

  # --- Step 2: Map year -> integer offset (1..n_years)
  year_to_offset <- new.env(hash = TRUE, parent = emptyenv(), size = n_years)
  for (j in seq_along(years)) {
    year_to_offset[[as.character(years[j])]] <- j
  }

  # --- Step 3: Build a fast row-index matrix: row_matrix[cell_ref, year_offset] = row in data
  #     This replaces the entire named-vector idx_lookup.
  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)

  data_ids   <- data$id
  data_years <- data$year

  # Vectorized fill
  cell_refs   <- integer(nrow(data))
  year_offsets <- integer(nrow(data))
  for (i in seq_len(nrow(data))) {
    cell_refs[i]    <- id_to_ref[[as.character(data_ids[i])]]
    year_offsets[i]  <- year_to_offset[[as.character(data_years[i])]]
  }
  # Single vectorized assignment
  row_matrix[cbind(cell_refs, year_offsets)] <- seq_len(nrow(data))

  # --- Step 4: Build neighbor lookup using integer matrix indexing
  #     For each row i, find neighbor cell refs, then look up their rows
  #     for the same year offset.

  # Pre-extract neighbor refs as integer vectors (avoids repeated list access overhead)
  neighbor_refs <- lapply(seq_along(neighbors), function(j) {
    nb <- neighbors[[j]]
    nb[nb > 0L]
  })

  # Main lookup construction
  lookup <- vector("list", nrow(data))

  for (i in seq_len(nrow(data))) {
    cr  <- cell_refs[i]
    yo  <- year_offsets[i]
    nb  <- neighbor_refs[[cr]]
    if (length(nb) == 0L) {
      lookup[[i]] <- integer(0)
      next
    }
    rows <- row_matrix[nb, yo]          # vectorized integer matrix index
    rows <- rows[!is.na(rows)]
    lookup[[i]] <- rows
  }

  lookup
}

# =============================================================================
# OPTIMIZED compute_neighbor_stats
# =============================================================================
# Replace do.call(rbind, ...) with pre-allocated matrix.
# Numerically identical output.

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n    <- length(neighbor_lookup)
  out  <- matrix(NA_real_, nrow = n, ncol = 3)

  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0L) next
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    out[i, 1] <- max(nv)
    out[i, 2] <- min(nv)
    out[i, 3] <- mean(nv)
  }
  out
}

# =============================================================================
# OPTIMIZED compute_and_add_neighbor_features (drop-in replacement)
# =============================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]
  data
}

# =============================================================================
# MAIN PIPELINE (drop-in replacement for the outer loop)
# =============================================================================
# Build the lookup ONCE â€” now O(n) instead of O(nÂ²)
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
# The 15 new columns (5 vars Ã— {max, min, mean}) are numerically identical
# to the original implementation, preserving the estimand.
```

---

## Expected Speedup

| Component | Original | Optimized | Why |
|---|---|---|---|
| `build_neighbor_lookup` | ~80+ hours (O(nÂ²) named-vector scan) | ~2â€“5 minutes (O(n) hash + matrix indexing) | Eliminated character-key linear search; replaced with integer matrix lookup |
| `compute_neighbor_stats` (Ã—5) | ~minutes each | ~minutes each (slightly faster) | Pre-allocated matrix instead of `do.call(rbind, ...)`; minor gain |
| **Total** | **~86+ hours** | **~10â€“20 minutes** | **~250â€“500Ã— speedup** |

The dominant improvement comes entirely from fixing `build_neighbor_lookup`, confirming that the colleague's diagnosis was wrong â€” the real bottleneck was the O(nÂ²) named-vector lookup, not `do.call(rbind, ...)`.
END_RESPONSE>>>
