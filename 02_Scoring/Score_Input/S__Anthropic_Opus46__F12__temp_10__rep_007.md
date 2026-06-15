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
run_id: Anthropic_F12_T10_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single matrix assembly step per variable (5 calls total). Each call binds ~6.46M rows of 3 columns â€” this completes in seconds.

**The true bottleneck is `build_neighbor_lookup()`.**

Here is the evidence from the code:

1. **Per-row `paste()` and character key lookups inside `lapply` over 6.46 million rows.** For every single row `i`, the function:
   - Converts `data$id[i]` to character and looks it up in `id_to_ref` (a named vector lookup â€” O(n) hash probe, but done 6.46M times).
   - Extracts neighbor cell IDs via `id_order[neighbors[[ref_idx]]]`.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” constructing character strings for every neighbor of every row.
   - Looks up each of those keys in `idx_lookup`, a named vector of length 6.46 million â€” a **hash table probe for every neighbor edge, repeated across all 28 years**.

2. **Scale calculation:** There are ~1,373,394 directed rook-neighbor relationships per year Ã— 28 years = **~38.5 million character-string paste + hash-lookup operations**, all inside a sequential R-level `lapply`. Character allocation, hashing, and `paste()` are extremely expensive in R at this scale.

3. **Redundancy:** The neighbor *structure* is identical across all 28 years (the grid doesn't change), yet the lookup recomputes string keys for every cell-year row. This is 28Ã— redundant work.

4. `compute_neighbor_stats()` by contrast does only numeric indexing (`vals[idx]`) and simple `max/min/mean` â€” these are vectorized C-level operations and are comparatively cheap.

**Conclusion:** The dominant bottleneck is the ~38.5 million `paste()` + named-vector lookups in `build_neighbor_lookup()`. The colleague's diagnosis is wrong.

---

## Optimization Strategy

1. **Eliminate all character key construction and hash lookups.** Replace the string-keyed lookup with direct integer-index arithmetic. Since data is a balanced panel (344,208 cells Ã— 28 years), if we sort by `(id, year)`, each cell's data occupies a contiguous block of 28 rows. The neighbor lookup for any cell-year `(c, y)` is simply: for each neighbor cell `c'`, the row index is `(position_of_c' - 1) * 28 + year_offset`. This is pure integer arithmetic â€” no `paste()`, no hash probes.

2. **Build the lookup as a precomputed integer-index list once**, using vectorized operations rather than row-by-row `lapply`.

3. **Replace `do.call(rbind, ...)` with direct pre-allocated matrix fills** (minor secondary gain).

4. **Preserve the trained Random Forest model** â€” we change only the feature-engineering pipeline, not the model. The numerical values produced are identical.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” replaces build_neighbor_lookup + compute_neighbor_stats
# =============================================================================

#' Optimized neighbor feature engineering.
#' Assumes cell_data is a balanced panel: every cell appears for every year.
#' Preserves exact numerical output (max, min, mean of neighbor values).

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # ---- Step 1: Ensure data is sorted by (id, year) ----
  # Create a cell-index map: id -> integer position in id_order
  n_cells <- length(id_order)
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_along(id_order)
  # If IDs are not contiguous integers, use a hash:
  # id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  years <- sort(unique(data$year))
  n_years <- length(years)
  year_to_offset <- setNames(seq_along(years), as.character(years))

  # Sort data by (id position, year) so that cell c occupies rows
  # [(pos_c - 1) * n_years + 1] through [pos_c * n_years]
  data$cell_pos <- id_to_pos[data$id]
  data$year_off <- year_to_offset[as.character(data$year)]
  sort_order <- order(data$cell_pos, data$year_off)
  data <- data[sort_order, ]

  # ---- Step 2: Build neighbor row-index list (vectorized) ----
  # For row r belonging to cell at position p and year-offset y:
  #   row_index = (p - 1) * n_years + y
  # Neighbor cells of position p are: neighbors[[p]]
  # Their row indices for same year y: (neighbors[[p]] - 1) * n_years + y

  # Pre-expand: for each cell position, store neighbor positions as integers
  # neighbors is an nb object (list of integer vectors of neighbor positions)
  neighbor_positions <- neighbors  # already indexed into id_order positions

  # Build lookup: list of length nrow(data), each element = integer vector of
  # row indices of neighbors in the sorted data.
  n_rows <- nrow(data)

  # Vectorized construction using rep + arithmetic
  # cell_pos and year_off for every row (already computed and sorted)
  cp <- data$cell_pos
  yo <- data$year_off

  # For speed, use Rcpp-style logic in pure R via vapply

  # But even an lapply here is over rows grouped by cell.
  # Key insight: all years within a cell share the same neighbor *cells*.
  # So we loop over cells (344K), not cell-years (6.46M).

  lookup <- vector("list", n_rows)

  for (p in seq_len(n_cells)) {
    nb_pos <- neighbor_positions[[p]]
    if (length(nb_pos) == 0L) {
      # All years for this cell get empty neighbors
      row_start <- (p - 1L) * n_years + 1L
      row_end   <- p * n_years
      for (r in row_start:row_end) {
        lookup[[r]] <- integer(0)
      }
    } else {
      # Base row indices for neighbor cells (year offset 0)
      nb_base <- (nb_pos - 1L) * n_years
      row_start <- (p - 1L) * n_years + 1L
      for (y in seq_len(n_years)) {
        r <- row_start + y - 1L
        lookup[[r]] <- nb_base + y  # same year offset for all neighbors
      }
    }
  }

  list(data = data, lookup = lookup, sort_order = sort_order)
}


compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  n <- length(neighbor_lookup)
  out <- matrix(NA_real_, nrow = n, ncol = 3)

  for (i in seq_len(n)) {
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


compute_and_add_neighbor_features_fast <- function(data, var_name,
                                                   neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data[[var_name]], neighbor_lookup)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]
  data
}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

# ---- Build the fast lookup (runs once) ----
message("Building optimized neighbor lookup...")
timing <- system.time({
  fast <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
})
message(sprintf("Lookup built in %.1f seconds", timing["elapsed"]))

cell_data_sorted <- fast$data
lookup           <- fast$lookup
sort_order       <- fast$sort_order

# ---- Compute neighbor features for all 5 variables ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  cell_data_sorted <- compute_and_add_neighbor_features_fast(
    cell_data_sorted, var_name, lookup
  )
}

# ---- Restore original row order so downstream code / RF model is unaffected --
cell_data[sort_order, ] <- cell_data_sorted
# Copy new columns back
for (var_name in neighbor_source_vars) {
  for (suffix in c("_neighbor_max", "_neighbor_min", "_neighbor_mean")) {
    col <- paste0(var_name, suffix)
    cell_data[[col]][sort_order] <- cell_data_sorted[[col]]
  }
}

message("Done. Neighbor features added. RF model untouched.")
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| **Key construction** | ~38.5M `paste()` calls | **Zero** â€” pure integer arithmetic |
| **Lookups** | Named-vector hash probes (character) | Direct integer indexing |
| **Loop granularity** | 6.46M row-level `lapply` in R | 344K cell-level loop (28Ã— fewer iterations) |
| **`do.call(rbind,...)`** | 6.46M 3-element vectors | Pre-allocated matrix fill |
| **Estimated time** | 86+ hours | **Minutes** (primarily the cell loop + numeric indexing) |
| **Numerical output** | Identical | Identical |
| **RF model** | Preserved | Preserved |

The core insight: the neighbor graph is **year-invariant**, so we loop over 344K cells (not 6.46M rows), and compute each year's neighbor row indices with a single integer addition (`nb_base + y`) â€” eliminating millions of string operations that were the true bottleneck.
END_RESPONSE>>>
