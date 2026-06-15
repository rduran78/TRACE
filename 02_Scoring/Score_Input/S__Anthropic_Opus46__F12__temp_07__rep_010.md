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
run_id: Anthropic_F12_T07_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The colleague is **partially correct but misses the deeper bottleneck**. Let me audit both functions against the data facts:

### `compute_neighbor_stats()` â€” Colleague's claimed bottleneck

- `do.call(rbind, result)`: Binding ~6.46M rows from a list of 3-element vectors is not cheap, but it's a **single call** per variable (5 total). With modern R, `do.call(rbind, ...)` on a list of equal-length numeric vectors is actually reasonably efficient â€” it's O(n) in memory. This is a **minor** bottleneck.
- "Repeated list binding": There is **no** repeated list binding inside `compute_neighbor_stats()`. It uses `lapply` to build the list in one pass, then a single `rbind`. The colleague's description of the code is factually inaccurate.

### `build_neighbor_lookup()` â€” The **true deep bottleneck**

This function runs `lapply` over **every row** (~6.46 million cell-year rows) and, for each row:

1. Calls `as.character(data$id[i])` â€” 6.46M character coercions.
2. Looks up `id_to_ref[as.character(...)]` â€” 6.46M named-vector lookups (linear hash probe each time).
3. Extracts `neighbor_cell_ids` via subsetting `id_order[neighbors[[ref_idx]]]`.
4. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” **6.46M paste calls**, each producing a vector of ~4 strings (avg ~4 rook neighbors per cell: 1,373,394 directed relationships / 344,208 cells â‰ˆ 4).
5. Looks up `idx_lookup[neighbor_keys]` â€” 6.46M named-vector lookups on a **6.46M-length named vector**. Named vector lookup in R is O(n) or at best O(1) amortized via internal hashing, but the sheer volume (~25.8M key lookups total) on a 6.46M-entry vector is extremely expensive.
6. Filters NAs and coerces to integer.

**The critical insight**: The neighbor relationships are **invariant across years**. There are only 344,208 unique cells, but the function redundantly recomputes the same neighbor sets **28 times** (once per year per cell). This means ~6.46M iterations when only ~344K unique spatial lookups are needed, with the year dimension being a simple offset calculation.

**Total key lookups**: ~6.46M rows Ã— ~4 neighbors = ~25.8 million string-paste-and-match operations against a 6.46M-entry named vector. This is the dominant bottleneck â€” not `do.call(rbind, ...)`.

### Verdict

**Reject the colleague's diagnosis.** The main bottleneck is `build_neighbor_lookup()`, specifically:
1. Redundant recomputation across 28 years of what is a purely spatial relationship.
2. Repeated string pasting and named-vector lookups on a 6.46M-entry vector inside a row-level loop.
3. `compute_neighbor_stats` is comparatively lightweight (vectorized subsetting of a numeric vector).

---

## Optimization Strategy

1. **Exploit temporal invariance**: Compute the neighbor structure only for the ~344,208 unique spatial cells, not for all ~6.46M cell-year rows. Then use integer arithmetic to map from "cell-level neighbor" to "cell-year row index" using year offsets.

2. **Replace named-vector lookups with integer indexing**: Use `match()` once to build an integer mapping, then use direct integer subsetting (O(1)) instead of repeated named-vector lookups (expensive hash probes on millions of entries).

3. **Vectorize `compute_neighbor_stats`**: Replace the per-row `lapply` with a single vectorized grouping operation using `data.table` or pre-allocated matrix arithmetic. Alternatively, flatten the neighbor lookup into a long-form table and use `data.table` grouped aggregation.

4. **Preserve the trained Random Forest model**: No changes to the model or its predictions. We only optimize the feature-engineering pipeline that feeds into prediction.

5. **Preserve the original numerical estimand**: The optimized code computes identical `max`, `min`, and `mean` neighbor statistics.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# OPTIMIZED build_neighbor_lookup
# =============================================================================
# Key idea: neighbor relationships are purely spatial and invariant across years.
# We compute the mapping once for unique cells, then expand by year via
# integer arithmetic.
#
# Assumptions (from the original code and pipeline facts):
#   - data has columns: id, year, and the variable columns
#   - data is a data.frame (or data.table) with ~6.46M rows
#   - id_order: vector of unique cell IDs (length 344,208)
#   - neighbors: spdep nb object (list of length 344,208), each element is
#     an integer vector of indices into id_order
# =============================================================================

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert to data.table for fast operations (non-destructive)
  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # --- Step 1: Build a cell-to-integer and year-to-integer mapping ---
  unique_years <- sort(unique(dt$year))
  n_years      <- length(unique_years)
  n_cells      <- length(id_order)

  # Map each cell ID to an integer 1..n_cells
  cell_int <- setNames(seq_along(id_order), as.character(id_order))

  # Map each year to an integer 1..n_years
  year_int <- setNames(seq_along(unique_years), as.character(unique_years))

  # --- Step 2: Build a fast row-index matrix: row_matrix[cell, year] = row in data ---
  # This replaces the expensive named-vector idx_lookup entirely.
  dt[, cell_i := cell_int[as.character(id)]]
  dt[, year_i := year_int[as.character(year)]]

  # Allocate matrix (344,208 x 28 â‰ˆ 9.6M entries, ~77 MB for integers â€” fine)
  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_matrix[cbind(dt$cell_i, dt$year_i)] <- dt$row_idx

  # --- Step 3: Build the neighbor lookup as a long-form data.table ---
  # For each cell, get its neighbor cell indices (spatial, year-invariant).
  # Then expand across all years using row_matrix.

  # Build a long table of (focal_cell_i, neighbor_cell_i) from the nb object
  # This is done ONCE for 344,208 cells, not 6.46M rows.
  focal_cells   <- rep(seq_len(n_cells), times = lengths(neighbors))
  neighbor_cells <- unlist(neighbors, use.names = FALSE)

  # Remove self-neighbors and zero entries (spdep convention: 0L means no neighbors)
  valid <- neighbor_cells > 0L
  focal_cells    <- focal_cells[valid]
  neighbor_cells <- neighbor_cells[valid]

  # Now expand across years: for each (focal_cell, neighbor_cell) pair,
  # and for each year, look up the row indices.
  # Total entries: ~1.37M pairs Ã— 28 years â‰ˆ 38.5M â€” manageable.

  # Build the expanded long table efficiently
  n_pairs <- length(focal_cells)

  # Repeat each pair n_years times
  expanded_focal    <- rep(focal_cells, each = n_years)
  expanded_neighbor <- rep(neighbor_cells, each = n_years)
  expanded_year     <- rep(seq_len(n_years), times = n_pairs)

  # Look up row indices for focal and neighbor
  focal_row    <- row_matrix[cbind(expanded_focal, expanded_year)]
  neighbor_row <- row_matrix[cbind(expanded_neighbor, expanded_year)]

  # Remove entries where either focal or neighbor is missing (cell-year doesn't exist)
  valid2 <- !is.na(focal_row) & !is.na(neighbor_row)

  neighbor_long <- data.table(
    focal_row    = focal_row[valid2],
    neighbor_row = neighbor_row[valid2]
  )

  # Return both the long table and the total number of rows
  # (needed for compute_neighbor_stats_fast)
  list(
    neighbor_long = neighbor_long,
    n_rows        = nrow(dt)
  )
}

# =============================================================================
# OPTIMIZED compute_neighbor_stats
# =============================================================================
# Uses data.table grouped aggregation on the long-form neighbor table.
# Computes max, min, mean in a single vectorized pass.
# =============================================================================

compute_neighbor_stats_fast <- function(data, neighbor_lookup_fast, var_name) {
  nl     <- neighbor_lookup_fast$neighbor_long
  n_rows <- neighbor_lookup_fast$n_rows
  vals   <- data[[var_name]]

  # Attach neighbor values
  nl_copy <- copy(nl)
  nl_copy[, nval := vals[neighbor_row]]

  # Remove NAs in neighbor values
  nl_valid <- nl_copy[!is.na(nval)]

  # Grouped aggregation â€” fully vectorized via data.table
  stats <- nl_valid[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row]

  # Allocate output matrix (same shape as original: n_rows x 3)
  out <- matrix(NA_real_, nrow = n_rows, ncol = 3)
  out[stats$focal_row, 1] <- stats$nb_max
  out[stats$focal_row, 2] <- stats$nb_min
  out[stats$focal_row, 3] <- stats$nb_mean

  out
}

# =============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# =============================================================================
# Drop-in replacement that preserves column naming conventions.
# =============================================================================

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup_fast) {
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup_fast, var_name)

  data[[paste0(var_name, "_nb_max")]]  <- stats_mat[, 1]
  data[[paste0(var_name, "_nb_min")]]  <- stats_mat[, 2]
  data[[paste0(var_name, "_nb_mean")]] <- stats_mat[, 3]

  data
}

# =============================================================================
# OPTIMIZED OUTER LOOP
# =============================================================================
# Drop-in replacement for the original pipeline.
# The trained Random Forest model is untouched.
# The numerical outputs (max, min, mean of neighbor values) are identical.
# =============================================================================

# Build the fast lookup ONCE (replaces build_neighbor_lookup)
neighbor_lookup_fast <- build_neighbor_lookup_fast(
  cell_data, id_order, rook_neighbors_unique
)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_lookup_fast
  )
}

# cell_data now has the same neighbor feature columns as the original pipeline.
# Feed directly into the existing (already trained) Random Forest for prediction.
```

---

## Performance Summary

| Component | Original | Optimized | Speedup Factor |
|---|---|---|---|
| **`build_neighbor_lookup`** | ~6.46M row-level `lapply` with string paste + named-vector lookup on 6.46M keys | Single vectorized expansion via integer matrix indexing; no string operations in the hot path | **~100â€“500Ã—** |
| **`compute_neighbor_stats`** | 6.46M-iteration `lapply` + `do.call(rbind, ...)` per variable | `data.table` grouped aggregation (vectorized C-level) | **~20â€“50Ã—** |
| **Overall pipeline** | ~86+ hours estimated | **Minutes** (estimated 5â€“20 min depending on disk I/O and RAM pressure) | **~250â€“1000Ã—** |
| **Memory** | 6.46M-element list of integer vectors (~large overhead from list structure) | Long-form `data.table` with ~38.5M rows Ã— 2 integer columns (~308 MB) + row_matrix (~37 MB) | Comparable or lower |

The trained Random Forest model is completely preserved â€” no retraining occurs. The numerical estimand (max, min, mean of rook-neighbor values per cell-year) is identical to the original implementation.
END_RESPONSE>>>
