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
run_id: Anthropic_F08_T03_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F08_Anthropic_temp_03_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup list of length ~6.46 million, where each element maps a cell-year row to the row indices of its neighbors *in that same year*. This is enormously wasteful because **the neighbor topology is purely spatial and static across all 28 years**. The same neighbor graph applies identically to every year, yet the current code:

1. **Rebuilds the neighbor mapping per row** (6.46M iterations) instead of per cell (344K iterations).
2. **Uses string-pasting and named-vector lookups** (`paste(id, year)` keys into `idx_lookup`) â€” extremely slow at scale.
3. **Produces a 6.46M-element list** that consumes large amounts of RAM and forces `compute_neighbor_stats` to iterate over 6.46M entries per variable.

The result: ~86+ hours of runtime dominated by the `build_neighbor_lookup` step and the subsequent per-row stat computation.

### Key Insight

- **Static (cell-level):** The neighbor adjacency structure. Cell *i*'s neighbors are always the same set of cells regardless of year.
- **Dynamic (cell-year-level):** The variable values (`ntl`, `ec`, `pop_density`, `def`, `usd_est_n2`) that change each year.

Therefore, we should:
- Build the neighbor lookup **once, at the cell level** (344K entries, not 6.46M).
- Compute neighbor stats **per year** using fast vectorized/matrix operations, reusing the cell-level adjacency.

---

## Optimization Strategy

1. **Cell-level adjacency (built once):** Convert `rook_neighbors_unique` (an `nb` object) into a cell-index-to-neighbor-cell-indices list. This is just the `nb` object itself (already indexed 1â€¦344,208). No string keys, no per-row expansion.

2. **Year-sliced, vectorized computation:** For each year, extract the variable column as a vector aligned to cell order. Then for each cell, pull neighbor values using the static adjacency list and compute max/min/mean. This reduces the outer loop from 6.46M to 344K per year, and we loop over only 28 years.

3. **Use `data.table` for fast split-by-year and column assignment**, avoiding repeated data-frame copies.

4. **Vectorize the inner stat computation** using a sparse-matrix multiply for the mean, and analogous approaches for max/min â€” or at minimum, use a tight `vapply` over 344K cells (not 6.46M rows).

5. **Preserve the numerical estimand exactly:** max, min, and mean of non-NA neighbor values, with NA when no valid neighbors exist â€” identical to the original.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Prepare data.table and establish canonical cell ordering
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# id_order is the canonical vector of cell IDs (length 344,208) that
# corresponds positionally to rook_neighbors_unique (the nb object).
# i.e., rook_neighbors_unique[[k]] gives neighbor indices into id_order
# for the cell id_order[k].

n_cells <- length(id_order)

# Create a fast lookup: cell_id -> position in id_order
id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the STATIC cell-level neighbor lookup (done ONCE)
#
# rook_neighbors_unique is already an nb object indexed 1..n_cells.
# Each element is an integer vector of neighbor positions in id_order.
# We just need to handle the nb "no-neighbor" convention (0L).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cell_neighbor_idx <- lapply(rook_neighbors_unique, function(nb_vec) {

  nb_vec <- nb_vec[nb_vec != 0L]  # nb objects use 0 for "no neighbors"
  as.integer(nb_vec)
})
# cell_neighbor_idx[[k]] = integer vector of positions in id_order
# that are neighbors of cell id_order[k].

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Ensure cell_data is keyed so we can extract year-slices
#         in canonical cell order efficiently.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Add a column for the cell's position in id_order (for fast ordering)
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# Key by year and cell_pos so that within each year, rows are in
# canonical cell order (positions 1..n_cells).
setkey(cell_data, year, cell_pos)

# Verify every year has exactly n_cells rows in the right order
# (the pipeline description implies a balanced panel).
years <- sort(unique(cell_data$year))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Define the fast neighbor-stat function (operates on one
#         year-slice at a time, using the static adjacency list).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_stats_fast <- function(values_vec, cell_neighbor_idx) {
  # values_vec: numeric vector of length n_cells, in canonical cell order
  # cell_neighbor_idx: list of length n_cells, each element = integer
  #                    vector of neighbor positions
  # Returns: n_cells x 3 matrix  [max, min, mean]

  n <- length(values_vec)
  out <- matrix(NA_real_, nrow = n, ncol = 3)

  for (k in seq_len(n)) {
    nb_idx <- cell_neighbor_idx[[k]]
    if (length(nb_idx) == 0L) next
    nb_vals <- values_vec[nb_idx]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) == 0L) next
    out[k, 1L] <- max(nb_vals)
    out[k, 2L] <- min(nb_vals)
    out[k, 3L] <- mean(nb_vals)
  }
  out
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Loop over variables and years, compute stats, assign columns
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate output columns with NA
for (var_name in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]
}

for (var_name in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  message("Processing neighbor stats for: ", var_name)

  for (yr in years) {
    # Extract the year-slice row indices (already sorted by cell_pos via key)
    yr_rows <- which(cell_data$year == yr)
    # Because of setkey(year, cell_pos), these rows are in canonical order.

    values_vec <- cell_data[[var_name]][yr_rows]

    stats_mat <- compute_neighbor_stats_fast(values_vec, cell_neighbor_idx)

    # Assign back using set() for zero-copy column update
    set(cell_data, i = yr_rows, j = col_max,  value = stats_mat[, 1L])
    set(cell_data, i = yr_rows, j = col_min,  value = stats_mat[, 2L])
    set(cell_data, i = yr_rows, j = col_mean, value = stats_mat[, 3L])
  }
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Clean up helper column, restore original row order if needed
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cell_data[, cell_pos := NULL]

# If downstream code expects a data.frame:
# cell_data <- as.data.frame(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 6: Predict with the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# The trained RF model object (e.g., `rf_model`) is used as-is.
# cell_data now contains the same neighbor_max_*, neighbor_min_*,
# neighbor_mean_* columns with identical numerical values as the
# original pipeline.
#
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup size | 6.46M-element list (per row) | 344K-element list (per cell, built once) |
| Lookup construction | String paste + named vector match over 6.46M rows | Direct reuse of `nb` object (trivial) |
| Stat computation iterations | 6.46M per variable (32.3M total) | 344K Ã— 28 years Ã— 5 vars = 48.2M cell-level ops, but **no string ops** |
| Estimated runtime | 86+ hours | **~10â€“30 minutes** (tight integer-indexed loop, no allocation overhead) |
| RAM | Huge intermediate list + string keys | ~negligible beyond the data itself |
| Numerical result | max/min/mean of non-NA neighbor values | **Identical** |
| RF model | Unchanged | Unchanged |

The key speedup comes from (a) eliminating the 6.46M-entry string-keyed lookup entirely, (b) operating on year-slices of a contiguous numeric vector with integer indexing, and (c) using `data.table::set()` for zero-copy column assignment.
END_RESPONSE>>>
