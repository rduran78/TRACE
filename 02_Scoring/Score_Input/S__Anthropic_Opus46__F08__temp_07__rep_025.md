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
run_id: Anthropic_F08_T07_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core bottleneck is in `build_neighbor_lookup`. It constructs a **row-level** lookup of length ~6.46 million, where each entry maps a cell-year row to the row indices of its neighbors **in that same year**. This is enormously wasteful because:

1. **The neighbor graph is static.** Cell `i`'s rook neighbors are the same in 1992 as in 2019. The `nb` object (`rook_neighbors_unique`) already encodes this topology once for all 344,208 cells. There is no reason to rebuild or re-traverse it for every year.

2. **The per-row lookup is O(n_rows).** Building 6.46 million list entriesâ€”each requiring string-pasting, hash lookups, and NA filteringâ€”is the dominant cost. The estimated 86+ hours comes almost entirely from this step.

3. **`compute_neighbor_stats` is fine in principle** but is called on a lookup that is 28Ã— larger than necessary. If we separate the static topology from the year-varying data, we can compute neighbor statistics year-by-year using fast vectorized/matrix operations on only 344,208 cells at a time.

**In summary:** The code conflates the *static spatial topology* with the *year-varying panel structure*, inflating the problem by a factor of 28 and forcing expensive string-key lookups on millions of rows.

---

## Optimization Strategy

1. **Build the neighbor lookup once, at the cell level (not cell-year level).** The `nb` object already provides this: `rook_neighbors_unique[[cell_ref]]` gives the neighbor indices for cell `cell_ref`. We just need a mapping from `cell_id` â†’ position in the data within each year.

2. **Loop over years (28 iterations), not over rows (6.46M iterations).** For each year, extract the 344,208-row slice, compute neighbor max/min/mean using the static `nb` object with vectorized operations, and write results back.

3. **Use matrix indexing or `vapply` over 344,208 cells per year** instead of 6.46M. This reduces the inner-loop size by 28Ã—.

4. **Use `data.table` for fast subsetting and assignment by year.** This avoids repeated copying of the full data frame.

5. **Pre-convert the `nb` object to an integer-index list once.** This is already done (`rook_neighbors_unique`), so no work needed.

**Expected speedup:** The current approach does ~6.46M list iterations Ã— 5 variables = ~32.3M iterations with string operations. The new approach does 28 years Ã— 344,208 cells Ã— 5 variables = ~48.2M lightweight integer-indexed vector lookupsâ€”but critically, *no string pasting or hash-map lookups*, and the inner operation is a simple `vals[nb_idx]` on a pre-aligned numeric vector. This should complete in **minutes, not hours**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Prepare the data and static structures
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table for fast grouped operations (non-destructive)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# id_order: the vector of cell IDs in the same order as rook_neighbors_unique.
# Build a fast lookup: cell_id -> position in id_order (i.e., index into nb list)
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

# Precompute the neighbor index list at the CELL level (static, built once).
# nb_list[[j]] gives the integer positions (in id_order) of cell j's neighbors.
# This is exactly what rook_neighbors_unique already is (spdep::nb object),
# but we ensure it's a clean integer list and handle the 0-neighbor case.
n_cells <- length(id_order)
nb_list <- lapply(seq_len(n_cells), function(j) {
  idx <- rook_neighbors_unique[[j]]
  # spdep::nb uses 0L to denote "no neighbors"
  if (length(idx) == 1L && idx[1] == 0L) {
    integer(0)
  } else {
    as.integer(idx)
  }
})

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Function to compute neighbor stats for one variable, one year
#          using the STATIC neighbor topology
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# For a given numeric vector `vals` of length n_cells (aligned to id_order),
# compute neighbor max, min, mean for every cell.
compute_neighbor_stats_vec <- function(vals, nb_list) {
  n <- length(nb_list)
  out <- matrix(NA_real_, nrow = n, ncol = 3)  # columns: max, min, mean

  for (j in seq_len(n)) {
    idx <- nb_list[[j]]
    if (length(idx) == 0L) next
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    out[j, 1L] <- max(nv)
    out[j, 2L] <- min(nv)
    out[j, 3L] <- mean(nv)
  }
  out
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Ensure cell_data is sorted by (year, id_order position)
#         so we can extract aligned vectors per year
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Add the cell-reference index (position in id_order / nb_list)
cell_data[, cell_ref := id_to_ref[as.character(id)]]

# Sort by year and cell_ref so that within each year the rows are aligned
# to the nb_list indexing.
setkey(cell_data, year, cell_ref)

# Verify alignment: within each year, cell_ref should be 1:n_cells
# (every cell appears exactly once per year).

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Main loop â€” years Ã— variables (28 Ã— 5 = 140 iterations)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate output columns
for (var_name in neighbor_source_vars) {
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  if (!max_col  %in% names(cell_data)) cell_data[, (max_col)  := NA_real_]
  if (!min_col  %in% names(cell_data)) cell_data[, (min_col)  := NA_real_]
  if (!mean_col %in% names(cell_data)) cell_data[, (mean_col) := NA_real_]
}

years <- sort(unique(cell_data$year))

for (yr in years) {
  message("Processing year: ", yr)

  # Boolean mask for this year's rows (already sorted by cell_ref within year)
  yr_mask <- cell_data$year == yr

  for (var_name in neighbor_source_vars) {
    # Extract the variable values aligned to id_order for this year
    vals <- cell_data[[var_name]][yr_mask]

    # Compute neighbor stats using the static topology
    stats <- compute_neighbor_stats_vec(vals, nb_list)

    # Write back into the data.table
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    set(cell_data, which(yr_mask), max_col,  stats[, 1L])
    set(cell_data, which(yr_mask), min_col,  stats[, 2L])
    set(cell_data, which(yr_mask), mean_col, stats[, 3L])
  }
}

# Clean up the helper column
cell_data[, cell_ref := NULL]

message("Neighbor feature computation complete.")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict using the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained RF model object and prediction call remain exactly as before.
# For example:
#   cell_data$prediction <- predict(rf_model, newdata = cell_data)$predictions
# Nothing about the model or its usage changes.
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets and the same variable values as before. The only change is *how* we find those neighbor rowsâ€”via a pre-aligned vector indexed by the static `nb` object, rather than via string-key hash lookups. The arithmetic is identical. |
| **Trained RF model** | Untouched. We only changed the feature-engineering step that feeds into `predict()`. Column names and semantics are preserved. |
| **Edge cases** | Zero-neighbor cells and `NA` values are handled identically (returning `NA` for all three statistics). |

## Expected Performance

| Metric | Before | After |
|---|---|---|
| Lookup construction | ~6.46M string-paste + hash entries | 0 (use existing `nb` object directly) |
| Inner loop iterations | ~6.46M per variable | ~344K per variable per year |
| Total inner iterations | ~32.3M with string ops | ~48.2M with integer vector indexing |
| Estimated wall time | 86+ hours | **~5â€“15 minutes** on a 16 GB laptop |
| Peak memory | Large list of 6.46M entries + string keys | One 344K Ã— 3 matrix + one numeric vector at a time |
END_RESPONSE>>>
