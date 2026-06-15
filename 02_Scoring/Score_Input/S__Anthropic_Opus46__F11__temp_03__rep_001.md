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
run_id: Anthropic_F11_T03_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** as the dominant bottleneck â€” specifically `build_neighbor_lookup` and `compute_neighbor_stats`.

**Evidence from the code:**

1. **`build_neighbor_lookup`** uses a top-level `lapply` over **~6.46 million rows**. Inside each iteration it performs `as.character()` coercion, `paste()` string construction, and named-vector lookups (`idx_lookup[neighbor_keys]`) â€” all character-based hash lookups repeated per row. With ~1.37 million neighbor relationships spread across 6.46M row iterations, this is enormously expensive in pure R.

2. **`compute_neighbor_stats`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million elements via `lapply`, subsetting a numeric vector, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` at the end binds 6.46 million 3-element vectors into a matrix â€” also slow.

3. The outer loop calls this machinery for **5 variables**, so the total work is: 6.46M iterations for the lookup build + 5 Ã— 6.46M iterations for stats = **~38.8 million R-level loop iterations** with per-iteration string operations and subsetting.

4. By contrast, Random Forest `predict()` on a pre-trained model with ~110 predictors and 6.46M rows is a single vectorized C/C++ call (in `ranger` or `randomForest`). Even for a large forest this typically completes in minutes, not hours.

**Conclusion:** The 86+ hour runtime is dominated by row-level R loops with string operations in the neighbor feature engineering, not by RF inference.

---

## Optimization Strategy

1. **Eliminate per-row string operations in `build_neighbor_lookup`:** Replace character-key lookups with integer-indexed lookups. Pre-build a matrix mapping `(cell_index, year_index) â†’ row_number` so neighbor row indices can be retrieved via integer matrix indexing â€” O(1) and vectorized.

2. **Vectorize `compute_neighbor_stats`:** Instead of `lapply` over 6.46M elements, unroll the neighbor lookup into a flat edge list (source_row, neighbor_row), extract all neighbor values at once, then use `data.table` grouped aggregation (`max`, `min`, `mean`) â€” fully vectorized in C.

3. **Preserve the trained RF model and the original numerical estimand** â€” no changes to modeling or prediction code.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED build_neighbor_lookup
# Returns a data.table edge list: (source_row, neighbor_row)
# instead of a list-of-vectors over 6.46M elements.
# ==============================================================================
build_neighbor_edgelist <- function(data, id_order, neighbors) {
  # data must have columns: id, year
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  dt <- as.data.table(data)
  dt[, row_idx := .I]

  n_ids   <- length(id_order)
  n_years <- uniqueN(dt$year)
  years   <- sort(unique(dt$year))

  # Map (cell_position_in_id_order, year) -> row index in data
  # cell_position: 1..n_ids;  year_position: 1..n_years
  id_to_pos  <- setNames(seq_along(id_order), as.character(id_order))
  year_to_pos <- setNames(seq_along(years), as.character(years))

  dt[, id_pos   := id_to_pos[as.character(id)]]
  dt[, year_pos := year_to_pos[as.character(year)]]

  # Build a lookup matrix: row_lookup[id_pos, year_pos] = row_idx in data
  row_lookup <- matrix(NA_integer_, nrow = n_ids, ncol = n_years)
  row_lookup[cbind(dt$id_pos, dt$year_pos)] <- dt$row_idx

  # Build flat edge list from the nb object
  # For each cell i (position in id_order), neighbors[[i]] gives neighbor positions
  src_pos <- rep(seq_len(n_ids), lengths(neighbors))
  nbr_pos <- unlist(neighbors)

  # Expand across all years: for each (src_cell, nbr_cell) pair, repeat for every year
  n_edges_per_year <- length(src_pos)
  n_year_vec       <- length(years)

  # Replicate edge list across years
  src_pos_all  <- rep(src_pos, times = n_year_vec)
  nbr_pos_all  <- rep(nbr_pos, times = n_year_vec)
  year_pos_all <- rep(seq_len(n_year_vec), each = n_edges_per_year)

  # Look up actual row indices
  source_row   <- row_lookup[cbind(src_pos_all, year_pos_all)]
  neighbor_row <- row_lookup[cbind(nbr_pos_all, year_pos_all)]

  # Remove edges where either source or neighbor row doesn't exist
  valid <- !is.na(source_row) & !is.na(neighbor_row)

  data.table(
    source_row   = source_row[valid],
    neighbor_row = neighbor_row[valid]
  )
}

# ==============================================================================
# OPTIMIZED compute_neighbor_stats (vectorized via data.table)
# ==============================================================================
compute_neighbor_stats_fast <- function(data_dt, edgelist, var_name) {
  # edgelist: data.table with (source_row, neighbor_row)
  # data_dt: data.table with row_idx or we use positional indexing

  vals <- data_dt[[var_name]]

  # Attach neighbor values
  el <- copy(edgelist)
  el[, nbr_val := vals[neighbor_row]]

  # Remove NAs

  el <- el[!is.na(nbr_val)]

  # Grouped aggregation â€” fully vectorized in C via data.table
  stats <- el[, .(
    nb_max  = max(nbr_val),
    nb_min  = min(nbr_val),
    nb_mean = mean(nbr_val)
  ), by = source_row]

  stats
}

# ==============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# ==============================================================================
compute_and_add_neighbor_features_fast <- function(data_dt, var_name, edgelist) {
  n <- nrow(data_dt)
  stats <- compute_neighbor_stats_fast(data_dt, edgelist, var_name)

  # Initialize with NA
  max_col  <- rep(NA_real_, n)
  min_col  <- rep(NA_real_, n)
  mean_col <- rep(NA_real_, n)

  max_col[stats$source_row]  <- stats$nb_max
  min_col[stats$source_row]  <- stats$nb_min
  mean_col[stats$source_row] <- stats$nb_mean

  set(data_dt, j = paste0("nb_max_",  var_name), value = max_col)
  set(data_dt, j = paste0("nb_min_",  var_name), value = min_col)
  set(data_dt, j = paste0("nb_mean_", var_name), value = mean_col)

  data_dt
}

# ==============================================================================
# MAIN PIPELINE (drop-in replacement for the outer loop)
# ==============================================================================

# Convert to data.table once
cell_data_dt <- as.data.table(cell_data)

# Build the edge list ONCE (replaces build_neighbor_lookup)
# This is vectorized â€” runs in seconds, not hours.
edgelist <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data_dt <- compute_and_add_neighbor_features_fast(
    cell_data_dt, var_name, edgelist
  )
}

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_data_dt)

# ==============================================================================
# Random Forest prediction is UNCHANGED â€” it was never the bottleneck.
# Example (assuming the model object is `rf_model`):
#   predictions <- predict(rf_model, newdata = cell_data)
# ==============================================================================
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup build** | 6.46M `lapply` iterations with `paste()` + character hash lookups | Vectorized integer matrix indexing; flat edge list built in seconds |
| **Neighbor stats** | 5 Ã— 6.46M `lapply` iterations + `do.call(rbind, ...)` | 5 Ã— one `data.table` grouped aggregation (C-level) |
| **Expected runtime** | ~86+ hours | **Minutes** (edge list build ~30sâ€“2min; each variable's stats ~10â€“30s) |
| **RF model** | Preserved (unchanged) | Preserved (unchanged) |
| **Numerical estimand** | max, min, mean of non-NA neighbor values per row | Identical â€” same semantics, same output columns |

The bottleneck was never Random Forest inference. It was millions of R-level loop iterations performing per-row string construction and character-keyed lookups in the neighbor feature engineering step.
END_RESPONSE>>>
