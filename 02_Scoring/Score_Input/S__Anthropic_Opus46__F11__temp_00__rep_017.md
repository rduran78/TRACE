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
run_id: Anthropic_F11_T00_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M iterations, this alone is extremely expensive.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level loop iterations total.

3. By contrast, Random Forest **prediction** on a pre-trained model is a single call to `predict()` on a matrix of ~6.46M Ã— 110 features. This is executed in optimized C/Fortran code internally and, while not trivial, is orders of magnitude faster than tens of millions of interpreted R loop iterations with per-element string operations.

**The 86+ hour runtime is dominated by the neighbor feature engineering, not by RF inference.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a vectorized `data.table` join approach. Instead of looping row-by-row with string key construction and named-vector lookups, we:
   - Expand the `nb` object into an edge list (cell_id â†’ neighbor_cell_id) once.
   - Join against the panel data on (neighbor_cell_id, year) to get row indices.
   - Group by the original row index to collect neighbor row indices as a list.

2. **Replace `compute_neighbor_stats()`** with a vectorized `data.table` grouped aggregation. Instead of `lapply` over millions of rows:
   - Unnest the neighbor lookup into a long table (row_idx, neighbor_row_idx).
   - Pull the variable values for all neighbor rows at once (vectorized subsetting).
   - Group-by the original row index and compute `max`, `min`, `mean` in one pass using `data.table`'s optimized grouped aggregation.

3. **Leave the Random Forest prediction code untouched**, since it is not the bottleneck.

This reduces the runtime from 86+ hours to an estimated **minutes** (dominated by the `data.table` joins and grouped aggregations, which are highly optimized in C).

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build neighbor lookup via vectorized data.table join
# ============================================================

build_neighbor_lookup_fast <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id', 'year', and an implicit row index
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  # --- 1a. Build edge list from the nb object ---
  # Each element neighbors[[i]] is an integer vector of indices into id_order.
  # We expand this into a two-column data.table: (cell_id, neighbor_cell_id)

  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells), lengths(neighbors))
  to_idx   <- unlist(neighbors)

  # Remove the spdep "no neighbors" sentinel (0)
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  edge_dt <- data.table(
    cell_id          = id_order[from_idx],
    neighbor_cell_id = id_order[to_idx]
  )

  # --- 1b. Add row index to the panel data ---
  data_dt[, row_idx := .I]

  # --- 1c. Join edges Ã— years to get (row_idx, neighbor_row_idx) ---
  # For every (cell_id, year) row, find all neighbor rows sharing the same year.

  # Keyed lookup table: given (id, year) -> row_idx
  id_year_key <- data_dt[, .(id, year, row_idx)]
  setkey(id_year_key, id, year)

  # Get all unique years
  years <- unique(data_dt$year)

  # For each cell_id row, its neighbors in the same year:
  # Approach: cross join edges with years, then look up row indices on both sides.

  # First, get (cell_id, year, row_idx) for the "from" side
  from_lookup <- id_year_key  # columns: id, year, row_idx
  setnames(from_lookup, c("id", "year", "row_idx"), c("cell_id", "year", "from_row_idx"))

  # Merge edges with from_lookup to get (cell_id, year, from_row_idx, neighbor_cell_id)
  setkey(edge_dt, cell_id)
  setkey(from_lookup, cell_id)

  # This is a many-to-many join: each edge Ã— each year the cell appears in
  edge_year <- merge(edge_dt, from_lookup, by = "cell_id", allow.cartesian = TRUE)
  # Columns: cell_id, neighbor_cell_id, year, from_row_idx

  # Now look up the neighbor's row index for the same year
  to_lookup <- data_dt[, .(neighbor_cell_id = id, year, neighbor_row_idx = row_idx)]
  setkey(to_lookup, neighbor_cell_id, year)
  setkey(edge_year, neighbor_cell_id, year)

  neighbor_map <- merge(edge_year, to_lookup, by = c("neighbor_cell_id", "year"))
  # Columns: neighbor_cell_id, year, cell_id, from_row_idx, neighbor_row_idx

  # --- 1d. Return as a list indexed by from_row_idx ---
  # (This format is compatible with downstream code, but we will also
  #  return the long-form table for the fast stats computation.)

  setkey(neighbor_map, from_row_idx)

  return(neighbor_map[, .(from_row_idx, neighbor_row_idx)])
}


# ============================================================
# STEP 2: Compute neighbor stats via vectorized aggregation
# ============================================================

compute_neighbor_stats_fast <- function(data_dt, neighbor_map_dt, var_name) {
  # data_dt:         data.table with row_idx column and the variable of interest
  # neighbor_map_dt: data.table with columns (from_row_idx, neighbor_row_idx)
  # var_name:        character, name of the variable to aggregate

  n_rows <- nrow(data_dt)

  # Pull neighbor values in one vectorized operation
  work <- copy(neighbor_map_dt)
  work[, val := data_dt[[var_name]][neighbor_row_idx]]

  # Drop NAs

  work <- work[!is.na(val)]

  # Grouped aggregation â€” data.table does this in C
  agg <- work[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = from_row_idx]

  # Allocate full result (NA for rows with no valid neighbors)
  result <- data.table(
    nb_max  = rep(NA_real_, n_rows),
    nb_min  = rep(NA_real_, n_rows),
    nb_mean = rep(NA_real_, n_rows)
  )
  result[agg$from_row_idx, `:=`(
    nb_max  = agg$nb_max,
    nb_min  = agg$nb_min,
    nb_mean = agg$nb_mean
  )]

  return(result)
}


# ============================================================
# STEP 3: Optimized outer loop
# ============================================================

compute_and_add_neighbor_features_fast <- function(data_dt, var_name, neighbor_map_dt) {
  stats <- compute_neighbor_stats_fast(data_dt, neighbor_map_dt, var_name)

  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)

  data_dt[, (max_col)  := stats$nb_max]
  data_dt[, (min_col)  := stats$nb_min]
  data_dt[, (mean_col) := stats$nb_mean]

  return(data_dt)
}


# ============================================================
# MAIN PIPELINE (drop-in replacement)
# ============================================================

# Convert to data.table if not already
cell_data_dt <- as.data.table(cell_data)

# Step 1: Build the neighbor map (replaces build_neighbor_lookup)
message("Building neighbor map...")
neighbor_map_dt <- build_neighbor_lookup_fast(cell_data_dt, id_order, rook_neighbors_unique)

# Step 2: Compute and attach neighbor features for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor features for: %s", var_name))
  cell_data_dt <- compute_and_add_neighbor_features_fast(cell_data_dt, var_name, neighbor_map_dt)
}

# Step 3: Random Forest prediction (UNCHANGED â€” not the bottleneck)
# e.g.:
# predictions <- predict(rf_model, newdata = cell_data_dt)
# cell_data_dt[, prediction := predictions]

# Convert back to data.frame if downstream code requires it
cell_data <- as.data.frame(cell_data_dt)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Bottleneck** | `build_neighbor_lookup` + `compute_neighbor_stats` (R-level `lapply` over ~6.46M rows, repeated 5Ã—) | Eliminated |
| **Lookup construction** | Per-row `paste` + named-vector lookup (~6.46M iterations) | Single vectorized `data.table` merge of edge list Ã— panel |
| **Stats computation** | `lapply` over 6.46M rows Ã— 5 vars = ~32.3M R iterations | Vectorized `data.table` grouped `max`/`min`/`mean` (C-optimized) |
| **RF inference** | Not the bottleneck; untouched | Untouched |
| **Numerical results** | `max`, `min`, `mean` of rook-neighbor values per cell-year | Identical â€” same aggregation, same values |
| **Estimated runtime** | 86+ hours | Minutes (dominated by the `data.table` joins) |
| **Trained RF model** | Preserved | Preserved |
END_RESPONSE>>>
