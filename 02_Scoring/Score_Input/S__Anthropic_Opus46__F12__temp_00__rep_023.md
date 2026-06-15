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
run_id: Anthropic_F12_T00_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single matrix assembly step per variable (5 calls total). Each call binds ~6.46M rows Ã— 3 columns â€” this completes in seconds in R.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Character key construction and named-vector lookup over 6.46M rows.** Inside the `lapply`, for every single row `i`, the function:
   - Calls `as.character(data$id[i])` â€” scalar character conversion, 6.46M times.
   - Indexes into `id_to_ref` (a named character vector) â€” named vector lookup is O(n) hash probe but done 6.46M times.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” constructs character keys for every neighbor of every row.
   - Indexes into `idx_lookup` â€” another named-vector lookup, but now for *every neighbor of every row*. With ~1.37M directed neighbor relationships across 344K cells and 28 years, this means roughly **~4 neighbor lookups Ã— 6.46M rows â‰ˆ 25.8 million** character-key hash lookups into a 6.46M-entry named vector.

2. **This is an inherently O(N Ã— K) character-hashing operation** where N = 6.46M and K â‰ˆ average neighbor count (~4 for rook). The `paste()` and named-vector indexing are extremely slow in a row-level `lapply` in R.

3. The neighbor lookup is **year-invariant in structure** â€” rook neighbors don't change across years â€” yet the code redundantly rebuilds neighbor indices for every cell-year row, inflating the work by a factor of 28.

`compute_neighbor_stats()`, by contrast, does only integer indexing into a numeric vector (`vals[idx]`) and simple arithmetic â€” this is fast even at scale.

## Optimization Strategy

1. **Build the neighbor lookup at the cell level (344K cells), not the cell-year level (6.46M rows).** Since rook neighbors are time-invariant, compute the mapping from each cell to its neighbor cells once.

2. **Map cell-level neighbor structure to row-level using integer indexing** via a `data.table` join (cell Ã— year â†’ row index), avoiding all `paste()` and named-character-vector lookups.

3. **Vectorize `compute_neighbor_stats()`** using `data.table` grouped operations or pre-allocated matrix arithmetic, eliminating the per-row `lapply` entirely.

4. **Preserve the trained Random Forest model** â€” we only change feature engineering speed, not values. The numerical results are identical.

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 1: Build neighbor lookup ONCE at the cell level (not row level)
#         This replaces build_neighbor_lookup() entirely.
# ---------------------------------------------------------------

build_neighbor_lookup_fast <- function(dt, id_order, neighbors) {
  # dt: data.table with columns 'id' and 'year' (and all predictor columns)
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices)

  # --- Cell-level neighbor edge list (time-invariant) ---
  # Map each cell's position in id_order to its neighbor cell IDs
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Build edge list: (focal_id, neighbor_id)
  edge_list <- rbindlist(lapply(seq_along(id_order), function(pos) {
    nb_pos <- neighbors[[pos]]
    if (length(nb_pos) == 0L || (length(nb_pos) == 1L && nb_pos[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[pos], neighbor_id = id_order[nb_pos])
  }))

  # --- Build row-index lookup: (id, year) -> row number in dt ---
  # Ensure dt has a row index column
  dt[, .row_idx := .I]

  # Create keyed lookup table
  row_lookup <- dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # --- For each focal row, find its neighbor rows via join ---
  # Expand edge_list across all years
  years <- sort(unique(dt$year))

  # Cross join edges Ã— years
  edge_year <- CJ_dt(edge_list, years)

  # Join to get focal row index
  setkey(edge_year, focal_id, year)
  edge_year[row_lookup, focal_row := i..row_idx, on = .(focal_id = id, year = year)]

  # Join to get neighbor row index
  edge_year[row_lookup, neighbor_row := i..row_idx, on = .(neighbor_id = id, year = year)]

  # Drop edges where either focal or neighbor row is missing
  edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row)]

  # Sort by focal_row for grouped operations
  setkey(edge_year, focal_row)

  # Clean up temporary column
  dt[, .row_idx := NULL]

  return(edge_year)
}

# Helper: cross join a data.table with a vector of years
CJ_dt <- function(edge_dt, years) {
  years_dt <- data.table(year = years)
  # Cross join
  result <- edge_dt[, .(year = years), by = .(focal_id, neighbor_id)]
  return(result)
}

# ---------------------------------------------------------------
# STEP 2: Vectorized neighbor stats computation
#         Replaces compute_neighbor_stats() and the outer loop.
# ---------------------------------------------------------------

compute_all_neighbor_features_fast <- function(dt, edge_year, neighbor_source_vars) {
  # dt: data.table of cell_data (6.46M rows)
  # edge_year: data.table with columns (focal_row, neighbor_row) â€” the edge list
  # neighbor_source_vars: character vector of variable names

  n <- nrow(dt)

  for (var_name in neighbor_source_vars) {
    message("Processing neighbor features for: ", var_name)

    # Extract neighbor values via integer indexing (fast)
    edge_year[, nval := dt[[var_name]][neighbor_row]]

    # Remove NA neighbor values
    valid <- edge_year[!is.na(nval)]

    # Grouped aggregation â€” fully vectorized
    agg <- valid[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_row]

    # Initialize result columns with NA
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign results by integer row index (preserves exact numerical values)
    set(dt, i = agg$focal_row, j = max_col,  value = agg$nb_max)
    set(dt, i = agg$focal_row, j = min_col,  value = agg$nb_min)
    set(dt, i = agg$focal_row, j = mean_col, value = agg$nb_mean)

    # Clean up
    edge_year[, nval := NULL]
  }

  return(dt)
}

# ---------------------------------------------------------------
# STEP 3: Main pipeline â€” drop-in replacement
# ---------------------------------------------------------------

# Convert to data.table if not already
cell_data <- as.data.table(cell_data)

# Build the fast edge-year lookup (runs once, ~344K cells Ã— ~4 neighbors Ã— 28 years)
message("Building neighbor edge-year index...")
edge_year <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge-year table: %s rows", format(nrow(edge_year), big.mark = ",")))

# Compute all neighbor features (vectorized, no row-level lapply)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- compute_all_neighbor_features_fast(cell_data, edge_year, neighbor_source_vars)

# The trained Random Forest model is untouched.
# Column names follow the same convention (neighbor_max_ntl, neighbor_min_ntl, etc.)
# Numerical results are identical â€” same max/min/mean over the same integer-indexed neighbor sets.

# Predict using the existing model as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup()`: 6.46M row-level `paste()` + named-vector hash lookups | Eliminated |
| **Neighbor structure** | Rebuilt per cell-year (Ã—28 redundancy) | Built once at cell level, expanded via join |
| **Key mechanism** | Character `paste()` + named vector indexing | Integer row indexing + `data.table` grouped aggregation |
| **`compute_neighbor_stats`** | Row-level `lapply` + `do.call(rbind, ...)` | Fully vectorized `data.table` `[, .(), by=]` |
| **Estimated runtime** | 86+ hours | **Minutes** (dominated by the single `data.table` grouped aggregation over ~25.8M edges Ã— 5 variables) |
| **Numerical output** | Identical | Identical (same `max`, `min`, `mean` over same neighbor sets) |
| **RF model** | Preserved | Preserved (no retraining) |

The colleague misidentified a cosmetic inefficiency (`do.call(rbind, ...)`) while the real cost was the **6.46-million-iteration character-key construction and hash-lookup loop** in `build_neighbor_lookup()`, compounded by the 28Ã— year redundancy. The fix is to work at the cell level with integer indexing and vectorized grouped aggregation.
END_RESPONSE>>>
