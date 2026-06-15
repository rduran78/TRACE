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
run_id: Anthropic_F04_T10_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is overwhelmingly in `build_neighbor_lookup`, not in `compute_neighbor_stats`. Here's why:

**`build_neighbor_lookup`** iterates over **~6.46 million rows** with an `lapply` call. Inside each iteration it:

1. Looks up the cell's reference index via a named character vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. **Constructs paste-based string keys** (`paste(neighbor_cell_ids, year, sep = "_")`) for every neighbor of every row.
4. **Matches those string keys** against a ~6.46-million-entry named character vector (`idx_lookup`).

Across 6.46 million rows, each with an average of ~4 rook neighbors, that's ~25.8 million `paste` + named-vector-lookup operations, all inside an interpreted R loop. Named vector lookup in R is **O(n)** hash-table probing on character strings, and `paste` allocates new strings each time. The combined cost is enormous: **string allocation, hashing, and matching dominate runtime**, easily explaining 86+ hours.

**`compute_neighbor_stats`** is comparatively cheap: it simply indexes into a numeric vector and computes `max`/`min`/`mean` over small integer index sets. Once the lookup is built, this part is fast.

**Summary of root causes:**

| Cause | Location | Impact |
|---|---|---|
| Per-row `paste` key construction (Ã—6.46M rows Ã— ~4 neighbors) | `build_neighbor_lookup` | Very high |
| Per-row character-key lookup in a 6.46M-entry named vector | `build_neighbor_lookup` | Very high |
| Entire computation is in a scalar R `lapply` loop | Both functions | High |
| `do.call(rbind, result)` on a 6.46M-element list of 3-vectors | `compute_neighbor_stats` | Moderate |

---

## Optimization Strategy

**Core idea:** Eliminate all string operations. Replace the string-keyed lookup with integer arithmetic and vectorized/`data.table` joins.

1. **Replace `paste`-based key construction with integer compound keys.** Since `year` is in a known range (1992â€“2019, 28 values) and `id` is a cell index, we can encode `(id, year) â†’ single integer` as `id * 100 + (year - 1991)` or use a `data.table` keyed join. This avoids all `paste` and character hashing.

2. **Vectorize the neighbor lookup construction.** Instead of looping over 6.46M rows, expand the neighbor list once per cell (344,208 cells), then join against all years simultaneously using `data.table`. This turns the problem into a single merge.

3. **Vectorize `compute_neighbor_stats`** using `data.table` grouped aggregation (`max`, `min`, `mean` by group) instead of `lapply` over 6.46M list elements.

4. **Memory check:** The expanded neighbor-pair table will have ~1.37M neighbor pairs Ã— 28 years â‰ˆ 38.5M rows with a few integer columnsâ€”roughly 1â€“2 GB, well within 16 GB RAM.

These changes reduce the estimated runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

#' Build neighbor features using vectorized data.table operations.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order         integer vector of cell IDs (the ordering used by the nb object)
#' @param rook_neighbors   spdep::nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names to compute neighbor stats for
#' @return data.table with original columns plus neighbor feature columns appended
add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      rook_neighbors,
                                      neighbor_source_vars) {

  # --- Step 0: Convert to data.table; preserve original row order ---
  dt <- as.data.table(cell_data)
  dt[, .row_order := .I]

  # --- Step 1: Build edge list (cell_id -> neighbor_cell_id) from nb object ---
  #     This is done once for the 344,208 cells, not per row.
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors), function(i) {
    nb_idx <- rook_neighbors[[i]]
    # nb objects use 0-length integer for no-neighbor; filter those
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1L] == 0L)) {
      return(NULL)
    }
    data.table(cell_id = id_order[i],
               neighbor_id = id_order[nb_idx])
  }))
  # edge_list has ~1.37M rows (directed rook-neighbor pairs)

  # --- Step 2: Expand edge list across all 28 years ---
  years <- sort(unique(dt$year))
  # Cross join edge_list with years: ~1.37M Ã— 28 â‰ˆ 38.5M rows
  edge_year <- CJ_dt(edge_list, years)

  # --- Step 3: Attach row indices for the focal cell (for later join-back) ---
  setkey(dt, id, year)
  # We need a mapping from (id, year) -> row index in dt
  dt_idx <- dt[, .(id, year, .row_order)]
  setkey(dt_idx, id, year)

  # Attach focal row order to edge_year
  setkey(edge_year, cell_id, year)
  edge_year <- dt_idx[edge_year,
                       .(focal_row = .row_order,
                         neighbor_id = i.neighbor_id,
                         year = i.year),
                       on = .(id = cell_id, year),
                       nomatch = 0L]

  # --- Step 4: Attach neighbor variable values ---
  # Build a slim table of just id, year, and the source vars from dt
  neighbor_vals_dt <- dt[, c("id", "year", neighbor_source_vars), with = FALSE]
  setkey(neighbor_vals_dt, id, year)

  # Join neighbor values onto edge_year
  setkey(edge_year, neighbor_id, year)
  edge_year <- neighbor_vals_dt[edge_year,
                                 on = .(id = neighbor_id, year),
                                 nomatch = 0L]

  # --- Step 5: Compute grouped stats (max, min, mean) per focal row per variable ---
  # Group by focal_row
  stat_exprs <- list()
  for (v in neighbor_source_vars) {
    sym_v <- as.name(v)
    stat_exprs[[paste0("neighbor_max_", v)]]  <-
      bquote(as.numeric(max(.(sym_v), na.rm = TRUE)))
    stat_exprs[[paste0("neighbor_min_", v)]]  <-
      bquote(as.numeric(min(.(sym_v), na.rm = TRUE)))
    stat_exprs[[paste0("neighbor_mean_", v)]] <-
      bquote(mean(.(sym_v), na.rm = TRUE))
  }
  # Evaluate all at once in a single grouped aggregation pass
  stats_dt <- edge_year[, eval(as.call(c(as.name("list"),
                                          stat_exprs))),
                         by = focal_row]

  # Replace Inf/-Inf from max/min of all-NA groups with NA
  inf_cols <- grep("^neighbor_(max|min)_", names(stats_dt), value = TRUE)
  for (col in inf_cols) {
    set(stats_dt, i = which(is.infinite(stats_dt[[col]])), j = col, value = NA_real_)
  }

  # --- Step 6: Join stats back to the original data in original row order ---
  setkey(stats_dt, focal_row)
  setkey(dt, .row_order)
  dt <- stats_dt[dt, on = .(focal_row = .row_order)]

  # Clean up helper columns
  dt[, focal_row := NULL]
  dt[, .row_order := NULL]

  return(dt[])
}

#' Helper: cross join a data.table with a vector of years
CJ_dt <- function(edge_dt, years_vec) {
  years_dt <- data.table(year = years_vec)
  # Cross join via allow.cartesian
  merge_key <- 1L
  edge_dt[, .merge_key := merge_key]
  years_dt[, .merge_key := merge_key]
  result <- edge_dt[years_dt, on = ".merge_key", allow.cartesian = TRUE]
  result[, .merge_key := NULL]
  return(result)
}


# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now contains the same 15 new columns that the original code produced:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... etc.
#
# Feed cell_data (with its ~110 predictors including these 15) into the
# already-trained Random Forest via predict(rf_model, cell_data) as before.
# The trained model is not modified.
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets. The `data.table` join `(neighbor_id, year)` replicates the original `paste(id, year)` key logic, so the same rows are matched. `na.rm = TRUE` replicates the `!is.na` filter. |
| **Trained RF model** | The code only constructs predictor columns. It never touches the model object. You call `predict(rf_model, cell_data)` afterward exactly as before. |
| **Column names** | Output columns are named `neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>` â€” adjust the naming pattern if your trained model expects different feature names (e.g., `compute_and_add_neighbor_features` used a different convention). |

---

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~80+ hours (paste + char match Ã— 6.46M) | ~30 seconds (vectorized edge-list expansion + data.table keyed join) |
| Neighbor stats computation | ~6 hours (lapply + do.call rbind Ã— 5 vars) | ~2 minutes (single grouped `data.table` aggregation) |
| Peak RAM | Lower but irrelevant at 86 h | ~4â€“6 GB (well within 16 GB) |
| **Total** | **86+ hours** | **~3â€“5 minutes** |
END_RESPONSE>>>
