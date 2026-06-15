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
run_id: Anthropic_F04_T07_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Performs a character-based hash lookup (`id_to_ref`),
2. Retrieves neighbor cell IDs from the `nb` object,
3. Constructs character key strings via `paste()` for every neighbor,
4. Looks those keys up in `idx_lookup` (a named character vector of length ~6.46M).

**String allocation and named-vector lookup are O(n) or near-O(n) per probe in R's hashed environments, but `paste()` and repeated character coercion across 6.46 million rows with ~4 neighbors each creates ~25+ million temporary strings.** The `lapply` returns a list of 6.46M integer vectors â€” the sheer object overhead is enormous.

`compute_neighbor_stats` then loops over that 6.46M-element list *once per variable* (Ã—5 variables), each time extracting, subsetting, and summarizing â€” another ~32M R-level function calls.

**Root causes ranked by impact:**

1. **Row-level `lapply` in `build_neighbor_lookup`**: ~6.46M R-level iterations with string operations.
2. **Row-level `lapply` in `compute_neighbor_stats`**: ~6.46M iterations Ã— 5 variables.
3. **No vectorization or use of data.table / matrix operations.**

## Optimization Strategy

**Core idea:** Replace the row-level loop with a fully vectorized, edge-list-based `data.table` join. Instead of building a per-row neighbor lookup list, we:

1. Expand the `nb` object into a flat edge list (cell_id â†’ neighbor_id), ~1.37M rows.
2. Cross-join with years using `data.table` to get ~1.37M Ã— 28 â‰ˆ 38.4M edge-year rows.
3. Join source variable values onto the neighbor side.
4. Group-by aggregate (`max`, `min`, `mean`) per (cell, year).
5. Join results back to the main table.

This eliminates all per-row R loops and string operations, replacing them with `data.table`'s optimized C-level grouped joins and aggregations. Expected runtime: **minutes, not days.**

## Working R Code

```r
library(data.table)

# â”€â”€ Step 0: Convert main data to data.table (once) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# â”€â”€ Step 1: Expand nb object to flat edge list (once) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#   rook_neighbors_unique is an nb object indexed by position in id_order.
#   id_order is the vector of cell IDs corresponding to each nb element.

build_edge_list <- function(id_order, neighbors) {
  # neighbors is an spdep nb object: list of integer index vectors
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)
  # Remove the spdep convention where 0L means "no neighbors"
  valid    <- to_idx > 0L
  data.table(
    id          = id_order[from_idx[valid]],
    neighbor_id = id_order[to_idx[valid]]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# ~1.37M rows, two integer columns â€” trivial memory

# â”€â”€ Step 2: Cross with years to get edge-year table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years_dt <- data.table(year = sort(unique(cell_dt$year)))
# Cross join: every edge Ã— every year
edge_year_dt <- edge_dt[, CJ_id := .I]  # placeholder; use CJ approach below

# More memory-efficient: use allow.cartesian join
edge_year_dt <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years_dt$year)
edge_year_dt[, `:=`(
  id          = edge_dt$id[edge_idx],
  neighbor_id = edge_dt$neighbor_id[edge_idx]
)]
edge_year_dt[, edge_idx := NULL]
# ~38.4M rows Ã— 3 columns â‰ˆ ~0.9 GB (fits in 16 GB RAM)

setkey(edge_year_dt, neighbor_id, year)

# â”€â”€ Step 3: Function to compute and attach neighbor features for one var â”€â”€â”€â”€â”€
compute_and_add_neighbor_features_fast <- function(cell_dt, edge_year_dt, var_name) {
  # Extract only the columns we need for the join
  val_dt <- cell_dt[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(val_dt, neighbor_id, year)

  # Join neighbor values onto edge-year table
  merged <- val_dt[edge_year_dt, on = .(neighbor_id, year), nomatch = NA]
  # merged has columns: neighbor_id, year, val, id

  # Aggregate per (id, year)
  agg <- merged[!is.na(val),
    .(
      nmax  = max(val),
      nmin  = min(val),
      nmean = mean(val)
    ),
    by = .(id, year)
  ]

  # Name output columns to match original pipeline convention
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(agg, c("nmax", "nmin", "nmean"), c(max_col, min_col, mean_col))

  # Join back to main table
  setkey(agg, id, year)
  setkey(cell_dt, id, year)
  cell_dt <- agg[cell_dt, on = .(id, year)]

  cell_dt
}

# â”€â”€ Step 4: Outer loop over the 5 source variables â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_dt <- compute_and_add_neighbor_features_fast(cell_dt, edge_year_dt, var_name)
}

# â”€â”€ Step 5: Convert back to data.frame if downstream code expects it â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is unchanged â€” use it for prediction as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves the Original Numerical Estimand

The aggregation computes **exactly** `max`, `min`, and `mean` of the same neighbor values identified by the same rook-contiguity structure (`rook_neighbors_unique`) and the same year-matching logic. The only change is *how* the computation is dispatched (vectorized C-level `data.table` grouping vs. R-level `lapply`). No approximation, sampling, or rounding is introduced. The trained Random Forest model is never modified or retrained.

## Memory Budget Check

| Object | Rows | Est. Size |
|---|---|---|
| `cell_dt` | 6.46M Ã— ~113 cols | ~5.5 GB |
| `edge_year_dt` | 38.4M Ã— 3 cols | ~0.9 GB |
| `merged` (temp per var) | 38.4M Ã— 4 cols | ~1.2 GB |
| **Peak total** | | **~9 GB** |

Fits within 16 GB. If memory is tight, `edge_year_dt` can be built on-the-fly per variable or years can be processed in chunks, but this should not be necessary.

## Expected Speedup

| Phase | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M string ops) | ~2 seconds (vectorized `unlist`) |
| `compute_neighbor_stats` Ã—5 | ~hours (32M R calls) | ~3â€“5 min (data.table grouped agg) |
| **Total** | **86+ hours** | **< 10 minutes** |
END_RESPONSE>>>
