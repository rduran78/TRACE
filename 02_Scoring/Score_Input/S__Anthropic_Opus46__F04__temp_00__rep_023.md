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
run_id: Anthropic_F04_T00_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Looks those keys up in another named character vector (`idx_lookup`).

String construction (`paste`) and named-vector lookup (which is O(n) hash probing on character keys) repeated ~6.46M times, each with a variable number of neighbors, is extremely expensive in interpreted R. The secondary bottleneck is `compute_neighbor_stats`, which runs an `lapply` over 6.46M list elements calling `max`/`min`/`mean` on small vectors â€” slow due to R function-call overhead, but less severe than the lookup construction.

**Root causes, ranked:**

| Rank | Cause | Impact |
|------|-------|--------|
| 1 | Per-row `paste()` + character-key lookup in `build_neighbor_lookup` over 6.46M rows | Dominant (~80%+ of runtime) |
| 2 | `lapply` + per-element `max/min/mean` in `compute_neighbor_stats` over 6.46M list elements | Significant (~15%) |
| 3 | Repeated extraction for 5 variables (Ã—3 stats = 15 new columns) | Multiplicative on #2 |

## Optimization Strategy

**Core idea:** Replace all per-row string operations and list-based lookups with a single vectorized merge/join using `data.table`, and replace the per-row `lapply` stats computation with grouped `data.table` aggregation.

**Steps:**

1. **Pre-expand the neighbor graph into an edge table** (cell_id â†’ neighbor_id), ~1.37M directed edges. This is done once.
2. **Join the edge table to the panel data by (neighbor_id, year)** to get neighbor variable values â€” a single keyed `data.table` join, fully vectorized in C.
3. **Group-by aggregate** (max, min, mean) per (cell_id, year) â€” also vectorized in C via `data.table`.
4. **Join the aggregated stats back** to the main data.

This eliminates all `lapply`, all `paste` key construction, and all named-vector lookups. Expected speedup: **~200â€“500Ã—** (minutes instead of 86+ hours).

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert panel data to data.table (if not already)
# ============================================================
cell_dt <- as.data.table(cell_data)

# ============================================================
# STEP 1: Build a vectorized edge table from the nb object
#         (done once; ~1.37M rows)
# ============================================================
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  # id_order is the vector mapping positional index -> cell id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)
  data.table(
    id       = id_order[from_idx],
    nb_id    = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# ============================================================
# STEP 2 & 3: For each source variable, compute neighbor
#             max/min/mean via keyed join + grouped aggregation
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key the main table for fast joins
setkey(cell_dt, id, year)

for (var_name in neighbor_source_vars) {

  # --- 2a. Build a slim lookup: (id, year, value) keyed for join ---
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setnames(val_dt, "id", "nb_id")
  setkey(val_dt, nb_id, year)

  # --- 2b. Expand edges Ã— years: join edge_dt to cell_dt's years,
  #         then join to get neighbor values.
  #         We need (id, year) for every cell-row, crossed with its neighbors.
  #         Efficient approach: join edges to the year column of cell_dt,
  #         then join neighbor values. ---

  # Get unique (id, year) pairs from cell_dt
  id_year <- cell_dt[, .(id, year)]
  setkey(id_year, id)

  # Merge: for each (id, year), attach all neighbor ids
  # edge_dt is keyed on 'id'
  setkey(edge_dt, id)
  expanded <- edge_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, nb_id, year

  # Merge: attach the neighbor's variable value
  setkey(expanded, nb_id, year)
  expanded <- val_dt[expanded, on = .(nb_id, year), nomatch = NA]
  # expanded now has: nb_id, year, val, id

  # --- 3. Aggregate per (id, year) ---
  agg <- expanded[!is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    by = .(id, year)
  ]

  # Name columns to match original pipeline's naming convention
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))

  # --- 4. Join aggregated stats back to main table ---
  setkey(agg, id, year)
  setkey(cell_dt, id, year)

  # Remove old columns if they exist (idempotent re-runs)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  cell_dt <- agg[cell_dt, on = .(id, year)]

  # Clean up per-iteration temporaries
  rm(val_dt, id_year, expanded, agg)
}

# ============================================================
# STEP 4: Convert back to data.frame if downstream code expects it
# ============================================================
cell_data <- as.data.frame(cell_dt)

# ============================================================
# STEP 5: Predict with the existing trained Random Forest
#         (model object is untouched)
# ============================================================
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---------|-----------|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets (rook contiguity Ã— same year, excluding `NA`). The `data.table` grouped aggregation is numerically identical to the original `lapply` approach. |
| **Trained RF model** | The model object is never modified. Only the input feature columns are reconstructed with identical values, so `predict()` produces the same output. |
| **Missing-data handling** | `!is.na(val)` in the aggregation and the left join (`agg[cell_dt, ...]`) ensure that cell-years with no valid neighbors receive `NA` for all three stats â€” matching the original `c(NA, NA, NA)` return. |

## Expected Performance

| Stage | Original | Optimized | Speedup |
|-------|----------|-----------|---------|
| Neighbor lookup construction | ~70+ hrs (lapply, paste, char lookup) | ~10 sec (vectorized edge table) | ~25,000Ã— |
| Stats computation (5 vars Ã— 6.46M rows) | ~16 hrs (lapply, per-row R calls) | ~2â€“5 min (data.table keyed join + groupby) | ~200Ã— |
| **Total** | **~86+ hrs** | **~3â€“6 min** | **~1,000Ã—** |

Peak memory for the largest intermediate (`expanded`) is approximately 1.37M edges Ã— 28 years â‰ˆ 38.4M rows Ã— 3 columns â‰ˆ ~0.9 GB, well within the 16 GB laptop constraint.
END_RESPONSE>>>
