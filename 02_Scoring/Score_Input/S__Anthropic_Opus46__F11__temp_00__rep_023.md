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
run_id: Anthropic_F11_T00_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M iterations, this alone is extremely expensive.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level loop iterations total.

3. By contrast, Random Forest **prediction** (`predict()`) on a pre-trained model is a single vectorized C/C++ call over the dataset. Even with 110 predictors and 6.46M rows, this is orders of magnitude faster than tens of millions of interpreted R loop iterations with per-element string operations.

**The 86+ hour runtime is dominated by the neighbor feature engineering, not by RF inference.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a fully vectorized `data.table` join approach. Instead of looping row-by-row, expand all neighbor relationships into an edge list, join against the data to resolve `(neighbor_id, year)` â†’ row index, and store the result as a grouped structure.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation per variable â€” no R-level `lapply` over millions of rows.

3. **Preserve the trained RF model** (no retraining) and **preserve the original numerical estimand** (max, min, mean of neighbor values are computed identically).

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Assume these objects already exist:
#       cell_data              â€“ data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, â€¦
#       id_order               â€“ integer/character vector of unique cell IDs (same order as rook_neighbors_unique)
#       rook_neighbors_unique  â€“ spdep nb object (list of integer index vectors)
#       rf_model               â€“ pre-trained Random Forest model
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# 1.  Build a global edge-list of directed neighbor relationships
#     (done once; replaces build_neighbor_lookup entirely)
# ---------------------------------------------------------------

# Expand the nb object into a two-column edge list of positional indices
#   from_pos : index into id_order  (the focal cell)
#   to_pos   : index into id_order  (the neighbor cell)
edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
    return(NULL)
  }
  data.table(from_pos = i, to_pos = nb)
}))

# Map positional indices to actual cell IDs
edge_list[, from_id := id_order[from_pos]]
edge_list[, to_id   := id_order[to_pos]]
edge_list[, c("from_pos", "to_pos") := NULL]

# ---------------------------------------------------------------
# 2.  Convert cell_data to data.table and add a row-index column
# ---------------------------------------------------------------
dt <- as.data.table(cell_data)
dt[, .row_idx := .I]          # preserve original row order

# Create a unique set of years
years <- sort(unique(dt$year))

# ---------------------------------------------------------------
# 3.  Build the full neighbor-lookup table:
#     For every (focal_id, year) row, find all (neighbor_id, year) rows.
#     This is a cross-join of edge_list Ã— years, then joined to dt.
# ---------------------------------------------------------------

# Keyed version of dt for fast joins
dt_key <- dt[, .(id, year, .row_idx)]
setkey(dt_key, id, year)

# Expand edges across all years  (edges Ã— years)
# Memory: ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows (manageable on 16 GB)
edges_by_year <- CJ_dt_edges(edge_list, years)

# --- helper to cross-join edges with years efficiently ---
# We avoid CJ on the full thing; instead replicate edge_list for each year.
edges_by_year <- edge_list[, .(from_id, to_id)][
  , CJ_year := list(years)          # won't work directly; use rep approach below
]

# Cleaner approach:
edges_by_year <- edge_list[
  rep(seq_len(.N), each = length(years))
][, year := rep(years, times = nrow(edge_list))]

# Now edges_by_year has columns: from_id, to_id, year
# Join to get the ROW INDEX of the focal cell and the neighbor cell

# Focal row index
setkey(edges_by_year, from_id, year)
edges_by_year[dt_key, focal_row := i..row_idx, on = .(from_id = id, year)]

# Neighbor row index
setkey(edges_by_year, to_id, year)
edges_by_year[dt_key, neighbor_row := i..row_idx, on = .(to_id = id, year)]

# Drop rows where either focal or neighbor is missing (cell-year doesn't exist)
edges_by_year <- edges_by_year[!is.na(focal_row) & !is.na(neighbor_row)]

# ---------------------------------------------------------------
# 4.  Compute neighbor stats for each variable (vectorized)
# ---------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  # Attach the neighbor's value for this variable
  edges_by_year[, nval := dt[[var_name]][neighbor_row]]

  # Aggregate: max, min, mean per focal row (excluding NAs)
  agg <- edges_by_year[!is.na(nval),
    .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ),
    by = focal_row
  ]

  # Initialize new columns with NA
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")

  dt[, (max_col)  := NA_real_]
  dt[, (min_col)  := NA_real_]
  dt[, (mean_col) := NA_real_]

  # Fill in aggregated values
  dt[agg$focal_row, (max_col)  := agg$nb_max]
  dt[agg$focal_row, (min_col)  := agg$nb_min]
  dt[agg$focal_row, (mean_col) := agg$nb_mean]
}

# Clean up the temporary column
edges_by_year[, nval := NULL]

# ---------------------------------------------------------------
# 5.  Random Forest prediction (unchanged â€” not the bottleneck)
# ---------------------------------------------------------------
dt[, .row_idx := NULL]
cell_data <- as.data.frame(dt)

predictions <- predict(rf_model, newdata = cell_data)
cell_data$predicted <- predictions

# ---------------------------------------------------------------
# Done. The trained RF model is preserved (never retrained).
# The numerical estimand (neighbor max, min, mean) is identical.
# ---------------------------------------------------------------
```

### Cleaner self-contained version (fixing the cross-join):

```r
library(data.table)

# === STEP 1: Edge list from nb object ===
edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {

  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) return(NULL)
  data.table(from_id = id_order[i], to_id = id_order[nb])
}))

# === STEP 2: Prepare data.table ===
dt <- as.data.table(cell_data)
dt[, .row_idx := .I]

# === STEP 3: Build full (focal_row, neighbor_row) mapping ===
# Join edges to data by (from_id, year) and (to_id, year)
# Strategy: join edges to dt on from_id to get all (from_id, year, focal_row),
#           then join to dt on (to_id, year) to get neighbor_row.

# Get focal side: every edge Ã— every year the focal cell appears
focal_side <- merge(
  edge_list,
  dt[, .(from_id = id, year, focal_row = .row_idx)],
  by = "from_id",
  allow.cartesian = TRUE
)

# Get neighbor side
setkey(dt, id, year)
focal_side[dt, neighbor_row := i..row_idx,
           on = .(to_id = id, year = year)]

# Keep only valid pairs
pairs <- focal_side[!is.na(neighbor_row), .(focal_row, neighbor_row)]
rm(focal_side); gc()

# === STEP 4: Vectorized neighbor stats ===
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  pairs[, nval := dt[[var_name]][neighbor_row]]

  agg <- pairs[!is.na(nval),
    .(nb_max = max(nval), nb_min = min(nval), nb_mean = mean(nval)),
    by = focal_row
  ]

  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")

  dt[, (max_col)  := NA_real_]
  dt[, (min_col)  := NA_real_]
  dt[, (mean_col) := NA_real_]

  dt[agg$focal_row, (max_col)  := agg$nb_max]
  dt[agg$focal_row, (min_col)  := agg$nb_min]
  dt[agg$focal_row, (mean_col) := agg$nb_mean]

  pairs[, nval := NULL]
}

# === STEP 5: RF prediction (preserved, not retrained) ===
dt[, .row_idx := NULL]
cell_data <- as.data.frame(dt)
cell_data$predicted <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Colleague's Claim (RF Inference) | Actual Bottleneck (Neighbor Features) |
|---|---|---|
| Operation type | Single vectorized C call | ~38M R-level interpreted iterations |
| Scales with | Rows Ã— trees (optimized in C) | Rows Ã— neighbors Ã— variables (R loops + string ops) |
| Estimated share of 86h runtime | Minor (minutes) | Dominant (hours to days) |

**Optimization impact**: The `data.table` join-and-aggregate approach replaces ~38 million R-level `lapply` iterations and millions of `paste()`/named-vector lookups with vectorized C-level joins and grouped aggregations. Expected speedup: **~100â€“500Ã—**, reducing the neighbor feature step from many hours to minutes.
END_RESPONSE>>>
