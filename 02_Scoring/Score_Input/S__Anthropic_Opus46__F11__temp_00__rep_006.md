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
run_id: Anthropic_F11_T00_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M iterations, this alone is extremely expensive.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level loop iterations total.

3. By contrast, Random Forest **prediction** (`predict()`) on a pre-trained model is a single vectorized C/C++ call over the dataset. Even with 6.46M rows and 110 predictors, this typically completes in minutes, not hours. Loading and writing the model/predictions is I/O-bound and also fast relative to the feature engineering.

**The bottleneck is the row-level R `lapply` loops over millions of rows in the neighbor lookup construction and neighbor statistics computation.** This is what drives the estimated 86+ hour runtime.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()` with a vectorized `data.table` equi-join.** Instead of looping row-by-row, explode the neighbor graph into an edge list (cell_id, neighbor_id), merge with year to get (cell_id, year, neighbor_id, year) pairs, and map everything to row indices via a keyed `data.table` join. This eliminates millions of `paste()` and named-vector lookups.

2. **Replace `compute_neighbor_stats()` with a single grouped `data.table` aggregation per variable.** Using the edge-list representation, join neighbor values and compute `max`, `min`, `mean` in one grouped operation â€” fully vectorized in C via `data.table`.

3. **Preserve the trained Random Forest model and the original numerical estimand.** The optimized code produces identical neighbor features (same max, min, mean of neighbor values), so predictions from the existing model are unchanged.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Assume these objects already exist in the workspace:
#       cell_data              â€“ data.frame / data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, â€¦
#       id_order               â€“ integer/character vector of unique cell IDs (same order as rook_neighbors_unique)
#       rook_neighbors_unique  â€“ spdep nb object (list of integer index vectors)
#       rf_model               â€“ the pre-trained Random Forest model
# ---------------------------------------------------------------

# Convert cell_data to data.table if not already
cell_data <- as.data.table(cell_data)

# ---------------------------------------------------------------
# STEP 1: Build a vectorized edge list from the nb object
#         This replaces build_neighbor_lookup() entirely.
# ---------------------------------------------------------------

# Expand the nb list into a two-column edge list of positional indices
#   from_pos -> index into id_order (the focal cell)
#   to_pos   -> index into id_order (the neighbor cell)
edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_i <- rook_neighbors_unique[[i]]
  # spdep nb objects use 0L to denote "no neighbors"
  nb_i <- nb_i[nb_i != 0L]
  if (length(nb_i) == 0L) return(NULL)
  data.table(from_id = id_order[i], to_id = id_order[nb_i])
}))
# edge_list now has columns: from_id, to_id
# This loop is over ~344K cells (not 6.46M) and runs in seconds.

cat(sprintf("Edge list built: %d directed neighbor pairs\n", nrow(edge_list)))

# ---------------------------------------------------------------
# STEP 2: Create a row-index mapping table (id, year) -> row_idx
# ---------------------------------------------------------------

cell_data[, row_idx := .I]

# Key for fast joins
row_map <- cell_data[, .(id, year, row_idx)]
setkey(row_map, id, year)

# ---------------------------------------------------------------
# STEP 3: Build the full neighbor-pair table at the cell-year level
#         by joining edge_list with the year dimension.
#
#   For every (from_id, year) row, find all (to_id, year) rows.
#   This is the vectorized equivalent of build_neighbor_lookup().
# ---------------------------------------------------------------

# Get focal rows: (from_id, year, focal_row_idx)
focal <- row_map[, .(from_id = id, year, focal_row_idx = row_idx)]
setkey(focal, from_id)

# Join edges onto focal rows  â†’  (from_id, year, to_id, focal_row_idx)
# This replicates each focal row once per neighbor.
neighbor_pairs <- edge_list[focal, on = .(from_id), allow.cartesian = TRUE, nomatch = 0L]
# columns: from_id, to_id, year, focal_row_idx

# Now attach the neighbor's row index by joining (to_id, year) -> row_idx
setnames(row_map, c("id", "year", "row_idx"), c("to_id", "year", "nb_row_idx"))
setkey(row_map, to_id, year)

neighbor_pairs <- row_map[neighbor_pairs, on = .(to_id, year), nomatch = 0L]
# columns: to_id, year, nb_row_idx, from_id, focal_row_idx

# Restore row_map names for potential later use
setnames(row_map, c("to_id", "year", "nb_row_idx"), c("id", "year", "row_idx"))

cat(sprintf("Neighbor-pair table built: %d cell-year-neighbor rows\n", nrow(neighbor_pairs)))

# ---------------------------------------------------------------
# STEP 4: Compute neighbor stats for each source variable
#         This replaces compute_neighbor_stats() and the outer loop.
# ---------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  cat(sprintf("Computing neighbor features for: %s\n", var_name))

  # Pull the variable values for neighbor rows (vectorized)
  neighbor_pairs[, nb_val := cell_data[[var_name]][nb_row_idx]]

  # Grouped aggregation â€” fully vectorized in data.table's C backend

  agg <- neighbor_pairs[!is.na(nb_val),
    .(
      nb_max  = max(nb_val),
      nb_min  = min(nb_val),
      nb_mean = mean(nb_val)
    ),
    keyby = .(focal_row_idx)
  ]

  # Initialize new columns with NA
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")

  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  # Assign aggregated values back by row index
  cell_data[agg$focal_row_idx, (max_col)  := agg$nb_max]
  cell_data[agg$focal_row_idx, (min_col)  := agg$nb_min]
  cell_data[agg$focal_row_idx, (mean_col) := agg$nb_mean]

  # Clean up the temporary column
  neighbor_pairs[, nb_val := NULL]
}

# Drop helper column
cell_data[, row_idx := NULL]

cat("Neighbor feature engineering complete.\n")

# ---------------------------------------------------------------
# STEP 5: Random Forest prediction (unchanged â€” not the bottleneck)
# ---------------------------------------------------------------

predictions <- predict(rf_model, newdata = cell_data)
cell_data[, prediction := predictions]

cat("Pipeline complete.\n")
```

---

## Why This Is Fast

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | ~6.46M `lapply` iterations with `paste()` + named-vector lookups | One `data.table` cartesian join over ~344K edges Ã— 28 years (vectorized C) |
| Neighbor stats | 5 Ã— 6.46M `lapply` iterations computing `max/min/mean` per row | 5 Ã— one grouped `data.table` aggregation (vectorized C) |
| Estimated time | **86+ hours** | **Minutes** (typically 5â€“20 min depending on RAM/disk) |
| RF inference | Minutes (unchanged) | Minutes (unchanged) |
| Numerical output | Original `max`, `min`, `mean` | **Identical** `max`, `min`, `mean` â€” estimand preserved |

The optimization targets the actual bottleneck (row-level R loops over 6.46M rows repeated across 5 variables) and replaces it with vectorized `data.table` joins and grouped aggregations, reducing runtime by roughly **two to three orders of magnitude** while producing numerically identical features and preserving the trained Random Forest model.
END_RESPONSE>>>
