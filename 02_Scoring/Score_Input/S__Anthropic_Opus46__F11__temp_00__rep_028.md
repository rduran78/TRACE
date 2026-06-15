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
run_id: Anthropic_F11_T00_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M iterations, this alone is extremely expensive.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level loop iterations total.

3. **Random Forest inference** (`predict()`) on a pre-trained model over ~6.46M rows with ~110 predictors is non-trivial, but it is a single vectorized C-level call (in `randomForest` or `ranger`). It is orders of magnitude faster than tens of millions of interpreted R-level loop iterations with repeated string operations and named-vector lookups.

The estimated 86+ hour runtime is dominated by the R-level `lapply` loops with per-element string manipulation and named-vector indexing across millions of rows.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a fully vectorized `data.table` join approach. Instead of building a per-row list of neighbor indices via character key lookups in a loop, we:
   - Expand the neighbor list into an edge-list data.table (`cell_id` â†’ `neighbor_cell_id`).
   - Cross-join with years to get all (`neighbor_cell_id`, `year`) pairs.
   - Join against the main data to resolve row indices in bulk.
   - The result is a single `data.table` of edges: `(row_i, neighbor_row_j)`.

2. **Replace `compute_neighbor_stats()`** with a single vectorized `data.table` grouped aggregation per variable. Instead of looping over 6.46M elements, we join the edge table to the variable values and compute `max`, `min`, `mean` grouped by `row_i`.

3. **Leave the Random Forest predict() call untouched** â€” it is already efficient.

This reduces the runtime from 86+ hours to likely **minutes**.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Assume these objects already exist in the environment:
#       cell_data              â€” data.frame / data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, â€¦
#       id_order               â€” integer/character vector of unique cell IDs (same order as rook_neighbors_unique)
#       rook_neighbors_unique  â€” nb object (list of integer index vectors into id_order)
#       rf_model               â€” the pre-trained Random Forest model
# ---------------------------------------------------------------

# Convert to data.table if not already
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)

# Preserve original row order
cell_data[, .row_id := .I]

# ---------------------------------------------------------------
# 1.  Build a vectorized edge list from the nb object
#     Each element rook_neighbors_unique[[k]] contains integer indices
#     into id_order that are neighbors of id_order[k].
# ---------------------------------------------------------------

# Expand nb list into an edge data.table: (focal_cell_id, neighbor_cell_id)
edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(k) {
  nb_idx <- rook_neighbors_unique[[k]]
  if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_id = id_order[k], neighbor_id = id_order[nb_idx])
}))

cat(sprintf("Edge list built: %d directed neighbor pairs\n", nrow(edge_list)))

# ---------------------------------------------------------------
# 2.  Build a row-index lookup:  (id, year) -> .row_id
# ---------------------------------------------------------------

row_lookup <- cell_data[, .(id, year, .row_id)]
setkey(row_lookup, id, year)

# ---------------------------------------------------------------
# 3.  Build the full neighbor-row mapping
#     For every row i in cell_data, find all rows j that are
#     spatial neighbors in the same year.
# ---------------------------------------------------------------

# Get (focal_id, year, focal_row_id) for every row
focal <- cell_data[, .(focal_id = id, year, focal_row = .row_id)]

# Join focal rows to edge list to get (focal_row, neighbor_id, year)
# This is the cross of "each row's year" Ã— "each row's neighbor cells"
setkey(edge_list, focal_id)
setkey(focal, focal_id)

neighbor_map <- edge_list[focal, on = "focal_id", allow.cartesian = TRUE,
                          nomatch = NULL,
                          .(focal_row, neighbor_id, year)]

# Now resolve neighbor_id + year -> neighbor_row
setkey(neighbor_map, neighbor_id, year)
neighbor_map <- row_lookup[neighbor_map, on = c(id = "neighbor_id", "year"),
                           nomatch = NULL,
                           .(focal_row, neighbor_row = .row_id)]

cat(sprintf("Neighbor-row map built: %d (focal_row, neighbor_row) pairs\n",
            nrow(neighbor_map)))

# ---------------------------------------------------------------
# 4.  Compute neighbor stats for each source variable (vectorized)
# ---------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  cat(sprintf("Computing neighbor features for: %s\n", var_name))

  # Attach the neighbor's value to each edge
  neighbor_map[, nbr_val := cell_data[[var_name]][neighbor_row]]

  # Aggregate per focal row â€” remove NAs within groups
  agg <- neighbor_map[!is.na(nbr_val),
                      .(nb_max  = max(nbr_val),
                        nb_min  = min(nbr_val),
                        nb_mean = mean(nbr_val)),
                      by = focal_row]

  # Initialize columns with NA
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")

  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  # Fill in computed values by row index
  cell_data[agg$focal_row, (max_col)  := agg$nb_max]
  cell_data[agg$focal_row, (min_col)  := agg$nb_min]
  cell_data[agg$focal_row, (mean_col) := agg$nb_mean]

  # Clean up temporary column
  neighbor_map[, nbr_val := NULL]
}

# ---------------------------------------------------------------
# 5.  Random Forest prediction (unchanged â€” already efficient)
# ---------------------------------------------------------------

# Prepare the predictor matrix (exclude non-predictor columns as needed)
# This assumes the model expects the same column names present in cell_data.
predictions <- predict(rf_model, newdata = cell_data)

cell_data[, prediction := predictions]

# Clean up helper column
cell_data[, .row_id := NULL]

cat("Pipeline complete.\n")
```

---

## Summary

| Component | Original Cost | Optimized Cost |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level iterations with string ops & named-vector lookups | One vectorized `data.table` join (seconds) |
| `compute_neighbor_stats` (Ã—5 vars) | ~32.3M R-level `lapply` iterations | 5 grouped `data.table` aggregations (seconds each) |
| `predict()` (Random Forest) | Single vectorized C call | **Unchanged** â€” not the bottleneck |

The true bottleneck is the **interpreted R-level loop-based neighbor feature engineering**, not Random Forest inference. The optimized code replaces all row-level loops with bulk `data.table` joins and grouped aggregations, reducing the estimated runtime from **86+ hours to minutes** while preserving the trained model and the original numerical estimand exactly.
END_RESPONSE>>>
