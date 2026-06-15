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
run_id: Anthropic_F11_T03_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates roughly **25.8 million string-paste and hash-lookup operations** inside a sequential `lapply`.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. That's **~32.3 million R-level function invocations** across the 5 variables.

3. By contrast, Random Forest **prediction** (`predict()`) on a pre-trained model is a single vectorized C-level call over the 6.46M rows. Even with 110 predictors, this typically completes in seconds to a few minutes. Loading the model from disk is a single `readRDS()`. Writing predictions is a single `write.csv()` or equivalent. None of these are row-level R loops.

**Conclusion:** The bottleneck is the row-level R `lapply` loops over millions of rows doing string operations and named-vector lookups. The estimated 86+ hours runtime is dominated by this feature engineering, not by RF inference.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()` entirely** with a vectorized `data.table` merge/join approach. Instead of building a per-row list of neighbor indices via string pasting and named-vector lookups, we construct a neighbor edge table and join it against the data to resolve row indices in bulk.

2. **Replace `compute_neighbor_stats()` with a grouped `data.table` aggregation.** Once we have an edge table mapping each row to its neighbor rows, we can extract neighbor values and compute `max`, `min`, `mean` with a single grouped aggregation â€” no R-level `lapply` over millions of rows.

3. **Preserve the trained RF model and the original numerical estimand.** We only change how features are computed, not what they compute. The resulting columns are numerically identical.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Convert cell_data to data.table and add a row index
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)
cell_data[, .row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Build the neighbor edge table (vectorized, replaces build_neighbor_lookup)
#
#    rook_neighbors_unique is an nb object: a list of length = number of
#    unique spatial cells (344,208). Element i contains integer indices
#    into id_order of cell i's rook neighbors.
#    id_order is the vector of cell IDs in the same order as the nb object.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# 2a. Expand the nb list into a two-column edge table of cell IDs
#     (focal_id, neighbor_id). This is done once.
n_cells <- length(rook_neighbors_unique)
focal_ref <- rep(seq_len(n_cells),
                 times = lengths(rook_neighbors_unique))
neighbor_ref <- unlist(rook_neighbors_unique, use.names = FALSE)

# Remove the spdep "no-neighbor" sentinel (0)
valid <- neighbor_ref != 0L
focal_ref    <- focal_ref[valid]
neighbor_ref <- neighbor_ref[valid]

edges <- data.table(
  focal_id    = id_order[focal_ref],
  neighbor_id = id_order[neighbor_ref]
)

# 2b. Create a lookup from (id, year) -> row index in cell_data
id_year_idx <- cell_data[, .(id, year, .row_idx)]

# 2c. For every (focal_id, year) pair, find the row index of the focal row
#     and for every (neighbor_id, year) pair, find the row index of the
#     neighbor row. We do this via keyed joins.
setkey(id_year_idx, id, year)

# Expand edges across all years: cross join edges Ã— years
years <- sort(unique(cell_data$year))
edge_year <- CJ(edge_idx = seq_len(nrow(edges)), year = years)
edge_year[, `:=`(focal_id    = edges$focal_id[edge_idx],
                 neighbor_id = edges$neighbor_id[edge_idx])]
edge_year[, edge_idx := NULL]

# Join to get focal row index
edge_year[id_year_idx, focal_row := i..row_idx,
          on = .(focal_id = id, year = year)]

# Join to get neighbor row index
edge_year[id_year_idx, neighbor_row := i..row_idx,
          on = .(neighbor_id = id, year = year)]

# Drop edges where either side has no matching row (boundary / missing years)
edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row)]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Compute neighbor stats for all 5 variables (vectorized,
#    replaces compute_neighbor_stats + the outer for-loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  # Attach the neighbor's value for this variable to every edge
  edge_year[, nval := cell_data[[var_name]][neighbor_row]]

  # Grouped aggregation: max, min, mean per focal row (excluding NAs)
  stats <- edge_year[!is.na(nval),
                     .(nmax  = max(nval),
                       nmin  = min(nval),
                       nmean = mean(nval)),
                     keyby = .(focal_row)]

  # Prepare NA-filled columns, then fill from stats
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  cell_data[stats$focal_row, (max_col)  := stats$nmax]
  cell_data[stats$focal_row, (min_col)  := stats$nmin]
  cell_data[stats$focal_row, (mean_col) := stats$nmean]

  # Clean up the temporary column
  edge_year[, nval := NULL]
}

# Drop the helper column
cell_data[, .row_idx := NULL]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Random Forest prediction (unchanged â€” this was never the bottleneck)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
library(randomForest)  # or ranger, etc.

rf_model    <- readRDS("trained_rf_model.rds")
predictions <- predict(rf_model, newdata = cell_data)
cell_data[, prediction := predictions]

# Write output
fwrite(cell_data, "predictions_output.csv")
```

---

## Why This Is Fast

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M `lapply` iterations with `paste` + named-vector hash lookups | One vectorized `data.table` cross-join + two keyed joins |
| Neighbor stats | 5 Ã— 6.46M `lapply` iterations with per-row `max`/`min`/`mean` | 5 grouped `data.table` aggregations (C-level `GForce`) |
| Estimated time | 86+ hours | **Minutes** (typically 5â€“15 min on a 16 GB laptop) |
| RF inference | Already fast (single vectorized C call) | Unchanged |
| Numerical results | Original values | **Identical** â€” same max, min, mean over same neighbor sets |

### Memory Note

The `edge_year` table has ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows with a few integer/numeric columns â€” roughly 1â€“2 GB, well within the 16 GB RAM budget. If memory is tight, the year loop can be chunked, but this is typically unnecessary.
END_RESPONSE>>>
