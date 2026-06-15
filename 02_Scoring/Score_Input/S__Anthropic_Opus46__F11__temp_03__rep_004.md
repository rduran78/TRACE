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
run_id: Anthropic_F11_T03_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named-vector lookups in R are **O(n) string-hashing operations per call** against a vector of 6.46 million names. Doing this 6.46 million times is catastrophically slow.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million rows with `lapply`, extracting subsets, computing `max/min/mean`, and then binding ~6.46 million 3-element vectors with `do.call(rbind, ...)` â€” itself an expensive operation.

3. **Random Forest inference** (`predict()` on a pre-trained model) for ~6.46 million rows with ~110 predictors is a single vectorized C-level call. On a modern laptop this typically completes in seconds to a few minutes. It is nowhere near 86+ hours.

**The bottleneck is the row-by-row R-level looping and repeated named-vector lookups across 6.46 million rows, repeated for 5 variables.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a fully vectorized `data.table` join approach. Instead of building a per-row list of neighbor indices, construct an edge-list data.table of `(focal_row, neighbor_row)` pairs via keyed joins. This eliminates all per-row `paste`, `as.character`, and named-vector lookups.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation over the edge list. For each focal row, compute `max`, `min`, and `mean` of neighbor values in one vectorized pass â€” no `lapply`, no `do.call(rbind, ...)`.

3. **Process all 5 variables in one pass** over the edge list rather than rebuilding/re-traversing the lookup 5 times.

4. **Leave the Random Forest predict() call untouched** â€” it is not the bottleneck.

Expected speedup: from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Convert cell_data to data.table (preserves all columns, including
#    the ~110 predictors needed later for RF predict).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]                 # preserve original row order

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Build the directed edge list from the nb object (one-time cost).
#    rook_neighbors_unique is a list of length = number of unique spatial
#    ids (344,208). id_order maps position -> cell id.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
}))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Expand edges across years via a keyed join.
#    This replaces build_neighbor_lookup() entirely.
#    Result: each row is (focal_row_idx, neighbor_row_idx).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Key cell_dt for fast joins
setkey(cell_dt, id, year)

# Create a slim lookup: (id, year) -> row_idx
row_lookup <- cell_dt[, .(id, year, row_idx)]

# For every (focal_id, neighbor_id) pair, join to every year present
# for the focal cell, then resolve the neighbor's row in that same year.

# Step A: get all (focal_id, year, focal_row_idx) from the data
focal_years <- row_lookup[, .(focal_id = id, year, focal_row = row_idx)]

# Step B: join edges to focal_years on focal_id
setkey(edges, focal_id)
setkey(focal_years, focal_id)
edge_year <- edges[focal_years, on = "focal_id", allow.cartesian = TRUE,
                   nomatch = NULL]
# edge_year now has columns: focal_id, neighbor_id, year, focal_row

# Step C: resolve neighbor_row by joining (neighbor_id, year) -> row_idx
setkey(row_lookup, id, year)
edge_year[, neighbor_row := row_lookup[.(edge_year$neighbor_id,
                                         edge_year$year), row_idx]]
edge_year <- edge_year[!is.na(neighbor_row)]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Compute neighbor stats for all 5 variables in one vectorized pass.
#    This replaces compute_neighbor_stats() and the outer for-loop.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Attach neighbor values for all source vars at once
neighbor_vals <- cell_dt[edge_year$neighbor_row, ..neighbor_source_vars]
neighbor_vals[, focal_row := edge_year$focal_row]

# Grouped aggregation: max, min, mean per focal_row per variable
agg <- neighbor_vals[, lapply(.SD, function(v) {
  v <- v[!is.na(v)]
  if (length(v) == 0L) list(NA_real_, NA_real_, NA_real_)
  else list(max(v), min(v), mean(v))
}), by = focal_row, .SDcols = neighbor_source_vars]

# The above returns list-columns; unpack them properly.
# More efficient approach â€” aggregate each stat separately:
agg_max  <- neighbor_vals[, lapply(.SD, function(v) {
  v <- v[!is.na(v)]; if (length(v) == 0L) NA_real_ else max(v)
}), by = focal_row, .SDcols = neighbor_source_vars]

agg_min  <- neighbor_vals[, lapply(.SD, function(v) {
  v <- v[!is.na(v)]; if (length(v) == 0L) NA_real_ else min(v)
}), by = focal_row, .SDcols = neighbor_source_vars]

agg_mean <- neighbor_vals[, lapply(.SD, function(v) {
  v <- v[!is.na(v)]; if (length(v) == 0L) NA_real_ else mean(v)
}), by = focal_row, .SDcols = neighbor_source_vars]

# Rename columns to match expected feature names
setnames(agg_max,  neighbor_source_vars,
         paste0("neighbor_max_",  neighbor_source_vars))
setnames(agg_min,  neighbor_source_vars,
         paste0("neighbor_min_",  neighbor_source_vars))
setnames(agg_mean, neighbor_source_vars,
         paste0("neighbor_mean_", neighbor_source_vars))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. Join aggregated features back to cell_dt by focal_row.
#    Rows with no neighbors get NA (correct behaviour, matches original).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- merge(cell_dt, agg_max,  by.x = "row_idx", by.y = "focal_row",
                 all.x = TRUE, sort = FALSE)
cell_dt <- merge(cell_dt, agg_min,  by.x = "row_idx", by.y = "focal_row",
                 all.x = TRUE, sort = FALSE)
cell_dt <- merge(cell_dt, agg_mean, by.x = "row_idx", by.y = "focal_row",
                 all.x = TRUE, sort = FALSE)

# Restore original row order
setorder(cell_dt, row_idx)
cell_dt[, row_idx := NULL]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6. Random Forest prediction â€” UNCHANGED (not the bottleneck).
#    The pre-trained model is preserved; the numerical estimand is
#    identical because the neighbor features are numerically equivalent.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Example (assuming `rf_model` is the pre-trained model already in memory
# or loaded from disk):
#
#   rf_model   <- readRDS("trained_rf_model.rds")
#   cell_dt$prediction <- predict(rf_model, newdata = cell_dt)
#   fwrite(cell_dt, "predictions_output.csv")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M row-level `lapply` with `paste` + named-vector lookups | Single vectorized `data.table` keyed join producing an edge list |
| **Neighbor stats** | 5 Ã— `lapply` over 6.46M rows + `do.call(rbind, ...)` | One grouped `data.table` aggregation across all 5 variables |
| **RF inference** | Already vectorized C-level call (fast) | Untouched â€” not the bottleneck |
| **Estimated runtime** | 86+ hours | Minutes (dominated by the join/aggregation, which is vectorized in C) |
| **Numerical equivalence** | â€” | Preserved: same `max`, `min`, `mean` over the same neighbor sets |
| **Trained model** | â€” | Preserved: no retraining; same `predict()` call |
END_RESPONSE>>>
