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
run_id: Anthropic_F11_T03_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** as the dominant bottleneck â€” specifically `build_neighbor_lookup` and `compute_neighbor_stats`.

Here's why:

1. **`build_neighbor_lookup`** uses a base-R `lapply` over **~6.46 million rows**, and inside each iteration it performs `as.character()` coercion, `paste()` string construction, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) hash-table probing per call, and doing this 6.46 million times with string key construction each time is extremely expensive. This single function likely accounts for the vast majority of the 86+ hour runtime.

2. **`compute_neighbor_stats`** then iterates over the 6.46 million-element lookup list again, subsetting a numeric vector and computing `max`, `min`, `mean` per element. This is called **5 times** (once per neighbor source variable), yielding ~32.3 million iterations total.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46 million rows and 110 predictors, a single RF predict call on a standard laptop typically completes in seconds to a few minutes â€” orders of magnitude less than the neighbor computation.

**The bottleneck is the row-level R-loop string-key spatial join in `build_neighbor_lookup` and the repeated row-level iteration in `compute_neighbor_stats`.**

---

## Optimization Strategy

1. **Replace string-key lookups with integer-indexed joins using `data.table`.** Instead of building string keys and doing named-vector lookups 6.46M times, we create an integer-keyed `data.table` and perform a single vectorized merge/join to resolve all neighbor row indices at once.

2. **Vectorize `compute_neighbor_stats`** by exploding the neighbor relationships into a long-form edge table, joining the variable values, and computing grouped aggregations (`max`, `min`, `mean`) in a single `data.table` operation â€” eliminating the per-row `lapply` entirely.

3. **Process all 5 neighbor source variables in one grouped aggregation pass** over the edge table, rather than looping 5 separate times.

This reduces the complexity from ~6.46M Ã— k R-level iterations to a handful of vectorized `data.table` joins and group-by operations, bringing estimated runtime from 86+ hours down to **minutes**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# INPUTS (assumed to already exist in the environment):
#   cell_data              : data.frame with columns id, year, ntl, ec,
#                            pop_density, def, usd_est_n2, ... (~6.46M rows)
#   id_order               : integer/numeric vector of unique cell IDs
#                            (length 344,208), index-aligned with
#                            rook_neighbors_unique
#   rook_neighbors_unique  : spdep nb object (list of length 344,208);
#                            rook_neighbors_unique[[i]] gives integer
#                            indices into id_order for neighbors of
#                            id_order[i]
#   rf_model               : pre-trained Random Forest model (untouched)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# === STEP 1: Build a long-form edge table of directed neighbor pairs ===
#
# Each element rook_neighbors_unique[[i]] contains the *positional indices*
# (into id_order) of the neighbors of cell id_order[i].
# We convert this to a data.table of (focal_id, neighbor_id) pairs.

edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  if (length(nb_idx) == 0L) return(NULL)
  data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
}))
# edge_list has ~1,373,394 rows (directed relationships)

# === STEP 2: Convert cell_data to data.table and add a row index ===

dt <- as.data.table(cell_data)
dt[, row_idx := .I]  # preserve original row order for later reassembly

# === STEP 3: Cross the edge list with years to get cell-year neighbor pairs ===
#
# For every (focal_id, neighbor_id) pair and every year present in the data,
# we need the neighbor's variable values in that same year.
# Instead of a full cross join (which would be huge), we join through the data.

# 3a. Create a keyed lookup: for each (id, year) â†’ row_idx + variable values
#     We only need the columns we'll aggregate.
cols_needed <- c("id", "year", "row_idx", neighbor_source_vars)
dt_key <- dt[, ..cols_needed]
setkey(dt_key, id, year)

# 3b. For each focal row, identify its neighbors via the edge_list,
#     then look up the neighbor's values in the same year.
#     We do this with two joins:

#     First, attach the focal cell's year (and row_idx) to each edge.
focal_info <- dt[, .(focal_row_idx = row_idx, focal_id = id, year)]

# Join edge_list to focal_info to get (focal_row_idx, neighbor_id, year)
edges_with_year <- edge_list[focal_info,
                             on = .(focal_id),
                             allow.cartesian = TRUE,
                             nomatch = NULL]
# edges_with_year columns: focal_id, neighbor_id, focal_row_idx, year
# This table has ~(avg_neighbors * 6.46M) rows. With ~4 rook neighbors on
# average: ~25.8M rows. Fits comfortably in 16 GB.

# 3c. Join neighbor variable values by (neighbor_id, year)
setnames(dt_key, "id", "neighbor_id")
setnames(dt_key, "row_idx", "neighbor_row_idx")
setkey(dt_key, neighbor_id, year)
setkey(edges_with_year, neighbor_id, year)

edges_full <- dt_key[edges_with_year, on = .(neighbor_id, year), nomatch = NA]
# edges_full now has the neighbor's variable values for each focal-row / neighbor pair.

# === STEP 4: Grouped aggregation â€” compute max, min, mean per focal row ===

agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

# Build the aggregation call dynamically
agg_call <- as.call(c(as.name("list"),
                       setNames(agg_exprs, agg_names)))

neighbor_stats <- edges_full[,
                             eval(agg_call),
                             by = .(focal_row_idx)]

# Replace -Inf/Inf (from max/min on all-NA groups) with NA
for (col in agg_names) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

# === STEP 5: Merge aggregated neighbor features back to the main table ===

setkey(neighbor_stats, focal_row_idx)
setkey(dt, row_idx)

# Remove any pre-existing neighbor columns to avoid conflicts
existing_neighbor_cols <- intersect(names(dt), agg_names)
if (length(existing_neighbor_cols) > 0) {
  dt[, (existing_neighbor_cols) := NULL]
}

dt <- neighbor_stats[dt, on = .(focal_row_idx = row_idx)]

# Clean up helper columns
dt[, focal_row_idx := NULL]
dt[, row_idx := NULL]

# === STEP 6: Random Forest prediction (unchanged â€” not the bottleneck) ===

predictions <- predict(rf_model, newdata = dt)
dt[, prediction := predictions]

# === STEP 7: Convert back to data.frame if downstream code expects it ===

cell_data <- as.data.frame(dt)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-loop iterations with string ops | Eliminated; replaced by vectorized `data.table` join |
| `compute_neighbor_stats` | 6.46M Ã— 5 = 32.3M R-loop iterations | Single grouped `data.table` aggregation |
| Estimated runtime | 86+ hours | ~2â€“10 minutes |
| RF model | Preserved âœ“ | Preserved âœ“ |
| Numerical estimand | Identical max/min/mean per row | Identical âœ“ |

The trained Random Forest model is never retrained or modified â€” only `predict()` is called, exactly as before. The numerical outputs (neighbor max, min, mean and final predictions) are identical to the original pipeline.
END_RESPONSE>>>
