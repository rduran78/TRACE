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
run_id: Anthropic_F11_T10_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **rejected**. The code evidence clearly shows the bottleneck is in the **spatial neighbor feature engineering**, specifically `build_neighbor_lookup` and `compute_neighbor_stats`.

**Evidence:**

1. **`build_neighbor_lookup`** uses a row-level `lapply` over **~6.46 million rows**. Inside every iteration it performs `as.character()` coercion, character-based hash lookups (`id_to_ref`, `idx_lookup`), `paste()` key construction, and subsetting â€” all in a sequential, interpreted R loop. This alone produces billions of elementary R operations.

2. **`compute_neighbor_stats`** is called **5 times** (once per neighbor source variable), each time iterating over **~6.46 million rows** with an inner `lapply` that subsets, removes NAs, and computes `max/min/mean`. That is ~32.3 million serial R-level loop iterations just for the stats.

3. By contrast, Random Forest inference (`predict()` on a pre-trained model) is a single vectorized C/C++ call over the data matrix. Even with 110 predictors and 6.46M rows, this typically completes in minutes on a laptop â€” orders of magnitude faster than the feature engineering above.

4. The estimated 86+ hour runtime is consistent with billions of interpreted R operations in nested `lapply` loops, not with a single `predict()` call.

**Root cause:** Row-level interpreted loops over millions of rows with repeated string operations and named-vector lookups.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup`** with a vectorized `data.table` equi-join approach: explode the neighbor list into an edge table once, join against the data keyed by `(id, year)`, and group.
2. **Replace `compute_neighbor_stats`** with a single grouped `data.table` aggregation per variable â€” no R-level row loop at all.
3. **Avoid all `paste()`-based key lookups and per-row `lapply`.**

Expected speedup: from 86+ hours to roughly **minutes**, because all work is pushed into `data.table`'s optimized C backend.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build a directed edge table from the nb object (done ONCE)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # neighbors is a list of integer index vectors (spdep nb object)
  # id_order maps position -> cell id
  from_lengths <- lengths(neighbors)
  from_idx     <- rep(seq_along(neighbors), from_lengths)
  to_idx       <- unlist(neighbors, use.names = FALSE)

  data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Compute all neighbor features via data.table joins + grouped agg
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_all_neighbor_features <- function(cell_data, edge_dt,
                                          neighbor_source_vars) {
  # Convert to data.table if needed (by reference is fine)
  dt <- as.data.table(cell_data)

  # We will join on (neighbor_id == id, year == year).
  # Step A: create a slim table of just id, year, and the source vars.
  keep_cols <- c("id", "year", neighbor_source_vars)
  neighbor_vals <- dt[, ..keep_cols]

  # Step B: cross edge_dt with years via a keyed join.
  #   For every (focal_id, neighbor_id) pair and every year in the data,
  #   look up the neighbor's value.
  #
  #   Efficient approach: join edge_dt to neighbor_vals on
  #   neighbor_id == id, broadcasting across years.

  setnames(neighbor_vals, "id", "neighbor_id")
  setkey(neighbor_vals, neighbor_id)
  setkey(edge_dt, neighbor_id)

  # Merge: each edge gets all years of the neighbor

  merged <- edge_dt[neighbor_vals, on = "neighbor_id",
                    allow.cartesian = TRUE, nomatch = NULL]
  # merged now has columns: focal_id, neighbor_id, year, ntl, ec, ...

  # Step C: aggregate by (focal_id, year) for each variable
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    sym_v <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]]  <-
      bquote(max(.(sym_v), na.rm = TRUE))
    agg_exprs[[paste0("neighbor_min_", v)]]  <-
      bquote(min(.(sym_v), na.rm = TRUE))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      bquote(mean(.(sym_v), na.rm = TRUE))
  }

  # Build the aggregation call
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  agg_dt   <- merged[, eval(agg_call), by = .(focal_id, year)]

  # Replace infinite values (from max/min on all-NA) with NA
  for (col_name in names(agg_dt)[-(1:2)]) {
    set(agg_dt, which(is.infinite(agg_dt[[col_name]])), col_name, NA_real_)
  }

  # Step D: merge back onto the original data
  setnames(agg_dt, "focal_id", "id")
  setkey(dt, id, year)
  setkey(agg_dt, id, year)

  dt <- agg_dt[dt, on = .(id, year)]

  # Return as data.frame if the pipeline expects one

  as.data.frame(dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Run it
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, edge_dt, neighbor_source_vars
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Random Forest prediction (UNCHANGED â€” model is preserved)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# e.g.:
# predictions <- predict(rf_model, newdata = cell_data)
# cell_data$predicted <- predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M-iteration `lapply` with `paste()` + named-vector lookups | One vectorized `data.table` edge-table construction |
| `compute_neighbor_stats` | 5 Ã— 6.46M-iteration `lapply` (32.3M R-level loops) | Single grouped `data.table` aggregation (C backend) |
| Estimated runtime | 86+ hours | Minutes |
| Random Forest model | Preserved (untouched) | Preserved (untouched) |
| Numerical estimand | Identical `max`, `min`, `mean` per (focal cell, year) | Identical â€” same arithmetic, same NA handling |

The true bottleneck is the interpreted, row-level spatial feature engineering â€” not Random Forest inference.
END_RESPONSE>>>
