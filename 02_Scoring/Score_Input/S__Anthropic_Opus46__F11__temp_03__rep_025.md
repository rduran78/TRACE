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
run_id: Anthropic_F11_T03_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that **Random Forest inference** is the bottleneck is **incorrect**. The code evidence points squarely to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

**Evidence from the code:**

1. **`build_neighbor_lookup()`** uses `lapply` over **~6.46 million rows**, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. With ~1.37 million neighbor relationships spread across 6.46M cell-years, this single function produces millions of string allocations and named-vector searches (which are hash lookups but still expensive at this scale in R).

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's **~32.3 million R-level loop iterations** just for the stats, each with allocation overhead from the anonymous function closure.

3. **`do.call(rbind, result)`** on a list of 6.46 million 3-element vectors is itself a well-known R performance anti-pattern â€” it creates a massive intermediate list and binds row-by-row.

4. By contrast, Random Forest `predict()` on a pre-trained model against 6.46M rows with ~110 predictors is a single vectorized C/C++ call (in `ranger` or `randomForest`). It is computationally non-trivial but is orders of magnitude faster than tens of millions of interpreted R loop iterations with per-element string operations.

**Conclusion:** The bottleneck is the **row-level R `lapply` loops with string-key lookups and per-row aggregation** in the neighbor feature engineering, not the RF inference. The estimated 86+ hour runtime is dominated by this interpreted-R overhead.

---

## Optimization Strategy

1. **Replace the string-keyed lookup with integer-indexed lookup.** Instead of pasting `id_year` strings and doing named-vector lookups, map `(id, year)` pairs to integer row indices using a pre-built integer matrix or `data.table` keyed join.

2. **Vectorize neighbor stats computation.** Expand the neighbor relationships into a long-form `data.table` of `(row_index, neighbor_row_index)`, join the variable values, and compute grouped `max`, `min`, `mean` in a single `data.table` aggregation â€” no R-level loop at all.

3. **Process all 5 variables simultaneously** in one grouped aggregation pass rather than 5 separate `lapply` calls over 6.46M rows.

This reduces the runtime from 86+ hours to an estimated **minutes** (typically 2â€“10 minutes on a 16 GB laptop).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build an integer-indexed neighbor lookup using data.table
#         (replaces build_neighbor_lookup)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_neighbor_edges_dt <- function(cell_data_dt, id_order, rook_neighbors_unique) {

  # Map each id to its position in id_order (reference index)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Expand the nb object into a long edge list: (focal_ref, neighbor_ref)
  # Each element of rook_neighbors_unique is an integer vector of neighbor
  # positions within id_order.
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep::nb encodes "no neighbors" as 0L of length 1
    if (length(nb_i) == 1L && nb_i[1] == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_i])
  }))

  # edges is now a data.table with columns: focal_id, neighbor_id
  # representing the ~1.37 M directed rook-neighbor relationships (id-level).
  return(edges)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Vectorized neighbor stats for ALL variables at once
#         (replaces compute_neighbor_stats + the outer for-loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  # Convert to data.table (non-destructive copy)
  dt <- as.data.table(cell_data)

  # Assign a row index so we can map results back
  dt[, .row_idx := .I]

  # Build id-level edge list
  edges <- build_neighbor_edges_dt(dt, id_order, rook_neighbors_unique)

  # Cross edges with years: for every year, a focal cell's neighbors are the
  # same set of ids.  Instead of crossing explicitly (which would be huge),
  # we join on (id, year).

  # Create a keyed lookup: (id, year) -> row index
  # We need to join twice: once for focal, once for neighbor.

  id_year_key <- dt[, .(id, year, .row_idx)]

  # Join focal side: attach focal row index to every (focal_id, year) combination
  # But we don't want to enumerate all (focal_id, year) pairs from edges Ã— years.
  # Instead, work from the data rows directly.

  # For each row in dt, find its neighbors' rows in the same year.
  # Approach: merge dt (focal rows) with edges on focal_id, then look up
  # neighbor rows by (neighbor_id, year).

  # Focal side: every row is a focal observation
  focal <- dt[, .(focal_row = .row_idx, focal_id = id, year)]

  # Attach neighbor ids via the edge list
  setkey(edges, focal_id)
  setkey(focal, focal_id)
  # This join replicates each focal row by its number of neighbors

  focal_nb <- edges[focal, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # Columns: focal_id, neighbor_id, focal_row, year

  # Now look up the neighbor's row index for the same year
  setkey(id_year_key, id, year)
  setnames(id_year_key, ".row_idx", "neighbor_row")
  setnames(id_year_key, "id", "neighbor_id_key")

  focal_nb[, neighbor_id_chr := neighbor_id]
  # keyed join to get neighbor_row
  neighbor_rows <- id_year_key[
    focal_nb,
    on = .(neighbor_id_key = neighbor_id, year),
    nomatch = 0L
  ]
  # Columns now include: focal_row, neighbor_row, year, ...

  # For each neighbor source variable, pull the neighbor's value
  # and aggregate per focal_row.
  # We do this in one pass by subsetting the columns we need.

  # Pull neighbor values from dt
  val_cols <- neighbor_source_vars
  neighbor_vals <- dt[neighbor_rows$neighbor_row, ..val_cols]
  neighbor_vals[, focal_row := neighbor_rows$focal_row]

  # Aggregate: max, min, mean per focal_row for each variable
  agg_exprs <- unlist(lapply(val_cols, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(val_cols, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the aggregation call
  agg_list <- setNames(agg_exprs, agg_names)

  # data.table aggregation
  stats <- neighbor_vals[,
    lapply(agg_list, eval, envir = .SD),
    by = focal_row,
    .SDcols = val_cols
  ]

  # --- simpler, more robust aggregation approach ---
  # Compute per-variable stats explicitly to avoid eval complexity
  agg_parts <- list()
  for (v in val_cols) {
    part <- neighbor_vals[, .(
      nb_max  = max(get(v), na.rm = TRUE),
      nb_min  = min(get(v), na.rm = TRUE),
      nb_mean = mean(get(v), na.rm = TRUE)
    ), by = focal_row]
    setnames(part,
             c("nb_max", "nb_min", "nb_mean"),
             paste0("neighbor_", c("max_", "min_", "mean_"), v))
    agg_parts[[v]] <- part
  }

  # Merge all variable stats together on focal_row
  stats <- agg_parts[[1]]
  for (k in seq_along(agg_parts)[-1]) {
    stats <- merge(stats, agg_parts[[k]], by = "focal_row", all = TRUE)
  }

  # Join back to dt
  dt <- merge(dt, stats, by.x = ".row_idx", by.y = "focal_row", all.x = TRUE)

  # Replace Inf/-Inf (from max/min on empty sets after na.rm) with NA
  new_cols <- setdiff(names(stats), "focal_row")
  for (col in new_cols) {
    dt[is.infinite(get(col)), (col) := NA_real_]
  }

  # Remove helper column
  dt[, .row_idx := NULL]

  return(as.data.frame(dt))
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Drop-in replacement call
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Random Forest prediction (UNCHANGED â€” not the bottleneck)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The pre-trained RF model is loaded and used as-is.
# Example (preserving the original estimand):
#
#   library(ranger)  # or randomForest
#   rf_model <- readRDS("trained_rf_model.rds")
#   cell_data$predicted <- predict(rf_model, data = cell_data)$predictions
#   write.csv(cell_data, "final_predictions.csv", row.names = FALSE)
#
# This step is already efficient (vectorized C++ internals) and is NOT
# modified.
```

---

## Summary

| Aspect | Colleague's Claim (RF Inference) | Actual Bottleneck (Neighbor Features) |
|---|---|---|
| **Operation** | `predict()` â€” single vectorized C/C++ call | `lapply` over 6.46M rows Ã— 6 passes (1 build + 5 stats) |
| **Key cost driver** | Matrix multiplication in compiled code | String pasting, named-vector hash lookups, per-row anonymous functions, `do.call(rbind, ...)` on 6.46M-element list |
| **Estimated time** | Minutes | 86+ hours |
| **Fix** | None needed | Vectorize via `data.table` joins and grouped aggregation |

The optimized code replaces all row-level R loops with `data.table` keyed joins and grouped aggregations, reducing the estimated runtime from **86+ hours to single-digit minutes** while preserving the trained Random Forest model and the original numerical estimand (neighbor max, min, mean for each of the 5 source variables).
END_RESPONSE>>>
