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
run_id: Anthropic_F12_T03_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized concatenation step per variable â€” called only 5 times total. That is not the 86-hour bottleneck.

**The true bottleneck is `build_neighbor_lookup()`.**

Specifically, inside its `lapply` over **all 6.46 million rows**, every iteration performs:

1. **`as.character()` conversion and named-vector lookup** (`id_to_ref[as.character(data$id[i])]`) â€” character coercion and hash lookup per row.
2. **`paste()` to build neighbor keys** â€” string concatenation for every neighbor of every row.
3. **Named-vector lookup on `idx_lookup`** â€” a named character vector with 6.46 million entries is searched per-neighbor-key via partial string matching/hashing.

This means roughly **6.46M Ã— ~4 neighbors Ã— (paste + hash lookup)** = ~25.8 million `paste` calls and named-vector lookups, all inside a sequential `lapply` with per-element R overhead. String operations in R are notoriously slow and not vectorizable in this row-by-row pattern. This single function dominates the runtime by orders of magnitude.

`compute_neighbor_stats()` is comparatively lightweight: it does integer indexing into a numeric vector (very fast) and computes `max/min/mean` on small neighbor sets.

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()` entirely.** Eliminate the row-level `lapply`. Instead, expand the neighbor graph (which is defined over cells) across all 28 years using vectorized operations â€” a merge/join rather than per-row string pasting and lookup.

2. **Use `data.table` for fast keyed joins** instead of named-vector lookups with `paste`-constructed keys.

3. **Vectorize `compute_neighbor_stats()`** using `data.table` grouped aggregation over the expanded neighbor-edge table, replacing the per-row `lapply` + `do.call(rbind, ...)`.

4. **Preserve the trained Random Forest model** â€” we only change feature engineering, producing numerically identical columns.

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Vectorized neighbor lookup + stats in one pipeline
# ---------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, id_order, neighbors, neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # Ensure a row index for final reassembly
  dt[, .row_id := .I]

  # --- Step A: Build a cell-level edge list from the nb object ---
  # neighbors is a list of integer index vectors (spdep::nb), indexed by position in id_order
  edge_list <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb <- neighbors[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(data.table(from_id = integer(0), to_id = integer(0)))
    }
    data.table(from_id = id_order[i], to_id = id_order[nb])
  }))

  # --- Step B: Create a keyed lookup: (id, year) -> row_id ---
  setkey(dt, id, year)

  # --- Step C: Expand edge list across all years ---
  # Get unique years
  years <- unique(dt$year)

  # Cross join edges Ã— years: each spatial edge exists in every year
  edge_year <- CJ_dt(edge_list, years)

  # Helper: cross join edge_list with years vector
  # We do this efficiently:
  edge_year <- edge_list[, .(from_id, to_id)][
    , .(year = years), by = .(from_id, to_id)
  ]

  # --- Step D: Attach row indices for "from" and "to" ---
  # Map (from_id, year) -> .row_id  (the focal row)
  id_year_to_row <- dt[, .(id, year, .row_id)]
  setkey(id_year_to_row, id, year)

  # Focal row index
  edge_year <- merge(edge_year, id_year_to_row,
                     by.x = c("from_id", "year"),
                     by.y = c("id", "year"),
                     all.x = TRUE, sort = FALSE)
  setnames(edge_year, ".row_id", "focal_row")

  # Neighbor row index
  edge_year <- merge(edge_year, id_year_to_row,
                     by.x = c("to_id", "year"),
                     by.y = c("id", "year"),
                     all.x = TRUE, sort = FALSE)
  setnames(edge_year, ".row_id", "neighbor_row")

  # Drop edges where either side is missing
  edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row)]

  # --- Step E: For each variable, compute grouped stats vectorized ---
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)

    # Attach neighbor values via integer indexing (very fast)
    edge_year[, nval := dt[[var_name]][neighbor_row]]

    # Grouped aggregation â€” single pass over the edge table
    stats <- edge_year[!is.na(nval),
                       .(nb_max  = max(nval),
                         nb_min  = min(nval),
                         nb_mean = mean(nval)),
                       by = focal_row]

    # Initialize columns to NA
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign by row index
    dt[stats$focal_row, (max_col)  := stats$nb_max]
    dt[stats$focal_row, (min_col)  := stats$nb_min]
    dt[stats$focal_row, (mean_col) := stats$nb_mean]

    # Clean up temp column
    edge_year[, nval := NULL]
  }

  # Drop helper column and return as data.frame to preserve downstream compatibility
  dt[, .row_id := NULL]
  return(as.data.frame(dt))
}

# ---------------------------------------------------------------
# 2. Usage (drop-in replacement for the original outer loop)
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Is Fast

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M `lapply` iterations with `paste` + named-vector hash | One vectorized `merge` on keyed `data.table` |
| String operations | ~25.8M `paste()` calls | Zero â€” joins on integer/numeric keys |
| Neighbor stats | 6.46M `lapply` iterations Ã— 5 vars | 5 grouped `data.table` aggregations |
| `do.call(rbind, ...)` | 6.46M 3-element vectors bound per var | Eliminated â€” direct column assignment |
| Estimated time | 86+ hours | **Minutes** (dominated by the edge Ã— year expansion, ~38M rows, handled in-memory by `data.table`) |

## Memory Note

The expanded edge table is approximately `1,373,394 directed edges Ã— 28 years â‰ˆ 38.5M rows Ã— 4 integer columns â‰ˆ 0.6 GB`, well within the 16 GB laptop constraint.

## Summary

**Reject the colleague's diagnosis.** The true bottleneck is `build_neighbor_lookup()`, which performs millions of per-row string concatenations and named-vector hash lookups in an R-level loop. The fix is to replace the entire row-level `lapply` pipeline with vectorized `data.table` keyed joins and grouped aggregations, reducing runtime from 86+ hours to minutes while producing numerically identical output columns.
END_RESPONSE>>>
