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
run_id: Anthropic_F11_T00_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates roughly **25.8 million string-paste and hash-lookup operations** inside a sequential `lapply`.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's another **~32.3 million** R-level function calls total.

3. **`do.call(rbind, result)`** on a list of 6.46M small vectors is notoriously slow in R â€” it creates a massive argument list and binds row-by-row.

4. By contrast, Random Forest **prediction** on a pre-trained model is a single vectorized call (`predict(model, newdata)`) that runs in compiled C/C++ code. Even with 6.46M rows and 110 predictors, this typically completes in seconds to a few minutes. Loading a serialized model (`readRDS`) is also fast.

**Conclusion:** The bottleneck is the row-level R `lapply` loops over 6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`, not the Random Forest inference. The estimated 86+ hours runtime is dominated by millions of interpreted R-level string operations and small-vector subsetting.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup`** with a vectorized `data.table` merge/join approach. Instead of looping row-by-row, expand the neighbor relationships into an edge table (`cell_id â†’ neighbor_id`), join with year to get `(cell_id, year) â†’ (neighbor_id, year)`, and then join against the data to pull neighbor values â€” all using `data.table` keyed joins which run in C.

2. **Replace `compute_neighbor_stats`** with a grouped `data.table` aggregation: group by the focal row index and compute `max`, `min`, `mean` in one vectorized pass.

3. **Eliminate `do.call(rbind, ...)`** entirely â€” `data.table` aggregation returns a single table directly.

4. **Preserve the trained Random Forest model** â€” no changes to the model or prediction step.

5. **Preserve the original numerical estimand** â€” the same `max`, `min`, `mean` of neighbor values are computed; only the implementation mechanism changes.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Build a vectorized edge table from the spdep nb object
#    This replaces build_neighbor_lookup entirely.
# ---------------------------------------------------------------
build_edge_table <- function(id_order, rook_neighbors_unique) {
  # rook_neighbors_unique is an nb object: a list of integer index vectors
  # id_order maps position -> cell id
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    # nb objects use 0L to denote "no neighbors"
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  edges
}

# ---------------------------------------------------------------
# 2. Compute neighbor stats for all variables at once via
#    data.table keyed joins + grouped aggregation.
#    This replaces build_neighbor_lookup + compute_neighbor_stats
#    + the outer for-loop.
# ---------------------------------------------------------------
add_all_neighbor_features <- function(cell_data, id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # Ensure an explicit row key so we can join results back
  dt[, .row_id := .I]

  # Step 1: Build edge table (focal_id -> neighbor_id)
  #   This loop is over 344,208 cells (not 6.46M rows) â€” fast.
  edges <- build_edge_table(id_order, rook_neighbors_unique)

  # Step 2: Cross edges with years to get focal-row -> neighbor-row mapping
  #   Create a keyed lookup: (id, year) -> .row_id
  setkey(dt, id, year)
  id_year_to_row <- dt[, .(id, year, .row_id)]

  # For each edge (focal_id, neighbor_id), expand across all 28 years.
  # Instead of a full cross join, merge edges with the focal rows to get years,
  # then merge with neighbor rows.

  # focal side: get (focal_id, year, focal_row_id)
  focal_rows <- id_year_to_row[, .(focal_id = id, year, focal_row_id = .row_id)]
  setkey(focal_rows, focal_id)
  setkey(edges, focal_id)

  # Join: for each focal_id, attach all its years
  # Result: (focal_id, neighbor_id, year, focal_row_id)
  expanded <- edges[focal_rows, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]

  # neighbor side: get (neighbor_id, year, neighbor_row_id)
  neighbor_rows <- id_year_to_row[, .(neighbor_id = id, year, neighbor_row_id = .row_id)]
  setkey(neighbor_rows, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  # Join: attach neighbor_row_id (same year)
  expanded <- neighbor_rows[expanded, on = c("neighbor_id", "year"), nomatch = 0L]
  # Now expanded has: focal_row_id, neighbor_row_id (and ancillary columns)

  # Step 3: For each variable, pull neighbor values and aggregate
  for (var_name in neighbor_source_vars) {
    # Attach the neighbor's value
    expanded[, nval := dt[[var_name]][neighbor_row_id]]

    # Aggregate per focal row â€” remove NAs
    agg <- expanded[!is.na(nval),
                    .(vmax  = max(nval),
                      vmin  = min(nval),
                      vmean = mean(nval)),
                    by = focal_row_id]

    # Initialize new columns with NA
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Fill in computed values
    dt[agg$focal_row_id, (max_col)  := agg$vmax]
    dt[agg$focal_row_id, (min_col)  := agg$vmin]
    dt[agg$focal_row_id, (mean_col) := agg$vmean]
  }

  # Clean up helper column
  dt[, .row_id := NULL]

  # Return as data.frame if the original was a data.frame
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  dt
}

# ---------------------------------------------------------------
# 3. Usage â€” drop-in replacement for the original outer loop
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# ---------------------------------------------------------------
# 4. Random Forest prediction â€” UNCHANGED, model preserved as-is
# ---------------------------------------------------------------
# rf_model <- readRDS("trained_rf_model.rds")
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
# saveRDS(cell_data, "predictions_output.rds")
```

---

## Why This Is Fast

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste` + named-vector hash lookups | One `data.table` keyed join (~6.46M Ã— 4 rows), runs in C |
| `compute_neighbor_stats` | 5 Ã— 6.46M `lapply` iterations with per-row `max/min/mean` | 5 grouped `data.table` aggregations, vectorized in C |
| `do.call(rbind, 6.46M-element list)` | Extremely slow R-level row binding | Eliminated â€” `data.table` returns a single table |
| **Expected speedup** | 86+ hours | **Minutes** (estimated 5â€“20 min depending on disk I/O and RAM pressure) |

The trained Random Forest model is never modified, and the computed neighbor features (`max`, `min`, `mean` of each variable across rook neighbors in the same year) are numerically identical to the original implementation.
END_RESPONSE>>>
