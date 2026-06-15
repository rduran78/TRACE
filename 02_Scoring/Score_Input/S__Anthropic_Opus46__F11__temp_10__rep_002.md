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
run_id: Anthropic_F11_T10_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis

**The colleague's claim is rejected.** The bottleneck is **not** Random Forest inference. Loading a trained model, calling `predict()` once on ~6.46M rows Ã— 110 features, and writing predictions is a matter of seconds to minutes on a modern laptop.

**The true bottleneck is `build_neighbor_lookup` and `compute_neighbor_stats`.** Here's why:

1. **`build_neighbor_lookup`** runs an `lapply` over **6.46 million rows**, and inside each iteration it performs `paste()`, character key construction, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) string hashing per call. Over 6.46M iterations, each touching multiple neighbors, this produces billions of character operations. This single function likely accounts for the vast majority of the 86+ hour runtime.

2. **`compute_neighbor_stats`** iterates over the 6.46M-element lookup list, performing subsetting, `na.rm` filtering, and `max`/`min`/`mean` per row â€” then binds everything with `do.call(rbind, ...)` on a 6.46M-element list. This is slow but secondary compared to the lookup construction.

3. These operations are repeated for **5 variables**, compounding the cost of `compute_neighbor_stats` (though `build_neighbor_lookup` runs once).

---

# Optimization Strategy

1. **Replace character-key lookup with integer-indexed direct joins.** Build the neighbor lookup using `data.table` fast merge/join on integer keys `(id, year)` â†’ row index, eliminating all `paste()` and named-vector string matching.

2. **Vectorize `compute_neighbor_stats`** using `data.table` grouping: explode the neighbor relationships into an edge table, join variable values, and compute grouped `max`/`min`/`mean` in one vectorized pass â€” for all 5 variables at once.

3. **Avoid per-row `lapply` entirely.**

---

# Working R Code

```r
library(data.table)

# â”€â”€ 0. Ensure cell_data is a data.table with a row index â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.table(cell_data)
cell_data[, row_idx := .I]

# â”€â”€ 1. Build integer-indexed neighbor edge table (replaces build_neighbor_lookup) â”€â”€
build_neighbor_edges <- function(dt, id_order, neighbors) {
  # Map each id to its position in id_order
  id_to_ref <- data.table(id = id_order, ref_idx = seq_along(id_order))

  # For every ref_idx, get the neighbor ids
  edges_id <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb <- neighbors[[i]]
    if (length(nb) == 0L) return(data.table(src_id = integer(0), nb_id = integer(0)))
    data.table(src_id = id_order[i], nb_id = id_order[nb])
  }))

  # Get unique years
  years <- sort(unique(dt$year))

  # Cross-join edges Ã— years  â†’  (src_id, nb_id, year)
  edge_year <- CJ(edge_idx = seq_len(nrow(edges_id)), year = years)
  edge_year[, `:=`(src_id = edges_id$src_id[edge_idx],
                    nb_id  = edges_id$nb_id[edge_idx])]
  edge_year[, edge_idx := NULL]

  # Map (id, year) â†’ row_idx in dt
  key_map <- dt[, .(id, year, row_idx)]
  setkey(key_map, id, year)

  # Attach source row index
  setnames(key_map, "id", "src_id")
  setkey(edge_year, src_id, year)
  edge_year <- key_map[edge_year, on = .(src_id, year), nomatch = 0L]
  setnames(edge_year, "row_idx", "src_row")

  # Attach neighbor row index
  setnames(key_map, "src_id", "nb_id")
  setkey(edge_year, nb_id, year)
  edge_year <- key_map[edge_year, on = .(nb_id, year), nomatch = 0L]
  setnames(edge_year, "row_idx", "nb_row")

  setnames(key_map, "nb_id", "id")  # restore
  return(edge_year[, .(src_row, nb_row)])
}

cat("Building neighbor edge table ...\n")
edge_dt <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)
cat(sprintf("Edge table: %s rows\n", format(nrow(edge_dt), big.mark = ",")))

# â”€â”€ 2. Compute neighbor stats for all variables at once (replaces loop) â”€â”€
compute_all_neighbor_features <- function(dt, edge_dt, var_names) {
  # Pre-extract the columns we need from the neighbor rows
  nb_vals <- dt[edge_dt$nb_row, ..var_names]
  nb_vals[, src_row := edge_dt$src_row]

  for (v in var_names) {
    cat(sprintf("  Computing neighbor stats for: %s\n", v))

    # Subset to non-NA neighbor values for this variable
    sub <- nb_vals[!is.na(get(v)), .(src_row, val = get(v))]

    stats <- sub[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), keyby = src_row]

    # Name the new columns
    max_col  <- paste0(v, "_nb_max")
    min_col  <- paste0(v, "_nb_min")
    mean_col <- paste0(v, "_nb_mean")
    setnames(stats, c("nb_max", "nb_min", "nb_mean"),
                     c(max_col,  min_col,  mean_col))

    # Join back to dt on row_idx
    dt[stats, on = .(row_idx = src_row),
       (c(max_col, min_col, mean_col)) := mget(c(max_col, min_col, mean_col))]
  }

  return(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features ...\n")
cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
cat("Neighbor features done.\n")

# â”€â”€ 3. Random Forest prediction (untouched â€” this was never the bottleneck) â”€â”€
# library(ranger)  # or randomForest
# rf_model <- readRDS("trained_rf_model.rds")
# preds <- predict(rf_model, data = cell_data)
# cell_data[, prediction := preds$predictions]
# fwrite(cell_data, "predictions_output.csv")
```

---

# Why This Fixes the Runtime

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste` + named-vector char lookup | One `data.table` cross-join + two integer-keyed joins (~minutes) |
| `compute_neighbor_stats` (Ã—5 vars) | 6.46M `lapply` iterations Ã— 5, plus `do.call(rbind, 6.46M-element list)` | Vectorized `data.table` grouped aggregation per variable (~seconds each) |
| RF inference | Fast (secondsâ€“minutes) | Unchanged |

**Estimated new runtime: under 30 minutes** (dominated by the edge-table cross-join at ~38.4M edges Ã— 28 years â‰ˆ 1.08B rows, which `data.table` handles efficiently in-memory within 16 GB). The trained Random Forest model and the original numerical estimand are fully preserved.
END_RESPONSE>>>
