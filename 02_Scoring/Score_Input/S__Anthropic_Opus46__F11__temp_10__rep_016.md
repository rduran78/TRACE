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
run_id: Anthropic_F11_T10_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis

**The colleague's claim is rejected.** The bottleneck is **not** Random Forest inference. Loading a trained model, calling `predict()` once on ~6.46M rows Ã— 110 features, and writing predictions is a matter of seconds to a few minutes on a standard laptop.

**The true bottleneck is the spatial neighbor feature engineering pipeline**, specifically:

1. **`build_neighbor_lookup`**: An `lapply` over **6.46 million rows**, each iteration performing character coercion, `paste`, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookups in R are hash-table operations, but doing millions of them with string construction inside a row-level loop is extremely slow.

2. **`compute_neighbor_stats`**: Called **5 times** (once per source variable), each time running `lapply` over 6.46 million entries, subsetting, removing NAs, and computing `max/min/mean`. That's ~32.3 million individual list iterations total.

3. **The `do.call(rbind, result)` pattern** on a 6.46-million-element list of 3-element vectors is itself a known R performance anti-pattern.

Combined, these pure-R row-level loops over millions of rows easily account for the estimated 86+ hour runtime. The Random Forest step is negligible by comparison.

---

# Optimization Strategy

1. **Vectorize `build_neighbor_lookup`** using `data.table` keyed joins instead of per-row `paste`/named-vector lookups.
2. **Vectorize `compute_neighbor_stats`** by "exploding" the neighbor relationships into an edge table, joining the variable values, and computing grouped aggregations with `data.table` â€” zero R-level row loops.
3. Leave the trained Random Forest model and `predict()` call untouched.

Expected speedup: from 86+ hours to **minutes**.

---

# Working R Code

```r
library(data.table)

# ===========================================================
# 1. Build an edge table (vectorised neighbor lookup)
# ===========================================================
build_neighbor_edges <- function(data_dt, id_order, rook_neighbors) {


# --- map each spatial id to its rook neighbors (cell-level, year-agnostic) ---
  edge_list <- rbindlist(lapply(seq_along(id_order), function(i) {
    nb <- rook_neighbors[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) return(NULL)
    data.table(id_from = id_order[i], id_to = id_order[nb])
  }))

# --- expand to cell-year level by joining on every year present ---
  years <- sort(unique(data_dt$year))

  # Cross join edges Ã— years
  edge_year <- edge_list[, CJ(id_to = id_to, year = years), by = id_from]

  # Add the row index of the *focal* cell-year
  data_dt[, row_idx := .I]
  focal_idx <- data_dt[, .(id, year, focal_row = row_idx)]
  setkey(focal_idx, id, year)

  edge_year <- merge(edge_year, focal_idx,
                     by.x = c("id_from", "year"),
                     by.y = c("id", "year"),
                     all.x = FALSE, allow.cartesian = TRUE)

  # Add the row index of the *neighbor* cell-year
  neighbor_idx <- data_dt[, .(id, year, neighbor_row = row_idx)]
  setkey(neighbor_idx, id, year)

  edge_year <- merge(edge_year, neighbor_idx,
                     by.x = c("id_to", "year"),
                     by.y = c("id", "year"),
                     all.x = FALSE, allow.cartesian = TRUE)

  edge_year[, .(focal_row, neighbor_row)]
}

# ===========================================================
# 2. Compute neighbor stats for one variable (vectorised)
# ===========================================================
compute_neighbor_stats_fast <- function(data_dt, edges, var_name) {

  # Pull the variable values for every neighbor row
  edges[, val := data_dt[[var_name]][neighbor_row]]

  # Drop NAs
  valid <- edges[!is.na(val)]

  # Grouped aggregation â€” one group per focal row
  stats <- valid[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = focal_row]

  # Build full-length columns (NA where a focal row had no valid neighbors)
  n <- nrow(data_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  out_max [stats$focal_row] <- stats$nb_max
  out_min [stats$focal_row] <- stats$nb_min
  out_mean[stats$focal_row] <- stats$nb_mean

  # Attach to data_dt in place
  set(data_dt, j = paste0(var_name, "_nb_max"),  value = out_max)
  set(data_dt, j = paste0(var_name, "_nb_min"),  value = out_min)
  set(data_dt, j = paste0(var_name, "_nb_mean"), value = out_mean)

  invisible(data_dt)
}

# ===========================================================
# 3. Full optimised pipeline (drop-in replacement)
# ===========================================================
run_optimised_pipeline <- function(cell_data,        # data.frame or data.table
                                   id_order,         # vector of spatial cell ids
                                   rook_neighbors_unique,
                                   rf_model) {       # pre-trained RF model
  # Convert once
  cell_dt <- as.data.table(cell_data)

  # --- Vectorised neighbor lookup (replaces build_neighbor_lookup) ----------
  message("Building edge table â€¦")
  edges <- build_neighbor_edges(cell_dt, id_order, rook_neighbors_unique)
  message(sprintf("  Edge table: %s rows", format(nrow(edges), big.mark = ",")))

  # --- Vectorised neighbor stats (replaces loop over compute_neighbor_stats)
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for '%s' â€¦", var_name))
    compute_neighbor_stats_fast(cell_dt, edges, var_name)
  }

  # Clean up helper column
  cell_dt[, row_idx := NULL]

  # --- Random Forest prediction (untouched) ---------------------------------
  message("Running Random Forest predict() â€¦")
  preds <- predict(rf_model, newdata = cell_dt)
  cell_dt[, prediction := preds]

  message("Done.")
  cell_dt
}

# ===========================================================
# Usage (preserves trained RF model & original estimand)
# ===========================================================
# result <- run_optimised_pipeline(cell_data,
#                                  id_order,
#                                  rook_neighbors_unique,
#                                  rf_model = my_trained_rf)
```

---

### Why this is correct and equivalent

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The same `max`, `min`, `mean` are computed over exactly the same neighbor sets; only the iteration strategy changed (grouped `data.table` aggregation vs. R-level `lapply`). |
| **Trained RF preserved** | The model object is passed in read-only; `predict()` is called identically. |
| **Estimand preserved** | No features are added or removed; the three neighbor-derived columns per source variable carry the same names and values. |
| **Performance** | The edge table construction is a one-time vectorised join (~tens of seconds). Each variable's stats are a single grouped aggregation over the edge table (~seconds each). Total expected wall time: **under 10 minutes** vs. 86+ hours. |
END_RESPONSE>>>
