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
run_id: Anthropic_F11_T07_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis

**The colleague's claim is rejected.** The bottleneck is **not** Random Forest inference. Loading a trained model, calling `predict()` once on ~6.46 million rows with ~110 features, and writing predictions takes minutes at most â€” it is a single vectorized operation.

**The actual bottleneck is the spatial neighbor feature engineering pipeline**, specifically:

1. **`build_neighbor_lookup`**: An `lapply` over **6.46 million rows**, each iteration performing character coercion, `paste()`, and named-vector lookups. Named vector lookup in R is O(n) string hashing per access, and this is done millions of times with millions of names â€” effectively O(nÂ²) behavior.

2. **`compute_neighbor_stats`**: Called **5 times** (once per variable), each time iterating over 6.46 million rows via `lapply`, subsetting and computing `max/min/mean` per row. That's ~32.3 million R-level loop iterations with per-element subsetting.

The combination of character-key lookups in `build_neighbor_lookup` and repeated row-level R loops in `compute_neighbor_stats` across 6.46M rows is what produces the 86+ hour runtime.

---

# Optimization Strategy

1. **Replace character-key named-vector lookups with integer-indexed operations** using `data.table` for fast equi-joins.
2. **Build the neighbor-row mapping as a two-column integer edge list** via a single vectorized merge/join â€” no `lapply` over 6.46M rows.
3. **Compute all neighbor statistics in one vectorized pass** using `data.table` grouped aggregation, eliminating per-row R loops entirely.
4. **Preserve the trained RF model and the original numerical estimand** â€” only the feature construction is rewritten.

Expected speedup: from 86+ hours to **minutes**.

---

# Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Build a vectorized edge list of (row_index, neighbor_row_index)
#    Replaces build_neighbor_lookup entirely.
# ---------------------------------------------------------------
build_neighbor_edges <- function(cell_data_dt, id_order, rook_neighbors) {
  # cell_data_dt must be a data.table with columns: id, year, and a row index
  # rook_neighbors is the spdep nb object (list of integer neighbor indices into id_order)

  # Map each grid-cell id to its position in id_order
  n_ids <- length(id_order)

  # Expand the nb list into a two-column edge table: (source_id, neighbor_id)
  # Each element k of rook_neighbors lists the neighbor positions in id_order
  source_pos <- rep(seq_len(n_ids), lengths(rook_neighbors))
  neighbor_pos <- unlist(rook_neighbors)

  edges_id <- data.table(
    source_id   = id_order[source_pos],
    neighbor_id = id_order[neighbor_pos]
  )

  # Add a row-index column to cell_data_dt
  cell_data_dt[, row_idx := .I]

  # Create lookup: (id, year) -> row_idx
  lookup <- cell_data_dt[, .(id, year, row_idx)]

  # Join edges with years: for every (source_id, year) row, find the
  # neighbor rows that share the same year.
  # Step A: attach row_idx and year for the source side
  setkey(lookup, id)
  source_lookup <- copy(lookup)
  setnames(source_lookup, c("id", "year", "row_idx"),
           c("source_id", "year", "src_row"))

  # Step B: attach row_idx for the neighbor side (same year)
  neighbor_lookup <- copy(lookup)
  setnames(neighbor_lookup, c("id", "year", "row_idx"),
           c("neighbor_id", "year", "nbr_row"))

  # Merge: edges_id Ã— source_lookup on source_id, then Ã— neighbor_lookup
  # on (neighbor_id, year)
  setkey(edges_id, source_id)
  setkey(source_lookup, source_id)

  # This gives every (source_id, year, neighbor_id) triple
  edge_year <- merge(edges_id, source_lookup, by = "source_id",
                     allow.cartesian = TRUE)

  setkey(edge_year, neighbor_id, year)
  setkey(neighbor_lookup, neighbor_id, year)

  edge_full <- merge(edge_year, neighbor_lookup,
                     by = c("neighbor_id", "year"))

  # Return only the essential columns
  edge_full[, .(src_row, nbr_row)]
}

# ---------------------------------------------------------------
# 2. Compute neighbor stats for ALL variables in one pass
#    Replaces compute_neighbor_stats + the outer for-loop.
# ---------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data_dt, edge_dt,
                                          neighbor_source_vars) {
  n <- nrow(cell_data_dt)

  for (var_name in neighbor_source_vars) {
    message("Processing neighbor features for: ", var_name)

    # Attach the neighbor's value to each edge
    vals <- cell_data_dt[[var_name]]
    edge_dt[, nbr_val := vals[nbr_row]]

    # Grouped aggregation by src_row (the focal row)
    stats <- edge_dt[!is.na(nbr_val),
                     .(nb_max  = max(nbr_val),
                       nb_min  = min(nbr_val),
                       nb_mean = mean(nbr_val)),
                     by = src_row]

    # Initialize new columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    cell_data_dt[, (max_col)  := NA_real_]
    cell_data_dt[, (min_col)  := NA_real_]
    cell_data_dt[, (mean_col) := NA_real_]

    # Fill in computed values
    cell_data_dt[stats$src_row, (max_col)  := stats$nb_max]
    cell_data_dt[stats$src_row, (min_col)  := stats$nb_min]
    cell_data_dt[stats$src_row, (mean_col) := stats$nb_mean]
  }

  cell_data_dt
}

# ---------------------------------------------------------------
# 3. Full pipeline (drop-in replacement for the original code)
# ---------------------------------------------------------------
run_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                  neighbor_source_vars) {
  cell_data_dt <- as.data.table(cell_data)

  message("Building vectorized neighbor edge list...")
  edge_dt <- build_neighbor_edges(cell_data_dt, id_order,
                                  rook_neighbors_unique)
  message(sprintf("Edge list: %s row-to-neighbor-row pairs", nrow(edge_dt)))

  message("Computing all neighbor features...")
  cell_data_dt <- compute_all_neighbor_features(cell_data_dt, edge_dt,
                                                 neighbor_source_vars)

  # Clean up helper column
  cell_data_dt[, row_idx := NULL]

  cell_data_dt
}

# ---------------------------------------------------------------
# Usage (preserves trained RF model and estimand)
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- run_neighbor_pipeline(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)

# Random Forest prediction is unchanged â€” not the bottleneck
# rf_model is the pre-trained model loaded from disk
predictions <- predict(rf_model, newdata = cell_data)
cell_data$predicted <- predictions
```

---

# Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste`/named-vector lookup | Single vectorized `data.table` merge producing an integer edge list |
| `compute_neighbor_stats` | 5 Ã— 6.46M `lapply` iterations (32.3M R-loop steps) | 5 Ã— one grouped `data.table` aggregation (fully vectorized C-level) |
| Estimated runtime | 86+ hours | Minutes |
| RF model | Untouched | Untouched |
| Numerical results | Identical `max/min/mean` per neighbor set | Identical |
END_RESPONSE>>>
