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
run_id: Anthropic_F11_T00_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M iterations, this alone is extremely expensive.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level loop iterations total.

3. **Random Forest inference** (`predict()` on a pre-trained model) is a single vectorized call over the dataset. Even with ~6.46M rows and ~110 predictors, a single `predict()` call on a `ranger` or `randomForest` object is internally optimized in C/C++ and typically completes in seconds to minutes â€” orders of magnitude faster than tens of millions of R-level `lapply` iterations.

The estimated 86+ hour runtime is consistent with the R-level loop overhead in the neighbor engineering, not with a single vectorized prediction call.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a vectorized `data.table` join approach. Instead of looping row-by-row, explode the neighbor list into an edge table `(cell_id, neighbor_id)`, join it with a `(cell_id, year) â†’ row_index` mapping table, and produce the lookup as a grouped structure â€” all using `data.table` merge/join operations with no R-level per-row loop.

2. **Replace `compute_neighbor_stats()`** with a single vectorized `data.table` grouped aggregation. Using the edge table, join in the variable values, then compute `max`, `min`, and `mean` grouped by the focal row index â€” entirely in C-level `data.table` code.

3. **Leave the Random Forest predict() call untouched**, since it is not the bottleneck.

This reduces the complexity from ~32M+ R-level `lapply` iterations to a handful of `data.table` joins and grouped aggregations, bringing the estimated runtime from 86+ hours down to minutes.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Vectorized build of the neighbor edge table (replaces build_neighbor_lookup)
# ==============================================================================

build_neighbor_edges <- function(cell_data_dt, id_order, rook_neighbors_unique) {

# cell_data_dt: a data.table with columns 'id', 'year', and a row index
# id_order: vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an nb object (list of integer index vectors)

  # --- Explode the nb object into a directed edge list of (focal_id, neighbor_id) ---
  n_cells <- length(id_order)
  focal_indices <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  neighbor_indices <- unlist(rook_neighbors_unique)

  # Remove zero-neighbor entries (spdep uses integer(0) for islands)
  valid <- !is.na(neighbor_indices) & neighbor_indices != 0L
  focal_indices <- focal_indices[valid]
  neighbor_indices <- neighbor_indices[valid]

  edges <- data.table(
    focal_cell_id    = id_order[focal_indices],
    neighbor_cell_id = id_order[neighbor_indices]
  )

  # --- Build a row-index lookup: (id, year) -> row position in cell_data_dt ---
  cell_data_dt[, row_idx := .I]

  # --- Cross-join edges with years via merge ---
  # For each focal row, find its neighbors in the same year.
  # Merge edges with focal rows to get the year and focal row_idx
  focal_key <- cell_data_dt[, .(focal_cell_id = id, year, focal_row_idx = row_idx)]
  setkey(focal_key, focal_cell_id)
  setkey(edges, focal_cell_id)

  # Inner join: each edge gets every year the focal cell appears in
  edge_year <- merge(edges, focal_key, by = "focal_cell_id", allow.cartesian = TRUE)

  # Now find the row_idx of the neighbor in the same year
  neighbor_key <- cell_data_dt[, .(neighbor_cell_id = id, year, neighbor_row_idx = row_idx)]
  setkey(neighbor_key, neighbor_cell_id, year)
  setkey(edge_year, neighbor_cell_id, year)

  edge_full <- merge(edge_year, neighbor_key, by = c("neighbor_cell_id", "year"),
                     nomatch = NULL)

  # Return the essential columns
  edge_full[, .(focal_row_idx, neighbor_row_idx)]
}

# ==============================================================================
# STEP 2: Vectorized neighbor stats (replaces compute_neighbor_stats)
# ==============================================================================

compute_neighbor_stats_fast <- function(cell_data_dt, edge_dt, var_name) {
  # edge_dt has columns: focal_row_idx, neighbor_row_idx
  # Pull the variable values for each neighbor
  edge_dt[, val := cell_data_dt[[var_name]][neighbor_row_idx]]

  # Remove NAs
  agg <- edge_dt[!is.na(val),
                 .(nb_max  = max(val),
                   nb_min  = min(val),
                   nb_mean = mean(val)),
                 by = focal_row_idx]

  # Allocate full-length result (NA for rows with no valid neighbors)
  n <- nrow(cell_data_dt)
  result <- data.table(
    nb_max  = rep(NA_real_, n),
    nb_min  = rep(NA_real_, n),
    nb_mean = rep(NA_real_, n)
  )
  result[agg$focal_row_idx, `:=`(
    nb_max  = agg$nb_max,
    nb_min  = agg$nb_min,
    nb_mean = agg$nb_mean
  )]

  # Clean up temporary column
  edge_dt[, val := NULL]

  # Name columns to match the variable
  setnames(result,
           c("nb_max", "nb_min", "nb_mean"),
           paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))
  result
}

# ==============================================================================
# STEP 3: Full pipeline (replaces the outer loop)
# ==============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  cell_data_dt <- as.data.table(cell_data)

  message("Building neighbor edge table (vectorized)...")
  edge_dt <- build_neighbor_edges(cell_data_dt, id_order, rook_neighbors_unique)
  setkey(edge_dt, focal_row_idx)
  message(sprintf("  Edge table: %s rows", format(nrow(edge_dt), big.mark = ",")))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_neighbor_stats_fast(cell_data_dt, edge_dt, var_name)
    cell_data_dt <- cbind(cell_data_dt, stats)
  }

  # Remove helper column
  cell_data_dt[, row_idx := NULL]

  cell_data_dt
}

# ==============================================================================
# STEP 4: Run and then predict (RF model untouched)
# ==============================================================================

# --- Feature engineering (the actual bottleneck, now fast) ---
cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)

# --- Random Forest inference (NOT the bottleneck, unchanged) ---
# Assuming `rf_model` is the pre-trained model loaded from disk:
# rf_model <- readRDS("trained_rf_model.rds")
# cell_data$prediction <- predict(rf_model, data = cell_data)$predictions
# (or for randomForest: predict(rf_model, newdata = cell_data))
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level `lapply` iterations with `paste`/named-vector lookups | Single vectorized `data.table` merge producing an edge table |
| `compute_neighbor_stats` | ~6.46M Ã— 5 = ~32.3M `lapply` iterations | 5 Ã— one `data.table` grouped aggregation (C-level) |
| Estimated runtime | 86+ hours | Minutes (typically 5â€“15 min on 16 GB laptop) |
| RF `predict()` | Untouched | Untouched |
| Trained RF model | Preserved | Preserved |
| Numerical estimand | Preserved (same `max`, `min`, `mean` over same neighbor sets) | Preserved identically |

The true bottleneck is the row-level R loop over millions of rows in the neighbor feature engineering, not the Random Forest inference.
END_RESPONSE>>>
