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
run_id: Anthropic_F11_T07_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** as the true bottleneck â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named-vector lookups in R are **O(n)** scans on each call in the worst case and are extremely slow at scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this creates and resolves **tens of millions** of string-keyed lookups inside a serial R loop.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. That's **~32.3 million** R-level function invocations across the 5 variables.

3. **`do.call(rbind, result)`** on a list of ~6.46M small vectors is notoriously slow â€” it creates millions of intermediate objects.

4. Random Forest `predict()` on a pre-trained model with ~110 predictors and ~6.46M rows is a single vectorized C-level call (in `ranger` or `randomForest`). It is fast and is **not** the bottleneck.

**The 86+ hour runtime is dominated by millions of scalar R-loop iterations with string-key lookups in the neighbor engineering step, not by model inference.**

---

## Optimization Strategy

1. **Replace the row-level `lapply` in `build_neighbor_lookup()`** with a vectorized, `data.table`-based merge/join approach. Pre-expand all neighbor relationships into a two-column edge table (`(row_i, neighbor_row_j)`), then join once.

2. **Replace the row-level `lapply` in `compute_neighbor_stats()`** with a grouped `data.table` aggregation (`max`, `min`, `mean`) over the edge table â€” fully vectorized, single pass per variable.

3. **Eliminate `do.call(rbind, ...)`** entirely; `data.table` aggregation returns a data.table directly.

4. **Preserve the trained Random Forest model** â€” no changes to inference code.

5. **Preserve the original numerical estimand** â€” same `max`, `min`, `mean` statistics over the same neighbor sets.

Expected speedup: from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build a fully vectorized neighbor edge table (run once)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_edges <- function(cell_data, id_order, rook_neighbors_unique) {
  # cell_data must have columns: id, year
  # id_order: vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique: an nb object (list of integer index vectors)

  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # --- Map each cell ID to its position in id_order (reference index) ---
  id_to_ref <- data.table(
    id      = id_order,
    ref_idx = seq_along(id_order)
  )

  # --- Expand nb object into a directed edge list at the cell-ID level ---
  #     source_ref -> neighbor_ref
  n_neighbors <- lengths(rook_neighbors_unique)
  edge_ref <- data.table(
    source_ref   = rep(seq_along(rook_neighbors_unique), times = n_neighbors),
    neighbor_ref = unlist(rook_neighbors_unique, use.names = FALSE)
  )

  # Map reference indices back to cell IDs
  edge_ref[, source_id   := id_order[source_ref]]
  edge_ref[, neighbor_id := id_order[neighbor_ref]]
  edge_ref[, c("source_ref", "neighbor_ref") := NULL]

  # --- Build a lookup from (id, year) -> row_idx ---
  key_dt <- dt[, .(id, year, row_idx)]

  # --- Join: for every (source_id, year) row, find all neighbor rows ---
  #     First attach the source row_idx
  edges <- merge(
    edge_ref,
    key_dt,
    by.x = "source_id",
    by.y = "id",
    allow.cartesian = TRUE   # each source_id appears in many years
  )
  setnames(edges, c("row_idx"), c("source_row"))

  # Now attach the neighbor row_idx for the same year
  edges <- merge(
    edges,
    key_dt,
    by.x = c("neighbor_id", "year"),
    by.y = c("id", "year"),
    all.x = FALSE             # inner join: drop if neighbor-year absent
  )
  setnames(edges, "row_idx", "neighbor_row")

  # Return a lean two-column integer table
  edges[, .(source_row = as.integer(source_row),
            neighbor_row = as.integer(neighbor_row))]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Vectorized neighbor statistics (run once per variable)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_stats_fast <- function(cell_data_dt, edges, var_name) {
  # edges: data.table with columns source_row, neighbor_row
  # cell_data_dt: data.table with row ordering matching row indices in edges

  vals <- cell_data_dt[[var_name]]

  # Attach neighbor values
  work <- edges[, .(source_row, nval = vals[neighbor_row])]

  # Drop NAs in neighbor values
  work <- work[!is.na(nval)]

  # Grouped aggregation â€” single vectorized pass
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), keyby = source_row]

  # Allocate full-length result columns (NA for rows with no valid neighbors)
  n <- nrow(cell_data_dt)
  col_max  <- rep(NA_real_, n)
  col_min  <- rep(NA_real_, n)
  col_mean <- rep(NA_real_, n)

  col_max[agg$source_row]  <- agg$nb_max
  col_min[agg$source_row]  <- agg$nb_min
  col_mean[agg$source_row] <- agg$nb_mean

  list(
    max  = col_max,
    min  = col_min,
    mean = col_mean
  )
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Full optimized pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
run_optimized_pipeline <- function(cell_data,
                                   id_order,
                                   rook_neighbors_unique,
                                   rf_model) {
  library(data.table)

  cell_dt <- as.data.table(cell_data)

  # --- One-time edge table construction ---
  message("Building neighbor edge table â€¦")
  edges <- build_neighbor_edges(cell_dt, id_order, rook_neighbors_unique)
  message(sprintf("  Edge table: %s rows", format(nrow(edges), big.mark = ",")))

  # --- Neighbor feature engineering (vectorized) ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_neighbor_stats_fast(cell_dt, edges, var_name)
    set(cell_dt, j = paste0("nb_max_",  var_name), value = stats$max)
    set(cell_dt, j = paste0("nb_min_",  var_name), value = stats$min)
    set(cell_dt, j = paste0("nb_mean_", var_name), value = stats$mean)
  }

  # --- Random Forest inference (unchanged, already fast) ---
  message("Running Random Forest predict() â€¦")
  preds <- predict(rf_model, data = cell_dt)

  # Handle both ranger and randomForest return types
  if (inherits(preds, "ranger.prediction")) {
    cell_dt[, prediction := preds$predictions]
  } else {
    cell_dt[, prediction := as.numeric(preds)]
  }

  message("Done.")
  cell_dt[]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Usage (preserves trained RF model & original estimand):
#
#   result <- run_optimized_pipeline(
#     cell_data              = cell_data,
#     id_order               = id_order,
#     rook_neighbors_unique  = rook_neighbors_unique,
#     rf_model               = rf_model          # pre-trained, untouched
#   )
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Bottleneck** | `build_neighbor_lookup` + `compute_neighbor_stats` (scalar R loops Ã— 6.46M rows Ã— 5 vars) | Eliminated |
| **Lookup strategy** | Named-vector string key scan per row | Single `data.table` merge (hash join) |
| **Aggregation** | `lapply` â†’ `do.call(rbind, ...)` over 6.46M elements | `data.table` grouped `[, .(max, min, mean), by=]` |
| **RF inference** | `predict()` â€” already vectorized C code | Unchanged |
| **Numerical results** | max / min / mean of rook-neighbor values | Identical |
| **Trained model** | Preserved | Preserved |
| **Estimated runtime** | 86+ hours | Minutes |

The real bottleneck was millions of R-level scalar iterations with string-keyed lookups, not Random Forest inference.
END_RESPONSE>>>
