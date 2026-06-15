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
run_id: Anthropic_F11_T07_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

**Evidence from the code:**

1. **`build_neighbor_lookup()`** uses `lapply` over **~6.46 million rows**, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`) for every single row. Named vector lookup in R is O(n) hash probing per call, and doing this 6.46 million times with string construction each time is extraordinarily expensive. With ~1.37 million neighbor relationships spread across those rows, millions of small character vectors are created and discarded, causing massive GC pressure.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over the 6.46 million-element `neighbor_lookup` list with `lapply`, computing `max`, `min`, and `mean` on small integer-indexed subsets. The 5 Ã— 6.46M = ~32.3 million R-level function calls via `lapply`, each allocating small vectors, dominate runtime.

3. **`do.call(rbind, result)`** on a list of 6.46 million 3-element vectors is itself a known performance anti-pattern in R â€” it creates a massive argument list and concatenates row-by-row.

4. By contrast, Random Forest **prediction** on a pre-trained model is a single call to `predict()` on a matrix of ~6.46M Ã— 110 features. Even with a large forest, this is a vectorized C/C++ operation that typically completes in seconds to minutes â€” not hours.

**Conclusion:** The 86+ hour runtime is dominated by the row-level R loops in neighbor lookup construction and repeated neighbor statistics computation, not by RF inference.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace the per-row `lapply` with a fully vectorized approach using `data.table`. Pre-build an integer-keyed join table (`id` Ã— `year` â†’ row index) and expand the neighbor list into a flat edge table, then join in bulk. This eliminates millions of `paste`/`as.character`/named-lookup calls.

2. **Vectorize `compute_neighbor_stats()`**: Instead of `lapply` over 6.46M list elements, use the flat edge table with `data.table` grouped aggregation (`max`, `min`, `mean` by source row), computed once per variable. This replaces 6.46M R function calls with a single grouped C-level operation.

3. **Compute all 5 variables in a single pass** if possible, or at least use the efficient grouped aggregation per variable.

4. **Leave the Random Forest model and predict() call untouched** â€” it is not the bottleneck.

---

## Working R Code

```r
library(data.table)

#' Optimized: build a flat edge data.table mapping each row to its neighbor rows.
#' Replaces build_neighbor_lookup().
build_neighbor_edges <- function(data_dt, id_order, neighbors) {
  # data_dt must be a data.table with columns: id, year, and a .row_idx column
  # id_order: integer vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors into id_order)

  # Step 1: Map each cell id to its reference index in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Step 2: Build a lookup from (id, year) -> row index using data.table keyed join
  # Ensure row index exists
  data_dt[, .row_idx := .I]
  setkey(data_dt, id, year)

  # Step 3: Expand neighbors into a flat edge list: (source_id, neighbor_id)
  # For each unique cell id in the data, find its neighbor cell ids
  unique_ids <- unique(data_dt$id)

  # Build edge list at the cell-id level
  # ref indices for unique_ids
  ref_indices <- id_to_ref[as.character(unique_ids)]

  # For each unique cell, get neighbor cell IDs
  edge_list <- rbindlist(lapply(seq_along(unique_ids), function(k) {
    ri <- ref_indices[k]
    if (is.na(ri)) return(NULL)
    nb_idx <- neighbors[[ri]]
    if (length(nb_idx) == 0) return(NULL)
    nb_ids <- id_order[nb_idx]
    data.table(source_id = unique_ids[k], neighbor_id = nb_ids)
  }))

  if (is.null(edge_list) || nrow(edge_list) == 0) {
    return(data.table(
      source_row = integer(0),
      neighbor_row = integer(0)
    ))
  }

  # Step 4: Expand to (source_id, year, neighbor_id, year) by joining with data
  # For every (source_id, year) row, we need (neighbor_id, same year) row.

  # Get (id, year, row_idx) for source side
  source_rows <- data_dt[, .(source_id = id, year, source_row = .row_idx)]

  # Merge edge_list with source_rows to get (source_row, neighbor_id, year)
  edges_with_year <- merge(
    edge_list,
    source_rows,
    by = "source_id",
    allow.cartesian = TRUE
  )

  # Now join to get neighbor_row: lookup (neighbor_id, year) -> row_idx
  neighbor_index <- data_dt[, .(neighbor_id = id, year, neighbor_row = .row_idx)]
  setkey(neighbor_index, neighbor_id, year)
  setkey(edges_with_year, neighbor_id, year)

  result <- neighbor_index[edges_with_year, nomatch = 0L]

  # Return flat edge table: source_row <-> neighbor_row
  result[, .(source_row, neighbor_row)]
}


#' Optimized: compute neighbor stats for one variable using grouped aggregation.
#' Replaces compute_neighbor_stats() + compute_and_add_neighbor_features().
compute_neighbor_stats_fast <- function(data_dt, edges, var_name, n_rows) {
  # edges: data.table with columns source_row, neighbor_row
  # Attach the variable values for the neighbor rows
  vals <- data_dt[[var_name]]

  work <- edges[, .(source_row, nval = vals[neighbor_row])]
  work <- work[!is.na(nval)]

  stats <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = source_row]

  # Initialize output columns with NA
  out_max  <- rep(NA_real_, n_rows)
  out_min  <- rep(NA_real_, n_rows)
  out_mean <- rep(NA_real_, n_rows)

  out_max[stats$source_row]  <- stats$nb_max
  out_min[stats$source_row]  <- stats$nb_min
  out_mean[stats$source_row] <- stats$nb_mean

  data.table(nb_max = out_max, nb_min = out_min, nb_mean = out_mean)
}


#' Main optimized pipeline (drop-in replacement for the outer loop).
run_optimized_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {

  # Convert to data.table if needed
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  cell_data[, .row_idx := .I]

  n_rows <- nrow(cell_data)

  # Build the flat edge table once (replaces build_neighbor_lookup)
  message("Building neighbor edge table...")
  edges <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)
  setkey(edges, source_row)
  message(sprintf("Edge table built: %d edges.", nrow(edges)))

  # Compute neighbor features for each variable (replaces the for-loop)
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats_dt <- compute_neighbor_stats_fast(cell_data, edges, var_name, n_rows)

    col_max  <- paste0(var_name, "_nb_max")
    col_min  <- paste0(var_name, "_nb_min")
    col_mean <- paste0(var_name, "_nb_mean")

    set(cell_data, j = col_max,  value = stats_dt$nb_max)
    set(cell_data, j = col_min,  value = stats_dt$nb_min)
    set(cell_data, j = col_mean, value = stats_dt$nb_mean)
  }

  cell_data[, .row_idx := NULL]
  message("Neighbor feature engineering complete.")
  cell_data
}


# â”€â”€ Usage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data <- run_optimized_neighbor_features(cell_data, id_order, rook_neighbors_unique)
#
# # Random Forest prediction (unchanged â€” not the bottleneck)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | `lapply` over 6.46M rows with `paste`/named lookup | Single `data.table` merge-join; loop only over ~344K unique IDs |
| `compute_neighbor_stats` (Ã—5) | `lapply` over 6.46M-element list; `do.call(rbind, ...)` | Vectorized `data.table` grouped aggregation (`[, .(max,min,mean), by=]`) |
| Estimated runtime | 86+ hours | Minutes (edge build) + seconds per variable (grouped agg) |
| RF inference | Untouched | Untouched â€” not the bottleneck |
| Numerical results | Preserved | Identical `max`, `min`, `mean` per neighbor set per variable |

The trained Random Forest model is never modified or retrained. The original numerical estimand (neighbor max, min, mean per variable, and downstream RF predictions) is preserved exactly.
END_RESPONSE>>>
