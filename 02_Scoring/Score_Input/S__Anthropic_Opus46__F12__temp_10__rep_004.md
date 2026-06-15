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
run_id: Anthropic_F12_T10_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: The Colleague Is Wrong

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized concatenation operation and runs in seconds to low minutes at most. The inner `lapply` in `compute_neighbor_stats` does not perform "repeated list binding" â€” it returns fixed-length vectors collected into a pre-allocated list.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** For every single row `i`, the function calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` and then indexes into `idx_lookup` â€” a named character vector with ~6.46 million entries. Named vector lookup in R is **O(n)** linear scan per query (R's named vectors use linear hashing that degrades badly at this scale). With ~6.46M rows and an average of ~4 rook neighbors each, that's ~25.8 million string-key lookups into a 6.46M-entry named vector.

2. **Repeated `as.character()` coercion** on every iteration.

3. **Character string construction (`paste`) inside a per-row loop** â€” ~6.46 million calls to `paste`, each producing ~4 strings.

This single function likely accounts for **>95% of the 86+ hour runtime**. Once the lookup is built, `compute_neighbor_stats` with 5 variables is comparatively trivial (5 Ã— one `lapply` of simple arithmetic over pre-resolved integer indices).

## Optimization Strategy

1. **Replace the named-vector lookup with an environment (hash map) or, better, a fully vectorized merge/join approach using `data.table`.** Eliminate the per-row `lapply` in `build_neighbor_lookup` entirely.

2. **Vectorize `build_neighbor_lookup`** by expanding the neighbor list into a two-column edge table `(row_i, neighbor_cell_id)`, joining on `(neighbor_cell_id, year)` to resolve target row indices in one bulk operation.

3. **Vectorize `compute_neighbor_stats`** by using the edge table with `data.table` grouped aggregation (`max`, `min`, `mean` by source row), eliminating the per-row `lapply` there too.

This reduces the runtime from ~86+ hours to **minutes**.

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED build_neighbor_lookup (returns an edge data.table)
# ============================================================
build_neighbor_edge_table <- function(data, id_order, neighbors) {
  # data must have columns: id, year (and be ordered by original row number)
  dt <- as.data.table(data)[, row_idx := .I]

  # Build a mapping from id_order position (ref_idx) to cell id
  # neighbors[[ref_idx]] gives neighbor positions in id_order
  # Expand neighbor list into an edge table: (cell_id, neighbor_cell_id)
  n_cells <- length(id_order)
  edge_list <- rbindlist(lapply(seq_len(n_cells), function(ref_idx) {
    nb <- neighbors[[ref_idx]]
    if (length(nb) == 0L) return(NULL)
    data.table(cell_id = id_order[ref_idx],
               neighbor_cell_id = id_order[nb])
  }))
  # edge_list now has ~1,373,394 rows (directed edges)

  # For every (cell_id, year) row, we need neighbor rows:
  # Join edges with the data on cell_id to get (row_idx_source, neighbor_cell_id, year)
  source <- dt[, .(row_idx_source = row_idx, cell_id = id, year)]
  edges_with_year <- merge(edge_list, source,
                           by = "cell_id", allow.cartesian = TRUE)
  # Now resolve neighbor_cell_id + year -> row_idx_target
  setnames(dt, "id", "cell_id_target")
  target_key <- dt[, .(cell_id_target, year, row_idx_target = row_idx)]
  setkey(target_key, cell_id_target, year)

  setnames(edges_with_year, "neighbor_cell_id", "cell_id_target")
  setkey(edges_with_year, cell_id_target, year)

  resolved <- target_key[edges_with_year, nomatch = 0L]
  # Columns: cell_id_target, year, row_idx_target, cell_id, row_idx_source

  resolved[, .(row_idx_source, row_idx_target)]
}

# ============================================================
# OPTIMIZED compute_neighbor_stats (vectorized via data.table)
# ============================================================
compute_neighbor_stats_fast <- function(data_dt, edge_dt, var_name, n_rows) {
  # edge_dt has columns: row_idx_source, row_idx_target
  # Extract neighbor values in one vectorized step
  vals <- data_dt[[var_name]]
  work <- edge_dt[, .(row_idx_source, nval = vals[row_idx_target])]
  work <- work[!is.na(nval)]

  stats <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = row_idx_source]

  # Build full-length result (NA for rows with no valid neighbors)
  out <- data.table(
    row_idx_source = seq_len(n_rows),
    nb_max  = NA_real_,
    nb_min  = NA_real_,
    nb_mean = NA_real_
  )
  out[stats, on = "row_idx_source",
      `:=`(nb_max = i.nb_max, nb_min = i.nb_min, nb_mean = i.nb_mean)]
  out
}

# ============================================================
# OPTIMIZED outer pipeline
# ============================================================
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  cell_dt <- as.data.table(cell_data)
  n_rows  <- nrow(cell_dt)

  message("Building neighbor edge table (vectorized)...")
  edge_dt <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
  setkey(edge_dt, row_idx_source)
  message(sprintf("  Edge table: %d resolved edges.", nrow(edge_dt)))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_neighbor_stats_fast(cell_dt, edge_dt, var_name, n_rows)

    # Attach columns with same naming convention as original code
    col_max  <- paste0(var_name, "_nb_max")
    col_min  <- paste0(var_name, "_nb_min")
    col_mean <- paste0(var_name, "_nb_mean")

    set(cell_dt, j = col_max,  value = stats$nb_max)
    set(cell_dt, j = col_min,  value = stats$nb_min)
    set(cell_dt, j = col_mean, value = stats$nb_mean)
  }

  message("Done. Returning augmented data.table.")
  cell_dt
}

# ============================================================
# USAGE (drop-in replacement; RF model is untouched)
# ============================================================
# cell_data <- run_optimized_pipeline(cell_data, id_order, rook_neighbors_unique)
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| Alleged bottleneck | `do.call(rbind, ...)` in `compute_neighbor_stats` | Negligible (single bind of pre-allocated list) |
| **True bottleneck** | â€” | `build_neighbor_lookup()`: 6.46M per-row `paste` + named-vector lookups (O(n) each) â‰ˆ 25.8M linear scans into a 6.46M-entry vector |
| Estimated speedup | â€” | From ~86+ hours â†’ **~5â€“15 minutes** via vectorized `data.table` joins |
| Numerical equivalence | â€” | Preserved: same `max`, `min`, `mean` over identical neighbor sets |
| Trained RF model | â€” | Untouched; only the feature-engineering pipeline is optimized |
END_RESPONSE>>>
