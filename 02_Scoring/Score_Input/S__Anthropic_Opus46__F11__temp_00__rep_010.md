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
run_id: Anthropic_F11_T00_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named-vector lookups in R are **O(n) string-matching scans** on each call when the vector is large. With ~6.46M keys in `idx_lookup`, this is catastrophically slow â€” effectively **O(nÂ²)** overall.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over ~6.46 million entries with `lapply`, computing `max`, `min`, and `mean` on subsetted vectors. While individually cheaper per iteration than `build_neighbor_lookup`, the repeated R-level loop over millions of rows is still very expensive.

3. **Random Forest inference** (`predict()` on a pre-trained model) for ~6.46M rows with ~110 predictors is a single vectorized C-level call (in `ranger` or `randomForest`). This typically completes in seconds to a few minutes â€” orders of magnitude faster than the neighbor computation.

**The bottleneck is the O(nÂ²)-behaving neighbor lookup construction and the repeated R-level row-wise loops for neighbor statistics across 6.46 million rows Ã— 5 variables.**

---

## Optimization Strategy

1. **Replace named-vector lookups with `data.table` hash joins** â€” `data.table` uses true hash-based indexing, turning O(n) scans into O(1) lookups.
2. **Vectorize `build_neighbor_lookup`** â€” Instead of building a per-row list, construct a flat edge-list (a two-column `data.table` of `[row_i, neighbor_row_j]`) using vectorized joins. This eliminates the 6.46M-iteration `lapply`.
3. **Vectorize `compute_neighbor_stats`** â€” Use the flat edge-list with `data.table` grouped aggregation (`[, .(max, min, mean), by = row_i]`) to compute all neighbor statistics in one vectorized pass per variable, eliminating another `lapply` over millions of rows.
4. **Preserve the trained Random Forest model and the original numerical estimand** â€” no changes to the model or prediction target.

Expected speedup: from **86+ hours to minutes**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build a vectorized neighbor edge-list using data.table hash joins
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_edgelist_dt <- function(data_dt, id_order, rook_neighbors) {

  # data_dt: a data.table with columns 'id' and 'year' (and all other columns)
  #          plus a column 'row_idx' = .I (row position in the original table)
  # id_order: integer vector of cell IDs in the order used by the nb object
  # rook_neighbors: spdep nb object (list of integer index vectors)

  # Step A: Expand the nb object into a flat edge-list of (ref_pos, neighbor_ref_pos)
  #   ref_pos indexes into id_order
  n_cells <- length(id_order)
  from_ref <- rep(seq_len(n_cells), times = lengths(rook_neighbors))
  to_ref   <- unlist(rook_neighbors, use.names = FALSE)

  # Remove zero-neighbor entries (spdep uses 0L as placeholder for no neighbors)
  valid <- to_ref != 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]

  # Map ref positions to actual cell IDs
  edges <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # Step B: Create a keyed lookup from (id, year) -> row_idx in data_dt
  #   We need to join edges with every year so that for each (from_id, year)
  #   we find the row_idx of (to_id, year).

  # Get the unique years
  years <- unique(data_dt$year)

  # Cross-join edges with years: each directed edge exists for every year
  edges_by_year <- edges[, .(year = years), by = .(from_id, to_id)]

  # Key the main data for fast join: (id, year) -> row_idx
  id_year_lookup <- data_dt[, .(id, year, row_idx)]
  setkey(id_year_lookup, id, year)

  # Join to get the source row index (the row whose neighbors we want)
  setnames(edges_by_year, "from_id", "id")
  edges_by_year <- id_year_lookup[edges_by_year, on = .(id, year), nomatch = 0L]
  setnames(edges_by_year, c("row_idx", "id"), c("src_row", "from_id"))

  # Join to get the neighbor row index
  setnames(edges_by_year, "to_id", "id")
  edges_by_year <- id_year_lookup[edges_by_year, on = .(id, year), nomatch = 0L]
  setnames(edges_by_year, c("row_idx", "id"), c("nbr_row", "to_id"))

  # Return a lean two-column edge-list: src_row -> nbr_row
  edges_by_year[, .(src_row, nbr_row)]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Compute neighbor stats for one variable using vectorized grouping
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_stats_dt <- function(data_dt, edge_dt, var_name) {
  # Pull the variable values for every neighbor row
  vals <- data_dt[[var_name]]
  work <- edge_dt[, .(src_row, nbr_val = vals[nbr_row])]

  # Drop NAs in the neighbor values before aggregation
  work <- work[!is.na(nbr_val)]

  # Grouped aggregation â€” single vectorized pass
  agg <- work[, .(
    nb_max  = max(nbr_val),
    nb_min  = min(nbr_val),
    nb_mean = mean(nbr_val)
  ), keyby = src_row]

  # Build full-length result columns (NA for rows with no valid neighbors)
  n <- nrow(data_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  out_max[agg$src_row]  <- agg$nb_max
  out_min[agg$src_row]  <- agg$nb_min
  out_mean[agg$src_row] <- agg$nb_mean

  suffix <- var_name
  new_names <- paste0(c("nb_max_", "nb_min_", "nb_mean_"), suffix)
  data_dt[, (new_names[1]) := out_max]
  data_dt[, (new_names[2]) := out_min]
  data_dt[, (new_names[3]) := out_mean]

  invisible(data_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Full optimized pipeline (drop-in replacement for the outer loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model) {
  # Convert to data.table in place (no deep copy if already data.table)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Add a row index column

  cell_data[, row_idx := .I]

  message("Building vectorized neighbor edge-list...")
  edge_dt <- build_neighbor_edgelist_dt(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("  Edge-list has %s rows.", format(nrow(edge_dt), big.mark = ",")))

  # Compute neighbor features for each source variable
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for '%s'...", var_name))
    compute_neighbor_stats_dt(cell_data, edge_dt, var_name)
  }

  # Clean up helper column
  cell_data[, row_idx := NULL]

  # â”€â”€ Random Forest inference (unchanged â€” this was never the bottleneck) â”€â”€
  message("Running Random Forest predictions...")
  preds <- predict(rf_model, data = cell_data)

  # Handle both ranger and randomForest return types
  if (inherits(preds, "ranger.prediction")) {
    cell_data[, prediction := preds$predictions]
  } else {
    cell_data[, prediction := as.numeric(preds)]
  }

  message("Done.")
  cell_data
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Usage (assumes objects already in the environment):
#   result <- run_optimized_pipeline(
#     cell_data              = cell_data,
#     id_order               = id_order,
#     rook_neighbors_unique  = rook_neighbors_unique,
#     rf_model               = rf_model
#   )
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows; named-vector string matching â‰ˆ O(nÂ²) | Vectorized `data.table` hash joins; O(n) |
| **Neighbor stats** | `lapply` over 6.46M rows Ã— 5 vars; R-level loop | `data.table` grouped aggregation; single vectorized pass per variable |
| **RF inference** | Already vectorized C-level call | Unchanged (not the bottleneck) |
| **Estimated runtime** | 86+ hours | Minutes |
| **Trained RF model** | Preserved | Preserved |
| **Numerical estimand** | max, min, mean of rook-neighbor values | Identical â€” preserved exactly |
END_RESPONSE>>>
