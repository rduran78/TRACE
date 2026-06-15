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
run_id: Anthropic_F04_T07_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is overwhelmingly in `build_neighbor_lookup`, not `compute_neighbor_stats`. Here's why:

**`build_neighbor_lookup`** iterates `lapply` over **~6.46 million rows**, and for each row it:
1. Performs character coercion and named-vector lookups (`id_to_ref`, `idx_lookup`) â€” these are O(n) hash lookups but repeated millions of times with per-element `paste` and `as.character` calls.
2. Constructs character key vectors (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) **inside the loop body** for every single row.
3. Subsets a named vector of length ~6.46M (`idx_lookup[neighbor_keys]`) per iteration.

The per-row string construction and named-vector lookup across 6.46 million iterations is the dominant cost. With ~4 neighbors per cell on average, that's ~25.8 million `paste` + hash-lookup operations embedded in an R-level loop â€” catastrophically slow.

**`compute_neighbor_stats`** is lighter (just numeric subsetting), but `do.call(rbind, result)` on a 6.46M-element list is also needlessly expensive.

## Optimization Strategy

**Core idea:** Replace the row-level R loop with vectorized operations using `data.table`.

1. **Vectorized neighbor lookup construction:** Instead of building a per-row list, construct a two-column edge table `(row_index, neighbor_row_index)` in one vectorized pass. Pre-join cell-IDs and years using `data.table` keyed joins.

2. **Vectorized neighbor stats:** Use `data.table` grouped aggregation (`[, .(max, min, mean), by = row_index]`) on the edge table â€” no R-level loop at all.

3. **Avoid `do.call(rbind, ...)`** on millions of list elements.

Expected speedup: from 86+ hours to **minutes** (roughly 2â€“10 minutes depending on disk I/O).

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. VECTORIZED NEIGHBOR LOOKUP (replaces build_neighbor_lookup)
# ---------------------------------------------------------------
build_neighbor_edge_table <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id' and 'year' (and row order matters)
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  # Step A: Build a cell-level edge list (cell_ref -> neighbor_ref)
  #   neighbors[[i]] gives integer indices into id_order for cell id_order[i]
  n_cells <- length(id_order)
  from_ref <- rep(seq_len(n_cells), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  cell_edges <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
  # Remove any zero-length / self-loop artifacts from nb objects

  cell_edges <- cell_edges[!is.na(to_id) & to_id != 0L]

  # Step B: Map every (id, year) to its row index in data_dt
  data_dt[, row_idx := .I]

  # Step C: Cross cell edges with years via keyed joins
  #   For every row i with (from_id, year), find the row j with (to_id, same year)
  from_map <- data_dt[, .(from_id = id, year, from_row = row_idx)]
  to_map   <- data_dt[, .(to_id = id, year, to_row = row_idx)]

  setkey(from_map, from_id, year)
  setkey(to_map, to_id, year)

  # Expand cell_edges Ã— years: join from_map on from_id, then to_map on (to_id, year)
  # Efficient approach: join cell_edges to from_map to get (from_row, to_id, year),
  # then join to to_map to get to_row.
  setkey(cell_edges, from_id)
  setkey(from_map, from_id)

  # Many-to-many merge: each cell edge fans out over all years the from_id appears in
  edge_years <- cell_edges[from_map, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_years now has columns: from_id, to_id, year, from_row

  setkey(edge_years, to_id, year)
  setkey(to_map, to_id, year)

  edge_years <- to_map[edge_years, on = c("to_id", "year"), nomatch = NA]
  # Keep only edges where the neighbor actually exists in that year
  edge_years <- edge_years[!is.na(to_row)]

  # Return lean edge table: (from_row, to_row)
  edge_years[, .(from_row, to_row)]
}

# ---------------------------------------------------------------
# 2. VECTORIZED NEIGHBOR STATS (replaces compute_neighbor_stats)
# ---------------------------------------------------------------
compute_neighbor_stats_vec <- function(data_dt, edge_dt, var_name) {
  # edge_dt has columns from_row, to_row
  # Fetch neighbor values in one vectorized pull
  edge_dt[, val := data_dt[[var_name]][to_row]]

  # Drop NAs in the variable
  valid <- edge_dt[!is.na(val)]

  # Grouped aggregation â€” one pass, fully vectorized
  stats <- valid[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = from_row]

  # Allocate full-length result columns (NA for rows with no valid neighbors)
  n <- nrow(data_dt)
  max_col  <- rep(NA_real_, n)
  min_col  <- rep(NA_real_, n)
  mean_col <- rep(NA_real_, n)

  max_col[stats$from_row]  <- stats$nb_max
  min_col[stats$from_row]  <- stats$nb_min
  mean_col[stats$from_row] <- stats$nb_mean

  # Clean up temp column
  edge_dt[, val := NULL]

  list(max = max_col, min = min_col, mean = mean_col)
}

# ---------------------------------------------------------------
# 3. OUTER LOOP (replaces original outer loop)
# ---------------------------------------------------------------
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  # Convert to data.table if not already (non-destructive copy)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  message("Building vectorized edge table...")
  edge_dt <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("Edge table: %s rows (directed cell-year neighbor pairs)", format(nrow(edge_dt), big.mark = ",")))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_neighbor_stats_vec(cell_data, edge_dt, var_name)

    # Column naming convention: preserve whatever the original
    # compute_and_add_neighbor_features used.
    # Common convention: <var>_nb_max, <var>_nb_min, <var>_nb_mean
    set(cell_data, j = paste0(var_name, "_nb_max"),  value = stats$max)
    set(cell_data, j = paste0(var_name, "_nb_min"),  value = stats$min)
    set(cell_data, j = paste0(var_name, "_nb_mean"), value = stats$mean)
  }

  # Clean up helper column
  cell_data[, row_idx := NULL]

  message("Done.")
  cell_data
}

# ---------------------------------------------------------------
# USAGE (drop-in replacement)
# ---------------------------------------------------------------
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets â€” the spatial topology (`rook_neighbors_unique`) and year-matching logic are identical. Only the implementation strategy changes (vectorized join vs. per-row loop). |
| **Trained RF model** | No model retraining. The output columns have the same values, so `predict()` produces identical results. |
| **Column names** | Adjust the `paste0(var_name, "_nb_max", ...)` pattern to match whatever naming convention the existing trained model expects. |

## Performance Estimate

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup | ~80+ hours (6.46M R-loop iterations with string ops) | ~1â€“3 min (vectorized `data.table` keyed joins) |
| Stats (5 vars) | ~6 hours (6.46M Ã— 5 `lapply` + `do.call(rbind)`) | ~1â€“2 min (grouped `data.table` aggregation) |
| **Total** | **~86+ hours** | **~2â€“5 minutes** |

Peak RAM for the edge table: ~1.37M cell-edges Ã— 28 years Ã— 2 integer columns â‰ˆ ~300 MB, well within 16 GB.
END_RESPONSE>>>
