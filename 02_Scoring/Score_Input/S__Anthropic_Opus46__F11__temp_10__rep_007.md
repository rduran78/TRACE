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
run_id: Anthropic_F11_T10_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points decisively to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and NA filtering. Named vector lookup in R is hash-based but still carries significant per-call overhead. With ~6.46M iterations, each doing multiple string allocations and lookups across vectors of millions of names, this alone can take many hours.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over ~6.46 million rows with `lapply`, subsetting, removing NAs, and computing `max/min/mean`. The final `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also expensive.

3. **Combined cost**: `build_neighbor_lookup` produces ~6.46M list elements, each containing integer index vectors. Then `compute_neighbor_stats` is called 5 times over all 6.46M rows. The total R-level interpreted loop iterations are approximately **6.46M Ã— (1 + 5) â‰ˆ 38.8 million**, all with per-element overhead from anonymous functions, string operations, and subsetting.

4. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, a single `predict()` call on a `ranger` or `randomForest` object is implemented in C/C++ and typically completes in seconds to minutes â€” orders of magnitude faster than 86+ hours.

**Verdict**: The bottleneck is the row-level R `lapply` loops over millions of rows with string operations and named-vector lookups in the neighbor feature engineering step.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace the row-by-row `lapply` with a fully vectorized join. Pre-expand all neighbor relationships into a flat edge table (`(source_row, neighbor_id)`), then join against the data to resolve `(neighbor_id, year)` â†’ `target_row`. Group by `source_row` to get the lookup. Use `data.table` for speed.

2. **Vectorize `compute_neighbor_stats()`**: Instead of iterating per row, use the flat edge table joined to variable values, then compute grouped `max/min/mean` in one `data.table` aggregation â€” a single vectorized pass per variable.

3. **Avoid string keys entirely**: Use integer-based joins (id + year as compound key) instead of paste-based string keys.

This should reduce runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ==============================================================
# OPTIMIZED: build_neighbor_edge_table
# Replaces build_neighbor_lookup with a flat data.table of edges
# mapping each (source_row) -> (neighbor_row) via integer joins.
# ==============================================================

build_neighbor_edge_table <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id', 'year', and a row index '.row_idx'
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  # Step 1: Build a flat edge list of (source_cell_id, neighbor_cell_id)
  #         from the nb object. This is independent of year.
  n_cells <- length(id_order)
  source_indices <- rep(seq_len(n_cells), lengths(neighbors))
  neighbor_indices <- unlist(neighbors)

  # Remove any zero-length / empty-neighbor entries (lengths == 0 produces nothing)
  cell_edges <- data.table(
    source_id   = id_order[source_indices],
    neighbor_id = id_order[neighbor_indices]
  )

  # Step 2: For each row in the data, we need its (id, year) and row index.
  #         The neighbor rows are those sharing the same year with a neighboring id.
  #         So we expand: for each data row, find all neighbor cell_ids, then join
  #         back to data to find the actual row index for (neighbor_id, year).

  # Create a keyed lookup: (id, year) -> row index
  row_lookup <- data_dt[, .(id, year, target_row = .row_idx)]
  setkey(row_lookup, id, year)

  # Expand: join data rows to cell_edges on source_id == id
  # This gives us (source_row, source_id, year, neighbor_id)
  source_info <- data_dt[, .(source_row = .row_idx, source_id = id, year)]
  setkey(cell_edges, source_id)
  setkey(source_info, source_id)

  # Merge: for each source row, get its neighbor cell IDs
  expanded <- cell_edges[source_info, on = .(source_id), allow.cartesian = TRUE,
                         nomatch = NULL]
  # expanded has columns: source_id, neighbor_id, source_row, year

  # Step 3: Resolve neighbor_id + year -> target_row
  expanded_resolved <- row_lookup[expanded,
                                   on = .(id = neighbor_id, year = year),
                                   nomatch = NULL]
  # This gives us: id (=neighbor_id), year, target_row, source_row, source_id

  # Return only the mapping we need
  result <- expanded_resolved[, .(source_row, target_row)]
  setkey(result, source_row)
  return(result)
}


# ==============================================================
# OPTIMIZED: compute_and_add_all_neighbor_features
# Computes max, min, mean for all neighbor source variables
# in vectorized grouped aggregations.
# ==============================================================

compute_and_add_all_neighbor_features <- function(data_dt, edge_table, neighbor_source_vars) {
  # edge_table: data.table with (source_row, target_row)
  # For each variable, join target values, then group by source_row

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)

    # Attach the variable values at the target rows
    edges_with_vals <- edge_table[, .(source_row, target_row)]
    edges_with_vals[, val := data_dt[[var_name]][target_row]]

    # Remove NAs
    edges_with_vals <- edges_with_vals[!is.na(val)]

    # Grouped aggregation
    agg <- edges_with_vals[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = source_row]

    # Initialize columns with NA
    max_col  <- paste0("nb_max_", var_name)
    min_col  <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)

    data_dt[, (max_col)  := NA_real_]
    data_dt[, (min_col)  := NA_real_]
    data_dt[, (mean_col) := NA_real_]

    # Assign aggregated values to the correct rows
    data_dt[agg$source_row, (max_col)  := agg$nb_max]
    data_dt[agg$source_row, (min_col)  := agg$nb_min]
    data_dt[agg$source_row, (mean_col) := agg$nb_mean]
  }

  return(data_dt)
}


# ==============================================================
# MAIN PIPELINE (drop-in replacement)
# ==============================================================

# Convert to data.table and add row index
cell_data_dt <- as.data.table(cell_data)
cell_data_dt[, .row_idx := .I]

# Build the vectorized edge table (replaces build_neighbor_lookup)
message("Building neighbor edge table...")
edge_table <- build_neighbor_edge_table(cell_data_dt, id_order, rook_neighbors_unique)
message("Edge table built: ", nrow(edge_table), " directed edges")

# Compute all neighbor features in vectorized fashion
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data_dt <- compute_and_add_all_neighbor_features(
  cell_data_dt, edge_table, neighbor_source_vars
)

# Clean up helper column
cell_data_dt[, .row_idx := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_data_dt)

# ----------------------------------------------------------
# Random Forest prediction (UNCHANGED â€” model is preserved)
# ----------------------------------------------------------
# predictions <- predict(rf_model, newdata = cell_data)
# cell_data$predicted <- predictions$predictions  # or predictions, depending on package
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `lapply` iterations with `paste()` + named vector lookup | One vectorized `data.table` join (integer keys, no strings) |
| **Neighbor stats** | 5 Ã— 6.46M `lapply` iterations with per-row subsetting | 5 Ã— one grouped `data.table` aggregation |
| **Estimated runtime** | 86+ hours | Minutes (typically 5â€“15 min on 16 GB laptop) |
| **RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical results** | Same max/min/mean estimands | Identical âœ“ |

The key insight: the bottleneck was never the Random Forest `predict()` call (which is compiled C/C++ code executing once). It was **millions of R-interpreted loop iterations** performing string concatenation and named-vector lookups. Replacing these with vectorized `data.table` joins and grouped aggregations eliminates the bottleneck entirely.
END_RESPONSE>>>
