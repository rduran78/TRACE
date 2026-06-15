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
run_id: Anthropic_F04_T10_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is squarely in `build_neighbor_lookup`, which uses an `lapply` loop over **~6.46 million rows**, performing per-row string pasting (`paste()`), character-keyed hash lookups into `idx_lookup`, and NA filtering. String allocation and named-vector lookups in R are slow at this scale. The `compute_neighbor_stats` function then loops over the same 6.46M rows again, extracting and summarizing neighbor values per row â€” repeated for each of the 5 variables (totaling ~32.3M summary operations).

**Key cost centers:**

1. **`build_neighbor_lookup`**: ~6.46M iterations, each doing `paste()` + named-vector subsetting. Named vector lookup in R is O(n) probing of a hashed environment per call, and `paste()` allocates a new string each time. Estimated: this single function accounts for the vast majority of the 86+ hour runtime.
2. **`compute_neighbor_stats`**: 5 Ã— 6.46M `lapply` iterations with per-row subsetting and `mean`/`min`/`max`. The `do.call(rbind, ...)` on a 6.46M-element list of 3-vectors is also expensive.
3. **String-keyed join logic**: The entire design maps `(id, year)` â†’ row via string concatenation and named-vector lookup. This can be replaced with integer arithmetic and `data.table` joins, eliminating all string operations.

## Optimization Strategy

1. **Replace string-keyed lookup with integer-keyed `data.table` join.** Map each `(id, year)` pair to its row index via a keyed `data.table`, then expand the neighbor list into a flat edge table (source_row, neighbor_row) using vectorized operations. This eliminates the per-row `lapply` in `build_neighbor_lookup` entirely.

2. **Vectorize `compute_neighbor_stats` using `data.table` grouped aggregation.** With a flat edge table `(source_row, neighbor_row)`, join the variable values, then compute `max`, `min`, `mean` grouped by `source_row` â€” all in one vectorized `data.table` operation per variable.

3. **Avoid `do.call(rbind, ...)` on millions of small vectors.** The `data.table` approach returns a single aggregated table directly.

4. **Memory consideration**: The flat edge table will have approximately `6.46M Ã— avg_neighbors` rows. With ~1.37M directed neighbor relationships per year Ã— 28 years â‰ˆ ~38.4M rows of `(source_row, neighbor_row)` integer pairs â‰ˆ ~307 MB. This fits comfortably in 16 GB.

## Optimized Working R Code

```r
library(data.table)

#' Build a flat edge table mapping each row to its neighbor rows.
#' Returns a data.table with columns: source_row, neighbor_row
build_neighbor_edge_table <- function(data_dt, id_order, neighbors) {
  # data_dt must be a data.table with columns: id, year (and others)
  # id_order: vector of cell IDs in the same order as the nb object
  # neighbors: spdep nb object (list of integer index vectors into id_order)

  n_rows <- nrow(data_dt)

  # Step 1: Map cell id -> position in id_order (integer)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Step 2: Build row index keyed by (id, year) using data.table
  data_dt[, row_idx := .I]
  row_lookup <- data_dt[, .(id, year, row_idx)]
  setkey(row_lookup, id, year)

  # Step 3: For each cell in id_order, expand its neighbor cell IDs

  # Build a cell-level edge list: (cell_id, neighbor_cell_id)
  # neighbors[[j]] gives integer indices into id_order for cell id_order[j]
  n_cells <- length(id_order)

  # Vectorized expansion of the nb object
  source_lengths <- lengths(neighbors)
  source_cell_idx <- rep(seq_len(n_cells), times = source_lengths)
  neighbor_cell_idx <- unlist(neighbors, use.names = FALSE)

  # Remove 0-neighbor entries (spdep uses integer(0) for islands)
  valid <- neighbor_cell_idx > 0L
  source_cell_idx <- source_cell_idx[valid]
  neighbor_cell_idx <- neighbor_cell_idx[valid]

  # Map indices back to actual cell IDs
  cell_edges <- data.table(
    source_id   = id_order[source_cell_idx],
    neighbor_id = id_order[neighbor_cell_idx]
  )
  rm(source_cell_idx, neighbor_cell_idx, valid)

  # Step 4: Get unique years
  years <- sort(unique(data_dt$year))

  # Step 5: Cross-join cell edges with years, then join to row indices

  # This produces the full (source_row, neighbor_row) edge table
  cell_edges_yr <- cell_edges[, CJ(year = years), by = .(source_id, neighbor_id)]

  # Join source side
  setnames(cell_edges_yr, "source_id", "id")
  cell_edges_yr <- row_lookup[cell_edges_yr, on = .(id, year), nomatch = 0L]
  setnames(cell_edges_yr, c("row_idx", "id"), c("source_row", "source_id"))

  # Join neighbor side
  setnames(cell_edges_yr, "neighbor_id", "id")
  cell_edges_yr <- row_lookup[cell_edges_yr, on = .(id, year), nomatch = 0L]
  setnames(cell_edges_yr, c("row_idx", "id"), c("neighbor_row", "neighbor_id"))

  # Return only what we need
  edge_table <- cell_edges_yr[, .(source_row, neighbor_row)]
  setkey(edge_table, source_row)

  # Clean up temporary column
  data_dt[, row_idx := NULL]

  return(edge_table)
}


#' Compute neighbor max, min, mean for a variable using the edge table.
#' Returns a data.table with columns: source_row, nb_max, nb_min, nb_mean
compute_neighbor_stats_fast <- function(data_dt, edge_table, var_name) {
  vals <- data_dt[[var_name]]

  # Attach variable value for each neighbor row
  work <- edge_table[, .(source_row, val = vals[neighbor_row])]

  # Remove NAs in the variable
  work <- work[!is.na(val)]

  # Grouped aggregation â€” single vectorized pass
  stats <- work[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = source_row]

  return(stats)
}


#' Compute and attach neighbor features for one variable to the dataset.
compute_and_add_neighbor_features_fast <- function(data_dt, var_name, edge_table) {
  stats <- compute_neighbor_stats_fast(data_dt, edge_table, var_name)

  # Prepare columns with proper names matching original pipeline output
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)

  # Initialize with NA, then fill matched rows
  n <- nrow(data_dt)
  data_dt[, (max_col)  := NA_real_]
  data_dt[, (min_col)  := NA_real_]
  data_dt[, (mean_col) := NA_real_]

  data_dt[stats$source_row, (max_col)  := stats$nb_max]
  data_dt[stats$source_row, (min_col)  := stats$nb_min]
  data_dt[stats$source_row, (mean_col) := stats$nb_mean]

  return(data_dt)
}


# ============================================================
# MAIN EXECUTION
# ============================================================

# Convert to data.table if not already (non-destructive to original data)
cell_data_dt <- as.data.table(cell_data)

# Build the edge table ONCE (replaces build_neighbor_lookup)
message("Building neighbor edge table...")
edge_table <- build_neighbor_edge_table(cell_data_dt, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %s rows", format(nrow(edge_table), big.mark = ",")))

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  cell_data_dt <- compute_and_add_neighbor_features_fast(cell_data_dt, var_name, edge_table)
}

# Convert back to data.frame if downstream code (e.g., predict.randomForest) expects one
cell_data <- as.data.frame(cell_data_dt)

message("Neighbor feature construction complete.")

# Prediction with existing trained RF model (unchanged)
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` computed on exactly the same neighbor sets with the same NA-removal logic. Results are identical to the original. |
| **Trained RF model** | Untouched. Only the feature construction step is optimized. Column names match the original output. |
| **Memory (16 GB laptop)** | Edge table â‰ˆ ~38M rows Ã— 2 integer cols â‰ˆ 307 MB. `data.table` intermediate for one variable â‰ˆ ~310 MB. Peak total â‰ˆ ~2â€“3 GB for this step, well within budget. |
| **Expected speedup** | The original 86+ hours is dominated by ~32M R-level `lapply` iterations with string operations. The replacement uses zero string operations, zero per-row R loops, and `data.table`'s C-level grouped aggregation. **Expected runtime: 5â€“15 minutes** on the same hardware. |

### Optional Further Gains

- **Parallel by variable**: The 5 `compute_neighbor_stats_fast` calls are independent and can be run via `future.apply::future_lapply` if memory permits (~1.5 GB per worker).
- **In-place column assignment**: The code already uses `:=` for zero-copy column addition.
- **Edge table caching**: Serialize `edge_table` to disk with `qs::qsave()` or `fst::write_fst()` so rebuilds are unnecessary across sessions.
END_RESPONSE>>>
