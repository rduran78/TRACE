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
run_id: Anthropic_F04_T03_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Looks those keys up in another named character vector (`idx_lookup`).

String construction (`paste`) and named-vector lookup (which is O(n) hash probing on character keys) repeated ~6.46M Ã— ~4 neighbors â‰ˆ 25.8M string operations is extremely expensive in interpreted R. The secondary bottleneck is `compute_neighbor_stats`, which uses an `lapply` over 6.46M elements calling `max`/`min`/`mean` on small vectors, then `do.call(rbind, ...)` on a 6.46M-element list â€” also slow.

**Root causes, ranked:**

1. **Row-level `lapply` with string key construction and lookup** in `build_neighbor_lookup` â€” O(N Ã— k) string operations in pure R.
2. **Row-level `lapply` with per-row summary stats** in `compute_neighbor_stats` â€” interpretive overhead on millions of tiny vectors.
3. **`do.call(rbind, list_of_6.46M_vectors)`** â€” known to be very slow for large lists.

## Optimization Strategy

1. **Vectorize the neighbor lookup entirely using `data.table` equi-joins.** Instead of building a per-row list, create an edge table `(row_i, neighbor_row_j)` via a merge on `(neighbor_cell_id, year)`. This replaces millions of `paste` + named-vector lookups with a single keyed join.

2. **Vectorize the neighbor stats using `data.table` grouped aggregation.** Once we have the edge table with the neighbor's variable value joined in, compute `max`, `min`, `mean` grouped by the focal row index â€” a single vectorized `data.table` operation.

3. **Reuse the edge table across all 5 variables.** The spatial topology doesn't change per variable, so the edge table is built once.

This reduces estimated runtime from 86+ hours to roughly **minutes** (the join is O(N log N) and the grouped aggregation is highly optimized in `data.table`).

## Optimized Working R Code

```r
library(data.table)

#' Build a data.table edge list mapping each focal row to its neighbor rows.
#' This replaces build_neighbor_lookup entirely.
#'
#' @param cell_data   data.frame/data.table with columns: id, year, and predictor vars
#' @param id_order    integer vector of cell IDs in the order matching the nb object
#' @param neighbors   spdep nb object (list of integer index vectors into id_order)
#' @return data.table with columns: focal_row, neighbor_row
build_neighbor_edge_table <- function(cell_data, id_order, neighbors) {
  
  # --- Step 1: Build a cell-level edge list (focal_cell_id -> neighbor_cell_id) ---
  # This is only ~1.37M rows (directed rook edges), very fast.
  n_cells <- length(id_order)
  focal_idx <- rep(seq_len(n_cells), lengths(neighbors))
  neighbor_idx <- unlist(neighbors)
  
  cell_edges <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
  rm(focal_idx, neighbor_idx)
  
  # --- Step 2: Convert cell_data to data.table and add a row index ---
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]
  
  # --- Step 3: Join to expand edges across years ---
  # For each focal row (id, year), find the neighbor rows that share the same year.
  # We do this with two keyed joins rather than 6.46M paste operations.
  
  # Focal side: attach focal row index to edges via (focal_id, year)
  # First, get the (id, year, row_idx) mapping
  id_year_map <- dt[, .(id, year, row_idx)]
  
  # Join cell_edges with focal rows: for each edge, expand across all years of the focal cell
  setkey(id_year_map, id)
  setkey(cell_edges, focal_id)
  
  # Merge: each cell-level edge Ã— each year the focal cell appears in
  edges_with_year <- cell_edges[id_year_map, 
                                 .(neighbor_id, year, focal_row = row_idx),
                                 on = .(focal_id = id),
                                 nomatch = 0L,
                                 allow.cartesian = TRUE]
  rm(cell_edges)
  
  # --- Step 4: Resolve neighbor rows by joining on (neighbor_id, year) ---
  setnames(id_year_map, c("id", "year", "neighbor_row"))
  setkey(id_year_map, id, year)
  setkey(edges_with_year, neighbor_id, year)
  
  edge_table <- edges_with_year[id_year_map,
                                 .(focal_row, neighbor_row),
                                 on = .(neighbor_id = id, year = year),
                                 nomatch = 0L]
  rm(edges_with_year, id_year_map)
  
  setkey(edge_table, focal_row)
  return(edge_table)
}


#' Compute neighbor max, min, mean for a variable and add columns to cell_data.
#' Replaces compute_neighbor_stats + compute_and_add_neighbor_features.
#'
#' @param dt          data.table version of cell_data (modified in place)
#' @param edge_table  output of build_neighbor_edge_table
#' @param var_name    character: name of the source variable
#' @return NULL (modifies dt by reference)
compute_and_add_neighbor_features_fast <- function(dt, edge_table, var_name) {
  
  # Attach the neighbor's variable value to each edge
  val_vec <- dt[[var_name]]
  edges <- copy(edge_table)
  edges[, val := val_vec[neighbor_row]]
  
  # Remove NA neighbor values
  edges <- edges[!is.na(val)]
  
  # Grouped aggregation â€” single vectorized pass
  stats <- edges[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = focal_row]
  
  # Initialize new columns with NA
  col_max  <- paste0("max_neighbor_",  var_name)
  col_min  <- paste0("min_neighbor_",  var_name)
  col_mean <- paste0("mean_neighbor_", var_name)
  
  set(dt, j = col_max,  value = NA_real_)
  set(dt, j = col_min,  value = NA_real_)
  set(dt, j = col_mean, value = NA_real_)
  
  # Fill in computed values at the correct rows
  set(dt, i = stats$focal_row, j = col_max,  value = stats$nb_max)
  set(dt, i = stats$focal_row, j = col_min,  value = stats$nb_min)
  set(dt, i = stats$focal_row, j = col_mean, value = stats$nb_mean)
  
  invisible(NULL)
}


# ============================================================
# MAIN EXECUTION â€” drop-in replacement for the original outer loop
# ============================================================

# Convert to data.table once
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]

# Build the edge table ONCE (replaces build_neighbor_lookup)
message("Building neighbor edge table...")
edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %s rows", format(nrow(edge_table), big.mark = ",")))

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor features for: %s", var_name))
  compute_and_add_neighbor_features_fast(cell_dt, edge_table, var_name)
}

# Remove helper column and convert back if needed
cell_dt[, row_idx := NULL]
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is untouched.
# Prediction proceeds as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per focal row. The edge table encodes the identical `(focal_row, neighbor_row)` pairs that the original `neighbor_lookup` list would contain. |
| **Trained RF model** | The model object is never touched. The output columns have the same names and semantics, so `predict()` works identically. |
| **NA handling** | Rows with zero valid neighbors get `NA` for all three stats, matching the original `c(NA, NA, NA)` return. |

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (6.46M `paste` + named-vector lookups) | ~1â€“3 min (one keyed `data.table` join) |
| Stats for 1 variable | ~hours (`lapply` + `do.call(rbind)`) | ~20â€“40 sec (grouped `data.table` aggregation) |
| Stats for 5 variables | ~days | ~2â€“4 min total |
| **Total** | **86+ hours** | **~5â€“10 minutes** |

Peak memory for the edge table: ~6.46M rows Ã— 4 neighbors Ã— 2 integer columns â‰ˆ ~200 MB, well within the 16 GB budget.
END_RESPONSE>>>
