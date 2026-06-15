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
run_id: Anthropic_F08_T03_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F08_Anthropic_temp_03_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup of length ~6.46 million, where each entry maps a cell-year row to its neighbor cell-year rows. This is enormously wasteful because **the neighbor topology is purely spatial and static across all 28 years**. The rook-neighbor relationships among the 344,208 cells never change â€” only the variable values attached to each cell change year-to-year. Yet the current code:

1. **Inflates the neighbor structure by 28Ã—**: Instead of storing ~1.37M directed neighbor pairs once (at the cell level), it replicates them into ~6.46M row-level entries, each containing year-specific row indices. The `build_neighbor_lookup` function does an expensive `paste`/`match` operation for every single cell-year row.

2. **Uses slow R-level `lapply` over millions of rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate with `lapply` over ~6.46M elements, which is extremely slow in interpreted R.

3. **Repeats the full scan per variable**: The neighbor lookup is reused, but `compute_neighbor_stats` still loops over 6.46M rows per variable (Ã—5 variables = ~32.3M iterations).

The fundamental insight: **separate the static topology (cell-to-cell neighbors) from the dynamic data (year-varying values)**. Build the neighbor index once at the cell level (344K entries), then for each year, slice the data, do a vectorized lookup, and compute stats.

## Optimization Strategy

1. **Build a cell-level neighbor lookup once** â€” a simple list of length 344,208 where each element contains integer indices into `id_order` (the cell-level position). This is just `rook_neighbors_unique` itself (an `nb` object already has this structure).

2. **Process year-by-year**: For each year, subset the data, create a cell-index â†’ row-index mapping, and use the static cell-level neighbor list to gather neighbor values.

3. **Vectorize the aggregation**: Convert the neighbor list into a flat vector of (focal_cell, neighbor_cell) pairs. Use vectorized indexing and `data.table` grouped aggregation (`max`, `min`, `mean`) to avoid any R-level `lapply` over millions of rows.

4. **Process all 5 variables simultaneously** per year to avoid redundant subsetting.

This reduces the problem from ~6.46M slow R-level iterations to 28 year-slices Ã— vectorized operations over ~344K cells, which should run in minutes rather than days.

## Working R Code

```r
library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  # ---------------------------------------------------------------
  # STEP 1: Build a STATIC cell-level edge list (once, ~1.37M rows)
  # ---------------------------------------------------------------
  # rook_neighbors_unique is an nb object: a list of integer vectors

  # where element i contains the indices (into id_order) of cell i's neighbors.
  
  n_cells <- length(id_order)
  
  # Build flat edge list: focal_cell_pos -> neighbor_cell_pos
  focal_pos <- rep(seq_len(n_cells),
                   times = lengths(rook_neighbors_unique))
  neighbor_pos <- unlist(rook_neighbors_unique, use.names = FALSE)
  
  # Remove any 0-entries (spdep uses 0 for "no neighbors")
  valid <- neighbor_pos > 0L
  focal_pos    <- focal_pos[valid]
  neighbor_pos <- neighbor_pos[valid]
  
  # Map cell positions to cell IDs
  focal_ids    <- id_order[focal_pos]
  neighbor_ids <- id_order[neighbor_pos]
  
  # Static edge table (cell-level, not row-level)
  edges <- data.table(focal_id    = focal_ids,
                      neighbor_id = neighbor_ids)
  
  # ---------------------------------------------------------------
  # STEP 2: Convert cell_data to data.table if needed

  # ---------------------------------------------------------------
  was_df <- !is.data.table(cell_data)
  if (was_df) {
    cell_data <- as.data.table(cell_data)
  }
  
  # Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    if (!col_max  %in% names(cell_data)) cell_data[, (col_max)  := NA_real_]
    if (!col_min  %in% names(cell_data)) cell_data[, (col_min)  := NA_real_]
    if (!col_mean %in% names(cell_data)) cell_data[, (col_mean) := NA_real_]
  }
  
  # Create a row-index column for direct assignment
  cell_data[, .row_idx := .I]
  
  # ---------------------------------------------------------------
  # STEP 3: Process year-by-year (28 iterations, fully vectorized)
  # ---------------------------------------------------------------
  years <- sort(unique(cell_data$year))
  
  for (yr in years) {
    # Subset rows for this year
    yr_idx <- cell_data[year == yr, .row_idx]
    yr_sub <- cell_data[yr_idx, c("id", neighbor_source_vars), with = FALSE]
    
    # Map cell id -> position within this year's subset
    # (so we can look up variable values by cell id)
    id_to_yr_pos <- setNames(seq_len(nrow(yr_sub)),
                             as.character(yr_sub$id))
    
    # Map edges to this year's subset positions
    focal_yr_pos    <- id_to_yr_pos[as.character(edges$focal_id)]
    neighbor_yr_pos <- id_to_yr_pos[as.character(edges$neighbor_id)]
    
    # Keep only edges where both focal and neighbor exist this year
    valid_edge <- !is.na(focal_yr_pos) & !is.na(neighbor_yr_pos)
    f_pos <- focal_yr_pos[valid_edge]
    n_pos <- neighbor_yr_pos[valid_edge]
    
    # Build a data.table of neighbor values for all variables at once
    # Columns: focal_pos, var1_val, var2_val, ...
    neighbor_vals_dt <- data.table(focal_pos = f_pos)
    
    for (var_name in neighbor_source_vars) {
      vals_vec <- yr_sub[[var_name]]
      neighbor_vals_dt[, (var_name) := vals_vec[n_pos]]
    }
    
    # Grouped aggregation: max, min, mean per focal cell, all vars
    agg_exprs <- list()
    for (var_name in neighbor_source_vars) {
      col_max  <- paste0("neighbor_max_",  var_name)
      col_min  <- paste0("neighbor_min_",  var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      agg_exprs[[col_max]]  <- call("max",  as.name(var_name), na.rm = TRUE)
      agg_exprs[[col_min]]  <- call("min",  as.name(var_name), na.rm = TRUE)
      agg_exprs[[col_mean]] <- call("mean", as.name(var_name), na.rm = TRUE)
    }
    
    agg_result <- neighbor_vals_dt[, eval(as.call(c(as.name("list"),
                                                     agg_exprs))),
                                    by = focal_pos]
    
    # Handle cells with no valid neighbors (they won't appear in agg_result)
    # They keep NA from pre-allocation.
    
    # Replace -Inf/Inf from max/min of empty-after-na.rm with NA
    for (var_name in neighbor_source_vars) {
      col_max <- paste0("neighbor_max_", var_name)
      col_min <- paste0("neighbor_min_", var_name)
      agg_result[is.infinite(get(col_max)), (col_max) := NA_real_]
      agg_result[is.infinite(get(col_min)), (col_min) := NA_real_]
    }
    
    # Map aggregated results back to cell_data rows
    # agg_result$focal_pos is the position within yr_sub;
    # yr_idx[focal_pos] gives the row in cell_data.
    target_rows <- yr_idx[agg_result$focal_pos]
    
    out_cols <- setdiff(names(agg_result), "focal_pos")
    for (col in out_cols) {
      set(cell_data, i = target_rows, j = col, value = agg_result[[col]])
    }
  }
  
  # Clean up helper column
  cell_data[, .row_idx := NULL]
  
  if (was_df) {
    cell_data <- as.data.frame(cell_data)
  }
  
  return(cell_data)
}
```

### Drop-in replacement for the outer loop

```r
# BEFORE (slow, ~86+ hours):
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# AFTER (fast, estimated ~5-15 minutes):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# Then proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same numerical estimand** | For each cell-year row, we compute `max`, `min`, `mean` of the same rook-neighbor values for the same 5 variables. The grouped `data.table` aggregation is numerically identical to the original `lapply` approach. |
| **Trained RF model untouched** | No model retraining occurs. The output columns have the same names (`neighbor_max_ntl`, `neighbor_min_ntl`, `neighbor_mean_ntl`, etc.), so `predict(rf_model, newdata = cell_data)` works identically. |
| **NA handling preserved** | `na.rm = TRUE` in aggregation mirrors the original `neighbor_vals[!is.na(neighbor_vals)]` filter. Cells with zero valid neighbors get `NA` (matching the original `return(c(NA, NA, NA))`). |

## Performance Summary

| Aspect | Original | Redesigned |
|---|---|---|
| Neighbor lookup size | ~6.46M row-level entries | ~1.37M cell-level edges (static) |
| R-level loop iterations | ~32.3M (`lapply` calls) | 28 (one per year) |
| Core computation | Interpreted R loops | Vectorized `data.table` grouped aggregation |
| Estimated runtime | 86+ hours | ~5â€“15 minutes |
| Peak memory | Moderate (large list) | Moderate (flat edge table + year slice) |
END_RESPONSE>>>
