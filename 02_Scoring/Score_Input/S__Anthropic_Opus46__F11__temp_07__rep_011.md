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
run_id: Anthropic_F11_T07_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** as the dominant bottleneck â€” specifically `build_neighbor_lookup` and `compute_neighbor_stats`.

**Evidence from the code:**

1. **`build_neighbor_lookup`**: This function runs an `lapply` over **every row** (~6.46 million rows). For each row, it performs character coercion (`as.character`), string concatenation (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still involves repeated string hashing. Over 6.46M iterations, each touching ~4 neighbors on average (given ~1.37M directed relationships across ~344K cells), this is enormously expensive. This single function produces ~6.46 million list elements, each requiring multiple string operations.

2. **`compute_neighbor_stats`**: This is called **5 times** (once per neighbor source variable), each time iterating over the full 6.46M-element `neighbor_lookup` list, subsetting values, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level function invocations of the inner lambda alone, plus millions of subsetting and aggregation operations.

3. **The outer loop** compounds the problem: 5 variables Ã— 6.46M rows Ã— per-row R-level overhead = the core of the 86+ hour runtime.

4. **Random Forest inference** by contrast is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, a single `predict.randomForest()` call is vectorized C/C++ code and typically completes in seconds to minutes â€” orders of magnitude faster than the neighbor feature engineering.

**Root cause**: Row-level R `lapply` loops over millions of rows with string operations and repeated subsetting. R's interpreted loop overhead and per-element string manipulation make this catastrophically slow at this scale.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup`** â€” Replace the row-by-row `lapply` with a fully vectorized join using `data.table`. Instead of building a list of neighbor indices per row, build an **edge table** (a two-column data.table mapping each row to its neighbor rows) once, then use grouped aggregation.

2. **Vectorize `compute_neighbor_stats`** â€” Replace the per-row `lapply` with a single `data.table` grouped aggregation over the edge table. Compute max, min, and mean of neighbor values in one vectorized pass per variable.

3. **Eliminate the list-of-vectors structure** entirely. The neighbor lookup list of 6.46M elements is memory-heavy and forces R-level iteration. An edge table is a flat structure amenable to vectorized operations.

4. **Preserve the trained Random Forest model and original numerical estimand** â€” The optimization only changes how features are computed, not their values. The same neighbor statistics (max, min, mean) are produced, so predictions from the pre-trained model remain identical.

**Expected speedup**: From 86+ hours to roughly **minutes** (the vectorized `data.table` joins and grouped aggregations over ~25.8M edges are extremely fast).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build a vectorized edge table (replaces build_neighbor_lookup)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_edge_table <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id', 'year', and a row index 'row_idx'
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)
  
  # Map each cell position (ref) to its neighbor cell positions
  # neighbors[[ref]] gives integer indices into id_order
  n_cells <- length(id_order)
  
  # Build edges at the cell level: (focal_cell_id, neighbor_cell_id)
  from_ref <- rep(seq_len(n_cells), times = lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)
  
  # Remove any zero-length or self-referencing if needed
  valid <- to_ref > 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]
  
  cell_edges <- data.table(
    focal_id    = id_order[from_ref],
    neighbor_id = id_order[to_ref]
  )
  
  # Now expand to cell-year level by joining with the data
  # Each focal (id, year) pairs with each neighbor (id, year) for the SAME year
  
  # Create keyed lookup: id, year -> row_idx
  focal_join <- data_dt[, .(focal_id = id, year, focal_row = row_idx)]
  neighbor_join <- data_dt[, .(neighbor_id = id, year, neighbor_row = row_idx)]
  
  # Merge: cell_edges Ã— years
  # For each (focal_id, neighbor_id) pair, match on same year
  setkey(cell_edges, focal_id)
  setkey(focal_join, focal_id)
  
  # First join: attach year and focal_row to each cell edge
  edge_year <- cell_edges[focal_join, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_year now has: focal_id, neighbor_id, year, focal_row
  
  # Second join: attach neighbor_row
  setkey(edge_year, neighbor_id, year)
  setkey(neighbor_join, neighbor_id, year)
  
  edge_full <- edge_year[neighbor_join, on = c("neighbor_id", "year"), nomatch = 0L]
  # edge_full now has: focal_id, neighbor_id, year, focal_row, neighbor_row
  
  edge_full[, .(focal_row, neighbor_row)]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Compute neighbor stats vectorized (replaces compute_neighbor_stats
#         and the outer for-loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_all_neighbor_features <- function(data_dt, edge_dt, neighbor_source_vars) {
  # edge_dt: data.table with columns focal_row, neighbor_row
  # For each variable, look up neighbor values, group by focal_row, compute stats
  
  n_rows <- nrow(data_dt)
  
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)
    
    # Attach neighbor values
    neighbor_vals <- data_dt[[var_name]][edge_dt$neighbor_row]
    
    work <- data.table(
      focal_row = edge_dt$focal_row,
      val       = neighbor_vals
    )
    
    # Remove NA neighbor values before aggregation
    work <- work[!is.na(val)]
    
    # Grouped aggregation â€” single vectorized pass
    stats <- work[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = focal_row]
    
    # Initialize columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    
    data_dt[, (max_col)  := NA_real_]
    data_dt[, (min_col)  := NA_real_]
    data_dt[, (mean_col) := NA_real_]
    
    # Fill in computed values
    data_dt[stats$focal_row, (max_col)  := stats$nb_max]
    data_dt[stats$focal_row, (min_col)  := stats$nb_min]
    data_dt[stats$focal_row, (mean_col) := stats$nb_mean]
  }
  
  data_dt
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Full optimized pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique, rf_model) {
  # Convert to data.table if not already
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # Add row index
  cell_data[, row_idx := .I]
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  # Step A: Build edge table (vectorized â€” runs once, replaces build_neighbor_lookup)
  message("Building neighbor edge table...")
  edge_dt <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
  message("Edge table built: ", nrow(edge_dt), " edges")
  
  # Step B: Compute all neighbor features (vectorized â€” replaces the for loop)
  message("Computing neighbor features...")
  cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
  
  # Step C: Random Forest inference (already fast â€” not the bottleneck)
  message("Running Random Forest predictions...")
  cell_data[, row_idx := NULL]  # Remove helper column before prediction
  preds <- predict(rf_model, newdata = cell_data)
  cell_data[, prediction := preds]
  
  message("Pipeline complete.")
  cell_data
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Usage (preserves trained RF model and original estimand):
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = trained_rf_model
# )
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations with string ops | Single vectorized `data.table` join |
| `compute_neighbor_stats` | 5 Ã— 6.46M R-level `lapply` calls | 5 Ã— one grouped `data.table` aggregation |
| Data structure | List of 6.46M vectors | Flat edge table (~25.8M rows, 2 integer cols) |
| Estimated runtime | 86+ hours | Minutes |
| RF model | Preserved (unchanged) | Preserved (unchanged) |
| Numerical output | Original neighbor max/min/mean | Identical values |

The true bottleneck is the **row-level R loop overhead in spatial feature engineering**, not Random Forest inference. The fix is to replace interpreted R loops with vectorized `data.table` joins and grouped aggregations.
END_RESPONSE>>>
