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
run_id: Anthropic_F11_T10_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that **Random Forest inference** is the bottleneck is **incorrect**. The code evidence points clearly to a different bottleneck: **the spatial neighbor feature engineering stage**, specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's the evidence:

1. **`build_neighbor_lookup()`** uses `lapply` over **~6.46 million rows**, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`) for every single row. Named-vector lookup in R is **O(n)** hash-probe per call, but doing it millions of times with millions of keys creates enormous overhead. This function alone likely accounts for the vast majority of the 86+ hour runtime.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million rows with `lapply`, subsetting vectors, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` on a 6.46-million-element list is also costly.

3. **Random Forest inference** is a single call to `predict()` on a pre-trained model. Even with 6.46M rows Ã— 110 predictors, modern `randomForest` or `ranger` predict calls complete in seconds to minutes â€” nowhere near 86 hours.

**The bottleneck is the row-by-row R-level loop constructing neighbor lookups and computing neighbor summary statistics over 6.46 million rows Ã— 5 variables.**

---

## Optimization Strategy

1. **Replace the character-key named-vector lookup in `build_neighbor_lookup()`** with a vectorized `data.table` join. Instead of iterating row-by-row with `lapply`, construct an edge-list of (focal_row, neighbor_id, year) and batch-join to get neighbor row indices.

2. **Replace the row-by-row `lapply` in `compute_neighbor_stats()`** with a grouped `data.table` aggregation over the edge-list: group by focal row index, compute `max`, `min`, `mean` of the neighbor values in one vectorized pass.

3. **This eliminates ~32 million R-level function calls** (6.46M Ã— 5 vars) and replaces them with 5 vectorized grouped aggregations.

Expected speedup: from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED PIPELINE: Neighbor Feature Engineering
# ==============================================================================

build_neighbor_edge_list <- function(data_dt, id_order, neighbors) {

  # Map each grid-cell ID to its position in id_order
  # neighbors[[k]] gives the neighbor indices (into id_order) for id_order[k]
  
  n_ids <- length(id_order)
  
  # Build a complete directed edge list: focal_id -> neighbor_id
  focal_ids <- rep(id_order, times = lengths(neighbors))
  neighbor_indices <- unlist(neighbors, use.names = FALSE)
  neighbor_ids <- id_order[neighbor_indices]
  
  edge_dt <- data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)
  
  # data_dt must have columns: id, year, and a row index
  data_dt[, .row_idx := .I]
  
  # For each (focal_id, year) pair, we need the focal row index
  # For each (neighbor_id, year) pair, we need the neighbor row index
  # Strategy: cross the edge list with years by joining on id
  
  # Step 1: Join edge list to data on focal_id = id to get (focal_row_idx, neighbor_id, year)
  focal_key <- data_dt[, .(focal_row_idx = .row_idx, focal_id = id, year)]
  setkey(focal_key, focal_id)
  setkey(edge_dt, focal_id)
  
  # This gives every (focal_row, year, neighbor_id) combination
  expanded <- edge_dt[focal_key, on = "focal_id", allow.cartesian = TRUE, nomatch = NULL]
  # Columns: focal_id, neighbor_id, focal_row_idx, year
  
  # Step 2: Join to data again to get neighbor_row_idx for (neighbor_id, year)
  neighbor_key <- data_dt[, .(neighbor_row_idx = .row_idx, neighbor_id = id, year)]
  setkey(neighbor_key, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  
  result <- neighbor_key[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # Keep only matched neighbors (neighbor exists in that year)
  result <- result[!is.na(neighbor_row_idx)]
  
  result[, .(focal_row_idx, neighbor_row_idx)]
}

compute_and_add_all_neighbor_features <- function(cell_data, neighbor_source_vars,
                                                   id_order, neighbors) {
  data_dt <- as.data.table(cell_data)
  
  message("Building neighbor edge list (vectorized)...")
  t0 <- proc.time()
  edge_list <- build_neighbor_edge_list(data_dt, id_order, neighbors)
  message(sprintf("  Edge list built: %d edges in %.1f seconds",
                  nrow(edge_list), (proc.time() - t0)[3]))
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for '%s'...", var_name))
    t1 <- proc.time()
    
    # Attach the neighbor's value to each edge
    vals <- data_dt[[var_name]]
    edge_list[, nval := vals[neighbor_row_idx]]
    
    # Grouped aggregation: max, min, mean per focal row
    stats <- edge_list[!is.na(nval),
                       .(nb_max  = max(nval),
                         nb_min  = min(nval),
                         nb_mean = mean(nval)),
                       by = focal_row_idx]
    
    # Initialize columns with NA
    max_col  <- paste0("nb_max_", var_name)
    min_col  <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)
    
    data_dt[, (max_col)  := NA_real_]
    data_dt[, (min_col)  := NA_real_]
    data_dt[, (mean_col) := NA_real_]
    
    # Fill in computed values
    data_dt[stats$focal_row_idx, (max_col)  := stats$nb_max]
    data_dt[stats$focal_row_idx, (min_col)  := stats$nb_min]
    data_dt[stats$focal_row_idx, (mean_col) := stats$nb_mean]
    
    edge_list[, nval := NULL]  # clean up for next variable
    
    message(sprintf("  Done in %.1f seconds", (proc.time() - t1)[3]))
  }
  
  data_dt[, .row_idx := NULL]  # remove helper column
  as.data.frame(data_dt)
}

# ==============================================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_and_add_all_neighbor_features(
  cell_data,
  neighbor_source_vars,
  id_order,
  rook_neighbors_unique
)

# ==============================================================================
# RANDOM FOREST INFERENCE (unchanged â€” this was never the bottleneck)
# ==============================================================================
# The trained RF model is preserved exactly as-is.
# predictions <- predict(rf_model, newdata = cell_data)
# cell_data$predicted <- predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | Row-by-row `lapply` over 6.46M rows with string key lookups | Single vectorized `data.table` join building an edge list |
| **Neighbor stats** | `lapply` Ã— 5 vars = ~32M R-level calls; `do.call(rbind, ...)` | 5 grouped `data.table` aggregations (vectorized C-level) |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes |
| **RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical results** | Original max/min/mean | Identical max/min/mean â€” numerically equivalent |

The Random Forest predict step was never the problem. The true bottleneck was millions of R-level loop iterations doing string manipulation and named-vector lookups for spatial neighbor feature construction.
END_RESPONSE>>>
