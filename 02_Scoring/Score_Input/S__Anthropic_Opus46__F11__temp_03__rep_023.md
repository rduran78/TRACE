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
run_id: Anthropic_F11_T03_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`) for every single row. Named vector lookup in R is hash-based but carries significant per-call overhead when done millions of times in an interpreted loop. With ~1.37 million neighbor relationships spread across 344K cells Ã— 28 years, this function alone generates tens of millions of string operations.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level function invocations just for the stats.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-loaded model object. Even with 6.46M rows and 110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in C/C++ and typically completes in seconds to minutes â€” orders of magnitude faster than the interpreted R loops above.

**The 86+ hour runtime is dominated by the neighbor feature engineering, not by model inference.**

---

## Optimization Strategy

1. **Replace the row-level `lapply` in `build_neighbor_lookup`** with a vectorized, `data.table`-based join approach. Instead of building a per-row list of neighbor indices via string keys, expand the neighbor relationships into an edge table and join against the data using integer keys (`id` + `year`), avoiding all string operations entirely.

2. **Replace the row-level `lapply` in `compute_neighbor_stats`** with a grouped `data.table` aggregation over the edge table. This computes `max`, `min`, and `mean` for all rows simultaneously in C-level code.

3. **Preserve the trained Random Forest model** â€” no retraining. The output columns are numerically identical (same `max`, `min`, `mean` of the same neighbor values), so the estimand is preserved.

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature engineering using data.table
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order        integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors  spdep::nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names to compute neighbor stats for
#' @return data.table with original columns plus neighbor feature columns appended

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors, neighbor_source_vars) {
  
  # --- Step 1: Convert to data.table and create a row index ---
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]
  
  # --- Step 2: Build an edge list (focal_id -> neighbor_id) from the nb object ---
  # Each element of rook_neighbors is an integer vector of indices into id_order
  edges <- rbindlist(lapply(seq_along(rook_neighbors), function(i) {
    nb_idx <- rook_neighbors[[i]]
    # spdep::nb uses 0 to indicate no neighbors; filter those out
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  
  # --- Step 3: Create a keyed lookup from (id, year) -> row index ---
  # This replaces the string-pasting named-vector lookup entirely
  setkey(dt, id, year)
  
  # --- Step 4: For each year, join edges to get focal_row and neighbor_row indices ---
  years <- sort(unique(dt$year))
  
  # Build a mapping: (id, year) -> .row_idx
  id_year_map <- dt[, .(id, year, .row_idx)]
  setkey(id_year_map, id)
  
  # Expand edges across all years at once using a cross join then keyed join
  # To avoid a massive cross join (edges Ã— years), we join per-year in a vectorized way
  # Actually, since every year has (potentially) every cell, we can do:
  
  # Create the full edge-year table by joining edges with id_year_map for focal and neighbor
  # Focal side
  setnames(id_year_map, c("id", "year", "focal_row"))
  edge_focal <- edges[id_year_map, on = .(focal_id = id), allow.cartesian = TRUE, nomatch = 0L]
  # edge_focal now has: focal_id, neighbor_id, year, focal_row
  
  # Neighbor side: get the row index for the neighbor in the same year
  neighbor_map <- dt[, .(neighbor_id = id, year, neighbor_row = .row_idx)]
  setkey(neighbor_map, neighbor_id, year)
  setkey(edge_focal, neighbor_id, year)
  
  edge_full <- neighbor_map[edge_focal, on = .(neighbor_id, year), nomatch = NA_integer_]
  # edge_full has: neighbor_id, year, neighbor_row, focal_id, focal_row
  
  # Drop rows where the neighbor doesn't exist in that year
  edge_full <- edge_full[!is.na(neighbor_row)]
  
  # --- Step 5: For each variable, compute grouped stats in one vectorized pass ---
  for (var_name in neighbor_source_vars) {
    
    # Extract neighbor values via direct integer indexing (very fast)
    edge_full[, nval := dt[[var_name]][neighbor_row]]
    
    # Remove NAs before aggregation
    edge_valid <- edge_full[!is.na(nval)]
    
    # Grouped aggregation: max, min, mean by focal_row
    stats <- edge_valid[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_row]
    
    # Assign back to dt using integer indexing
    max_col  <- paste0("max_neighbor_",  var_name)
    min_col  <- paste0("min_neighbor_",  var_name)
    mean_col <- paste0("mean_neighbor_", var_name)
    
    # Initialize with NA
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    # Fill in computed values
    dt[stats$focal_row, (max_col)  := stats$nb_max]
    dt[stats$focal_row, (min_col)  := stats$nb_min]
    dt[stats$focal_row, (mean_col) := stats$nb_mean]
  }
  
  # Clean up helper column
  dt[, .row_idx := NULL]
  
  return(dt)
}

# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# --- Random Forest prediction (unchanged, model preserved) ---
# library(ranger)  # or randomForest
# predictions <- predict(trained_rf_model, data = cell_data)
```

---

## Why This Is Fast

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M `lapply` iterations with `paste` + named vector lookup | One `data.table` keyed join (C-level merge) |
| Neighbor stats | 5 Ã— 6.46M `lapply` iterations with per-row `max/min/mean` | 5 grouped `data.table` aggregations (GForce-optimized) |
| String operations | ~6.46M `paste()` + `as.character()` calls per build | Zero â€” all joins on integer keys |
| Expected runtime | 86+ hours | **Minutes** (typically 5â€“15 min on 16 GB laptop) |

The trained Random Forest model is never modified, and the computed `max`, `min`, and `mean` neighbor statistics are numerically identical to the originals, preserving the estimand.
END_RESPONSE>>>
