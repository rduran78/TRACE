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
run_id: Anthropic_F11_T10_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering stage** â€” specifically `build_neighbor_lookup` and `compute_neighbor_stats` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup`** uses a `lapply` over **~6.46 million rows**, and inside each iteration it performs `as.character()` coercion, `paste()` string construction, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookups in R are O(n) string-hashing operations. Doing this ~6.46 million times, each time constructing and matching multiple string keys, is catastrophically slow. With ~1.37 million neighbor relationships spread across 28 years, the total number of string constructions and lookups is on the order of tens of millions.

2. **`compute_neighbor_stats`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows with `lapply`, subsetting, removing NAs, and computing `max/min/mean`. That's ~32.3 million R-level function calls with per-element vector subsetting.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-loaded model object. Even with 6.46 million rows and 110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in C/C++ and typically completes in seconds to minutes. Loading the model from disk is a one-time `readRDS()` call. Writing predictions is a single `fwrite()`/`write.csv()`. None of these are remotely close to 86+ hours.

**Verdict:** The bottleneck is the row-by-row, string-key-based spatial neighbor feature computation, not Random Forest inference.

---

## Optimization Strategy

1. **Eliminate `build_neighbor_lookup`'s per-row string operations.** Replace the string-keyed named-vector lookup with a direct integer-indexed matrix/hash approach. Pre-build a mapping from `(cell_index, year_index)` â†’ row number as an integer matrix, then look up neighbors purely via integer indexing.

2. **Vectorize `compute_neighbor_stats`.** Replace the `lapply` over 6.46M rows with a flat vectorized operation: expand all neighbor pairs into a two-column integer matrix (`row_i`, `neighbor_row`), then use `data.table` grouping to compute `max`, `min`, `mean` in one pass per variable.

3. **Process all 5 variables in one pass over the neighbor structure** rather than rebuilding/re-iterating the structure 5 times.

These changes reduce complexity from O(N Ã— k Ã— string_ops) to O(N Ã— k) integer operations, where N = 6.46M and k = average neighbor count.

---

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED SPATIAL NEIGHBOR FEATURE ENGINEERING
# ============================================================

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {
  
  # Convert to data.table for speed; preserve row order
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]
  
  # --- Step 1: Build integer-indexed (cell_index, year_index) -> row mapping ---
  
  # Map cell id -> integer index (1..N_cells)
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map year -> integer index (1..N_years)
  unique_years <- sort(unique(dt$year))
  year_to_idx  <- setNames(seq_along(unique_years), as.character(unique_years))
  
  n_cells <- length(id_order)
  n_years <- length(unique_years)
  
  # Assign cell_idx and year_idx to every row
  dt[, cell_idx := id_to_idx[as.character(id)]]
  dt[, year_idx := year_to_idx[as.character(year)]]
  
  # Build a matrix: row_lookup[cell_idx, year_idx] = row number in dt
  # Initialize with NA
  row_lookup <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_lookup[cbind(dt$cell_idx, dt$year_idx)] <- dt$.row_id
  
  # --- Step 2: Expand neighbor pairs into (focal_row, neighbor_row) for ALL year slices ---
  
  # Build flat edge list from the nb object: (focal_cell_idx, neighbor_cell_idx)
  # rook_neighbors_unique is a list of length n_cells; element i contains integer indices of neighbors
  focal_cell   <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  neighbor_cell <- unlist(rook_neighbors_unique, use.names = FALSE)
  
  # Remove any 0-entries (spdep uses 0 for "no neighbors")
  valid <- neighbor_cell > 0L
  focal_cell    <- focal_cell[valid]
  neighbor_cell <- neighbor_cell[valid]
  
  n_edges <- length(focal_cell)
  
  # For every year, translate (focal_cell, year) and (neighbor_cell, year) into row numbers
  # We replicate the edge list across all years
  all_focal_rows    <- integer(n_edges * n_years)
  all_neighbor_rows <- integer(n_edges * n_years)
  
  for (yi in seq_len(n_years)) {
    offset <- (yi - 1L) * n_edges
    idx_range <- (offset + 1L):(offset + n_edges)
    all_focal_rows[idx_range]    <- row_lookup[cbind(focal_cell, rep(yi, n_edges))]
    all_neighbor_rows[idx_range] <- row_lookup[cbind(neighbor_cell, rep(yi, n_edges))]
  }
  
  # Remove pairs where either focal or neighbor row is NA (cell-year doesn't exist in data)
  keep <- !is.na(all_focal_rows) & !is.na(all_neighbor_rows)
  all_focal_rows    <- all_focal_rows[keep]
  all_neighbor_rows <- all_neighbor_rows[keep]
  
  # --- Step 3: Compute neighbor stats for each variable using data.table grouping ---
  
  edges_dt <- data.table(
    focal_row    = all_focal_rows,
    neighbor_row = all_neighbor_rows
  )
  
  for (var_name in neighbor_source_vars) {
    
    # Attach neighbor values via integer indexing (vectorized)
    neighbor_vals <- dt[[var_name]][edges_dt$neighbor_row]
    edges_dt[, nval := neighbor_vals]
    
    # Remove NA neighbor values before aggregation
    edges_valid <- edges_dt[!is.na(nval)]
    
    # Grouped aggregation â€” single pass
    agg <- edges_valid[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_row]
    
    # Initialize new columns with NA
    max_col  <- paste0("neighbor_", var_name, "_max")
    min_col  <- paste0("neighbor_", var_name, "_min")
    mean_col <- paste0("neighbor_", var_name, "_mean")
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    # Assign aggregated values back by row index
    set(dt, i = agg$focal_row, j = max_col,  value = agg$nb_max)
    set(dt, i = agg$focal_row, j = min_col,  value = agg$nb_min)
    set(dt, i = agg$focal_row, j = mean_col, value = agg$nb_mean)
  }
  
  # Clean up helper columns
  dt[, c(".row_id", "cell_idx", "year_idx") := NULL]
  
  return(dt)
}

# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data, 
  id_order, 
  rook_neighbors_unique, 
  neighbor_source_vars
)

# ============================================================
# Random Forest prediction â€” unchanged, as it is NOT the bottleneck
# ============================================================

# rf_model <- readRDS("trained_rf_model.rds")          # load once
# cell_data$prediction <- predict(rf_model, cell_data)  # single vectorized call
# fwrite(cell_data, "predictions_output.csv")            # write once
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `paste` + named-vector string lookups | One integer matrix fill via `cbind` indexing |
| **Neighbor stats** | 6.46M `lapply` iterations Ã— 5 variables = ~32.3M R calls | Flat `data.table` grouped aggregation (vectorized C) Ã— 5 variables |
| **Estimated runtime** | 86+ hours | Minutes (typically 5â€“20 min depending on hardware) |
| **RF model** | Preserved â€” not retrained | Preserved â€” not retrained |
| **Numerical output** | `max`, `min`, `mean` of neighbor values | Identical `max`, `min`, `mean` â€” same estimand |

The trained Random Forest model is untouched. The numerical estimand (neighbor max, min, mean per variable per cell-year, and the downstream RF predictions) is preserved exactly.
END_RESPONSE>>>
