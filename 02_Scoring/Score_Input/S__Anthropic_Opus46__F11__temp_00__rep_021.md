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
run_id: Anthropic_F11_T00_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named-vector lookups in R are **O(n) scans** on each call when the vector is large. With ~6.46M keys in `idx_lookup`, this is catastrophically slow â€” effectively **O(nÂ²)** in aggregate.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over ~6.46 million entries with `lapply`, computing `max`, `min`, and `mean` on subsetted vectors. While individually cheaper per iteration than the lookup build, the 5Ã— repetition over millions of rows adds up.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with ~6.46M rows and ~110 predictors, modern RF implementations (e.g., `ranger`) perform prediction via optimized C++ tree traversal. This typically completes in **seconds to a few minutes** â€” negligible compared to the 86+ hours consumed by the feature engineering.

**The bottleneck is the row-by-row `lapply` loops with repeated named-vector lookups over millions of rows, not the RF prediction.**

---

## Optimization Strategy

1. **Replace named-vector lookups with `data.table` hash joins** â€” `data.table` uses hash-based indexing, turning O(n) scans into O(1) amortized lookups.

2. **Vectorize `build_neighbor_lookup()`** â€” Instead of looping row-by-row, expand all neighbor relationships into a flat edge table, join to get row indices, and group. This replaces ~6.46M R-level iterations with a single vectorized merge.

3. **Vectorize `compute_neighbor_stats()`** â€” Use `data.table` grouped aggregation (`max`, `min`, `mean` by source row) instead of `lapply` over millions of list elements.

4. **Compute all 5 variables' stats in one pass** over the neighbor edge table rather than 5 separate passes.

These changes should reduce runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED PIPELINE â€” replaces build_neighbor_lookup + compute_neighbor_stats
# ==============================================================================

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {
  
  # Convert to data.table if not already; preserve original order
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]
  
  # --- Step 1: Build a flat edge table of (cell_id, neighbor_cell_id) ----------
  # rook_neighbors_unique is an nb object: a list of integer index vectors
  # id_order[i] is the cell id for the i-th element of the nb list
  
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_indices <- rook_neighbors_unique[[i]]
    if (length(nb_indices) == 0L || (length(nb_indices) == 1L && nb_indices[1] == 0L)) {
      return(NULL)
    }
    data.table(
      focal_cell_id    = id_order[i],
      neighbor_cell_id = id_order[nb_indices]
    )
  }))
  
  # --- Step 2: Map (cell_id, year) -> row_idx via keyed join --------------------
  # Create a lookup: for every (id, year) in dt, what is the row index?
  
  id_year_lookup <- dt[, .(id, year, .row_idx)]
  setkey(id_year_lookup, id, year)
  
  # Get unique years
  unique_years <- sort(unique(dt$year))
  
  # Cross-join edges Ã— years to get all (focal_row, neighbor_row) pairs
  # For each edge (focal_cell_id, neighbor_cell_id) and each year,
  # look up the focal row index and the neighbor row index.
  
  # Expand edges by year
  edges_by_year <- CJ_dt(edge_list, data.table(year = unique_years))
  
  # Join to get focal row index
  setnames(edges_by_year, "focal_cell_id", "id")
  setkey(edges_by_year, id, year)
  edges_by_year <- id_year_lookup[edges_by_year, nomatch = 0L]
  setnames(edges_by_year, c(".row_idx", "id"), c("focal_row", "focal_cell_id"))
  
  # Join to get neighbor row index
  setnames(edges_by_year, "neighbor_cell_id", "id")
  setkey(edges_by_year, id, year)
  edges_by_year <- id_year_lookup[edges_by_year, nomatch = 0L]
  setnames(edges_by_year, c(".row_idx", "id"), c("neighbor_row", "neighbor_cell_id"))
  
  # --- Step 3: Compute neighbor stats for all variables at once -----------------
  # Extract the variable columns we need from dt as a matrix for fast indexing
  
  for (var_name in neighbor_source_vars) {
    vals <- dt[[var_name]]
    
    # Get neighbor values
    edges_by_year[, nval := vals[neighbor_row]]
    
    # Aggregate by focal_row: max, min, mean (excluding NAs)
    agg <- edges_by_year[!is.na(nval), .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_row]
    
    # Create full-length result columns (NA for rows with no valid neighbors)
    max_col  <- rep(NA_real_, nrow(dt))
    min_col  <- rep(NA_real_, nrow(dt))
    mean_col <- rep(NA_real_, nrow(dt))
    
    max_col[agg$focal_row]  <- agg$nb_max
    min_col[agg$focal_row]  <- agg$nb_min
    mean_col[agg$focal_row] <- agg$nb_mean
    
    # Add to dt with the same naming convention as the original pipeline
    set(dt, j = paste0(var_name, "_nb_max"),  value = max_col)
    set(dt, j = paste0(var_name, "_nb_min"),  value = min_col)
    set(dt, j = paste0(var_name, "_nb_mean"), value = mean_col)
  }
  
  # Clean up helper column
  dt[, .row_idx := NULL]
  
  return(dt)
}

# Helper: cross join two data.tables (since data.table::CJ is for vectors)
CJ_dt <- function(dt1, dt2) {
  k <- NULL
  dt1[, k := 1L]
  dt2[, k := 1L]
  result <- merge(dt1, dt2, by = "k", allow.cartesian = TRUE)
  result[, k := NULL]
  result
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# This single call replaces the original build_neighbor_lookup +
# the for-loop over compute_and_add_neighbor_features
cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# --- Random Forest prediction (unchanged â€” not the bottleneck) ----------------
# library(ranger)  # or randomForest, whichever was used
# predictions <- predict(trained_rf_model, data = cell_data)
# cell_data$predicted <- predictions$predictions
```

### Memory-Conscious Variant

If the full cross-join of ~1.37M edges Ã— 28 years (~38.4M rows) strains the 16 GB laptop RAM, process year-by-year:

```r
build_neighbor_features_fast_chunked <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {
  
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]
  
  # Build flat edge table (cell-level, ~1.37M rows â€” small)
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_indices <- rook_neighbors_unique[[i]]
    if (length(nb_indices) == 0L || (length(nb_indices) == 1L && nb_indices[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_cell_id = id_order[i], neighbor_cell_id = id_order[nb_indices])
  }))
  
  # Initialize result columns
  for (var_name in neighbor_source_vars) {
    set(dt, j = paste0(var_name, "_nb_max"),  value = NA_real_)
    set(dt, j = paste0(var_name, "_nb_min"),  value = NA_real_)
    set(dt, j = paste0(var_name, "_nb_mean"), value = NA_real_)
  }
  
  unique_years <- sort(unique(dt$year))
  
  for (yr in unique_years) {
    # Subset to this year
    dt_yr <- dt[year == yr, c("id", ".row_idx", neighbor_source_vars), with = FALSE]
    setkey(dt_yr, id)
    
    # Join edges to get focal and neighbor row indices for this year
    edges_yr <- copy(edge_list)
    
    # Focal join
    setnames(edges_yr, "focal_cell_id", "id")
    edges_yr <- dt_yr[, .(id, focal_row = .row_idx)][edges_yr, on = "id", nomatch = 0L]
    setnames(edges_yr, "id", "focal_cell_id")
    
    # Neighbor join
    setnames(edges_yr, "neighbor_cell_id", "id")
    edges_yr <- dt_yr[edges_yr, on = "id", nomatch = 0L]
    setnames(edges_yr, "id", "neighbor_cell_id")
    
    for (var_name in neighbor_source_vars) {
      nvals <- edges_yr[[var_name]]
      edges_yr[, nval := nvals]
      
      agg <- edges_yr[!is.na(nval), .(
        nb_max  = max(nval),
        nb_min  = min(nval),
        nb_mean = mean(nval)
      ), by = focal_row]
      
      set(dt, i = agg$focal_row, j = paste0(var_name, "_nb_max"),  value = agg$nb_max)
      set(dt, i = agg$focal_row, j = paste0(var_name, "_nb_min"),  value = agg$nb_min)
      set(dt, i = agg$focal_row, j = paste0(var_name, "_nb_mean"), value = agg$nb_mean)
    }
  }
  
  dt[, .row_idx := NULL]
  return(dt)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `lapply` iterations with O(n) named-vector scans | Single vectorized `data.table` keyed join |
| **Neighbor stats** | 5 Ã— 6.46M `lapply` iterations | 5 grouped `data.table` aggregations (vectorized C) |
| **Estimated runtime** | 86+ hours | ~5â€“20 minutes |
| **RF model** | Preserved (unchanged) | Preserved (unchanged) |
| **Numerical results** | max/min/mean per neighbor set | Identical max/min/mean per neighbor set |

The trained Random Forest model and the original numerical estimand (neighbor max, min, mean â†’ RF prediction) are fully preserved. Only the feature engineering code path is replaced.
END_RESPONSE>>>
