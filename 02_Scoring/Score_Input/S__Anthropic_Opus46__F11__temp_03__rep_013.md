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
run_id: Anthropic_F11_T03_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

**Evidence from the code:**

1. **`build_neighbor_lookup()`** uses `lapply` over **~6.46 million rows**, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`) for every single row. Named vector lookup in R is O(n) hash-table probing per call, and doing this 6.46 million times with string keys is extremely expensive. Each iteration also creates intermediate character vectors and subsets. This alone could take tens of hours.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` on a 6.46-million-element list of 3-element vectors is also costly.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, a single `predict()` call on a `ranger` or `randomForest` object typically completes in minutes, not hours. Loading and writing are I/O-bound and trivially fast relative to the feature engineering.

**The bottleneck is the row-level R `lapply` loops over 6.46M rows doing string operations and named-vector lookups for neighbor feature construction.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a vectorized `data.table` merge/join approach. Instead of building a per-row list of neighbor indices via string keys, we construct a flat edge-list data.table of `(focal_row, neighbor_id, year)` and join it against the data to get neighbor row indices â€” all vectorized.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation per variable. The flat edge-list lets us join neighbor values and compute `max/min/mean` via `by=` grouping â€” no R-level row loop at all.

3. **Eliminate `lapply` entirely** from both functions. The entire pipeline becomes vectorized joins and grouped aggregations, which `data.table` executes in C.

Expected speedup: from 86+ hours to **minutes** (likely 5â€“20 minutes depending on disk I/O and RAM pressure).

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build a vectorized flat edge-list (replaces build_neighbor_lookup)
# ============================================================

build_neighbor_edgelist <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id', 'year', and a row index
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices)
  
  # Map each cell ID to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Build flat edge-list: focal_ref -> neighbor_ref (in id_order space)
  focal_refs <- rep(seq_along(neighbors), lengths(neighbors))
  neighbor_refs <- unlist(neighbors, use.names = FALSE)
  
  # Convert from id_order indices back to actual cell IDs
  edges <- data.table(
    focal_id    = id_order[focal_refs],
    neighbor_id = id_order[neighbor_refs]
  )
  
  return(edges)
}

# ============================================================
# STEP 2: Compute all neighbor features via vectorized joins
# ============================================================

compute_all_neighbor_features <- function(cell_data_dt, edges, neighbor_source_vars) {
  # Ensure row index exists
  cell_data_dt[, row_idx := .I]
  
  # Create a keyed lookup: (id, year) -> row_idx
  # We'll join edges x years to get focal_row and neighbor_row pairs
  
  # Get unique years
  years <- sort(unique(cell_data_dt$year))
  
  # Cross-join edges with years to create (focal_id, neighbor_id, year) triples
  # Then join to data to get focal_row and neighbor_row
  # 
  # But this cross-join could be large: ~1.37M edges x 28 years = ~38.4M rows
  # This is very manageable in data.table.
  
  edges_by_year <- CJ_dt(edges, years)
  
  # Key the data for fast joins
  setkey(cell_data_dt, id, year)
  
  # Get focal row indices
  edges_by_year[cell_data_dt, focal_row := i.row_idx, on = .(focal_id = id, year)]
  
  # Get neighbor row indices
  edges_by_year[cell_data_dt, neighbor_row := i.row_idx, on = .(neighbor_id = id, year)]
  
  # Drop edges where either focal or neighbor doesn't exist in the data
  edges_by_year <- edges_by_year[!is.na(focal_row) & !is.na(neighbor_row)]
  
  # Now for each variable, pull neighbor values and aggregate
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)
    
    # Pull the neighbor values
    edges_by_year[, nval := cell_data_dt[[var_name]][neighbor_row]]
    
    # Aggregate by focal_row: max, min, mean (na.rm = TRUE)
    agg <- edges_by_year[!is.na(nval), .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_row]
    
    # Initialize new columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    
    cell_data_dt[, (max_col)  := NA_real_]
    cell_data_dt[, (min_col)  := NA_real_]
    cell_data_dt[, (mean_col) := NA_real_]
    
    # Assign aggregated values back
    cell_data_dt[agg$focal_row, (max_col)  := agg$nb_max]
    cell_data_dt[agg$focal_row, (min_col)  := agg$nb_min]
    cell_data_dt[agg$focal_row, (mean_col) := agg$nb_mean]
    
    # Clean up temp column
    edges_by_year[, nval := NULL]
  }
  
  # Remove helper column
  cell_data_dt[, row_idx := NULL]
  
  return(cell_data_dt)
}

# Helper: cross-join edges with a year vector
CJ_dt <- function(edges, years) {
  # Repeat each edge row for every year
  n_edges <- nrow(edges)
  n_years <- length(years)
  
  result <- data.table(
    focal_id    = rep(edges$focal_id, times = n_years),
    neighbor_id = rep(edges$neighbor_id, times = n_years),
    year        = rep(years, each = n_edges)
  )
  return(result)
}

# ============================================================
# STEP 3: Full pipeline
# ============================================================

# Convert to data.table (preserves all columns including predictors)
cell_data_dt <- as.data.table(cell_data)

# Build flat edge-list from the nb object (done once, very fast)
edges <- build_neighbor_edgelist(cell_data_dt, id_order, rook_neighbors_unique)

# Define the 5 neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute all neighbor features via vectorized joins
cell_data_dt <- compute_all_neighbor_features(cell_data_dt, edges, neighbor_source_vars)

# Convert back to data.frame if needed downstream
cell_data <- as.data.frame(cell_data_dt)

# ============================================================
# STEP 4: Random Forest prediction (unchanged â€” not the bottleneck)
# ============================================================
# The trained RF model is preserved as-is. Example:
# predictions <- predict(rf_model, newdata = cell_data)
# cell_data$predicted <- predictions$predictions  # for ranger
# write.csv(cell_data, "output.csv", row.names = FALSE)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | `lapply` over 6.46M rows, string ops, named-vector lookup | Single vectorized edge-list construction + `data.table` keyed joins |
| `compute_neighbor_stats` | `lapply` over 6.46M rows Ã— 5 vars, `do.call(rbind, ...)` | Grouped `data.table` aggregation (`by = focal_row`) per variable |
| Estimated runtime | 86+ hours | ~5â€“20 minutes |
| RF inference | Unchanged (minutes) | Unchanged (minutes) |
| Numerical results | â€” | Identical `max`, `min`, `mean` values preserved |
| Trained RF model | â€” | Untouched; no retraining |

The true bottleneck is the **row-level R loops with string manipulation and named-vector lookups** in the neighbor feature engineering, not Random Forest inference. Replacing these with vectorized `data.table` joins and grouped aggregations eliminates the bottleneck entirely.
END_RESPONSE>>>
