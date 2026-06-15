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
run_id: Anthropic_F11_T07_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and an `is.na` filter. Named-vector lookups in R are hash-based but carry overhead per call. Across 6.46M rows with an average of ~4 rook neighbors each, this creates and looks up **~25.8 million key strings**. The function is pure R with no vectorization of the outer loop.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over the same 6.46M rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's **~32.3 million** R-level function invocations for the stats alone. The final `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also expensive.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-loaded model object. Even with 6.46M rows and 110 predictors, modern `ranger` or `randomForest` predict calls are implemented in C/C++ and typically complete in seconds to a few minutes. Loading the model from disk is a one-time deserialization. Writing predictions is a single vector write. This is orders of magnitude cheaper than the neighbor computation.

**Conclusion:** The bottleneck is the row-level R `lapply` loops in `build_neighbor_lookup()` and `compute_neighbor_stats()`, not Random Forest inference. The estimated 86+ hours runtime is dominated by millions of interpreted R-loop iterations with per-element string operations and named-vector lookups.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a fully vectorized `data.table` join approach. Instead of building a per-row list of neighbor indices, construct a long-format edge table (`source_row` â†’ `neighbor_row`) via keyed joins. This eliminates millions of `paste` and named-lookup calls.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation over the long edge table. For each source row and variable, join to get neighbor values, then aggregate with `max`, `min`, `mean` in one pass â€” all in C-level `data.table` internals.

3. **Process all 5 variables simultaneously** in the aggregation step to avoid repeated iteration.

4. **Preserve the trained Random Forest model** â€” no changes to the model or predict step.

5. **Preserve the original numerical estimand** â€” the same `max`, `min`, `mean` of rook-neighbor values per cell-year are computed; only the implementation mechanism changes.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 1: Build a vectorized edge table (replaces build_neighbor_lookup)
# ---------------------------------------------------------------
build_neighbor_edges <- function(data_dt, id_order, rook_neighbors) {
  # Map each cell id to its position in id_order
  id_to_ref <- data.table(
    cell_id = id_order,
    ref_idx = seq_along(id_order)
  )
  
  # Expand rook_neighbors (an nb list) into a long edge list: source_cell -> neighbor_cell
  # Each element of rook_neighbors is an integer vector of indices into id_order
  edges <- rbindlist(lapply(seq_along(rook_neighbors), function(i) {
    nb <- rook_neighbors[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(source_cell = id_order[i], neighbor_cell = id_order[nb])
  }))
  
  # Now join edges to the data to get row indices.
  # data_dt must have columns: row_id, id, year
  # We need: for each row in data_dt, find all rows that are
  # (neighbor_cell, same year).
  
  # Key the data for fast join
  data_key <- data_dt[, .(row_id, id, year)]
  setkey(data_key, id, year)
  
  # For each edge (source_cell -> neighbor_cell), expand across all years
  # by joining source_cell to data to get (source_row, year),
  # then joining neighbor_cell + year to data to get neighbor_row.
  
  # Step A: get all (source_row_id, source_cell, year) combos
  source_rows <- data_key[edges, on = .(id = source_cell), 
                          .(source_row_id = row_id, 
                            neighbor_cell = neighbor_cell, 
                            year = year),
                          allow.cartesian = TRUE, nomatch = NULL]
  
  # Step B: join to get neighbor_row_id
  neighbor_rows <- data_key[source_rows, on = .(id = neighbor_cell, year = year),
                            .(source_row_id = source_row_id,
                              neighbor_row_id = row_id),
                            nomatch = NULL]
  
  return(neighbor_rows)
}

# ---------------------------------------------------------------
# STEP 2: Compute all neighbor stats at once (replaces compute_neighbor_stats loop)
# ---------------------------------------------------------------
compute_all_neighbor_features <- function(data_dt, edge_table, neighbor_source_vars) {
  # edge_table has columns: source_row_id, neighbor_row_id
  # For each variable, look up the neighbor value, then aggregate per source_row_id
  
  n_rows <- nrow(data_dt)
  
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)
    
    # Attach the neighbor's value to each edge
    vals <- data_dt[[var_name]]
    work <- edge_table[, .(source_row_id, nval = vals[neighbor_row_id])]
    
    # Remove NAs
    work <- work[!is.na(nval)]
    
    # Aggregate
    agg <- work[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = source_row_id]
    
    # Initialize result columns with NA
    max_col  <- rep(NA_real_, n_rows)
    min_col  <- rep(NA_real_, n_rows)
    mean_col <- rep(NA_real_, n_rows)
    
    # Fill in computed values
    max_col[agg$source_row_id]  <- agg$nb_max
    min_col[agg$source_row_id]  <- agg$nb_min
    mean_col[agg$source_row_id] <- agg$nb_mean
    
    # Add to data_dt using standard naming convention
    set(data_dt, j = paste0(var_name, "_nb_max"),  value = max_col)
    set(data_dt, j = paste0(var_name, "_nb_min"),  value = min_col)
    set(data_dt, j = paste0(var_name, "_nb_mean"), value = mean_col)
  }
  
  return(data_dt)
}

# ---------------------------------------------------------------
# STEP 3: Main pipeline (drop-in replacement for the outer loop)
# ---------------------------------------------------------------

# Convert to data.table if not already; add row_id
cell_data_dt <- as.data.table(cell_data)
cell_data_dt[, row_id := .I]

# Build the edge table once (replaces build_neighbor_lookup)
message("Building vectorized neighbor edge table...")
edge_table <- build_neighbor_edges(cell_data_dt, id_order, rook_neighbors_unique)
setkey(edge_table, source_row_id)

# Compute all 5 variables' neighbor features in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor features for all variables...")
cell_data_dt <- compute_all_neighbor_features(cell_data_dt, edge_table, neighbor_source_vars)

# Remove helper column
cell_data_dt[, row_id := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_data_dt)

# ---------------------------------------------------------------
# STEP 4: Random Forest prediction (UNCHANGED â€” not the bottleneck)
# ---------------------------------------------------------------
# The trained RF model is loaded and used as-is.
# Example (preserving original logic):
#   rf_model <- readRDS("trained_rf_model.rds")
#   cell_data$predictions <- predict(rf_model, newdata = cell_data)$predictions
#   write.csv(cell_data, "final_predictions.csv", row.names = FALSE)
```

---

## Expected Improvement

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level `lapply` iterations with `paste`/named lookup | Single vectorized `data.table` join (~seconds to low minutes) |
| `compute_neighbor_stats` | 5 Ã— 6.46M `lapply` iterations (~32.3M R calls) | 5 grouped `data.table` aggregations on a keyed edge table (C-level) |
| **Estimated total** | **86+ hours** | **Minutes** (dominated by the edge-table join and grouped aggregation, both in `data.table`'s C backend) |
| RF inference | Fast (unchanged) | Fast (unchanged) |

The Random Forest model and the original numerical estimand (per-cell-year neighbor max, min, and mean for each of the 5 source variables) are fully preserved.
END_RESPONSE>>>
