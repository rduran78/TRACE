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
run_id: Anthropic_F11_T03_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but carries significant per-call overhead when done 6.46 million times. This single function alone likely accounts for the majority of the 86+ hour runtime.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over the same 6.46 million rows with an `lapply` that subsets, removes NAs, and computes `max/min/mean`. That's ~32.3 million R-level loop iterations total.

3. **Random Forest inference** is a single call to `predict()` on a pre-trained model. Even with 6.46 million rows and 110 predictors, `predict.randomForest` or `predict.ranger` is implemented in C/C++ and typically completes in seconds to a few minutes. Loading and writing predictions is trivially fast by comparison.

**The bottleneck is the row-level R `lapply` loops over millions of rows with repeated string operations and named-vector lookups.**

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace the per-row `lapply` with a fully vectorized `data.table` merge/join approach. Instead of building a lookup list of length 6.46M, expand the neighbor relationships into an edge table and join on `(neighbor_id, year)` to get row indices â€” all in one vectorized operation.

2. **Vectorize `compute_neighbor_stats()`**: Instead of looping over each row's neighbor indices, use `data.table` grouped aggregation on the edge table to compute `max`, `min`, and `mean` in one pass per variable.

3. **Eliminate string key construction entirely**: Use integer-based composite keys or `data.table` joins on `(id, year)` pairs directly, avoiding millions of `paste()` calls.

Expected speedup: from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build a vectorized edge table from the nb object (once)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer vectors (indices into id_order)
  # Expand into a two-column data.table of (focal_id, neighbor_id)
  n_neighbors <- vapply(neighbors, length, integer(1))
  focal_idx <- rep(seq_along(neighbors), n_neighbors)
  neighbor_idx <- unlist(neighbors, use.names = FALSE)
  
  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Vectorized neighbor stats computation via data.table joins
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_and_add_all_neighbor_features <- function(cell_data, neighbor_source_vars,
                                                   id_order, rook_neighbors_unique) {
  
  # Convert to data.table if not already (by reference if possible)
  dt <- as.data.table(cell_data)
  
  # Assign a row index to each cell-year observation
  dt[, .row_idx := .I]
  
  # Build edge table: focal_id <-> neighbor_id (no year dimension yet)
  edges <- build_edge_table(id_order, rook_neighbors_unique)
  
  # Cross-join edges with years: for each (focal_id, neighbor_id) pair,
  # we need every year. But instead of a full cross-join (expensive in memory),
  # we join edges to the data twice: once for focal, once for neighbor.
  
  # Create a keyed lookup: (id, year) -> row_idx and variable values
  # We only need id, year, row_idx, and the neighbor source variables
  cols_needed <- c("id", "year", neighbor_source_vars)
  lookup <- dt[, ..cols_needed]
  lookup[, .row_idx := .I]
  
  # For each edge (focal_id, neighbor_id), join with each year present
  # in the focal's data to find the neighbor's row in the same year.
  
  # Step A: Get all (focal_id, year) combinations from the data
  focal_keys <- dt[, .(focal_id = id, year, focal_row = .row_idx)]
  
  # Step B: Join edges to focal_keys to get (focal_id, neighbor_id, year, focal_row)
  setkey(edges, focal_id)
  setkey(focal_keys, focal_id)
  
  # This is the big join: each focal cell-year gets its neighbor IDs
  expanded <- edges[focal_keys, on = .(focal_id), allow.cartesian = TRUE, nomatch = 0L]
  # Result columns: focal_id, neighbor_id, year, focal_row
  
  # Step C: Join to lookup to get neighbor variable values in the same year
  setkey(lookup, id, year)
  setkey(expanded, neighbor_id, year)
  
  matched <- lookup[expanded, on = .(id = neighbor_id, year = year), nomatch = 0L]
  # matched now has: focal_id, year, focal_row, and all neighbor source variable values
  
  # Step D: Aggregate by focal_row to compute max, min, mean for each variable
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (var_name in neighbor_source_vars) {
    v <- as.name(var_name)
    agg_exprs[[paste0("neighbor_max_", var_name)]] <- 
      bquote(as.numeric(max(.(v), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_min_", var_name)]] <- 
      bquote(as.numeric(min(.(v), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_mean_", var_name)]] <- 
      bquote(mean(.(v), na.rm = TRUE))
  }
  
  # Suppress -Inf/Inf warnings from max/min on all-NA groups
  agg_stats <- matched[, lapply(agg_exprs, eval), by = .(focal_row)]
  
  # Replace Inf/-Inf with NA (from groups where all neighbor values were NA)
  inf_cols <- names(agg_stats)[names(agg_stats) != "focal_row"]
  for (col in inf_cols) {
    set(agg_stats, which(is.infinite(agg_stats[[col]])), col, NA_real_)
  }
  
  # Step E: Merge aggregated stats back into the main data.table by row index
  setkey(agg_stats, focal_row)
  
  # Pre-allocate new columns with NA
  for (col in inf_cols) {
    set(dt, j = col, value = NA_real_)
  }
  
  # Assign values for rows that have neighbors
  rows_with_data <- agg_stats$focal_row
  for (col in inf_cols) {
    set(dt, i = rows_with_data, j = col, value = agg_stats[[col]])
  }
  
  # Clean up helper column
  dt[, .row_idx := NULL]
  
  return(dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Run the optimized pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_and_add_all_neighbor_features(
  cell_data,
  neighbor_source_vars,
  id_order,
  rook_neighbors_unique
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 4: Random Forest prediction (unchanged â€” not the bottleneck)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rf_model is the pre-trained Random Forest (preserved, not retrained)
predictions <- predict(rf_model, newdata = cell_data)
cell_data$predicted <- predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste()` + named vector lookup | Single vectorized `data.table` join |
| `compute_neighbor_stats` | 5 Ã— 6.46M `lapply` iterations | One grouped `data.table` aggregation for all 5 variables |
| String operations | ~19M+ `paste()` calls | Zero |
| Estimated runtime | 86+ hours | Minutes (dominated by the large join, which `data.table` handles efficiently in C) |
| RF model | Preserved | Preserved (no retraining) |
| Numerical results | Identical `max/min/mean` | Identical `max/min/mean` |

The true bottleneck is the millions of R-level loop iterations with per-element string construction and named-vector lookups in the neighbor feature engineering, not the Random Forest inference.
END_RESPONSE>>>
