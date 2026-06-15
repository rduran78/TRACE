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
run_id: Anthropic_F11_T10_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points to a completely different bottleneck: **the spatial neighbor feature engineering stage**, specifically `build_neighbor_lookup()` and the repeated calls to `compute_neighbor_stats()`.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but the overhead of creating ~6.46 million character keys, pasting strings, and doing repeated name-matching on a vector of 6.46 million names is enormous. This single function produces a list of 6.46 million integer vectors.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over the 6.46 million-element neighbor lookup list, subsetting a numeric vector, removing NAs, and computing max/min/mean. That's ~32.3 million R-level anonymous-function invocations total, each with allocation and subsetting overhead.

3. **The `do.call(rbind, result)` pattern** on a list of 6.46 million small vectors is notoriously slow â€” it essentially calls `rbind` iteratively and causes massive memory allocation churn.

4. **Random Forest inference** by contrast is a single call to `predict()` on a pre-loaded model object. Even for 6.46 million rows Ã— 110 predictors, `predict.randomForest` or `predict.ranger` runs in compiled C/C++ code and typically completes in seconds to a few minutes. Loading the model from disk is a single `readRDS()` call. Writing predictions is a single vector write. None of these are iterative R-level loops over millions of elements.

**Conclusion:** The bottleneck is the O(N Ã— K) R-level looping in the neighbor feature construction, where N â‰ˆ 6.46 million and K = 5 variables, amplified by slow string operations and list-to-matrix conversion. This is what produces the 86+ hour runtime.

---

## Optimization Strategy

1. **Eliminate string-key lookups entirely.** Replace the `paste()`/named-vector approach with direct integer-index arithmetic. Since the data is a panel of 344,208 cells Ã— 28 years, we can map any `(cell, year)` pair to a row index arithmetically if we sort the data by `(id, year)`.

2. **Vectorize neighbor stats computation.** Replace the per-row `lapply` with a single vectorized operation using `data.table` grouped aggregation over an edge list, or use matrix-based sparse operations.

3. **Build a sparse adjacency matrix** and use matrix multiplication / column operations to compute neighbor means, and vectorized sparse-matrix operations for min/max, avoiding any row-level R loop.

4. **Preserve the trained Random Forest model** â€” no changes to the prediction step.

5. **Preserve the original numerical estimand** â€” the neighbor max, min, and mean values remain identical.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED SPATIAL NEIGHBOR FEATURE ENGINEERING
# Replaces build_neighbor_lookup + compute_neighbor_stats loop
# Estimated speedup: 86+ hours -> minutes
# =============================================================================

library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {
  
  # ------------------------------------------------------------------
  # STEP 0: Convert to data.table for speed; record original order
  # ------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, ..orig_row_idx := .I]  # preserve original row order
  
  # ------------------------------------------------------------------
  # STEP 1: Sort data by (id, year) so we can use arithmetic indexing
  # ------------------------------------------------------------------
  # Get unique sorted ids and years
  unique_ids   <- sort(unique(dt$id))
  unique_years <- sort(unique(dt$year))
  n_cells <- length(unique_ids)
  n_years <- length(unique_years)
  
  # Create integer mappings
  id_to_int   <- setNames(seq_along(unique_ids), as.character(unique_ids))
  year_to_int <- setNames(seq_along(unique_years), as.character(unique_years))
  
  # Sort dt by (id, year) and record the mapping back to original order
  dt[, id_int   := id_to_int[as.character(id)]]
  dt[, year_int := year_to_int[as.character(year)]]
  setorder(dt, id_int, year_int)
  dt[, sorted_row := .I]
  
  # Now row index for (id_int=i, year_int=t) = (i - 1) * n_years + t
  # Verify:
  stopifnot(nrow(dt) == n_cells * n_years)
  
  # ------------------------------------------------------------------
  # STEP 2: Build directed edge list from rook_neighbors_unique
  #         using id_order (the original cell ID ordering in the nb object)
  # ------------------------------------------------------------------
  # id_order[k] is the cell ID for the k-th entry in rook_neighbors_unique
  # rook_neighbors_unique[[k]] gives integer indices into id_order of neighbors
  
  # Map id_order positions to our id_int positions
  id_order_to_int <- id_to_int[as.character(id_order)]
  
  # Build edge list: from_id_int -> to_id_int
  # Pre-allocate based on total neighbor count
  total_edges <- sum(lengths(rook_neighbors_unique))
  from_id_int <- integer(total_edges)
  to_id_int   <- integer(total_edges)
  
  pos <- 1L
  for (k in seq_along(rook_neighbors_unique)) {
    nb_indices <- rook_neighbors_unique[[k]]
    if (length(nb_indices) == 0L) next
    n_nb <- length(nb_indices)
    from_id_int[pos:(pos + n_nb - 1L)] <- id_order_to_int[k]
    to_id_int[pos:(pos + n_nb - 1L)]   <- id_order_to_int[nb_indices]
    pos <- pos + n_nb
  }
  
  # Remove any NAs (cells in id_order not present in data)
  valid <- !is.na(from_id_int) & !is.na(to_id_int)
  from_id_int <- from_id_int[valid]
  to_id_int   <- to_id_int[valid]
  n_edges <- length(from_id_int)
  
  cat(sprintf("Edge list built: %d directed edges\n", n_edges))
  
  # ------------------------------------------------------------------
  # STEP 3: Expand edge list across all years (vectorized)
  #         For each year t, row index = (id_int - 1) * n_years + t
  # ------------------------------------------------------------------
  # We create the full from_row -> to_row edge list across all years
  # This will have n_edges * n_years entries
  
  year_ints <- seq_len(n_years)
  
  # Vectorized expansion: rep edges for each year
  from_rows <- rep((from_id_int - 1L) * n_years, times = n_years) +
               rep(year_ints, each = n_edges)
  to_rows   <- rep((to_id_int - 1L) * n_years, times = n_years) +
               rep(year_ints, each = n_edges)
  
  cat(sprintf("Expanded edge list: %d edges across all years\n", length(from_rows)))
  
  # ------------------------------------------------------------------
  # STEP 4: Compute neighbor stats using data.table grouped operations
  # ------------------------------------------------------------------
  # For each (from_row, var), gather neighbor values via to_rows, then
  # compute max, min, mean grouped by from_row.
  
  # Build edge data.table once
  edges_dt <- data.table(from_row = from_rows, to_row = to_rows)
  
  # Free memory
  rm(from_rows, to_rows, from_id_int, to_id_int)
  gc()
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor features for: %s\n", var_name))
    
    # Extract the variable values in sorted order
    vals <- dt[[var_name]]
    
    # Look up neighbor values
    edges_dt[, nb_val := vals[to_row]]
    
    # Compute grouped stats (excluding NAs)
    stats <- edges_dt[!is.na(nb_val),
                      .(nb_max  = max(nb_val),
                        nb_min  = min(nb_val),
                        nb_mean = mean(nb_val)),
                      by = from_row]
    
    # Initialize columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    # Assign computed stats
    dt[stats$from_row, (max_col)  := stats$nb_max]
    dt[stats$from_row, (min_col)  := stats$nb_min]
    dt[stats$from_row, (mean_col) := stats$nb_mean]
    
    cat(sprintf("  Done: %s\n", var_name))
  }
  
  # Clean up temporary edge column
  edges_dt[, nb_val := NULL]
  
  # ------------------------------------------------------------------
  # STEP 5: Restore original row order and return as data.frame
  # ------------------------------------------------------------------
  setorder(dt, ..orig_row_idx)
  
  # Drop helper columns
  dt[, c("..orig_row_idx", "id_int", "year_int", "sorted_row") := NULL]
  
  as.data.frame(dt)
}

# =============================================================================
# USAGE â€” drop-in replacement for the original loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# =============================================================================
# RANDOM FOREST PREDICTION â€” unchanged, as it is NOT the bottleneck
# =============================================================================

# rf_model <- readRDS("path/to/trained_rf_model.rds")
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
# write.csv(cell_data, "output_predictions.csv", row.names = FALSE)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Bottleneck identified** | Colleague claimed RF inference | Actually: neighbor feature engineering |
| **`build_neighbor_lookup`** | 6.46M `lapply` iterations with `paste()`/string lookup | Eliminated; integer arithmetic indexing via sorted panel |
| **`compute_neighbor_stats`** | 5 Ã— 6.46M `lapply` + `do.call(rbind, ...)` | `data.table` grouped aggregation over pre-built edge list |
| **String operations** | ~6.46M `paste()` + named vector lookups per call | Zero string operations in hot path |
| **Memory pattern** | Millions of small list allocations + rbind | Columnar `data.table` operations, single edge `data.table` reused |
| **Expected runtime** | 86+ hours | Minutes (estimated 5â€“20 min depending on hardware) |
| **RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical estimand** | max/min/mean of rook-neighbor values per cell-year | Identical âœ“ |
END_RESPONSE>>>
