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
run_id: Anthropic_F11_T07_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering pipeline** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()` uses `lapply` over ~6.46 million rows**, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`) inside the loop. Named vector lookup in R is hash-based but carries significant per-call overhead when done millions of times. With ~1.37M neighbor relationships spread across 6.46M rows, this function alone generates tens of millions of string operations.

2. **`compute_neighbor_stats()` is called 5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max/min/mean`. That's ~32.3 million R-level anonymous function invocations combined, each with small-vector allocation overhead.

3. **The `do.call(rbind, result)` on a list of 6.46 million 3-element vectors** is notoriously slow â€” it effectively builds a matrix one row at a time.

4. By contrast, Random Forest `predict()` on a pre-trained model with ~110 predictors and 6.46M rows is a single vectorized C/C++ call (in `ranger` or `randomForest`). It is computationally non-trivial but is nowhere near 86+ hours. Typical RF prediction on this scale takes minutes to low tens of minutes, not days.

**Conclusion:** The bottleneck is the row-level R `lapply` loops over millions of rows doing string operations and small-vector statistics. This is a classic "death by a million R-level iterations" problem.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup` with a vectorized `data.table` equi-join.** Instead of building a per-row list of neighbor indices via string pasting and named-vector lookup, create an edge-list data.table of `(id, neighbor_id)` from the `nb` object, then merge with the panel data on `(neighbor_id, year)` to get all neighbor-row indices at once â€” fully vectorized.

2. **Replace `compute_neighbor_stats` with a grouped `data.table` aggregation.** Once we have the long-form edge table joined to the data, computing `max`, `min`, and `mean` per `(row_index, variable)` is a single grouped aggregation â€” no R-level loop needed.

3. **Eliminate `do.call(rbind, ...)` entirely** â€” `data.table` returns results in columnar form directly.

These changes convert O(N) R-level iterations (N â‰ˆ 6.46M Ã— 5) into a small number of vectorized C-level operations, reducing runtime from 86+ hours to likely **minutes**.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Convert nb object to a vectorized edge-list data.table
# ---------------------------------------------------------------
build_neighbor_edges <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object (list of integer index vectors)
  # id_order is the vector of cell IDs corresponding to each nb element
  
  # Pre-calculate lengths for pre-allocation
  lens <- lengths(neighbors_nb)
  total_edges <- sum(lens)
  
  # Build source (focal) and target (neighbor) index vectors
  focal_idx <- rep(seq_along(neighbors_nb), times = lens)
  neighbor_idx <- unlist(neighbors_nb, use.names = FALSE)
  
  # Map from positional index back to actual cell ID
  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

# ---------------------------------------------------------------
# 2. Compute all neighbor features in vectorized fashion
# ---------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, id_order, neighbors_nb,
                                          neighbor_source_vars) {
  
  # Convert to data.table if not already; add a row index for later join-back
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]
  
  # Step 1: Build edge list (focal_id -> neighbor_id)
  edges <- build_neighbor_edges(id_order, neighbors_nb)
  
  # Step 2: Create a keyed lookup from (id, year) -> row index
  #         and the values of the neighbor source variables
  keep_cols <- c("id", "year", ".row_idx", neighbor_source_vars)
  lookup <- dt[, ..keep_cols]
  setnames(lookup, "id", "neighbor_id")
  setkey(lookup, neighbor_id, year)
  
  # Step 3: Build the focal table: (focal_row_idx, focal_id, year)
  focal_info <- dt[, .(focal_row_idx = .row_idx, focal_id = id, year)]
  
  # Step 4: Join focal_info with edges to get (focal_row_idx, year, neighbor_id)
  #          for every focal-row Ã— neighbor combination
  setkey(edges, focal_id)
  setkey(focal_info, focal_id)
  
  # This is the large join: each focal row gets its neighbor IDs

  expanded <- edges[focal_info, on = "focal_id", allow.cartesian = TRUE,
                    nomatch = NULL]
  # expanded has columns: focal_id, neighbor_id, focal_row_idx, year
  
  # Step 5: Join with lookup to get the neighbor variable values
  #          matched on (neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  expanded <- lookup[expanded, on = .(neighbor_id, year), nomatch = NA]
  # Now expanded has the neighbor variable values for each focal-row Ã— neighbor pair
  
  # Step 6: Grouped aggregation â€” compute max, min, mean per focal row per variable
  #          We aggregate by focal_row_idx
  
  # Pre-allocate result columns in dt
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
  }
  
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (var_name in neighbor_source_vars) {
    v <- as.name(var_name)
    agg_exprs[[paste0("neighbor_max_", var_name)]]  <-
      bquote(as.double(max(.(v), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_min_", var_name)]]  <-
      bquote(as.double(min(.(v), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_mean_", var_name)]] <-
      bquote(mean(.(v), na.rm = TRUE))
  }
  
  # Single grouped aggregation over all variables at once
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  agg_result <- expanded[, eval(agg_call), by = focal_row_idx]
  
  # Replace Inf/-Inf (from max/min of all-NA groups) with NA
  for (col_name in names(agg_result)[-1]) {
    vals <- agg_result[[col_name]]
    set(agg_result, i = which(is.infinite(vals)), j = col_name, value = NA_real_)
  }
  
  # Step 7: Map aggregated results back to dt by row index
  new_cols <- setdiff(names(agg_result), "focal_row_idx")
  dt[agg_result$focal_row_idx, (new_cols) := agg_result[, ..new_cols]]
  
  # Clean up helper column
  dt[, .row_idx := NULL]
  
  return(dt)
}

# ---------------------------------------------------------------
# 3. Drop-in replacement for the outer loop
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# ---------------------------------------------------------------
# 4. Random Forest prediction (UNCHANGED â€” not the bottleneck)
# ---------------------------------------------------------------
# library(ranger)  # or randomForest
# trained_model <- readRDS("path/to/trained_rf_model.rds")
# cell_data$predictions <- predict(trained_model, data = cell_data)$predictions
# fwrite(cell_data, "output_predictions.csv")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows, string paste + named-vector lookup | Single vectorized `data.table` join on `(neighbor_id, year)` |
| **Neighbor stats** | 5 Ã— `lapply` over 6.46M rows + `do.call(rbind, ...)` | One grouped `data.table` aggregation across all 5 variables simultaneously |
| **Estimated runtime** | 86+ hours | Minutes (dominated by the large equi-join and grouped aggregation, both executed in C) |
| **RF model** | Preserved âœ“ | Preserved âœ“ â€” `predict()` call unchanged |
| **Numerical estimand** | max, min, mean of rook-neighbor values per cell-year | Identical â€” same grouping, same aggregation functions, same NA handling |

The trained Random Forest model is untouched. The numerical outputs (neighbor max, min, mean per variable per cell-year) are identical to the original. The only change is *how* those neighbor statistics are computed â€” replacing millions of R-level loop iterations with vectorized `data.table` operations.
END_RESPONSE>>>
