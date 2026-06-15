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
run_id: Anthropic_F11_T07_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence clearly points to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M iterations, each doing multiple string operations and lookups, this alone can take many hours.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over **~6.46M rows** with an `lapply` that subsets a numeric vector, removes NAs, and computes `max`, `min`, and `mean`. That's ~32.3 million R-level function calls total, each with allocation overhead from subsetting and the `do.call(rbind, ...)` at the end which binds ~6.46M small vectors.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with ~6.46M rows and ~110 predictors, the `ranger` or `randomForest` predict step is implemented in C/C++ and is typically completed in seconds to minutes â€” orders of magnitude faster than the neighbor feature construction described above.

**Conclusion:** The 86+ hour runtime is dominated by the row-level R `lapply` loops in `build_neighbor_lookup()` and `compute_neighbor_stats()`, not by model inference.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()` with a vectorized `data.table` merge/join.** Instead of looping over every row and doing string-key lookups, we expand the neighbor list into a two-column edge table `(cell_row, neighbor_row)` and resolve row indices via a keyed `data.table` join. This eliminates millions of individual `paste()` and named-vector lookups.

2. **Replace `compute_neighbor_stats()` with a grouped `data.table` aggregation.** Using the edge table, we join neighbor values and compute `max`, `min`, `mean` in a single grouped-by operation â€” fully vectorized in C via `data.table`.

3. **Leave the Random Forest predict step untouched**, as it is not the bottleneck.

Expected speedup: from 86+ hours to **minutes** (typically 5â€“20 minutes depending on hardware).

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert cell_data to data.table (if not already)
# ============================================================
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]  # preserve original row order

# ============================================================
# STEP 1: Vectorized neighbor lookup construction
#
# Instead of looping over 6.46M rows, we:
#   (a) Expand the nb object into an edge list of (focal_id, neighbor_id)
#   (b) Join with cell_dt to map (neighbor_id, year) -> row_idx
# ============================================================

build_neighbor_edges_dt <- function(cell_dt, id_order, neighbors) {
  # --- (a) Build edge list from the nb object ---
  # neighbors is a list of length = length(id_order).
  # neighbors[[k]] gives the indices (into id_order) of the neighbors of id_order[k].
  
  n_neighbors <- vapply(neighbors, length, integer(1))
  focal_pos   <- rep(seq_along(neighbors), times = n_neighbors)
  neigh_pos   <- unlist(neighbors, use.names = FALSE)
  
  # Map positional indices to actual cell IDs
  edge_dt <- data.table(
    focal_id    = id_order[focal_pos],
    neighbor_id = id_order[neigh_pos]
  )
  
  # --- (b) For every (focal row in cell_dt), find its neighbor rows ---
  # cell_dt has columns: id, year, row_idx, and all predictor columns.
  
  # Create a keyed lookup: (id, year) -> row_idx
  id_year_key <- cell_dt[, .(id, year, row_idx)]
  setkey(id_year_key, id)
  
  # Join focal rows: get (focal_id, year, focal_row_idx)
  focal_rows <- cell_dt[, .(focal_id = id, year, focal_row_idx = row_idx)]
  
  # Merge focal rows with edge list to get (focal_row_idx, neighbor_id, year)
  # For each focal row, attach all its spatial neighbors
  setkey(edge_dt, focal_id)
  setkey(focal_rows, focal_id)
  
  expanded <- edge_dt[focal_rows, 
                      .(neighbor_id, year, focal_row_idx), 
                      on = "focal_id", 
                      allow.cartesian = TRUE, 
                      nomatch = NULL]
  
  # Now resolve neighbor_id + year -> neighbor_row_idx
  setnames(id_year_key, c("id", "year", "row_idx"), 
           c("neighbor_id", "year", "neighbor_row_idx"))
  setkey(id_year_key, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  
  edges <- id_year_key[expanded, 
                       .(focal_row_idx, neighbor_row_idx), 
                       on = c("neighbor_id", "year"), 
                       nomatch = NULL]
  
  return(edges)
}

message("Building neighbor edge table (vectorized)...")
t0 <- proc.time()
edges <- build_neighbor_edges_dt(cell_dt, id_order, rook_neighbors_unique)
message("Edge table built: ", nrow(edges), " edges in ", 
        round((proc.time() - t0)[3], 1), "s")

# ============================================================
# STEP 2: Vectorized neighbor stats via grouped data.table ops
#
# For each variable, we look up the neighbor values via the edge
# table and compute max/min/mean grouped by focal_row_idx.
# ============================================================

compute_and_add_neighbor_features_dt <- function(cell_dt, var_name, edges) {
  # Attach neighbor values to the edge table
  edges_var <- edges[, .(focal_row_idx, neighbor_row_idx)]
  edges_var[, val := cell_dt[[var_name]][neighbor_row_idx]]
  
  # Remove NAs
  edges_var <- edges_var[!is.na(val)]
  
  # Grouped aggregation
  stats <- edges_var[, .(
    var_max  = max(val),
    var_min  = min(val),
    var_mean = mean(val)
  ), by = focal_row_idx]
  
  # Initialize columns with NA
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  # Assign computed stats back by row index
  cell_dt[stats$focal_row_idx, (max_col)  := stats$var_max]
  cell_dt[stats$focal_row_idx, (min_col)  := stats$var_min]
  cell_dt[stats$focal_row_idx, (mean_col) := stats$var_mean]
  
  invisible(cell_dt)
}

# ============================================================
# STEP 3: Outer loop â€” compute neighbor features for all 5 vars
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor features for ", length(neighbor_source_vars), " variables...")
t0 <- proc.time()
for (var_name in neighbor_source_vars) {
  message("  Processing: ", var_name)
  compute_and_add_neighbor_features_dt(cell_dt, var_name, edges)
}
message("All neighbor features computed in ", 
        round((proc.time() - t0)[3], 1), "s")

# ============================================================
# STEP 4: Remove helper column and convert back if needed
# ============================================================
cell_dt[, row_idx := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(cell_dt)

# ============================================================
# STEP 5: Random Forest prediction (UNCHANGED â€” not the bottleneck)
# ============================================================
# The pre-trained model and predict() call remain exactly as-is.
# Example (preserving the original estimand):
#
#   predictions <- predict(rf_model, newdata = cell_data)
#   cell_data$predicted <- predictions$predictions   # for ranger
#
# No changes needed here.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste()` + named vector lookups | Single vectorized `data.table` join on edge list |
| `compute_neighbor_stats` | 5 Ã— 6.46M `lapply` iterations + `do.call(rbind, ...)` | 5 Ã— single grouped `data.table` aggregation |
| RF inference | Single `predict()` call (fast) | **Unchanged** â€” not the bottleneck |
| Estimated runtime | 86+ hours | ~5â€“20 minutes |
| Numerical results | Original estimand | **Preserved** â€” identical `max`, `min`, `mean` logic |
| Trained RF model | Loaded from disk | **Preserved** â€” no retraining |

The bottleneck was never Random Forest inference. It was the row-level R loops constructing and querying the neighbor lookup across 6.46 million rows. Vectorizing these operations with `data.table` joins and grouped aggregations eliminates the bottleneck entirely.
END_RESPONSE>>>
