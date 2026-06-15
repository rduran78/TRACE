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
run_id: Anthropic_F05_T07_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full chain of waste:

1. **`build_neighbor_lookup` uses `lapply` over ~6.46M rows**, and for each row it constructs string keys via `paste()` and does named-vector lookups. The `idx_lookup` named vector (6.46M entries) is built once, but the per-row `paste()` + named-lookup is O(n) Ã— average-neighbor-count character operations. With ~6.46M rows and ~4 rook neighbors each, that's ~25.8M `paste` calls plus hash lookups inside a sequential `lapply`.

2. **The neighbor topology is time-invariant.** Every cell has the same neighbors in every year. Yet the code re-discovers neighbors per cell-year row rather than per cell, then simply broadcasting across years.

3. **`compute_neighbor_stats` is called 5 times** (once per variable) each iterating over 6.46M rows. Each call extracts neighbor values by integer index, which is fine, but the repeated `lapply` over 6.46M rows with R-level function-call overhead is slow.

4. **The entire pattern can be replaced with a vectorized matrix/data.table approach**: build a sparse neighbor matrix once (344K Ã— 344K), reshape variables into cell Ã— year matrices, and compute neighbor aggregates via sparse matrix multiplication â€” turning all 5 variables into a single set of matrix operations with no per-row R loops.

## Optimization Strategy

| Step | Current | Proposed |
|------|---------|----------|
| Neighbor discovery | Per cell-year string lookup in 6.46M-entry named vector | Sparse adjacency matrix (344K Ã— 344K), built once |
| Neighbor aggregation | `lapply` over 6.46M rows Ã— 5 vars = 32.3M R function calls | Sparse matrixâ€“dense matrix multiply: `A %*% V`, `A %*% (V != NA)` for counts, etc. |
| Max/Min | R-level `max`/`min` per row in `lapply` | Vectorized via `data.table` grouped operations or iterative sparse approach |
| Complexity | ~86+ hours | Minutes |

**Key insight**: Since `max` and `min` are not linear operators, they can't be computed directly via matrix multiplication. However, we can use `data.table` with a pre-built edge list (cell_i, cell_j) joined against the panel, grouped by (cell_i, year), which is fully vectorized and avoids any per-row R function calls.

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Preserves the original numerical estimand (max, min, mean of rook-neighbor
# values for each cell-year) and does not touch the trained RF model.
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # 1. Build a directed edge list from the nb object (time-invariant)
  #    Each entry in rook_neighbors_unique[[i]] is a vector of neighbor

  #    indices into id_order.
  # -------------------------------------------------------------------------
  message("Building edge list from nb object...")

  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))

  message(sprintf("  Edge list: %s directed edges", format(nrow(edge_list), big.mark = ",")))

  # -------------------------------------------------------------------------
  # 2. Convert cell_data to data.table (if not already) and key it
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Preserve original row order for safe re-attachment
  dt[, .row_order := .I]

  # -------------------------------------------------------------------------
  # 3. For each source variable, join neighbors and compute grouped stats
  #    We process one variable at a time to limit peak memory.
  # -------------------------------------------------------------------------

  # Prepare a slim lookup: (id, year, var_value) for joining
  # We join edge_list with the panel on neighbor_id == id AND same year.

  for (var_name in neighbor_source_vars) {
    message(sprintf("Processing neighbor variable: %s", var_name))

    # Slim table of (id, year, value) for the neighbor side
    val_dt <- dt[, .(neighbor_id = id, year, .var_val = get(var_name))]
    setkey(val_dt, neighbor_id, year)

    # Expand edges Ã— years: join edge_list with val_dt to get neighbor values
    # First, add year from the focal cell's panel rows.
    # Strategy: join focal rows to edge_list, then join neighbor values.

    # Focal keys: (focal_id, year) â€” we need one row per (focal_id, year, neighbor_id)
    focal_dt <- dt[, .(focal_id = id, year)]
    setkey(focal_dt, focal_id)
    setkey(edge_list, focal_id)

    # Merge: for each (focal_id, year), get all neighbor_ids
    # This creates ~6.46M Ã— ~4 = ~25.8M rows (fits in memory)
    expanded <- edge_list[focal_dt, on = "focal_id", allow.cartesian = TRUE]
    # expanded has columns: focal_id, neighbor_id, year

    # Now join neighbor values
    setkey(expanded, neighbor_id, year)
    expanded[val_dt, .var_val := i..var_val, on = .(neighbor_id, year)]

    # Compute grouped stats: max, min, mean per (focal_id, year)
    stats <- expanded[!is.na(.var_val),
                      .(nb_max  = max(.var_val),
                        nb_min  = min(.var_val),
                        nb_mean = mean(.var_val)),
                      by = .(focal_id, year)]

    # Rename columns to match original naming convention
    # Original code: compute_and_add_neighbor_features presumably creates
    # columns like: <var>_nb_max, <var>_nb_min, <var>_nb_mean
    new_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
    setnames(stats, c("nb_max", "nb_min", "nb_mean"), new_names)

    # Join back to dt
    dt[stats, (new_names) := mget(paste0("i.", new_names)),
       on = .(id = focal_id, year)]

    # Cells with no valid neighbors remain NA (matches original behavior)

    # Clean up
    rm(val_dt, focal_dt, expanded, stats)
    gc()

    message(sprintf("  Done: added %s", paste(new_names, collapse = ", ")))
  }

  # -------------------------------------------------------------------------
  # 4. Return in original row order as a data.frame (or data.table)
  # -------------------------------------------------------------------------
  setorder(dt, .row_order)
  dt[, .row_order := NULL]

  message("All neighbor features computed.")
  return(dt)
}


# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================
#
# # Original code:
# # neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# # for (var_name in neighbor_source_vars) {
# #   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# # }
#
# # Replacement:
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- compute_all_neighbor_features(
#   cell_data            = cell_data,
#   id_order             = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   neighbor_source_vars = neighbor_source_vars
# )
#
# # cell_data now has the same neighbor feature columns as before.
# # The trained Random Forest model is untouched â€” just use it for prediction:
# # preds <- predict(rf_model, newdata = cell_data)
```

## Why This Is Correct and Fast

| Concern | Explanation |
|---------|-------------|
| **Numerical equivalence** | The `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per (focal_id, year) as the original code. The edge list is derived from the same `rook_neighbors_unique` nb object and `id_order` mapping. |
| **RF model preserved** | No model retraining. The output columns are identical; just plug into `predict()`. |
| **Memory** | The expanded edge table is ~25.8M rows Ã— 4 columns (~0.8 GB). With one variable processed at a time, peak RAM stays well under 16 GB. |
| **Speed** | `data.table` keyed joins and grouped aggregations on 25.8M rows complete in seconds per variable. Total: **~1â€“3 minutes** vs. 86+ hours. |
| **Column naming** | Adjust the `new_names` pattern (e.g., `"nb_max_ntl"` vs `"ntl_nb_max"`) to match whatever your trained RF model expects. |

### Optional: If Column Names Differ

If the original `compute_and_add_neighbor_features` used a different naming scheme (e.g., `nb_max_ntl` instead of `ntl_nb_max`), simply change the `new_names` line:

```r
new_names <- paste0("nb_", c("max", "min", "mean"), "_", var_name)
```

Verify against `names(rf_model$forest$xlevels)` or the model's expected feature names to ensure alignment.
END_RESPONSE>>>
