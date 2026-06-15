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
run_id: Anthropic_F11_T10_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that **Random Forest inference** is the bottleneck is **incorrect**. The code evidence points overwhelmingly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** uses `lapply` over **~6.46 million rows**, and inside each iteration it performs `as.character()` coercion, character key construction with `paste()`, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector indexing in R is hash-table lookup, but doing it ~6.46 million times Ã— multiple neighbors per cell, with repeated string construction and `paste()` calls, is extremely expensive. This single function generates on the order of **billions of character operations**.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million rows with `lapply`, subsetting, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` on a 6.46-million-element list of 3-element vectors is also costly.

3. By contrast, Random Forest **prediction** on a pre-trained model is a single `predict()` call on a matrix of ~6.46M Ã— 110 features. Libraries like `ranger` or `randomForest` do this in compiled C/C++ code. This is orders of magnitude faster than the R-level row-wise loops described above.

**Conclusion:** The bottleneck is the R-level, row-by-row, string-heavy spatial neighbor feature computation, not RF inference.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace the per-row `lapply` with a fully vectorized approach using `data.table` joins. Instead of building a lookup per row, create a flat edge-list (cell-year â†’ neighbor-cell-year) and join to get row indices in bulk.

2. **Vectorize `compute_neighbor_stats()`**: Instead of `lapply` over millions of rows, use `data.table` grouped aggregation (`max`, `min`, `mean`) on the edge-list, which runs in compiled C code.

3. **Eliminate all `paste()`-based key construction and named-vector lookups** â€” use integer joins exclusively.

These changes reduce the estimated runtime from **86+ hours to minutes**.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED: build_neighbor_edge_list
# Produces a data.table with columns: row_idx, neighbor_row_idx
# This replaces build_neighbor_lookup() entirely.
# ==============================================================================
build_neighbor_edge_list <- function(data_dt, id_order, neighbors) {
  # data_dt must be a data.table with columns: id, year, and a .row_idx column
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices)

  # Step 1: Build flat edge list at the cell level (id -> neighbor_id)
  #   neighbors[[i]] gives the indices into id_order that are neighbors of id_order[i]
  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells), times = lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  cell_edges <- data.table(
    id            = id_order[from_idx],
    neighbor_id   = id_order[to_idx]
  )

  # Step 2: For each (id, year) row, join to get (neighbor_id, year) rows
  #   We need the row index of the focal row and the row index of the neighbor row.

  # Add row indices to data
  data_dt[, .row_idx := .I]

  # Create a keyed lookup: for each (id, year) -> row_idx
  id_year_lookup <- data_dt[, .(id, year, .row_idx)]
  setkey(id_year_lookup, id)

  # Join cell_edges with id_year_lookup on focal id to get (focal_row_idx, neighbor_id, year)
  # This cross-joins each cell-level edge with all years that the focal cell appears in.
  edges_with_focal <- cell_edges[id_year_lookup, on = "id", allow.cartesian = TRUE,
                                  nomatch = 0L]
  # edges_with_focal now has: id, neighbor_id, year, .row_idx (focal row)
  setnames(edges_with_focal, ".row_idx", "focal_row_idx")

  # Step 3: Join to get neighbor's row index for the same year
  setkey(id_year_lookup, id, year)
  setnames(id_year_lookup, c("id", "year", "neighbor_row_idx"))

  edges_full <- edges_with_focal[id_year_lookup,
                                  on = c("neighbor_id" = "id", "year" = "year"),
                                  nomatch = 0L]
  # Keep only what we need
  edges_full <- edges_full[, .(focal_row_idx, neighbor_row_idx)]

  return(edges_full)
}

# ==============================================================================
# OPTIMIZED: compute_and_add_all_neighbor_features
# Computes max, min, mean of neighbor values for all source vars at once.
# ==============================================================================
compute_and_add_all_neighbor_features <- function(cell_data_dt, edge_list,
                                                   neighbor_source_vars) {
  # edge_list: data.table with (focal_row_idx, neighbor_row_idx)
  # For each variable, attach the neighbor's value, then aggregate.

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)

    # Attach the neighbor's value
    edges_var <- edge_list[, .(focal_row_idx, neighbor_row_idx)]
    edges_var[, neighbor_val := cell_data_dt[[var_name]][neighbor_row_idx]]

    # Remove NAs in neighbor values
    edges_var <- edges_var[!is.na(neighbor_val)]

    # Grouped aggregation â€” runs in C via data.table
    agg <- edges_var[, .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ), by = focal_row_idx]

    # Create output columns initialized to NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    cell_data_dt[, (max_col)  := NA_real_]
    cell_data_dt[, (min_col)  := NA_real_]
    cell_data_dt[, (mean_col) := NA_real_]

    # Assign aggregated values to the correct rows
    cell_data_dt[agg$focal_row_idx, (max_col)  := agg$nb_max]
    cell_data_dt[agg$focal_row_idx, (min_col)  := agg$nb_min]
    cell_data_dt[agg$focal_row_idx, (mean_col) := agg$nb_mean]
  }

  return(cell_data_dt)
}

# ==============================================================================
# MAIN PIPELINE (drop-in replacement for the outer loop)
# ==============================================================================

# Convert to data.table if not already
cell_data_dt <- as.data.table(cell_data)

# Build the edge list ONCE (replaces build_neighbor_lookup)
message("Building neighbor edge list...")
edge_list <- build_neighbor_edge_list(cell_data_dt, id_order, rook_neighbors_unique)
message("Edge list built: ", nrow(edge_list), " directed edges across all cell-years.")

# Compute all neighbor features in vectorized fashion
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data_dt <- compute_and_add_all_neighbor_features(cell_data_dt, edge_list,
                                                       neighbor_source_vars)

# --- Random Forest inference (UNCHANGED, as it is NOT the bottleneck) ---
# Example (preserving the trained model and the original numerical estimand):
# library(ranger)  # or randomForest
# predictions <- predict(rf_model, data = cell_data_dt)
# cell_data_dt[, predicted := predictions$predictions]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_data_dt)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | `lapply` over 6.46M rows, `paste()` + named-vector lookup | Single `data.table` join producing a flat integer edge list |
| `compute_neighbor_stats` | `lapply` over 6.46M rows Ã— 5 vars, `do.call(rbind, ...)` | `data.table` grouped `max/min/mean` in compiled C, per variable |
| Key construction | ~6.46M `paste()` calls per invocation | Zero `paste()` â€” integer-keyed joins only |
| Expected runtime | **86+ hours** | **Minutes** (typically 5â€“15 min on a 16 GB laptop) |
| RF model | Preserved (untouched) | Preserved (untouched) |
| Numerical estimand | Preserved (same max/min/mean logic, same NA handling) | Preserved |

The true bottleneck is the **row-level R-interpreted spatial feature engineering**, not Random Forest inference. The fix is full vectorization via `data.table` joins and grouped aggregation.
END_RESPONSE>>>
