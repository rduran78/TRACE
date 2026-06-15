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
run_id: Anthropic_F11_T03_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

**Evidence from the code:**

1. **`build_neighbor_lookup()`** uses `lapply` over **~6.46 million rows**, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`) for every single row. Named vector lookup in R is O(n) hash-table probing repeated millions of times, and string concatenation/allocation at this scale is extremely expensive. This function alone likely accounts for the majority of the 86+ hour runtime.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million rows with `lapply`, subsetting vectors, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` on a 6.46-million-element list of 3-element vectors is also costly.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, modern RF implementations (e.g., `ranger`) perform prediction in minutes, not hours. Loading a serialized model and writing predictions are I/O-bound and trivially fast relative to the neighbor computation.

**Quantitative reasoning:** `build_neighbor_lookup` performs ~6.46M string paste operations and ~6.46M named-vector lookups (each touching multiple neighbor keys). `compute_neighbor_stats` performs ~6.46M Ã— 5 = ~32.3M list iterations with subsetting. The cumulative cost of these R-level interpreted loops with string operations dwarfs a single vectorized `predict()` call.

---

## Optimization Strategy

1. **Eliminate per-row string operations in `build_neighbor_lookup()`**: Replace the `paste(id, year, sep="_")` keying strategy with integer-arithmetic indexing. Map each `(id, year)` pair to a row index using a direct integer lookup matrix or `data.table` keyed join, avoiding all string allocation.

2. **Vectorize `compute_neighbor_stats()`**: Replace the R-level `lapply` loop with a grouped vectorized operation. Flatten the neighbor lookup into a two-column data.table (source row, neighbor row), join the variable values, and compute `max/min/mean` via `data.table` grouped aggregation â€” a single pass in C-optimized code.

3. **Build the neighbor lookup once using `data.table` keyed joins** instead of named-vector lookups.

4. **Leave the Random Forest inference untouched** â€” it is not the bottleneck.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED build_neighbor_lookup
# ==============================================================================
# Returns a data.table with columns: src_row, nbr_row
# This replaces the list-of-vectors representation with a flat edge table.
# ==============================================================================

build_neighbor_edge_table <- function(data_dt, id_order, neighbors) {
  # data_dt must be a data.table with columns: id, year, and a .row_idx column
  # id_order: integer vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors into id_order)

  # Step 1: Map each cell ID to its reference index in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Step 2: Build a keyed lookup from (id, year) -> row index in data_dt
  # Use integer keys, no string pasting
  data_dt[, .row_idx := .I]
  setkey(data_dt, id, year)

  # Step 3: Get unique cell IDs present in data and their ref indices
  unique_ids <- unique(data_dt$id)

  # Step 4: Build the full edge list (cell_id -> neighbor_cell_id) from nb object
  # This is done once, independent of year
  ref_indices <- id_to_ref[as.character(unique_ids)]
  # Keep only IDs that exist in the nb object
  valid <- !is.na(ref_indices)
  unique_ids <- unique_ids[valid]
  ref_indices <- ref_indices[valid]

  # Expand neighbor relationships: for each cell, list its neighbor cell IDs
  edge_list <- rbindlist(lapply(seq_along(unique_ids), function(i) {
    nbr_refs <- neighbors[[ref_indices[i]]]
    if (length(nbr_refs) == 0) return(NULL)
    data.table(src_id = unique_ids[i], nbr_id = id_order[nbr_refs])
  }))

  if (nrow(edge_list) == 0) {
    return(data.table(src_row = integer(0), nbr_row = integer(0)))
  }

  # Step 5: Cross with years to get (src_id, year, nbr_id, year) pairs
  # Then join to data_dt to resolve row indices
  years <- sort(unique(data_dt$year))

  # Create all (src_id, nbr_id, year) combinations
  # Since every cell-year row needs neighbor-year rows for the SAME year,
  # we cross the edge_list with all years
  edge_years <- edge_list[, CJ(src_id = src_id, nbr_id = nbr_id, year = years),
                          by = .EACHI][, .(src_id = src_id, nbr_id = nbr_id, year)]

  # Actually, more memory-efficient: cross edge_list with years vector
  edge_years <- CJ_edge_years(edge_list, years)

  # Join to get src_row
  setkey(data_dt, id, year)
  edge_years[data_dt, src_row := i..row_idx, on = .(src_id = id, year = year)]
  edge_years[data_dt, nbr_row := i..row_idx, on = .(nbr_id = id, year = year)]

  # Remove edges where either side is missing
  edge_years <- edge_years[!is.na(src_row) & !is.na(nbr_row)]

  return(edge_years[, .(src_row, nbr_row)])
}

# Helper: memory-efficient cross of edge_list with years
CJ_edge_years <- function(edge_list, years) {
  n_edges <- nrow(edge_list)
  n_years <- length(years)
  data.table(
    src_id = rep(edge_list$src_id, each = n_years),
    nbr_id = rep(edge_list$nbr_id, each = n_years),
    year   = rep(years, times = n_edges)
  )
}

# ==============================================================================
# OPTIMIZED compute_neighbor_stats (fully vectorized via data.table)
# ==============================================================================

compute_neighbor_stats_dt <- function(data_dt, edge_dt, var_name) {
  # edge_dt: data.table with columns src_row, nbr_row
  # Returns a data.table with columns: src_row, <var>_max, <var>_min, <var>_mean

  # Get neighbor values by joining
  nbr_vals <- edge_dt[, .(src_row, nbr_val = data_dt[[var_name]][nbr_row])]

  # Remove NAs in neighbor values
  nbr_vals <- nbr_vals[!is.na(nbr_val)]

  # Grouped aggregation â€” single pass in C
  stats <- nbr_vals[, .(
    v_max  = max(nbr_val),
    v_min  = min(nbr_val),
    v_mean = mean(nbr_val)
  ), by = src_row]

  setnames(stats, c("v_max", "v_min", "v_mean"),
           paste0(var_name, c("_max_nb", "_min_nb", "_mean_nb")))

  return(stats)
}

# ==============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# ==============================================================================

compute_and_add_neighbor_features_dt <- function(data_dt, var_name, edge_dt) {
  stats <- compute_neighbor_stats_dt(data_dt, edge_dt, var_name)

  # Merge back to data_dt by src_row (rows without neighbors get NA)
  new_cols <- paste0(var_name, c("_max_nb", "_min_nb", "_mean_nb"))

  # Initialize columns with NA
  for (col in new_cols) {
    set(data_dt, j = col, value = NA_real_)
  }

  # Fill in computed values
  for (col in new_cols) {
    set(data_dt, i = stats$src_row, j = col, value = stats[[col]])
  }

  return(data_dt)
}

# ==============================================================================
# MAIN OPTIMIZED PIPELINE
# ==============================================================================

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model_path, output_path) {

  # --- Step 1: Convert to data.table ---
  cell_dt <- as.data.table(cell_data)
  cell_dt[, .row_idx := .I]

  # --- Step 2: Build neighbor edge table (ONCE) ---
  # Build spatial edge list (cell-to-cell, no year dimension yet)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  unique_ids <- unique(cell_dt$id)
  ref_indices <- id_to_ref[as.character(unique_ids)]
  valid <- !is.na(ref_indices)
  unique_ids_valid <- unique_ids[valid]
  ref_indices_valid <- ref_indices[valid]

  spatial_edges <- rbindlist(lapply(seq_along(unique_ids_valid), function(i) {
    nbr_refs <- rook_neighbors_unique[[ref_indices_valid[i]]]
    if (length(nbr_refs) == 0) return(NULL)
    data.table(src_id = unique_ids_valid[i], nbr_id = id_order[nbr_refs])
  }))

  # Build (id, year) -> row_idx lookup
  setkey(cell_dt, id, year)
  row_lookup <- cell_dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # Expand spatial edges across all years
  years <- sort(unique(cell_dt$year))
  n_edges <- nrow(spatial_edges)
  n_years <- length(years)

  edge_dt <- data.table(
    src_id = rep(spatial_edges$src_id, each = n_years),
    nbr_id = rep(spatial_edges$nbr_id, each = n_years),
    year   = rep(years, times = n_edges)
  )

  # Resolve row indices via keyed join
  edge_dt[row_lookup, src_row := i..row_idx, on = .(src_id = id, year)]
  edge_dt[row_lookup, nbr_row := i..row_idx, on = .(nbr_id = id, year)]
  edge_dt <- edge_dt[!is.na(src_row) & !is.na(nbr_row), .(src_row, nbr_row)]

  # --- Step 3: Compute neighbor features (vectorized) ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor stats for:", var_name, "\n")
    cell_dt <- compute_and_add_neighbor_features_dt(cell_dt, var_name, edge_dt)
  }

  # --- Step 4: Random Forest inference (unchanged â€” not the bottleneck) ---
  rf_model <- readRDS(rf_model_path)
  predictions <- predict(rf_model, data = cell_dt)  # ranger syntax
  # For randomForest package, use: predict(rf_model, newdata = cell_dt)
  cell_dt[, prediction := if (inherits(predictions, "ranger.prediction"))
                            predictions$predictions
                          else
                            as.numeric(predictions)]

  # --- Step 5: Write output ---
  fwrite(cell_dt, output_path)
  cat("Pipeline complete. Output written to:", output_path, "\n")

  return(cell_dt)
}
```

---

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| **Alleged bottleneck** | RF inference (`predict()`) | Neighbor feature engineering |
| **Root cause** | â€” | Per-row `lapply` over 6.46M rows with string ops and named-vector lookups, repeated 5Ã— |
| **Estimated original time** | â€” | 86+ hours |
| **Expected optimized time** | â€” | ~5â€“20 minutes (data.table vectorized joins + grouped aggregation) |
| **RF model** | Preserved âœ“ | Not modified; single `predict()` call retained |
| **Numerical estimand** | Preserved âœ“ | Same max/min/mean neighbor stats; same predictions |

The optimization replaces ~32.3 million R-level interpreted loop iterations (with string allocation) with a flat edge table built via `data.table` keyed joins and C-level grouped aggregation, reducing runtime by roughly **three orders of magnitude**.
END_RESPONSE>>>
