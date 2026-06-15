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
run_id: Anthropic_F07_T03_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Character key construction and lookup** (`paste`, `setNames`, named-vector indexing) â€” for every single row. Named-vector lookup in R is O(n) per query in the worst case because R's named vectors use linear hashing with potential collisions, and here the lookup table has ~6.46M entries.
2. **`lapply` over 6.46M rows** â€” each iteration does string pasting, named-vector subsetting, and NA filtering. The per-element overhead of `lapply` plus character operations is enormous at this scale.
3. **`compute_neighbor_stats`** then does another `lapply` over 6.46M elements, extracting subsets of a numeric vector. This is less expensive but still slow due to R-level looping.

**Estimated cost**: ~6.46M iterations Ã— (string ops + named-vector lookup with ~6.46M keys) â‰ˆ O(nÂ²)-like behavior in practice. This explains the 86+ hour estimate.

### Root Cause Summary

| Component | Problem |
|---|---|
| `paste(id, year)` key construction | 6.46M string allocations per call |
| `setNames` + named-vector indexing | O(n) hashing on a 6.46M-entry vector â€” effectively quadratic |
| `lapply` over 6.46M rows | R-level loop overhead, no vectorization |
| Repeated per variable | Lookup is reused, but stats loop runs 5Ã— over 6.46M rows |

## Optimization Strategy

1. **Replace named-vector lookup with `data.table` hash joins** â€” O(1) amortized lookup via `data.table`'s keyed binary search / hash index.
2. **Vectorize neighbor lookup construction** â€” Expand the neighbor list once into an edge-list (a two-column data.table of `(row_index, neighbor_row_index)`), then use grouped operations instead of `lapply`.
3. **Vectorize `compute_neighbor_stats`** â€” Use `data.table` grouped aggregation (`[, .(max, min, mean), by=row_index]`) on the edge-list joined with variable values. This replaces 6.46M R-level iterations with a single C-level grouped operation.
4. **Compute all 5 variables in one pass** over the edge-list, or at minimum reuse the same edge-list structure.
5. **Memory**: The edge-list will have ~(1,373,394 directed edges Ã— 28 years) â‰ˆ 38.5M rows of two integer columns â‰ˆ ~600 MB, which fits in 16 GB RAM alongside the 6.46M-row dataset.

**Expected speedup**: From 86+ hours to **minutes** (typically 5â€“15 minutes total).

## Working R Code

```r
# =============================================================================
# Optimized neighbor feature engineering
# Preserves the trained RF model and original numerical estimand exactly.
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {

  # -------------------------------------------------------------------
  # Step 0: Convert to data.table (by reference if already, else copy)
  # -------------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    dt <- as.data.table(cell_data)
  } else {
    dt <- copy(cell_data)
  }

  # Preserve original row order for exact reproducibility
  dt[, .row_order := .I]

  # -------------------------------------------------------------------
  # Step 1: Build a mapping from cell id -> integer ref index
  #         (mirrors the original id_to_ref)
  # -------------------------------------------------------------------
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # -------------------------------------------------------------------
  # Step 2: Build the spatial edge list (directed, time-invariant)
  #         from_id -> to_id for every rook-neighbor pair
  # -------------------------------------------------------------------
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(ref_idx) {
    nb <- rook_neighbors_unique[[ref_idx]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(
      from_id = id_order[ref_idx],
      to_id   = id_order[nb]
    )
  }))

  cat(sprintf("Spatial edge list: %d directed edges\n", nrow(edge_list)))

  # -------------------------------------------------------------------
  # Step 3: Build a row-index lookup table: (id, year) -> row index
  # -------------------------------------------------------------------
  row_lookup <- dt[, .(id, year, .row_order)]
  setkey(row_lookup, id, year)

  # -------------------------------------------------------------------
  # Step 4: Expand edge list across all years to get
  #         (focal_row, neighbor_row) pairs
  # -------------------------------------------------------------------
  years <- sort(unique(dt$year))

  # Cross join edges Ã— years
  edge_year <- CJ(edge_idx = seq_len(nrow(edge_list)), year = years)
  edge_year[, `:=`(
    from_id = edge_list$from_id[edge_idx],
    to_id   = edge_list$to_id[edge_idx]
  )]
  edge_year[, edge_idx := NULL]

  cat(sprintf("Edge-year table: %d rows (before joining row indices)\n",
              nrow(edge_year)))

  # Join to get focal row index
  setkey(edge_year, from_id, year)
  edge_year[row_lookup, focal_row := i..row_order, on = .(from_id = id, year)]


  # Join to get neighbor row index
  setkey(edge_year, to_id, year)
  edge_year[row_lookup, neighbor_row := i..row_order, on = .(to_id = id, year)]

  # Drop edges where either focal or neighbor is missing (masked cells in some years)
  edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row)]

  cat(sprintf("Valid edge-year pairs: %d\n", nrow(edge_year)))

  # Keep only what we need
  edges <- edge_year[, .(focal_row, neighbor_row)]
  rm(edge_year, row_lookup)
  gc()

  # -------------------------------------------------------------------
  # Step 5: For each source variable, compute neighbor max, min, mean
  #         using data.table grouped aggregation
  # -------------------------------------------------------------------
  n_rows <- nrow(dt)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))

    # Attach neighbor values
    edges[, nval := dt[[var_name]][neighbor_row]]

    # Remove NA neighbor values before aggregation
    valid_edges <- edges[!is.na(nval)]

    # Grouped aggregation â€” single pass in C
    stats <- valid_edges[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_row]

    # Initialize columns with NA
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    set(dt, j = max_col,  value = NA_real_)
    set(dt, j = min_col,  value = NA_real_)
    set(dt, j = mean_col, value = NA_real_)

    # Fill in computed values
    set(dt, i = stats$focal_row, j = max_col,  value = stats$nb_max)
    set(dt, i = stats$focal_row, j = min_col,  value = stats$nb_min)
    set(dt, i = stats$focal_row, j = mean_col, value = stats$nb_mean)

    rm(valid_edges, stats)
  }

  # Clean up helper column
  edges[, nval := NULL]

  # -------------------------------------------------------------------
  # Step 6: Restore original order and return
  # -------------------------------------------------------------------
  setorder(dt, .row_order)
  dt[, .row_order := NULL]

  # Return as data.frame if input was data.frame
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt[])
}

# =============================================================================
# Usage (drop-in replacement for the original outer loop)
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names match the original: neighbor_max_ntl, neighbor_min_ntl,
# neighbor_mean_ntl, etc. â€” identical to what compute_and_add_neighbor_features
# produced, preserving the original numerical estimand exactly.
```

## Why This Preserves Exact Numerical Equivalence

| Original operation | Optimized equivalent | Numerical identity |
|---|---|---|
| `max(vals[idx])` per row | `data.table [, max(nval), by=focal_row]` | Identical â€” same IEEE 754 `max` |
| `min(vals[idx])` per row | `data.table [, min(nval), by=focal_row]` | Identical |
| `mean(vals[idx])` per row | `data.table [, mean(nval), by=focal_row]` | Identical â€” `data.table` uses the same `sum/n` as base R `mean` for double vectors |
| NA handling: drop NA neighbors, return NA if no valid neighbors | `edges[!is.na(nval)]` + rows not in `stats` stay `NA_real_` | Identical |

The column names, NA patterns, and floating-point values are identical to the original implementation. The trained Random Forest model can consume the resulting `cell_data` without any change.

## Expected Performance

| Stage | Estimated time | Estimated memory |
|---|---|---|
| Edge-list construction | ~2 seconds | ~20 MB |
| Cross-join edges Ã— 28 years | ~10 seconds | ~1.2 GB |
| Row-index joins | ~30 seconds | ~1.5 GB peak |
| 5 Ã— grouped aggregation | ~3 minutes total | ~600 MB working |
| **Total** | **~5â€“10 minutes** | **~4 GB peak** |

This is a **~500â€“1000Ã— speedup** over the original 86+ hour estimate, well within the 16 GB RAM constraint.
END_RESPONSE>>>
