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
run_id: Anthropic_F07_T07_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) list construction via row-by-row `lapply`

The function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's neighbor IDs.
2. Constructs paste-based keys (`"id_year"`).
3. Indexes into a named character vector (`idx_lookup`).

Named character vector lookup in R is **O(n)** per query in the worst case (hashing helps, but with 6.46M keys the overhead is enormous). Constructing 6.46 million individual key strings inside an `lapply` is also extremely slow. The total cost is roughly **O(N Ã— k)** where N â‰ˆ 6.46M and k = average neighbor count, but with massive constant factors from string operations and R-level looping.

### Bottleneck 2: `compute_neighbor_stats` â€” repeated R-level loops

For each of the 5 variables, the function loops over 6.46M rows in R, subsetting a numeric vector by index, removing NAs, and computing max/min/mean. This is called 5 times. While less catastrophic than Bottleneck 1, it's still ~32M R-level function calls.

### Why 86+ hours?
The `build_neighbor_lookup` alone does ~6.46M iterations of string construction and named-vector lookup. The named-vector lookup on a 6.46M-element vector is the killer â€” R's internal `match` on character vectors for named indexing is not O(1) per call when done this way repeatedly.

---

## Optimization Strategy

### Strategy: Vectorize everything; eliminate row-level R loops entirely.

**Key insight:** The neighbor structure is **time-invariant** â€” the same cell neighbors apply to every year. So we can:

1. **Expand the spatial neighbor list to a panel-level edge list once**, using vectorized integer arithmetic (no string keys).
2. **Compute neighbor stats using vectorized grouping** (`data.table` grouped aggregation over the edge list), which is C-level and cache-friendly.

**Specific steps:**

1. Convert `rook_neighbors_unique` (an `nb` object indexed by position in `id_order`) into a two-column integer edge list of **(from_cell_pos, to_cell_pos)**.
2. Build a lookup from `(cell_id, year)` â†’ row index using `data.table` keyed joins (integer-based, O(n log n) once).
3. Expand the spatial edge list across all 28 years to get a panel-level edge list: **(from_row, to_row)**. This is ~1.37M Ã— 28 â‰ˆ ~38.5M edges â€” easily fits in RAM.
4. For each variable, do a single vectorized `data.table` grouped operation: group by `from_row`, aggregate `value[to_row]` â†’ max, min, mean. This runs in seconds.

**Memory estimate:** The panel edge list is ~38.5M Ã— 2 integers = ~308 MB. With the data (~6.46M Ã— 110 cols), total is well within 16 GB.

**Time estimate:** Minutes instead of days.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars) {

  # ---------------------------------------------------------------
  # STEP 1: Convert cell_data to data.table, preserve original order

# ---------------------------------------------------------------
  setDT(cell_data)
  cell_data[, .row_idx := .I]
  
  # ---------------------------------------------------------------
  # STEP 2: Build spatial edge list from nb object
  #         rook_neighbors_unique[[i]] gives neighbor positions
 #         in id_order for cell at position i in id_order.
  # ---------------------------------------------------------------
  from_pos <- rep(
    seq_along(rook_neighbors_unique),
    lengths(rook_neighbors_unique)
  )
  to_pos <- unlist(rook_neighbors_unique, use.names = FALSE)
  
  # Remove 0-entries (spdep nb convention for no neighbors)
  valid <- to_pos > 0L
  from_pos <- from_pos[valid]
  to_pos   <- to_pos[valid]
  
  # Map positions to cell IDs
  spatial_edges <- data.table(
    from_id = id_order[from_pos],
    to_id   = id_order[to_pos]
  )
  
  cat(sprintf("Spatial edges: %d directed relationships\n", nrow(spatial_edges)))
  
  # ---------------------------------------------------------------
  # STEP 3: Build (cell_id, year) -> row_idx lookup
  # ---------------------------------------------------------------
  row_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)
  
  # ---------------------------------------------------------------
  # STEP 4: Get unique years
  # ---------------------------------------------------------------
  all_years <- sort(unique(cell_data$year))
  cat(sprintf("Years: %d (%d-%d)\n", length(all_years), min(all_years), max(all_years)))
  
  # ---------------------------------------------------------------
  # STEP 5: Expand spatial edges across all years to panel edges
  #         (from_row_idx, to_row_idx)
  # ---------------------------------------------------------------
  # Cross join spatial_edges Ã— years
  year_dt <- data.table(year = all_years)
  panel_edges <- spatial_edges[, CJ_idx := .I]  # just need the cross
  
  # More memory-efficient: expand year by year and join
  panel_edge_list <- vector("list", length(all_years))
  
  for (yi in seq_along(all_years)) {
    yr <- all_years[yi]
    
    # Look up from_row
    from_lookup <- row_lookup[.(spatial_edges$from_id, yr), .row_idx, nomatch = NA]
    to_lookup   <- row_lookup[.(spatial_edges$to_id,   yr), .row_idx, nomatch = NA]
    
    # Keep only edges where both endpoints exist in this year
    both_valid <- !is.na(from_lookup) & !is.na(to_lookup)
    
    panel_edge_list[[yi]] <- data.table(
      from_row = from_lookup[both_valid],
      to_row   = to_lookup[both_valid]
    )
  }
  
  panel_edges <- rbindlist(panel_edge_list)
  rm(panel_edge_list)
  
  # Clean up temporary column
  spatial_edges[, CJ_idx := NULL]
  
  cat(sprintf("Panel edges: %s directed relationships\n",
              formatC(nrow(panel_edges), format = "d", big.mark = ",")))
  cat(sprintf("Memory for panel edges: %.1f MB\n",
              object.size(panel_edges) / 1e6))
  
  # ---------------------------------------------------------------
  # STEP 6: For each variable, compute neighbor max, min, mean
  #         via vectorized data.table grouping
  # ---------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))
    
    # Attach the neighbor's value to each edge
    vals <- cell_data[[var_name]]
    panel_edges[, neighbor_val := vals[to_row]]
    
    # Group by from_row, compute stats (excluding NAs)
    stats <- panel_edges[!is.na(neighbor_val),
                         .(nmax  = max(neighbor_val),
                           nmin  = min(neighbor_val),
                           nmean = mean(neighbor_val)),
                         by = from_row]
    
    # Initialize new columns with NA
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")
    
    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]
    
    # Assign results by row index
    cell_data[stats$from_row, (max_col)  := stats$nmax]
    cell_data[stats$from_row, (min_col)  := stats$nmin]
    cell_data[stats$from_row, (mean_col) := stats$nmean]
    
    cat(sprintf("  Done. Non-NA rows: %d / %d\n",
                sum(!is.na(cell_data[[max_col]])), nrow(cell_data)))
  }
  
  # Clean up
  panel_edges[, neighbor_val := NULL]
  cell_data[, .row_idx := NULL]
  
  cat("All neighbor features computed.\n")
  return(cell_data)
}

# ===================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ===================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data             = cell_data,
  id_order              = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars  = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# The output columns (e.g., ntl_neighbor_max, ntl_neighbor_min,
# ntl_neighbor_mean, etc.) are numerically identical to the
# original implementation's output.
```

---

## Summary of Guarantees

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | Same `max`, `min`, `mean` over the same non-NA neighbor values per cell-year. No approximation. |
| **Trained RF model** | Untouched â€” code only computes input features. |
| **Irregular topology** | Fully preserved â€” uses the exact `nb` object; no raster focal assumptions. |
| **Masked/missing cells** | Edges with missing endpoints are dropped (`nomatch = NA`, then filtered). NAs in variable values are excluded, matching original behavior. |
| **RAM (16 GB)** | Panel edge list â‰ˆ ~308 MB; `data.table` ops are in-place. Total well under 16 GB. |
| **Runtime** | ~2â€“10 minutes total (vs. 86+ hours), dominated by the `rbindlist` of 28 edge tables and 5 grouped aggregations. |
END_RESPONSE>>>
