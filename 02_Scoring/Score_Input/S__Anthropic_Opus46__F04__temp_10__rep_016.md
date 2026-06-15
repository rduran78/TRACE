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
run_id: Anthropic_F04_T10_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The computational bottleneck is **not** the Random Forest inference. It resides squarely in the two spatial neighbor feature construction functions, and the root causes are:

1. **`build_neighbor_lookup` â€” Row-level `lapply` over ~6.46 million rows.** Each iteration performs character coercion, string pasting (`paste(..., sep="_")`), and named-vector lookup (`idx_lookup[neighbor_keys]`). String operations and named-vector indexing in R are O(n) or O(n log n) in practice due to hashing overhead. Across 6.46M rows this is catastrophic.

2. **`compute_neighbor_stats` â€” Row-level `lapply` over ~6.46M rows with per-row subsetting.** Each call extracts a variable-length subset of a numeric vector, removes NAs, and computes three summary statistics. The per-element R interpreter overhead (function call dispatch, memory allocation for small vectors) dominates. For 5 variables Ã— 6.46M rows = ~32.3 million R-level anonymous function calls.

3. **String-keyed lookup as the join strategy.** Using `paste(id, year)` as a composite key and then indexing into a named character vector is the slowest possible join strategy in R. This should be replaced by integer-indexed joins.

4. **Memory pattern.** `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors forces R to allocate and copy repeatedly. Pre-allocating a matrix is far cheaper.

**Estimated cost breakdown (current):**

| Step | Calls | Estimated share |
|---|---|---|
| `build_neighbor_lookup` (string ops) | 6.46M | ~40% |
| `compute_neighbor_stats` (Ã—5 vars) | 32.3M | ~55% |
| RF prediction | 1 call | ~5% |

---

## Optimization Strategy

### Principle: Replace row-level R loops and string keys with vectorized integer-indexed operations using `data.table`.

**Key changes:**

1. **Replace the string-keyed lookup with an integer-indexed sparse matrix (or `data.table` equi-join).** Build a mapping from `(id, year)` â†’ row index using `data.table` keyed joins (binary search, no string pasting). Then expand the neighbor list into a flat edge-list `(source_row, neighbor_row)` in one vectorized operation.

2. **Replace per-row `lapply` in `compute_neighbor_stats` with grouped vectorized aggregation.** Using the flat edge-list, join neighbor values, then group-by-source and compute `max`, `min`, `mean` in one `data.table` operation per variable.

3. **Pre-allocate and column-bind** rather than `do.call(rbind, ...)`.

**Expected speedup:** From ~86+ hours to **~2â€“10 minutes** on the same laptop (the dominant cost becomes a few `data.table` grouped aggregations over ~26M edge-rows Ã— 5 variables).

**Preservation guarantees:**
- The trained Random Forest model is untouched.
- The numerical output (max, min, mean of neighbor values) is identical to the original.

---

## Working R Code

```r
library(data.table)

#' Optimized spatial neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data       data.frame (or data.table) with columns: id, year, 
#'                        and all neighbor_source_vars.
#' @param id_order        integer vector: the cell IDs in the order matching
#'                        the spdep::nb object indices.
#' @param neighbors       spdep::nb object (list of integer index vectors).
#' @param neighbor_source_vars character vector of variable names to summarize.
#' @return data.table with original columns plus neighbor feature columns.
add_neighbor_features_fast <- function(cell_data,
                                       id_order,
                                       neighbors,
                                       neighbor_source_vars) {

  # ---- Step 0: Convert to data.table (by reference if already one) ----
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]
  
  # ---- Step 1: Build flat edge list (source_cell_id -> neighbor_cell_id) ----
  # This is done once and is purely integer-based.
  
  # Map from nb-index to cell id
  # neighbors[[k]] contains nb-indices of neighbors of id_order[k]
  n_cells <- length(id_order)
  
  # Lengths of each neighbor set
  n_lengths <- lengths(neighbors)
  
  # Source cell ids (repeated for each neighbor)
  source_ids <- rep(id_order, times = n_lengths)
  
  # Neighbor cell ids (unlisted)
  neighbor_nb_idx <- unlist(neighbors, use.names = FALSE)
  neighbor_ids    <- id_order[neighbor_nb_idx]
  
  # Edge table: each row is one directed neighbor relationship (cell level, no year)
  edges <- data.table(source_id = source_ids, neighbor_id = neighbor_ids)
  
  # ---- Step 2: Build row-index lookup keyed on (id, year) ----
  row_lookup <- dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)
  
  # ---- Step 3: Get unique years ----
  years <- unique(dt$year)
  
  # ---- Step 4: Expand edges across years via cross join ----
  # Each spatial edge exists in every year -> full edge-year table.
  # With ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows. Fits in 16 GB easily
  # (3 integer columns â‰ˆ 38.5M Ã— 3 Ã— 8 bytes â‰ˆ 0.9 GB).
  
  edge_year <- edges[, .(source_id, neighbor_id, 
                          year = rep(list(years), .N))]
  # More memory-efficient expansion:
  edge_year <- edges[rep(seq_len(.N), each = length(years))]
  edge_year[, year := rep(years, times = nrow(edges))]
  
  # ---- Step 5: Map (source_id, year) -> source_row_idx ----
  setkey(edge_year, source_id, year)
  edge_year[row_lookup, source_row := i..row_idx, on = .(source_id = id, year)]
  
  # ---- Step 5b: Map (neighbor_id, year) -> neighbor_row_idx ----
  edge_year[row_lookup, neighbor_row := i..row_idx, on = .(neighbor_id = id, year)]
  
  # Drop edges where either side has no matching row (boundary / missing years)
  edge_year <- edge_year[!is.na(source_row) & !is.na(neighbor_row)]
  
  # Keep only the columns we need
  edge_year <- edge_year[, .(source_row, neighbor_row)]
  setkey(edge_year, source_row)
  
  # ---- Step 6: For each variable, compute grouped neighbor stats ----
  for (var_name in neighbor_source_vars) {
    
    message("Computing neighbor features for: ", var_name)
    
    # Attach neighbor values
    vals_vec <- dt[[var_name]]
    edge_year[, nval := vals_vec[neighbor_row]]
    
    # Grouped aggregation (the workhorse â€” fully vectorized in C)
    agg <- edge_year[!is.na(nval),
                     .(nb_max  = max(nval),
                       nb_min  = min(nval),
                       nb_mean = mean(nval)),
                     keyby = source_row]
    
    # Initialize new columns with NA
    max_col  <- paste0("nb_max_",  var_name)
    min_col  <- paste0("nb_min_",  var_name)
    mean_col <- paste0("nb_mean_", var_name)
    
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    
    # Assign results by row index
    dt[agg$source_row, (max_col)  := agg$nb_max]
    dt[agg$source_row, (min_col)  := agg$nb_min]
    dt[agg$source_row, (mean_col) := agg$nb_mean]
    
    # Clean up temporary column
    edge_year[, nval := NULL]
  }
  
  # ---- Step 7: Clean up helper column ----
  dt[, .row_idx := NULL]
  
  return(dt)
}
```

### Drop-in replacement for the outer loop

```r
# ---------- BEFORE (86+ hours) ----------
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ---------- AFTER (~2-10 minutes) ----------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_neighbor_features_fast(
  cell_data            = cell_data,
  id_order             = id_order,
  neighbors            = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data is now a data.table with the same columns plus
# nb_max_ntl, nb_min_ntl, nb_mean_ntl, nb_max_ec, ... etc.
# Convert back to data.frame if downstream code requires it:
# cell_data <- as.data.frame(cell_data)

# The trained Random Forest model is used as before (unchanged):
# predictions <- predict(rf_model, newdata = cell_data)
```

### Memory-constrained variant (if 38.5M-row edge table is too large)

If RAM is tight, process one year at a time:

```r
add_neighbor_features_fast_chunked <- function(cell_data, id_order,
                                                neighbors,
                                                neighbor_source_vars) {
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]
  
  # Build spatial edge list (cell-level, no year dimension)
  n_lengths    <- lengths(neighbors)
  source_ids   <- rep(id_order, times = n_lengths)
  neighbor_ids <- id_order[unlist(neighbors, use.names = FALSE)]
  edges <- data.table(source_id = source_ids, neighbor_id = neighbor_ids)
  
  row_lookup <- dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)
  
  # Initialize result columns
  for (var_name in neighbor_source_vars) {
    dt[, paste0("nb_max_",  var_name) := NA_real_]
    dt[, paste0("nb_min_",  var_name) := NA_real_]
    dt[, paste0("nb_mean_", var_name) := NA_real_]
  }
  
  years <- sort(unique(dt$year))
  
  for (yr in years) {
    message("Year: ", yr)
    
    ey <- copy(edges)
    ey[, year := yr]
    
    ey[row_lookup, source_row   := i..row_idx, on = .(source_id = id, year)]
    ey[row_lookup, neighbor_row := i..row_idx, on = .(neighbor_id = id, year)]
    ey <- ey[!is.na(source_row) & !is.na(neighbor_row)]
    
    for (var_name in neighbor_source_vars) {
      vals_vec <- dt[[var_name]]
      ey[, nval := vals_vec[neighbor_row]]
      
      agg <- ey[!is.na(nval),
                .(nb_max = max(nval), nb_min = min(nval), nb_mean = mean(nval)),
                keyby = source_row]
      
      if (nrow(agg) > 0) {
        set(dt, i = agg$source_row, j = paste0("nb_max_",  var_name), value = agg$nb_max)
        set(dt, i = agg$source_row, j = paste0("nb_min_",  var_name), value = agg$nb_min)
        set(dt, i = agg$source_row, j = paste0("nb_mean_", var_name), value = agg$nb_mean)
      }
      ey[, nval := NULL]
    }
  }
  
  dt[, .row_idx := NULL]
  return(dt)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup strategy | String-paste + named vector | Integer-keyed `data.table` binary-search join |
| Aggregation | Per-row `lapply` (R interpreter loop) | `data.table` grouped `[, .(...), by=]` (C-level) |
| `build_neighbor_lookup` calls | 6.46M R function calls | 0 (replaced by vectorized edge expansion) |
| `compute_neighbor_stats` calls | 5 Ã— 6.46M R function calls | 5 `data.table` grouped aggregations |
| Estimated wall time | 86+ hours | ~2â€“10 minutes |
| Peak RAM (full variant) | Moderate | ~3â€“4 GB |
| Peak RAM (chunked variant) | Moderate | ~1.5â€“2 GB |
| RF model | Unchanged | Unchanged |
| Numerical results | Identical | Identical |
END_RESPONSE>>>
