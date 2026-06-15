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
run_id: Anthropic_F02_T03_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and index vectors, causing massive overhead from:

1. **Repeated string allocation and hashing**: `paste(neighbor_cell_ids, data$year[i], sep = "_")` is called 6.46 million times, each producing a small character vector. String operations in R are slow and memory-hungry.
2. **R-level loop overhead**: `lapply` over 6.46M elements with non-trivial closures is inherently slow in interpreted R.
3. **Redundant work across years**: The neighbor *structure* is purely spatial (rook contiguity) and identical for every year. Yet the lookup is rebuilt per cell-year row by pasting year onto each neighbor ID. The same spatial neighbor set is re-resolved 28 times (once per year).
4. **`compute_neighbor_stats`** then iterates over the 6.46M-element list *again* for each of the 5 variables, applying `max`/`min`/`mean` to small vectors â€” another 32.3 million R function calls.
5. **Memory**: Storing a 6.46M-element list of integer vectors (the neighbor lookup) plus intermediate character vectors can easily exceed 16 GB.

---

## Optimization Strategy

**Key insight**: Because the neighbor graph is purely spatial and time-invariant, we can separate the spatial and temporal dimensions. Instead of building a 6.46M-row lookup, we build a ~344K-cell spatial lookup once, then use vectorized joins and grouped matrix operations per year.

**Concrete steps**:

| # | Technique | Benefit |
|---|-----------|---------|
| 1 | **Use `data.table`** for keyed joins instead of named-vector hash lookups and `paste` keys. | 10â€“50Ã— faster joins, lower memory. |
| 2 | **Expand the spatial neighbor list into an edge-list data.table once** (~1.37M rows), then join to cell-year data per variable. | Eliminates the 6.46M-element list entirely. |
| 3 | **Vectorized grouped aggregation** (`data.table` `[, .(max, min, mean), by=...]`) instead of `lapply` + per-element `max`/`min`/`mean`. | Replaces millions of R calls with a single C-level grouped operation. |
| 4 | **Process one year at a time** (28 iterations) to cap peak memory. Each year slice is ~344K rows; the edge-list join produces ~1.37M rows â€” trivially small. | Keeps RAM well under 16 GB. |
| 5 | **Preserve the original numerical result exactly**: same `max`, `min`, `mean` of the same neighbor values â†’ identical features â†’ no need to retrain the Random Forest. | Model compatibility. |

**Expected speedup**: From 86+ hours to roughly **5â€“15 minutes** on the same laptop.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a spatial edge-list (done once, ~1.37M rows)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# id_order        : integer vector of cell IDs in the order matching the nb object
# rook_neighbors_unique : spdep nb object (list of integer index vectors)

build_spatial_edge_list <- function(id_order, neighbors) {
  # For each cell, expand its neighbor indices into (focal_id, neighbor_id) pairs
  n <- length(neighbors)
  focal_idx <- rep(seq_len(n), lengths(neighbors))
  neigh_idx <- unlist(neighbors)
  
  # Remove any 0-length / empty-neighbor cells (they simply won't appear)
  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neigh_idx]
  )
}

edge_dt <- build_spatial_edge_list(id_order, rook_neighbors_unique)
# edge_dt has columns: focal_id, neighbor_id   (~1.37M rows)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# Ensure id and year columns are keyed for fast joins
setkey(cell_data, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Compute neighbor features â€” one variable at a time,
#          one year at a time to control memory
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate result columns with NA_real_
for (var in neighbor_source_vars) {
  cell_data[, paste0("neighbor_max_",  var) := NA_real_]
  cell_data[, paste0("neighbor_min_",  var) := NA_real_]
  cell_data[, paste0("neighbor_mean_", var) := NA_real_]
}

# Create a row-index column for fast assignment
cell_data[, .row_idx := .I]

# Key a lookup table: (id, year) -> row index in cell_data
# We will use this to write results back in place.
setkey(cell_data, id, year)

years <- sort(unique(cell_data$year))

for (var in neighbor_source_vars) {
  
  col_max  <- paste0("neighbor_max_",  var)
  col_min  <- paste0("neighbor_min_",  var)
  col_mean <- paste0("neighbor_mean_", var)
  
  for (yr in years) {
    # --- Slice this year's data (only the columns we need) ---
    yr_data <- cell_data[year == yr, .(id, val = get(var))]
    setkey(yr_data, id)
    
    # --- Join neighbor values via the edge list ---
    # For every (focal_id, neighbor_id) edge, attach the neighbor's value
    # edge_dt:  focal_id | neighbor_id
    # yr_data:  id       | val
    
    merged <- edge_dt[yr_data, on = .(neighbor_id = id), nomatch = 0L,
                      allow.cartesian = TRUE]
    # merged now has: focal_id | neighbor_id | val  (val = neighbor's value)
    
    # Remove NA neighbor values before aggregation (matches original logic)
    merged <- merged[!is.na(val)]
    
    # --- Grouped aggregation (single vectorized pass) ---
    if (nrow(merged) > 0L) {
      stats <- merged[, .(
        nmax  = max(val),
        nmin  = min(val),
        nmean = mean(val)
      ), by = focal_id]
      
      # --- Write results back into cell_data ---
      # Match on (id, year)
      stats[, year := yr]
      setkey(stats, focal_id, year)
      
      # Use cell_data's key (id, year) for indexed update
      idx <- cell_data[stats, on = .(id = focal_id, year), which = TRUE]
      
      set(cell_data, i = idx, j = col_max,  value = stats$nmax)
      set(cell_data, i = idx, j = col_min,  value = stats$nmin)
      set(cell_data, i = idx, j = col_mean, value = stats$nmean)
    }
  }
  
  message("Done: ", var)
}

# Clean up helper column
cell_data[, .row_idx := NULL]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict using the existing trained Random Forest
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The cell_data now has the same neighbor feature columns as the original
# pipeline produced (same names, same numerical values), so the trained
# model can be applied directly:
#
#   predictions <- predict(trained_rf_model, newdata = cell_data)
#
# No retraining is needed.
```

---

## Why This Preserves the Original Numerical Estimand

| Original code | Optimized code | Equivalence |
|---|---|---|
| For each cell-year row, collects neighbor cell-year rows via index lookup | For each (focal_id, year) group, joins to the same neighbor cells in the same year via the identical spatial edge list | Same set of neighbors |
| Removes `NA` neighbor values, computes `max`, `min`, `mean` | `merged[!is.na(val)]` then `data.table` grouped `max`, `min`, `mean` | Identical aggregation functions on identical inputs |
| Cells with no valid neighbors get `NA` | Cells absent from `stats` retain the pre-allocated `NA_real_` | Identical NA handling |

The features are numerically identical, so the trained Random Forest model produces the same predictions without retraining.

---

## Summary

| Metric | Before | After |
|---|---|---|
| Neighbor lookup structure | 6.46M-element R list (string hashing) | 1.37M-row `data.table` edge list (integer keys) |
| Aggregation | 6.46M Ã— 5 = 32.3M `lapply` calls | 28 Ã— 5 = 140 vectorized `data.table` group-by operations |
| Peak memory | >16 GB (string vectors + list) | ~2â€“4 GB (one year-slice + edge list at a time) |
| Estimated wall time | 86+ hours | **5â€“15 minutes** |
| Model impact | â€” | None; features are numerically identical |
END_RESPONSE>>>
