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
run_id: Anthropic_F05_T07_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: Per-Row `paste()`/Hash Lookup Inside `lapply` Over 6.46M Rows

1. **`idx_lookup` construction** (`paste` over 6.46M rows, then `setNames`) happens once â€” that's fine.
2. **But inside the `lapply` over every row `i`**, the code calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` and then indexes into the character-keyed named vector `idx_lookup[neighbor_keys]`. With ~6.46M rows and an average of ~4 rook neighbors each, that's **~25.8M `paste` + character hash lookups**, all in an interpreted R loop.

### Why It's Broader Than Just String Keys

The entire algorithmic structure is **row-centric** when it should be **cell-centric then joined**:

- The neighbor topology is **time-invariant** â€” cell `A`'s neighbors are the same in 1992 and 2019.
- Yet `build_neighbor_lookup` recomputes neighbor row-indices **per cell-year row**, doing 28Ã— redundant work for each cell.
- After that, `compute_neighbor_stats` loops over 6.46M entries again per variable (5 times).

**Total wasted work**: The `lapply` in `build_neighbor_lookup` alone is O(6.46M Ã— k) string operations. The `compute_neighbor_stats` loop is O(6.46M Ã— 5) R-level list iterations.

### Summary

| Layer | What's repeated | Multiplier |
|-------|----------------|------------|
| String key construction | `paste()` per neighbor per row | ~25.8M calls |
| Character hash lookup | Named vector indexing | ~25.8M lookups |
| Temporal redundancy | Same topology resolved 28Ã— per cell | 28Ã— |
| Variable loop | Separate `lapply` over 6.46M per variable | 5Ã— |

---

## Optimization Strategy

### Principle: Separate topology (time-invariant) from data (time-varying), vectorize everything.

1. **Build a cell-to-row mapping once** using integer indexing, not string hashing.
2. **Expand neighbor pairs into a flat edge table** (cell_i, cell_j) â€” ~1.37M directed edges.
3. **Join by year using vectorized `data.table` operations**: for each year, the edge table maps to row pairs. Compute all neighbor stats in one grouped aggregation per variable (or all at once).
4. **No `lapply` over 6.46M rows. No `paste`. No character lookups.**

Expected speedup: from ~86 hours to **minutes**.

---

## Working R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {

  # -----------------------------------------------------------
  # 1. Convert to data.table and create integer cell index
  # -----------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # Ensure a row-id column for final reordering
  dt[, .row_orig := .I]
  
  # Map cell id -> integer index matching id_order position
  id_map <- data.table(
    id     = as.integer(id_order),
    ref_idx = seq_along(id_order)
  )
  
  # -----------------------------------------------------------
  # 2. Build flat directed edge table from nb object (time-invariant)
  #    rook_neighbors_unique[[k]] gives integer indices into id_order
  #    for the neighbors of id_order[k].
  # -----------------------------------------------------------
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(k) {
    nb <- rook_neighbors_unique[[k]]
    # spdep nb encodes no-neighbor as 0L; skip those
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(
      focal_id    = id_order[k],
      neighbor_id = id_order[nb]
    )
  }))
  # edges now has ~1,373,394 rows: (focal_id, neighbor_id)
  
  # -----------------------------------------------------------
  # 3. Key the main table for fast join
  # -----------------------------------------------------------
  # We need to look up neighbor values by (neighbor_id, year).
  # Create a slim lookup table with only the columns we need.
  lookup_cols <- c("id", "year", neighbor_source_vars)
  lookup <- dt[, ..lookup_cols]
  setnames(lookup, "id", "neighbor_id")
  setkeyv(lookup, c("neighbor_id", "year"))
  
  # Also create a focal table to know which (focal_id, year) pairs exist
  focal_keys <- dt[, .(focal_id = id, year, .row_orig)]
  
  # -----------------------------------------------------------
  # 4. Expand edges Ã— years via join
  #    For each focal row (focal_id, year), attach its neighbor rows.
  # -----------------------------------------------------------
  # Merge focal keys with edges to get (focal_id, year, neighbor_id)
  # This is the cartesian product of each focal's year-rows with its neighbors.
  edge_year <- merge(focal_keys, edges, by = "focal_id", allow.cartesian = TRUE)
  # edge_year: (.row_orig, focal_id, year, neighbor_id)
  # Rows: ~1.37M edges Ã— 28 years â‰ˆ 38.4M â€” fits in RAM (~2-3 GB)
  
  # -----------------------------------------------------------
  # 5. Attach neighbor variable values
  # -----------------------------------------------------------
  setkeyv(edge_year, c("neighbor_id", "year"))
  edge_year <- lookup[edge_year, on = .(neighbor_id, year), nomatch = NA]
  # Now edge_year has the neighbor's variable values for the matching year
  
  # -----------------------------------------------------------
  # 6. Grouped aggregation: compute max, min, mean per focal row
  # -----------------------------------------------------------
  # Build aggregation expressions programmatically
  agg_exprs <- list()
  for (var in neighbor_source_vars) {
    vmax  <- paste0("neighbor_max_", var)
    vmin  <- paste0("neighbor_min_", var)
    vmean <- paste0("neighbor_mean_", var)
    agg_exprs[[vmax]]  <- substitute(
      suppressWarnings(max(v[!is.na(v)], na.rm = FALSE)),
      list(v = as.name(var))
    )
    agg_exprs[[vmin]]  <- substitute(
      suppressWarnings(min(v[!is.na(v)], na.rm = FALSE)),
      list(v = as.name(var))
    )
    agg_exprs[[vmean]] <- substitute(
      mean(v, na.rm = TRUE),
      list(v = as.name(var))
    )
  }
  
  # More efficient: use data.table's native aggregation
  # We aggregate by .row_orig (unique focal cell-year row)
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  
  stats <- edge_year[, eval(agg_call), by = .row_orig]
  
  # -----------------------------------------------------------
  # 7. Handle rows with no neighbors (they won't appear in stats)
  #    â€” they should get NA for all neighbor features.
  # -----------------------------------------------------------
  # Merge back to the original row ordering
  setkeyv(stats, ".row_orig")
  setkeyv(dt, ".row_orig")
  
  new_cols <- names(stats)[names(stats) != ".row_orig"]
  dt[stats, (new_cols) := mget(paste0("i.", new_cols)), on = ".row_orig"]
  
  # Rows not in stats already have NA from the join (data.table default)
  
  # -----------------------------------------------------------
  # 8. Clean up helper columns and convert back
  # -----------------------------------------------------------
  # Replace NaN from mean(na.rm=TRUE) on empty sets with NA
  for (col in new_cols) {
    dt[is.nan(get(col)), (col) := NA_real_]
    # Also fix -Inf/Inf from max/min on empty vectors
    dt[is.infinite(get(col)), (col) := NA_real_]
  }
  
  dt[, .row_orig := NULL]
  
  # Return as data.frame if the original was a data.frame
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# -----------------------------------------------------------
# Usage â€” drop-in replacement for the original outer loop
# -----------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names follow the pattern: neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, etc.
# Rename if needed to match the RF model's expected feature names:
# setnames(cell_data, old_names, new_names)
```

---

## Memory Budget Check

| Object | Approximate Size |
|--------|-----------------|
| `dt` (6.46M Ã— 110 cols) | ~5.7 GB |
| `edges` (1.37M Ã— 2 int cols) | ~11 MB |
| `edge_year` (38.4M Ã— ~8 cols) | ~2.5 GB |
| `stats` (6.46M Ã— 16 cols) | ~0.8 GB |
| **Peak** | **~9â€“10 GB** âœ“ fits in 16 GB |

If memory is tight, process variables in batches of 2â€“3 instead of all 5 at once (reduces `edge_year` width), or use the chunked variant below:

```r
# Memory-conservative variant: process one variable at a time
for (var in neighbor_source_vars) {
  lookup_one <- dt[, .(neighbor_id = id, year, val = get(var))]
  setkeyv(lookup_one, c("neighbor_id", "year"))
  ey <- lookup_one[edge_year_skeleton, on = .(neighbor_id, year), nomatch = NA]
  
  s <- ey[, .(
    vmax  = {v <- val[!is.na(val)]; if(length(v)) max(v) else NA_real_},
    vmin  = {v <- val[!is.na(val)]; if(length(v)) min(v) else NA_real_},
    vmean = mean(val, na.rm = TRUE)
  ), by = .row_orig]
  
  setnames(s, c(".row_orig",
    paste0("neighbor_max_", var),
    paste0("neighbor_min_", var),
    paste0("neighbor_mean_", var)))
  
  dt[s, (names(s)[-1]) := mget(paste0("i.", names(s)[-1])), on = ".row_orig"]
  rm(ey, s, lookup_one); gc()
}
```

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Algorithm** | Row-level `lapply` with string hashing | Vectorized `data.table` join + grouped agg |
| **String ops** | ~25.8M `paste()` calls | **Zero** |
| **Topology reuse** | Recomputed per year (28Ã—) | Computed once, joined |
| **Variable loop** | Separate 6.46M-row `lapply` per var | Single grouped aggregation |
| **Estimated time** | ~86+ hours | **~2â€“10 minutes** |
| **RF model** | Untouched âœ“ | Untouched âœ“ |
| **Numerical output** | max/min/mean per neighbor set | Identical estimand âœ“ |
END_RESPONSE>>>
