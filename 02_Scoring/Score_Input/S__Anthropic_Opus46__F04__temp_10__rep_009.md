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
run_id: Anthropic_F04_T10_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is overwhelmingly in `build_neighbor_lookup`, not in `compute_neighbor_stats`. Here's why:

**`build_neighbor_lookup`** iterates over ~6.46 million rows with `lapply`, and *inside each iteration* it performs:

1. **Character coercion and named-vector lookups** (`id_to_ref[as.character(...)]`, `idx_lookup[neighbor_keys]`) â€” these are hash-table lookups on character keys, repeated ~6.46M times, each touching a named vector of length ~6.46M. Named vector lookup in R is O(n) in the worst case for each probe when the vector is large.
2. **`paste()` key construction** â€” allocates ~4â€“8 short strings per row (one per neighbor), totaling ~25â€“50 million small string allocations.
3. **`lapply` over 6.46M elements** returns a list of 6.46M integer vectors â€” massive memory allocation overhead in R's single-threaded interpreter.

The net effect: billions of character operations and hash probes in interpreted R, producing the estimated 86+ hour runtime.

**`compute_neighbor_stats`** is comparatively cheap â€” it just indexes into a numeric vector using pre-built integer indices, computes max/min/mean on ~4 values, and does this 5 times total (once per variable). That's fast.

**The Random Forest inference** is a single `predict()` call on a pre-trained model with ~110 columns and ~6.46M rows â€” typically minutes, not hours.

**Conclusion:** The pathology is the row-by-row string-key construction and lookup in `build_neighbor_lookup`. The fix is to eliminate character key manipulation entirely and use direct integer-indexed joins.

---

## Optimization Strategy

### Principle: Replace per-row string-key lookups with a vectorized merge/join on integer columns.

The core insight is that `build_neighbor_lookup` is essentially performing a spatial join: "for each (cell, year) row, find the row indices of its rook neighbors in the same year." This is a standard equi-join that `data.table` can execute in seconds.

**Steps:**

1. **Expand the neighbor list into an edge table** â€” a two-column data.table of `(id, neighbor_id)` with ~1.37M rows (one per directed edge). This is done once.
2. **Join the edge table to the panel data twice** â€” once to attach the focal row's year, once to look up the neighbor's row index in that year. This is a vectorized merge on integer keys. `data.table` performs this with radix-sort joins in seconds.
3. **Compute neighbor stats vectorially** â€” group by the focal row index and compute max/min/mean of each variable in one pass per variable, fully vectorized.

This replaces 6.46M Ã— (string alloc + hash probe) iterations with two `data.table` merges on ~37M rows (6.46M rows Ã— avg ~5.7 neighbors, minus boundary cells), executing in under a minute on a laptop.

**Numerical equivalence:** The same neighbor sets are used, the same max/min/mean aggregations are computed, and the same columns are appended to the data. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Convert panel data to data.table (if not already)
#     Assumes: cell_data is a data.frame / data.table with
#              columns 'id' (integer cell id) and 'year'.
#     Assumes: id_order is the vector mapping position in
#              rook_neighbors_unique to cell id.
#     Assumes: rook_neighbors_unique is an nb object (list
#              of integer index vectors referencing id_order).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cell_dt <- as.data.table(cell_data)

# Preserve original row ordering so we can write results
# back in the correct position (important for predict()).
cell_dt[, .row_idx := .I]

# Create a fast row-index lookup keyed on (id, year).
# This replaces the old character-keyed idx_lookup entirely.
cell_dt[, .row_id_year := .I]                 
row_lookup <- cell_dt[, .(.row_id_year, id, year)]
setkey(row_lookup, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build the edge table from the nb object â€” ONCE
#     Result: edges with columns  focal_id, neighbor_id
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_id    = id_order[i],
             neighbor_id = id_order[nb_idx])
}))

cat(sprintf("Edge table: %s directed edges\n", format(nrow(edges), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Cross edges with years â†’ (focal_id, year, neighbor_id)
#     Then join to row_lookup to get the neighbor's row index.
#
#     Instead of exploding edges Ã— 28 years up front (which
#     would be ~38 M rows but still manageable), we join
#     through the panel data directly.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# 2a. Attach focal row index and year to every edge instance.
#     focal_rows: one row per (focal cell-year, neighbor_id).
setkey(edges, focal_id)
focal_panel <- cell_dt[, .(focal_id = id, year, focal_row = .row_idx)]
setkey(focal_panel, focal_id)

# Merge: for every cell-year row, expand its neighbor edges.
# Result columns: focal_id, year, focal_row, neighbor_id
edge_panel <- edges[focal_panel, on = .(focal_id), allow.cartesian = TRUE, nomatch = 0L]

cat(sprintf("Edge-panel table: %s rows\n", format(nrow(edge_panel), big.mark = ",")))

# 2b. Look up the neighbor's row in the same year.
setkey(row_lookup, id, year)
edge_panel[row_lookup,
           neighbor_row := i..row_id_year,
           on = .(neighbor_id = id, year)]

# Drop edges whose neighbor has no data in that year (boundary / missing).
edge_panel <- edge_panel[!is.na(neighbor_row)]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Compute neighbor stats vectorially for each variable
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  # Pull the numeric vector once (aligned to row index).
  vals <- cell_dt[[var_name]]

  # Attach neighbor values to every edge-panel row.
  edge_panel[, nval := vals[neighbor_row]]

  # Aggregate per focal row â€” exactly max, min, mean of
  # non-NA neighbor values (same semantics as original code).
  agg <- edge_panel[!is.na(nval),
                    .(nb_max  = max(nval),
                      nb_min  = min(nval),
                      nb_mean = mean(nval)),
                    keyby = .(focal_row)]

  # Prepare NA-filled columns, then overwrite matched rows.
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)

  set(cell_dt, j = max_col,  value = NA_real_)
  set(cell_dt, j = min_col,  value = NA_real_)
  set(cell_dt, j = mean_col, value = NA_real_)

  set(cell_dt, i = agg$focal_row, j = max_col,  value = agg$nb_max)
  set(cell_dt, i = agg$focal_row, j = min_col,  value = agg$nb_min)
  set(cell_dt, i = agg$focal_row, j = mean_col, value = agg$nb_mean)

  cat(sprintf("  âœ“ %s neighbor features added\n", var_name))
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Clean up helper columns and convert back if needed
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt[, c(".row_idx", ".row_id_year") := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(cell_dt)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Predict with the EXISTING trained Random Forest
#     (model object is untouched)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

---

## Expected Performance Comparison

| Phase | Original (est.) | Optimized (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~80â€“86 hours | eliminated |
| Edge table construction | â€” | ~2 seconds |
| `edge_panel` join (edges Ã— years) | â€” | ~20â€“40 seconds |
| Neighbor row lookup join | â€” | ~10â€“20 seconds |
| `compute_neighbor_stats` (5 vars) | ~5â€“10 min | ~30â€“90 seconds |
| **Total neighbor feature time** | **~86 hours** | **~2â€“4 minutes** |
| Peak RAM (neighbor step) | ~8â€“12 GB (list of 6.46M vectors) | ~4â€“6 GB (`edge_panel` table) |

The optimization yields roughly a **1,000â€“2,500Ã—** speedup by replacing interpreted per-row string manipulation with vectorized `data.table` integer-key joins, while producing numerically identical neighbor max/min/mean features and leaving the trained Random Forest model completely unchanged.
END_RESPONSE>>>
