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
run_id: Anthropic_F04_T00_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Looks those keys up in another named character vector (`idx_lookup`).

String concatenation (`paste`), named-vector lookups on character keys, and per-row `lapply` iteration over 6.46M rows are all extremely slow in R. The `compute_neighbor_stats` function then loops over the same 6.46M entries again per variable, using R-level `lapply` with `max`/`min`/`mean` on small vectorsâ€”slow but secondary.

**Root causes, ranked by impact:**

1. **Row-level `lapply` with string operations over 6.46M rows** in `build_neighbor_lookup` â€” O(N Ã— avg_neighbors) string pastes and named-vector lookups.
2. **Named character vector lookup** (`idx_lookup[neighbor_keys]`) is O(n) hash probing per call but the constant factor for 6.46M Ã— ~4 neighbors is enormous.
3. **`compute_neighbor_stats`** uses R-level loop over 6.46M entries per variable (Ã—5 variables).
4. **No vectorization or use of data.table/matrix indexing.**

## Optimization Strategy

**Core idea:** Replace all row-level R loops and string-key lookups with fully vectorized `data.table` joins and grouped aggregations.

- **Step 1:** Build an edge table (a two-column `data.table`) of all directed neighbor pairs from the `nb` object â€” done once, ~1.37M rows.
- **Step 2:** Join this spatial edge table to the panel data by year, producing a long table of (focal_row, neighbor_row) pairs â€” this is a single equi-join, fully vectorized.
- **Step 3:** For each variable, compute `max`, `min`, `mean` of neighbor values via a single grouped aggregation on the long table.

This eliminates all `lapply`, all `paste` key construction, and all named-vector lookups. Expected runtime: **minutes, not hours**.

## Working R Code

```r
library(data.table)

# â”€â”€ Step 0: Convert panel data to data.table (if not already) â”€â”€
cell_dt <- as.data.table(cell_data)

# Ensure id and year columns exist; create a row index for later re-merge
cell_dt[, .row_idx := .I]

# â”€â”€ Step 1: Build a spatial edge table from the nb object â”€â”€
# rook_neighbors_unique is a list of integer vectors (spdep nb object).
# id_order is the vector mapping position in the nb list â†’ cell id.
# neighbors[[i]] gives the positions (in id_order) of neighbors of cell id_order[i].

edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_i <- rook_neighbors_unique[[i]]
  if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_id = id_order[i], neighbor_id = id_order[nb_i])
}))
# edge_list has ~1.37M rows: (focal_id, neighbor_id)

# â”€â”€ Step 2: Build the full (focal_row, neighbor_row) mapping via join â”€â”€
# We need, for every cell-year row, the rows of its spatial neighbors in the SAME year.
# Strategy: join edge_list to cell_dt twice â€” once for focal, once for neighbor â€” keyed on year.

# Create lean lookup: id, year â†’ row index (and the variable columns we need)
keep_cols <- c("id", "year", ".row_idx", neighbor_source_vars)
lookup <- cell_dt[, ..keep_cols]

# Focal side: attach year (and focal row index) to each edge
#   For each focal_id, it appears in many years. We expand edges Ã— years.
focal_keys <- cell_dt[, .(id, year, focal_row = .row_idx)]

# Merge edges with focal keys to get (focal_row, neighbor_id, year)
setkey(edge_list, focal_id)
setkey(focal_keys, id)
expanded <- edge_list[focal_keys, on = .(focal_id = id),
                      .(focal_row, neighbor_id, year),
                      allow.cartesian = TRUE, nomatch = NULL]

# Now attach neighbor row index and neighbor variable values
# Build neighbor key table
neighbor_keys <- copy(lookup)
setnames(neighbor_keys, "id", "neighbor_id")
setnames(neighbor_keys, ".row_idx", "neighbor_row")

setkey(expanded, neighbor_id, year)
setkey(neighbor_keys, neighbor_id, year)

# Join to get neighbor variable values aligned to each (focal_row, neighbor_row)
joined <- neighbor_keys[expanded, on = .(neighbor_id, year), nomatch = NULL]
# joined now has columns: neighbor_id, year, neighbor_row, <var columns>, focal_row

# â”€â”€ Step 3: Grouped aggregation per variable â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (vname in neighbor_source_vars) {
  agg <- joined[!is.na(get(vname)),
                 .(nb_max  = max(get(vname)),
                   nb_min  = min(get(vname)),
                   nb_mean = mean(get(vname))),
                 by = focal_row]

  # Initialize columns with NA
  max_col  <- paste0(vname, "_nb_max")
  min_col  <- paste0(vname, "_nb_min")
  mean_col <- paste0(vname, "_nb_mean")

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  cell_dt[agg$focal_row, (max_col)  := agg$nb_max]
  cell_dt[agg$focal_row, (min_col)  := agg$nb_min]
  cell_dt[agg$focal_row, (mean_col) := agg$nb_mean]
}

# â”€â”€ Step 4: Convert back to data.frame if downstream code expects it â”€â”€
cell_dt[, .row_idx := NULL]
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is untouched.
# Numerical results are identical (same max/min/mean over same neighbor sets).
```

## Memory Note for 16 GB RAM

The `joined` table will have ~6.46M Ã— ~4 neighbors â‰ˆ 25.8M rows with ~7 columns of numeric/integer data â€” roughly **1.5â€“2 GB**. This fits in 16 GB RAM. If memory is tight, process variables one at a time inside the loop and keep only the needed column in `neighbor_keys` per iteration:

```r
for (vname in neighbor_source_vars) {
  nk <- cell_dt[, .(neighbor_id = id, year, val = get(vname))]
  setkey(nk, neighbor_id, year)
  j <- nk[expanded, on = .(neighbor_id, year), nomatch = NULL]
  agg <- j[!is.na(val), .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)),
           by = focal_row]
  cell_dt[agg$focal_row, paste0(vname, "_nb_max")  := agg$nb_max]
  cell_dt[agg$focal_row, paste0(vname, "_nb_min")  := agg$nb_min]
  cell_dt[agg$focal_row, paste0(vname, "_nb_mean") := agg$nb_mean]
  rm(nk, j, agg); gc()
}
```

This peaks at ~800 MB for the temporary tables and is safe on a 16 GB laptop.

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | `lapply` over 6.46M rows, `paste` keys | Single `data.table` equi-join |
| Neighbor stats | `lapply` per variable per row | Grouped `data.table` aggregation |
| Estimated runtime | 86+ hours | ~5â€“15 minutes |
| Numerical output | max/min/mean of neighbor values | Identical |
| RF model | Unchanged | Unchanged |
| Peak RAM | Moderate (but slow) | ~2 GB temporary (fits 16 GB) |
END_RESPONSE>>>
