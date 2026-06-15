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
run_id: Anthropic_F07_T07_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` creates a list of 6.46 million elements, each produced by an `lapply` call that performs per-row string pasting, hash lookups, and NA filtering.** This is an O(n) loop with expensive string operations at every iteration (~6.46M iterations). Then `compute_neighbor_stats` loops over the same 6.46M entries again, once per variable. Combined, this yields approximately:

1. **~6.46M string `paste` + named-vector lookups** in `build_neighbor_lookup` â€” the dominant cost. Named vector lookup in R via `[` on character keys is O(n) in pathological cases and involves repeated hashing. With ~6.46M rows, the `idx_lookup` named vector is enormous, and each lookup into it is slow.
2. **~6.46M Ã— 5 = ~32.3M small `lapply` iterations** in `compute_neighbor_stats`, each allocating a tiny vector â€” death by a thousand cuts from R-level loop overhead and GC pressure.
3. **Memory pressure**: A 6.46M-element list of integer vectors, plus repeated copies of data columns, can easily push past comfortable limits on 16 GB RAM.

**Estimated cost of current approach**: The 86+ hour estimate is consistent with per-row string operations and R-level loops at this scale.

## Optimization Strategy

The key insight: **the neighbor graph is static across years, and the panel is balanced (every cell appears in every year).** Therefore we can:

1. **Vectorize the neighbor lookup entirely** using `data.table` joins instead of string-keyed named vectors. Map `(cell_id, year)` â†’ row index via a keyed `data.table`, then expand the neighbor list into an edge table `(source_row, target_row)` with a single equi-join on `(neighbor_cell_id, year)`.

2. **Compute all neighbor statistics in one vectorized pass per variable** using `data.table` grouped aggregation on the edge table â€” no R-level loops at all.

3. **Avoid creating a 6.46M-element list**. Instead, represent the lookup as a two-column integer matrix (edge list of row indices), which is compact and feeds directly into grouped operations.

**Expected speedup**: From 86+ hours to **minutes** (typically 5â€“15 minutes depending on disk I/O and RAM).

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Ensure cell_data is a data.table with original row order tracked
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
 cell_data <- as.data.table(cell_data)
}
cell_data[, .row_id := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a compact directed edge list of (cell_i, neighbor_cell_j)
#     from the spdep nb object, using the id_order mapping.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# rook_neighbors_unique is a list of length n_cells (344,208).
# rook_neighbors_unique[[k]] gives the integer indices (into id_order)
# of the rook-neighbors of cell id_order[k].

edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(k) {
 nb <- rook_neighbors_unique[[k]]
 if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
   return(NULL)
 }
 data.table(src_cell = id_order[k], dst_cell = id_order[nb])
}))

cat(sprintf("Edge list: %d directed rook-neighbor pairs\n", nrow(edges)))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Expand edge list across years by joining to the panel.
#     For each (src_cell, year) row, find the row indices of all
#     (dst_cell, same year) neighbor rows.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Minimal lookup: cell id + year -> row index
row_lookup <- cell_data[, .(id, year, .row_id)]
setkey(row_lookup, id, year)

# Attach source row ids
src <- row_lookup[edges, on = .(id = src_cell), allow.cartesian = TRUE,
                  nomatch = 0L]
setnames(src, ".row_id", "src_row")
# src now has columns: id (=src_cell), year, src_row, dst_cell

# Attach destination row ids
setkey(row_lookup, id, year)
edge_rows <- row_lookup[src, on = .(id = dst_cell, year = year),
                        allow.cartesian = TRUE, nomatch = 0L]
setnames(edge_rows, ".row_id", "dst_row")
# edge_rows has: src_row, dst_row  (plus id, year, etc.)

# Keep only what we need â€” compact integer edge table
edge_dt <- edge_rows[, .(src_row = src_row, dst_row = dst_row)]
setkey(edge_dt, src_row)

cat(sprintf("Expanded edge table: %d row-pairs across all years\n",
            nrow(edge_dt)))

# Free temporaries
rm(src, edge_rows, row_lookup)
gc()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Compute neighbor max, min, mean for each source variable
#     in one vectorized grouped aggregation per variable.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
 cat(sprintf("Computing neighbor stats for: %s\n", var_name))

 # Pull the variable values aligned to dst_row
 edge_dt[, val := cell_data[[var_name]][dst_row]]

 # Grouped aggregation â€” drops NAs within each group
 stats <- edge_dt[!is.na(val),
                  .(nb_max  = max(val),
                    nb_min  = min(val),
                    nb_mean = mean(val)),
                  keyby = src_row]

 # Initialize columns as NA (handles cells with no valid neighbors)
 max_col  <- paste0("n_max_",  var_name)
 min_col  <- paste0("n_min_",  var_name)
 mean_col <- paste0("n_mean_", var_name)

 cell_data[, (max_col)  := NA_real_]
 cell_data[, (min_col)  := NA_real_]
 cell_data[, (mean_col) := NA_real_]

 # Assign results by row index
 cell_data[stats$src_row, (max_col)  := stats$nb_max]
 cell_data[stats$src_row, (min_col)  := stats$nb_min]
 cell_data[stats$src_row, (mean_col) := stats$nb_mean]

 # Clean up the temporary column
 edge_dt[, val := NULL]

 cat(sprintf("  Done. Non-NA rows: %d / %d\n",
             sum(!is.na(cell_data[[max_col]])), nrow(cell_data)))
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Clean up helper column; convert back to data.frame if needed
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data[, .row_id := NULL]

# If downstream code (e.g., the trained Random Forest predict method)
# expects a plain data.frame:
# cell_data <- as.data.frame(cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | The edge list is built from the identical `rook_neighbors_unique` nb object and `id_order` mapping. No relationships are added or removed. |
| **Same year matching** | The join condition `on = .(id, year)` enforces that neighbors are matched within the same year, exactly as the original `paste(id, year)` key did. |
| **Same statistics** | `max`, `min`, `mean` with `NA` removal replicates the original `compute_neighbor_stats` logic identically. Cells with zero valid neighbors get `NA` in all three columns, matching the original `c(NA, NA, NA)` return. |
| **Same column names** | The naming convention `n_max_<var>`, `n_min_<var>`, `n_mean_<var>` matches what `compute_and_add_neighbor_features` would produce. Adjust the prefixes if your wrapper used different names. |
| **Trained RF model untouched** | No model object is modified. The code only computes predictor columns that already exist in the trained model's expected feature set. |
| **Numerical estimand preserved** | All operations are exact (no approximations, sampling, or rounding). The floating-point results are identical to the original code. |

## Expected Performance

| Stage | Estimated Time | Estimated Peak RAM |
|---|---|---|
| Edge list construction (344K cells) | ~2 seconds | ~50 MB |
| Edge expansion across 28 years (join) | ~1â€“3 minutes | ~2â€“4 GB |
| Per-variable grouped aggregation (Ã—5) | ~1â€“2 min each | ~1â€“2 GB working |
| **Total** | **~10â€“15 minutes** | **~6â€“8 GB peak** |

This replaces the 86+ hour runtime with a solution that completes in minutes on the specified 16 GB laptop.
END_RESPONSE>>>
