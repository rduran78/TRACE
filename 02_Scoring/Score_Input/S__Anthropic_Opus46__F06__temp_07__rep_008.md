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
run_id: Anthropic_F06_T07_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list of per-row neighbor indices

This function iterates over **~6.46 million rows** one at a time in an R `lapply`, performing character key lookups (`paste` + named-vector indexing) for every single cell-year observation. The `paste(id, year)` key construction and named-vector lookup is repeated millions of times, which is extremely slow in interpreted R.

### Bottleneck B: `compute_neighbor_stats` â€” per-row subsetting and summary

For each of ~6.46 million rows, the code subsets a numeric vector by variable-length index vectors, removes NAs, and computes `max`, `min`, `mean`. This is called **5 times** (once per neighbor source variable), yielding ~32.3 million R-level loop iterations total. Each iteration has overhead from anonymous function dispatch, subsetting, NA removal, and three summary function calls.

### Why raster focal/kernel operations are *not* directly applicable

Focal operations assume a regular rectangular grid with a fixed kernel shape. Here, the grid cells have a **panel (year) dimension**, the neighbor structure comes from a **precomputed `spdep::nb` object** (which may have irregular boundaries, islands, or missing cells in certain years), and we need to preserve exact numerical agreement with the original pipeline. A raster focal approach would require reshaping data into a 3D array and handling missing cells/years carefully â€” it could introduce subtle edge-case differences. The correct strategy is to **vectorize the existing logic using `data.table` joins and grouped aggregation**, which preserves exact results.

---

## 2. Optimization Strategy

| Step | Current | Proposed | Speedup source |
|---|---|---|---|
| Neighbor lookup | Per-row `lapply` with character key matching | Expand `spdep::nb` into a `data.table` edge list; merge on `(id, year)` to get row indices | Vectorized join, no per-row R loop |
| Neighbor stats | Per-row `lapply` computing max/min/mean | `data.table` grouped aggregation: `[, .(max, min, mean), by = source_row]` | C-level grouped aggregation |
| Repeat Ã—5 vars | 5 separate full passes | Single edge-list built once; 5 grouped aggregations (cheap) | Edge list reused |

**Expected runtime: ~1â€“5 minutes** instead of 86+ hours.

**Numerical equivalence**: The `max`, `min`, `mean` operations on the identical set of non-NA neighbor values produce bit-identical results. The Random Forest model is never retouched.

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Convert cell_data to data.table (if not already) and add row index
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)
cell_data[, .row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a vectorised edge list from the spdep::nb object
#     Each element of rook_neighbors_unique[[i]] gives the *positional*
#     indices (into id_order) of cell i's rook neighbors.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_dt <- function(id_order, nb_obj) {
  # nb_obj is a list of integer vectors (spdep::nb); 0L means no neighbors
  from_id <- rep(
    id_order,
    times = vapply(nb_obj, function(x) {
      if (length(x) == 1L && x[1] == 0L) 0L else length(x)
    }, integer(1))
  )
  to_idx <- unlist(lapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) integer(0) else x
  }), use.names = FALSE)
  to_id <- id_order[to_idx]
  data.table(focal_id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_dt(id_order, rook_neighbors_unique)
cat(sprintf("Edge list rows (directed): %s\n", format(nrow(edge_dt), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Create a lean lookup: (id, year) â†’ .row_idx
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
id_year_lookup <- cell_data[, .(id, year, .row_idx)]
setkey(id_year_lookup, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  For every year, expand the edge list so each row of cell_data
#     knows which rows are its neighbors.
#     Result: edge_full has columns  focal_row, neighbor_row
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years <- sort(unique(cell_data$year))

# Cross-join edges Ã— years, then map ids â†’ row indices
edge_year <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
edge_year[, `:=`(
  focal_id    = edge_dt$focal_id[edge_idx],
  neighbor_id = edge_dt$neighbor_id[edge_idx]
)]
edge_year[, edge_idx := NULL]

# Map focal (id, year) â†’ row index
edge_year[id_year_lookup, focal_row := i..row_idx,
          on = .(focal_id = id, year = year)]

# Map neighbor (id, year) â†’ row index
edge_year[id_year_lookup, neighbor_row := i..row_idx,
          on = .(neighbor_id = id, year = year)]

# Drop edges where either side is missing (cell absent in that year)
edge_full <- edge_year[!is.na(focal_row) & !is.na(neighbor_row),
                       .(focal_row, neighbor_row)]
rm(edge_year); gc()

cat(sprintf("Expanded edge rows: %s\n", format(nrow(edge_full), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Compute neighbor stats for each source variable and join back
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  cat(sprintf("Computing neighbor stats for: %s\n", var_name))

  # Attach the neighbor's value to each edge
  edge_full[, nval := cell_data[[var_name]][neighbor_row]]

  # Grouped aggregation â€” only over non-NA neighbor values
  stats <- edge_full[!is.na(nval),
                     .(nb_max  = max(nval),
                       nb_min  = min(nval),
                       nb_mean = mean(nval)),
                     keyby = .(focal_row)]

  # Prepare column names matching original pipeline
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  # Initialize with NA, then fill matched rows
  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)

  set(cell_data, i = stats$focal_row, j = max_col,  value = stats$nb_max)
  set(cell_data, i = stats$focal_row, j = min_col,  value = stats$nb_min)
  set(cell_data, i = stats$focal_row, j = mean_col, value = stats$nb_mean)

  edge_full[, nval := NULL]   # clean up temp column
}

# Clean up helper column
cell_data[, .row_idx := NULL]

cat("Done. All 15 neighbor feature columns added.\n")
```

### Memory note for 16 GB RAM

The `edge_full` table is the largest new object. Approximate size:

- ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows Ã— 2 integer columns = **~307 MB**

This fits comfortably alongside the ~6.46M-row `cell_data` (~5.4 GB at 110 columns of doubles). Total peak memory stays well under 16 GB.

### If memory is tight â€” chunked-year variant

If the full cross-join is too large (e.g., many more edges than stated), process year-by-year:

```r
for (yr in years) {
  idx_yr <- cell_data[year == yr, .(.row_idx, id)]
  setkey(idx_yr, id)

  edges_yr <- edge_dt[idx_yr, on = .(focal_id = id), nomatch = 0L,
                      .(focal_row = i..row_idx, neighbor_id)]
  edges_yr[idx_yr, neighbor_row := i..row_idx, on = .(neighbor_id = id)]
  edges_yr <- edges_yr[!is.na(neighbor_row), .(focal_row, neighbor_row)]

  for (var_name in neighbor_source_vars) {
    edges_yr[, nval := cell_data[[var_name]][neighbor_row]]
    stats <- edges_yr[!is.na(nval),
                      .(nb_max = max(nval), nb_min = min(nval), nb_mean = mean(nval)),
                      keyby = .(focal_row)]

    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    set(cell_data, i = stats$focal_row, j = max_col,  value = stats$nb_max)
    set(cell_data, i = stats$focal_row, j = min_col,  value = stats$nb_min)
    set(cell_data, i = stats$focal_row, j = mean_col, value = stats$nb_mean)

    edges_yr[, nval := NULL]
  }
}
```

---

## Summary

| Aspect | Guarantee |
|---|---|
| **Numerical equivalence** | Identical `max`, `min`, `mean` over the same non-NA neighbor value sets â€” no approximation. |
| **Random Forest model** | Untouched; only the feature-engineering step is optimized. |
| **Runtime** | ~1â€“5 minutes (vs. 86+ hours) on the same laptop. |
| **RAM** | Peak â‰ˆ 8â€“10 GB; fits in 16 GB. Year-chunked variant available if needed. |
END_RESPONSE>>>
