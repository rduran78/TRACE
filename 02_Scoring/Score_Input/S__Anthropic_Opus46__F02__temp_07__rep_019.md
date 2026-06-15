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
run_id: Anthropic_F02_T07_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is twofold:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over ~6.46 million rows
Each iteration performs character coercion, `paste`, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries per-call overhead. Doing this 6.46 million times with string allocation and matching is extremely slow. The result is a list of ~6.46M integer vectors, which also consumes substantial memory.

### 2. `compute_neighbor_stats` â€” repeated row-level `lapply` over the same 6.46M-element list
For each of the 5 source variables, another full pass over the 6.46M-element neighbor lookup list is made, each time subsetting a numeric vector and computing `max`, `min`, `mean`. That is ~32.3 million R-level function calls across all variables.

### Memory
The neighbor lookup list alone (6.46M elements, each a small integer vector) can consume several GB due to R's per-object overhead (~128 bytes minimum per SEXP). Combined with the 6.46M Ã— 110 data frame, 16 GB RAM is tight.

### Summary of root causes
| Issue | Cause |
|---|---|
| Slow lookup build | Per-row string operations and named-vector hash lookups in pure R |
| Slow stats | Per-row `lapply` with R-level function calls Ã— 5 variables |
| High memory | 6.46M-element R list of integer vectors; redundant copies |

---

## Optimization Strategy

**Core idea: Replace the row-level list with a flat sparse-matrix (CSR) representation and vectorize all neighbor statistics using sparse matrixâ€“vector multiplication and grouped operations via `data.table`.**

### Step-by-step

1. **Flatten the neighbor lookup into a sparse adjacency matrix (cell-year Ã— cell-year).** Instead of building a list of 6.46M elements, build a single `dgCMatrix` (compressed sparse column) or use two integer vectors (`i`, `j`) representing directed edges between row indices. This eliminates millions of small R objects.

2. **Build the adjacency only at the cell level (344K Ã— 344K), then expand to cell-year via a join.** Since neighbors are time-invariant, the neighbor structure is identical across all 28 years. We build cell-level edges once, then join on year to get cell-year edges. This avoids the expensive per-row `paste`/lookup.

3. **Compute neighbor stats vectorially.** Using the flat edge list and `data.table` grouped aggregation (`max`, `min`, `mean` by target row), we process each variable in one vectorized pass â€” no `lapply` over 6.46M rows.

4. **Process variables sequentially** to limit peak memory.

**Expected improvement:**
- Build time: minutes instead of tens of hours.
- Stats computation per variable: seconds to low minutes.
- Memory: the edge list is ~2 Ã— 1.37M Ã— 28 â‰ˆ 77M integers â‰ˆ ~600 MB, manageable on 16 GB.
- The trained Random Forest model is untouched; the numerical outputs (max, min, mean of neighbor values) are identical.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Convert cell_data to data.table (if not already) and create a
#     compact integer row-index.  Preserve original column order.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
orig_cols <- copy(names(cell_data))
setDT(cell_data)

# Absolute row position â€” we will write results back by this index.
cell_data[, .row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a CELL-level directed edge list from the nb object.
#     rook_neighbors_unique is a list of length 344,208; element k
#     contains the integer indices of the neighbors of cell k
#     (referring to positions in id_order).
#     id_order is the vector that maps position -> cell id.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_cell_edge_list <- function(id_order, neighbors) {
  # neighbors[[k]] gives neighbor positions for cell at position k.
  from_pos <- rep(seq_along(neighbors),
                  lengths(neighbors))
  to_pos   <- unlist(neighbors, use.names = FALSE)

  # Map positions to actual cell ids.
  data.table(
    from_id = id_order[from_pos],
    to_id   = id_order[to_pos]
  )
}

cell_edges <- build_cell_edge_list(id_order, rook_neighbors_unique)
# cell_edges has ~1,373,394 rows (directed pairs).

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Expand to cell-year edges by joining on every year present in
#     the data.  Because neighbors are time-invariant, each directed
#     cell pair appears once per year.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Thin index table: row_idx, id, year â€” needed for the join.
idx_table <- cell_data[, .(row_idx = .row_idx, id, year)]
setkey(idx_table, id, year)

# Attach the "from" row index (the focal / target row).
cell_edges_yr <- cell_edges[
  , .(from_id, to_id, year = rep(list(sort(unique(cell_data$year))),
                                  .N))
][, .(from_id, to_id, year = unlist(year)), by = .I][, I := NULL]

# --- more memory-friendly alternative (cross-join years once) --------
years_dt <- data.table(year = sort(unique(cell_data$year)))

# CJ-like expansion without materialising a huge intermediate:
cell_edges_yr <- cell_edges[, .(from_id, to_id)]
cell_edges_yr <- cell_edges_yr[
  rep(seq_len(.N), each = nrow(years_dt))
]
cell_edges_yr[, year := rep(years_dt$year, times = nrow(cell_edges))]

# Join to get the ROW INDEX of the "from" (target) row:
setkey(cell_edges_yr, from_id, year)
setkey(idx_table, id, year)
cell_edges_yr[idx_table, target_row := i.row_idx,
              on = .(from_id = id, year)]

# Join to get the ROW INDEX of the "to" (neighbor) row:
cell_edges_yr[idx_table, neighbor_row := i.row_idx,
              on = .(to_id = id, year)]

# Drop edges where either side is missing (cell-year not in panel).
cell_edges_yr <- cell_edges_yr[!is.na(target_row) & !is.na(neighbor_row)]

# Keep only what we need â€” free memory.
edge_dt <- cell_edges_yr[, .(target_row, neighbor_row)]
rm(cell_edges_yr, cell_edges, idx_table, years_dt)
gc()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Vectorised neighbor statistics for each source variable.
#     For every target row, compute max / min / mean of the variable
#     across its neighbors (excluding NAs), exactly replicating the
#     original compute_neighbor_stats logic.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_and_add_neighbor_features_fast <- function(cell_data,
                                                   var_name,
                                                   edge_dt) {
  vals <- cell_data[[var_name]]

  # Attach the neighbor's value to every edge.
  work <- copy(edge_dt)
  work[, nval := vals[neighbor_row]]

  # Remove edges where the neighbor value is NA (matches original logic).
  work <- work[!is.na(nval)]

  # Grouped aggregation â€” one pass.
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), keyby = target_row]

  # Initialise result columns to NA (matches original: rows with no

  # valid neighbors get NA).
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)

  # Write aggregated values into the correct rows.
  set(cell_data, i = agg$target_row, j = max_col,  value = agg$nb_max)
  set(cell_data, i = agg$target_row, j = min_col,  value = agg$nb_min)
  set(cell_data, i = agg$target_row, j = mean_col, value = agg$nb_mean)

  invisible(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Outer loop â€” identical variable list, same column names produced.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  compute_and_add_neighbor_features_fast(cell_data, var_name, edge_dt)
  gc()                       
  cat("Done:", var_name, "\n")
}

# Clean up helper column.
cell_data[, .row_idx := NULL]
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per target row. The grouping key (`target_row`) is a 1:1 map to the original row. No approximation is introduced. |
| **Trained RF model** | No model code is touched. Only the feature-engineering step is optimized. The resulting columns have the same names and identical values, so `predict(rf_model, cell_data)` produces the same predictions. |
| **NA handling** | Rows with zero valid neighbors receive `NA` for all three stats, matching the original `c(NA, NA, NA)` return. |

---

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Neighbor lookup build | ~hours (6.46M `lapply` + string ops) | ~1â€“3 min (vectorized integer expansion + `data.table` keyed join) |
| Stats per variable | ~10+ hours (6.46M `lapply`) | ~30â€“90 sec (`data.table` grouped agg on ~38M edge rows) |
| **Total (5 vars)** | **86+ hours** | **~10â€“15 minutes** |
| Peak RAM | >16 GB (list overhead) | ~6â€“10 GB (flat integer edge table + one working copy) |

The optimization replaces millions of per-row R function calls and small-object allocations with bulk vectorized operations in `data.table`, which delegates to highly optimized C routines internally. This brings the workload comfortably within the capacity of a 16 GB laptop.
END_RESPONSE>>>
