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
run_id: Anthropic_F07_T00_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Per-row string pasting and named-vector lookup** â€” `paste()` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called ~6.46 million times inside an `lapply`. Each call does string allocation, hashing, and matching.
2. **The lookup is O(n) in list overhead** â€” `idx_lookup` is a named character vector of length ~6.46M. Named lookups on such a vector are O(n) or at best O(log n) per query, repeated for every neighbor of every row.
3. **`compute_neighbor_stats`** then iterates over the same 6.46M-element list, extracting values one-at-a-time with `lapply` â€” no vectorization.

Together these two functions produce ~86+ hours of runtime dominated by millions of small R-level string operations and hash lookups.

### Root causes (ranked):
| # | Cause | Impact |
|---|-------|--------|
| 1 | `build_neighbor_lookup`: per-row `paste` + named-vector lookup over 6.46M keys | ~95% of wall time |
| 2 | `compute_neighbor_stats`: R-level `lapply` over 6.46M rows | ~4% |
| 3 | Repeating the full lookup build once (not per variable) | Already correct â€” minor |

## Optimization Strategy

1. **Replace string-key lookups with integer join via `data.table`.** Build a `data.table` keyed on `(id, year)` with a row-index column. Then for each row, neighbor row-indices are resolved by a single keyed join â€” O(1) amortized per lookup.

2. **Vectorize the neighbor-stat computation.** Expand the neighbor list into a long `data.table` of `(row_i, neighbor_row_j)`, join the variable values, and compute `max/min/mean` with a single grouped aggregation â€” fully vectorized in C via `data.table`.

3. **Compute all 5 variables' stats in one pass over the edge list** (or 5 fast passes), avoiding any per-row R-level iteration.

4. **Memory budget.** The edge list has ~6.46M rows Ã— ~4 neighbors â‰ˆ 26M edges (directed). At 2 integer columns + 1 double column â‰ˆ ~600 MB peak, well within 16 GB.

**Expected speedup: from 86+ hours â†’ ~2â€“5 minutes.**

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build the directed edge list (row_i  â†’  row_j) ONCE
#     This replaces build_neighbor_lookup entirely.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_list <- function(cell_data, id_order, rook_neighbors_unique) {
  # cell_data must already be a data.table (or will be converted)
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]                        # preserve original row order

  # Map each cell id to its position in id_order (the nb object index)
  id_to_ref <- data.table(
    id  = id_order,
    ref = seq_along(id_order)
  )


  # Expand the nb object into a long edge table:  ref_from â†’ id_to
  # rook_neighbors_unique is a list of integer vectors (spdep nb object)
  edges_ref <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(r) {
    nb <- rook_neighbors_unique[[r]]
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(ref_from = r, ref_to = nb)
  }))

  # Translate ref indices back to cell ids
  edges_ref[, id_from := id_order[ref_from]]
  edges_ref[, id_to   := id_order[ref_to]]
  edges_ref[, c("ref_from", "ref_to") := NULL]

  # For every year, the edge (id_from, year) â†’ (id_to, year) exists.
  # We cross-join edges with years, then map to row indices.
  years <- sort(unique(dt$year))

  # Create a keyed lookup:  (id, year) â†’ row_idx
  setkey(dt, id, year)
  lookup <- dt[, .(id, year, row_idx)]

  # Cross join edges Ã— years
  edge_year <- CJ_edges_years(edges_ref, years)

  # Map from-side
  setnames(lookup, "row_idx", "row_from")
  setkey(lookup, id, year)
  edge_year <- lookup[edge_year, on = .(id = id_from, year), nomatch = 0L]
  setnames(edge_year, "id", "id_from")

  # Map to-side
  setnames(lookup, c("id", "year", "row_to"))
  setkey(lookup, id, year)
  edge_year <- lookup[edge_year, on = .(id = id_to, year), nomatch = 0L]

  # Keep only the two row-index columns

  edge_year[, .(row_from, row_to)]
}

# Helper: memory-efficient cross join of edges Ã— years
CJ_edges_years <- function(edges_ref, years) {
  # edges_ref has columns id_from, id_to
  # Replicate each edge for every year
  n_edges <- nrow(edges_ref)
  n_years <- length(years)
  dt <- data.table(
    id_from = rep(edges_ref$id_from, each = n_years),
    id_to   = rep(edges_ref$id_to,   each = n_years),
    year    = rep(years, times = n_edges)
  )
  dt
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Vectorized neighbor stats for one variable
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_stats_fast <- function(cell_data_dt, edge_dt, var_name) {
  # edge_dt has columns: row_from, row_to
  # Attach the neighbor (to-side) values
  vals <- cell_data_dt[[var_name]]
  work <- edge_dt[, .(row_from, nval = vals[row_to])]
  work <- work[!is.na(nval)]

  # Grouped aggregation â€” fully vectorized in C
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), keyby = row_from]

  # Allocate full-length result columns (NA for cells with no neighbors)
  n <- nrow(cell_data_dt)
  res <- data.table(
    nb_max  = rep(NA_real_, n),
    nb_min  = rep(NA_real_, n),
    nb_mean = rep(NA_real_, n)
  )
  res[agg$row_from, `:=`(
    nb_max  = agg$nb_max,
    nb_min  = agg$nb_min,
    nb_mean = agg$nb_mean
  )]

  # Name columns to match original convention
  prefix <- paste0("neighbor_", var_name, "_")
  setnames(res, c(
    paste0(prefix, "max"),
    paste0(prefix, "min"),
    paste0(prefix, "mean")
  ))
  res
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Main pipeline â€” drop-in replacement for the outer loop
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

## Convert once
cell_data_dt <- as.data.table(cell_data)
cell_data_dt[, row_idx := .I]

## Build edge list once  (~1-2 min, ~600 MB)
message("Building edge list â€¦")
edge_dt <- build_edge_list(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge list: %s directed edges", format(nrow(edge_dt), big.mark = ",")))

## Compute neighbor features for every source variable  (~seconds each)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("  neighbor stats: ", var_name)
  stats <- compute_neighbor_stats_fast(cell_data_dt, edge_dt, var_name)
  cell_data_dt <- cbind(cell_data_dt, stats)
}

## Drop helper column and convert back if needed
cell_data_dt[, row_idx := NULL]
cell_data <- as.data.frame(cell_data_dt)   # or keep as data.table downstream

message("Done. Neighbor features added for all variables.")
```

### Why this preserves correctness

| Requirement | Guarantee |
|---|---|
| **Same numerical estimand** | `max`, `min`, `mean` are computed over exactly the same neighbor sets â€” the edge list encodes the identical rook-neighbor topology from `rook_neighbors_unique`, filtered to rows that exist in the panel. NA handling is identical (NAs excluded before aggregation; all-NA â†’ NA). |
| **Trained Random Forest untouched** | No model code is modified. The output columns have the same names and values, so `predict(rf_model, cell_data)` produces identical predictions. |
| **Irregular / masked topology** | The edge list is built directly from the `nb` object â€” no rectangular-raster assumption. Missing cell-years (if any) are handled by the `nomatch = 0L` join. |

### Performance estimate

| Step | Time (approx.) | Peak RAM |
|---|---|---|
| `build_edge_list` | 1â€“2 min | ~600 MB |
| `compute_neighbor_stats_fast` Ã— 5 vars | ~30 sec total | ~200 MB working |
| **Total** | **~2â€“3 minutes** | **< 4 GB** |

This replaces the original ~86-hour runtime with a ~2â€“3 minute fully vectorized pipeline.
END_RESPONSE>>>
