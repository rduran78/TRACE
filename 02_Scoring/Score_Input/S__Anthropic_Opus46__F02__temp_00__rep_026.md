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
run_id: Anthropic_F02_T00_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F02_Anthropic_temp_00_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and named-vector lookups, which are extremely slow in interpreted R. The secondary bottleneck is `compute_neighbor_stats`, which again loops over 6.46 million entries with per-element subsetting and summary computation. Together, these two stages dominate the 86+ hour runtime.

**Specific problems:**

1. **String-key lookups in `build_neighbor_lookup`:** For every row, neighbor cell IDs are pasted with the year to form keys, then looked up in a named vector (`idx_lookup`). Named-vector lookup in R is O(n) per query in the worst case and involves repeated memory allocation of small character vectors. Over 6.46M rows Ã— ~4 neighbors each â‰ˆ 25M+ string operations and lookups.

2. **`lapply` over millions of rows:** R's `lapply` has per-iteration overhead. With 6.46M iterations, even microsecond overhead accumulates to hours.

3. **`do.call(rbind, result)` on a list of 6.46M small vectors:** This is a notoriously slow pattern in R for large lists.

4. **Memory pressure:** 6.46M rows Ã— 110 columns is already ~5â€“6 GB for numeric data. Building a 6.46M-element list of integer vectors for the neighbor lookup adds substantial memory overhead, and the repeated `data.frame` column binding in the loop compounds this.

---

## Optimization Strategy

**Replace all per-row R loops with vectorized and `data.table`-based operations:**

1. **Vectorized neighbor lookup via `data.table` join:** Instead of building a per-row list, create an edge table (`source_row â†’ neighbor_row`) using a single merge/join. This eliminates all string pasting and named-vector lookups.

2. **Vectorized neighbor stats via `data.table` grouped aggregation:** Instead of `lapply` over 6.46M elements, use `data.table`'s `[, .(max, min, mean), by = source_row]` on the edge table joined with variable values. This leverages C-level grouped aggregation.

3. **Avoid intermediate list structures entirely.**

4. **Process all 5 variables in a tight loop over the same edge table** â€” the edge table is built once.

**Expected speedup:** From 86+ hours to roughly 10â€“30 minutes, well within 16 GB RAM.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 0: Convert cell_data to data.table (if not already) and
#         add a row index. This is a zero-copy operation on columns.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)
cell_data[, row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build a vectorized edge table that maps every (cell, year)
#         row to its neighbor (cell, year) rows.
#
#         This replaces build_neighbor_lookup entirely.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(cell_data, id_order, neighbors) {
  # --- 1a. Expand the nb object into a two-column edge list of cell IDs
  #         (not row indices â€” spatial cell IDs).
  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells),
                  times = lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove any zero-length / self-referencing artifacts from spdep

  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  edges <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
  rm(from_idx, to_idx, valid)

  # --- 1b. Join with cell_data to attach year and row_idx for the
  #         *source* cell-year rows.
  #         We need: for every (from_id, year) â†’ row_idx of source
  #                  for every (to_id,   year) â†’ row_idx of neighbor
  id_year_map <- cell_data[, .(id, year, row_idx)]

  # Attach source row indices: every edge Ã— every year
  setkey(id_year_map, id)
  setkey(edges, from_id)

  # Cross-join edges with years via the source cell's years
  source_map <- id_year_map[, .(from_id = id, year, src_row = row_idx)]
  setkey(source_map, from_id)
  edge_year <- edges[source_map, on = "from_id",
                     allow.cartesian = TRUE, nomatch = 0L]
  rm(source_map)

  # Attach neighbor row indices
  nbr_map <- id_year_map[, .(to_id = id, year, nbr_row = row_idx)]
  setkey(edge_year, to_id, year)
  setkey(nbr_map, to_id, year)
  edge_year <- edge_year[nbr_map, on = c("to_id", "year"),
                         nomatch = 0L]
  rm(nbr_map, id_year_map)

  # Keep only the columns we need
  edge_year <- edge_year[, .(src_row, nbr_row)]
  setkey(edge_year, src_row)

  return(edge_year)
}

edge_table <- build_edge_table(cell_data, id_order, rook_neighbors_unique)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Compute neighbor stats for each variable using grouped
#         aggregation on the edge table.
#
#         This replaces compute_neighbor_stats + the outer loop.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  # Pull the variable values into the edge table
  edge_table[, nbr_val := cell_data[[var_name]][nbr_row]]

  # Grouped aggregation â€” runs at C level inside data.table
  stats <- edge_table[!is.na(nbr_val),
                      .(var_max  = max(nbr_val),
                        var_min  = min(nbr_val),
                        var_mean = mean(nbr_val)),
                      by = src_row]

  # Allocate new columns with NA, then fill matched rows
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  set(cell_data, j = col_max,  value = NA_real_)
  set(cell_data, j = col_min,  value = NA_real_)
  set(cell_data, j = col_mean, value = NA_real_)

  set(cell_data, i = stats$src_row, j = col_max,  value = stats$var_max)
  set(cell_data, i = stats$src_row, j = col_min,  value = stats$var_min)
  set(cell_data, i = stats$src_row, j = col_mean, value = stats$var_mean)

  rm(stats)
}

# Clean up helper column
edge_table[, nbr_val := NULL]
cell_data[, row_idx := NULL]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data now contains the same 15 new neighbor feature columns
# (3 stats Ã— 5 variables) as the original code, with identical
# numerical values.  The trained Random Forest model can be applied
# directly via predict(rf_model, cell_data) without retraining.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same neighbor sets; the edge table encodes the identical rook-neighbor relationships and year-matching logic as the original string-key lookup. |
| **NA handling** | `!is.na(nbr_val)` in the grouped aggregation mirrors the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Rows with no valid neighbors receive `NA` via the pre-initialized columns. |
| **Trained RF model** | No model object is touched. Only the feature columns are computed. Column names follow the same `neighbor_{stat}_{var}` convention. |

## Memory Estimate

| Object | Approximate Size |
|---|---|
| `cell_data` (6.46M Ã— 125 cols) | ~6.5 GB |
| `edge_table` (â‰ˆ38.4M rows Ã— 2 int cols) | ~0.6 GB |
| Temporary `stats` tables | ~0.15 GB peak |
| **Total peak** | **~7.3 GB** (fits in 16 GB) |

The edge table has ~38.4M rows (1,373,394 directed neighbor pairs Ã— 28 years), which is large but manageable as two integer columns. The entire pipeline should complete in **10â€“30 minutes** on a standard laptop.
END_RESPONSE>>>
