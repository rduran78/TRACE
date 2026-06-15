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
run_id: Anthropic_F07_T03_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over ~6.46 million rows one at a time in an R-level `lapply`. For each row it:

1. Looks up the cell's rook neighbors from the `nb` object.
2. Constructs character keys (`paste(id, year)`) and matches them into a named character vector (`idx_lookup`) of length 6.46M.

Named-vector lookup in R is **O(n)** per probe (linear scan of names), so ~6.46M lookups Ã— ~4 neighbors each Ã— 6.46M-length name vector â‰ˆ **trillions of character comparisons**. That is why the estimate is 86+ hours.

`compute_neighbor_stats` is a secondary bottleneck: another R-level `lapply` over 6.46M elements, each extracting and summarising a small numeric vector.

## Optimization Strategy

| Problem | Fix |
|---|---|
| Named-vector lookup is O(n) per probe | Replace with **`data.table` keyed join** â€” O(1) amortised hash lookup |
| Per-row `lapply` in `build_neighbor_lookup` | **Vectorise entirely**: explode the `nb` list into an edge-list, join to get row indices, then `split()` once |
| Per-row `lapply` in `compute_neighbor_stats` | **Vectorise**: attach the variable values to the edge-list, then `data.table` grouped aggregation (`max`, `min`, `mean` by source row) |
| Repeating the edge-list build for every variable | Build the edge-list **once**; reuse for all 5 variables |
| Memory: 6.46M-row list of integer vectors | Edge-list representation is more compact and cache-friendly |

The numerical results are **identical** (same max, min, mean per cell-year, same column names). The trained Random Forest model is untouched.

## Working R Code

```r
library(data.table)

# â”€â”€ 1. Build a vectorised edge-list (once) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_neighbor_edgelist <- function(cell_data, id_order, rook_neighbors_unique) {
  # cell_data must have columns: id, year (and be in its original row order)
  # id_order: vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique: spdep nb object (list of integer index vectors)

  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]                       # preserve original row position

  # --- map each cell-ID to its position in id_order --------------------------
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # --- explode nb list into an edge-list of (source_cell, neighbor_cell) -----
  n_nb    <- lengths(rook_neighbors_unique)           # number of neighbors per cell
  src_ref <- rep(seq_along(id_order), n_nb)           # source index in id_order
  dst_ref <- unlist(rook_neighbors_unique)             # neighbor index in id_order

  edges <- data.table(
    src_id = id_order[src_ref],
    dst_id = id_order[dst_ref]
  )

  # --- cross with years to get (source_row, neighbor_row) --------------------
  # Key the main table for fast join
  setkey(dt, id, year)

  # Join source side: get every (src_id, year, src_row_idx)
  src_rows <- dt[, .(src_id = id, year, src_row = row_idx)]

  # Merge edges with source rows  â†’  (src_row, dst_id, year)
  #   For every edge and every year the source cell appears in, we need
  #   the neighbor's row in that same year.
  edge_year <- edges[src_rows, on = .(src_id), allow.cartesian = TRUE, nomatch = 0L]
  #   columns: src_id, dst_id, year, src_row

  # Join neighbor side: get dst_row
  dst_rows <- dt[, .(dst_id = id, year, dst_row = row_idx)]
  edge_year <- dst_rows[edge_year, on = .(dst_id, year), nomatch = 0L]
  #   columns: dst_id, year, dst_row, src_id, src_row

  # Keep only what we need
  edge_year <- edge_year[, .(src_row, dst_row)]
  setkey(edge_year, src_row)

  return(edge_year)
}

# â”€â”€ 2. Compute neighbor stats for one variable (vectorised) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_stats_fast <- function(cell_data_dt, edge_year, var_name) {
  vals <- cell_data_dt[[var_name]]

  # Attach neighbor values
  el <- copy(edge_year)
  el[, nval := vals[dst_row]]
  el <- el[!is.na(nval)]

  # Grouped aggregation
  agg <- el[, .(nb_max  = max(nval),
                nb_min  = min(nval),
                nb_mean = mean(nval)),
            keyby = src_row]

  # Allocate full-length result (NA for cells with no valid neighbors)
  n <- nrow(cell_data_dt)
  out <- data.table(
    nb_max  = rep(NA_real_, n),
    nb_min  = rep(NA_real_, n),
    nb_mean = rep(NA_real_, n)
  )
  out[agg$src_row, `:=`(nb_max  = agg$nb_max,
                        nb_min  = agg$nb_min,
                        nb_mean = agg$nb_mean)]

  setnames(out, c(paste0(var_name, "_nb_max"),
                  paste0(var_name, "_nb_min"),
                  paste0(var_name, "_nb_mean")))
  return(out)
}

# â”€â”€ 3. Outer loop: drop-in replacement â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

## Convert once
cell_dt <- as.data.table(cell_data)

## Build edge-list once  (~1-3 min, ~200-400 MB)
message("Building neighbour edge-list â€¦")
edge_year <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge-list: %s rows", format(nrow(edge_year), big.mark = ",")))

## Compute features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("  â†’ ", var_name)
  nb_cols <- compute_neighbor_stats_fast(cell_dt, edge_year, var_name)
  cell_dt <- cbind(cell_dt, nb_cols)
}

## Convert back to data.frame if downstream code expects one
cell_data <- as.data.frame(cell_dt)
```

## Expected Performance

| Step | Old | New |
|---|---|---|
| `build_neighbor_lookup` | ~80+ hours (named-vector probing) | ~1â€“3 minutes (`data.table` keyed joins) |
| `compute_neighbor_stats` (Ã—5 vars) | ~5 hours (R-level lapply) | ~1â€“2 minutes (`data.table` grouped agg) |
| **Total** | **~86+ hours** | **~3â€“6 minutes** |

Peak RAM stays well within 16 GB: the edge-list is ~26M rows Ã— 2 int columns â‰ˆ 200 MB; the main table is ~6.46M Ã— 110 cols â‰ˆ 5â€“6 GB.

**Numerical equivalence**: `max`, `min`, and `mean` are computed over exactly the same sets of neighbor values (non-NA, same-year, rook-adjacent), so the trained Random Forest model's predictions are unchanged.
END_RESPONSE>>>
