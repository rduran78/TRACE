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
run_id: Anthropic_F02_T10_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is twofold:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows
For every row, the function performs character coercion, `paste` key construction, and named-vector lookups. Named-vector lookups in R are hash-tableâ€“based but the constant factor is large when called 6.46 million times inside `lapply`. The result is a **list of 6.46M integer vectors**, which is both slow to build and memory-heavy (each list element carries R object overhead â‰ˆ 100â€“200 bytes, so ~1â€“1.3 GB just in list overhead alone, before the actual index data).

### 2. `compute_neighbor_stats` â€” O(n) `lapply` iterated 5 times
For each of the 5 source variables, another 6.46M-element `lapply` computes `max/min/mean` over small integer-index subsets. The `do.call(rbind, ...)` at the end materialises a 6.46M Ã— 3 matrix from a list of 6.46M length-3 vectors â€” an extremely expensive pattern in R.

### Combined effect
- ~6.46M Ã— (1 + 5) = ~38.8M R-level anonymous-function calls.
- `paste`/character matching repeated millions of times.
- `do.call(rbind, list_of_millions)` is notoriously slow.
- Peak RAM easily exceeds 16 GB once you account for intermediate copies.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Row-level `lapply` + `paste` key lookup | Replace with **vectorised `data.table` merge/join** â€” build a flat `data.table` of `(row_index, neighbor_row_index)` in one shot using integer keys, not character keys. |
| Per-row `max/min/mean` in a loop | Compute all three stats in **one grouped `data.table` aggregation** per variable â€” completely vectorised in C. |
| `do.call(rbind, list_of_millions)` | Eliminated; `data.table` returns a single result table directly. |
| 5 separate passes over the edge list | Melt or loop is fine (only 5 iterations over a vectorised operation). |
| Memory: 6.46M-element list | Replaced by a flat two-column integer `data.table` of edges (~1.37M Ã— 28 â‰ˆ 38.5M rows Ã— 2 cols Ã— 8 bytes â‰ˆ 0.6 GB). |

**Expected speed-up**: from 86+ hours to roughly **5â€“20 minutes** on the same laptop, well within 16 GB RAM.

**Preservation guarantees**:
- The trained Random Forest model is not touched.
- The output columns are numerically identical (`max`, `min`, `mean` of the same neighbor values), preserving the original estimand.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a flat edge table:  (row_i, neighbor_row_i)
#     Completely replaces build_neighbor_lookup().
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_edge_table <- function(cell_data_dt, id_order, neighbors) {
 
  # --- Map each cell id to its position in id_order (integer key) ----
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # --- Build a data.table that maps (id, year) â†’ row index -----------
  #     Using integer keys avoids paste / character matching entirely.
  row_key <- cell_data_dt[, .(id, year, row_i = .I)]
  setkey(row_key, id, year)

  # --- Expand the nb object into a flat edge list of cell-id pairs ---
  #     neighbors[[k]] gives the *positions in id_order* of the
  #     neighbours of cell id_order[k].
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  edge_cells <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
  rm(from_ref, to_ref)

  # --- Cross-join edges with every year present in the data ----------
  years <- sort(unique(cell_data_dt$year))
  edge_cells_years <- edge_cells[, .(year = years), by = .(from_id, to_id)]
  rm(edge_cells)

  # --- Attach the originating row index (the row that *needs* the
  #     neighbour feature) ------------------------------------------------
  setnames(edge_cells_years, c("from_id"), c("id"))
  edge_cells_years <- row_key[edge_cells_years, on = .(id, year), nomatch = 0L]
  setnames(edge_cells_years, c("row_i", "id"), c("focal_row", "focal_id"))

  # --- Attach the neighbour row index ------------------------------------
  setnames(edge_cells_years, c("to_id"), c("id"))
  edge_cells_years <- row_key[edge_cells_years, on = .(id, year), nomatch = 0L]
  setnames(edge_cells_years, c("row_i", "id"), c("nbr_row", "nbr_id"))

  # Keep only the two columns we need for aggregation
  edge_cells_years[, .(focal_row, nbr_row)]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Vectorised neighbour stats for one variable.
#     Replaces compute_neighbor_stats() + compute_and_add_neighbor_features().
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_features_vec <- function(cell_data_dt, var_name, edges) {
  # Attach the neighbour's value to every edge
  edges[, nbr_val := cell_data_dt[[var_name]][nbr_row]]

  # Grouped aggregation â€” all three stats in one pass
  stats <- edges[!is.na(nbr_val),
                 .(nb_max  = max(nbr_val),
                   nb_min  = min(nbr_val),
                   nb_mean = mean(nbr_val)),
                 keyby = .(focal_row)]

  # Initialise output columns with NA (rows that have no valid neighbours)
  max_col  <- rep(NA_real_, nrow(cell_data_dt))
  min_col  <- rep(NA_real_, nrow(cell_data_dt))
  mean_col <- rep(NA_real_, nrow(cell_data_dt))

  max_col [stats$focal_row] <- stats$nb_max
  min_col [stats$focal_row] <- stats$nb_min
  mean_col[stats$focal_row] <- stats$nb_mean

  # Clean up the temporary column on the shared edges table
  edges[, nbr_val := NULL]

  # Return a named list of three vectors
  setNames(
    list(max_col, min_col, mean_col),
    paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  )
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Main pipeline â€” drop-in replacement for the outer loop.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
run_neighbor_feature_pipeline <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique) {
  cell_data_dt <- as.data.table(cell_data)

  message("Building flat edge table â€¦")
  edges <- build_neighbor_edge_table(cell_data_dt, id_order,
                                     rook_neighbors_unique)
  setkey(edges, focal_row)
  message(sprintf("  Edge table: %s rows (%.1f MB)",
                  formatC(nrow(edges), big.mark = ","),
                  object.size(edges) / 1e6))

  neighbor_source_vars <- c("ntl", "ec", "pop_density",
                            "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbour stats for '%s' â€¦", var_name))
    new_cols <- compute_neighbor_features_vec(cell_data_dt, var_name, edges)
    for (nm in names(new_cols)) {
      set(cell_data_dt, j = nm, value = new_cols[[nm]])
    }
  }

  message("Done.")
  # Return a data.table (also a data.frame) â€” compatible downstream
 cell_data_dt[]
}

# â”€â”€ Usage (preserves trained RF model; drop-in replacement) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- run_neighbor_feature_pipeline(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique
)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbour values as before (same rook-neighbor list, same year matching, same NA removal). |
| **Column naming** | Output columns follow the same `{var}_nb_max / _nb_min / _nb_mean` convention â€” adjust the suffix pattern if your RF model expects a different naming scheme. |
| **Trained RF model** | The code only adds feature columns to `cell_data`. The model object is never loaded, modified, or retrained. |
| **No approximation** | No sampling, no tiling, no lossy compression â€” the computation is exact. |

---

## Summary of Expected Gains

| Metric | Original | Optimised |
|---|---|---|
| R-level function calls | ~38.8 M | ~5 (one `data.table` grouped aggregation per variable) |
| Key construction | `paste` on 6.46M rows Ã— 6 | Integer join keys only |
| Memory pattern | 6.46M-element R list (~1.3 GB overhead) | Flat integer `data.table` (~0.6 GB) |
| Estimated wall time | 86+ hours | **5â€“20 minutes** |
| Peak RAM | >16 GB (swapping) | ~8â€“12 GB (fits in 16 GB) |
END_RESPONSE>>>
