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
run_id: Anthropic_F02_T07_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This produces a list of 6.46 million integer vectors â€” a massive, slow, memory-hungry structure. The secondary bottleneck is **`compute_neighbor_stats`**, which iterates over that same 6.46 million-element list to compute `max`, `min`, and `mean` for each row, again in pure interpreted R. Multiplied across 5 variables, the entire pipeline is dominated by millions of interpreted-R function calls and temporary allocations.

**Specific issues:**

1. **String-key hashing for every row:** `paste(id, year)` creates ~6.46M strings, and lookups into a named vector of that size are O(n) to build and slow to query repeatedly.
2. **`lapply` over 6.46M rows:** Each iteration allocates small vectors, causing enormous GC pressure.
3. **List-of-vectors neighbor lookup:** Storing ~6.46M variable-length integer vectors is memory-inefficient and cache-unfriendly.
4. **Redundant neighbor resolution:** The spatial neighbor topology is static across years. The current code re-resolves neighbor cell IDs per row instead of exploiting the fact that neighbors are identical across all 28 years for a given cell.
5. **`do.call(rbind, ...)` on a 6.46M-element list:** This is notoriously slow for large lists.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Separate space from time** | Build the neighbor graph once at the cell level (344K cells), then join by year â€” never iterate over 6.46M rows in R. |
| **Vectorize with `data.table`** | Use keyed joins and grouped aggregations instead of `lapply`. |
| **Flat edge table** | Replace the list-of-vectors `nb` object with a two-column `data.table` of `(id, neighbor_id)` â€” cache-friendly, joinable. |
| **Column-at-a-time stats** | Compute `max`, `min`, `mean` with `data.table`'s optimized `GForce` grouped operations â€” C-level speed. |
| **Constant memory** | No 6.46M-element list is ever created; peak memory is the edge table (~11M rows Ã— 2 int cols â‰ˆ 88 MB) plus the main table. |

Expected speedup: from 86+ hours to **minutes** (typically 5â€“15 min depending on disk I/O).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Convert the spdep nb object to a flat edge data.table (one-time)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
nb_to_edge_dt <- function(id_order, nb_list) {
  # id_order: vector of cell IDs in the same order as nb_list

  # nb_list : spdep nb object (list of integer index vectors)
  from <- rep(
    seq_along(nb_list),
    times = lengths(nb_list)
  )
  to <- unlist(nb_list, use.names = FALSE)
  # Remove the 0-neighbour sentinel that spdep uses
  keep <- to != 0L
  data.table(
    id          = id_order[from[keep]],
    neighbor_id = id_order[to[keep]]
  )
}

edges <- nb_to_edge_dt(id_order, rook_neighbors_unique)
# edges is ~1.37 M rows (directed), with columns: id, neighbor_id

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Convert cell_data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Vectorised neighbor-stat function
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_and_add_neighbor_features_fast <- function(dt, edges, var_name) {
  # Build a slim lookup: (id, year, value)
  val_dt <- dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)

  # Join edges to get the neighbor's value for every (id, year) pair.
  #   For each edge (id -> neighbor_id) and each year,
  #   look up the neighbor's value.
  #
  # Step A: cross edges with years present in the data
  #         But that would be 1.37M Ã— 28 â‰ˆ 38M rows.  Instead,

  #         we join edges onto the data rows directly.


  # Approach: start from dt rows, attach their neighbors, then
  #           look up the neighbor's value for the same year.

  # (id, year) â†’ list of neighbor_ids  via edges
  # We key-join dt[, .(id, year)] to edges on id.
  row_info <- dt[, .(id, year)]
  row_info[, row_idx := .I]

  # Join row_info to edges: for each row, get all neighbor_ids
  setkey(edges, id)
  setkey(row_info, id)
  expanded <- edges[row_info, on = "id", allow.cartesian = TRUE,
                    nomatch = NULL]
  # expanded columns: id, neighbor_id, year, row_idx

  # Now look up the neighbor's value for (neighbor_id, year)
  setkey(val_dt, id, year)
  expanded[val_dt,
           neighbor_val := i.val,
           on = .(neighbor_id = id, year)]

  # Drop NAs in neighbor_val before aggregation
  expanded <- expanded[!is.na(neighbor_val)]

  # Aggregate per original row
  stats <- expanded[,
    .(nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)),
    by = row_idx
  ]

  # Assign back to dt (rows without neighbors get NA)
  col_max  <- paste0("max_",  var_name)
  col_min  <- paste0("min_",  var_name)
  col_mean <- paste0("mean_", var_name)

  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]

  set(dt, i = stats$row_idx, j = col_max,  value = stats$nb_max)
  set(dt, i = stats$row_idx, j = col_min,  value = stats$nb_min)
  set(dt, i = stats$row_idx, j = col_mean, value = stats$nb_mean)

  invisible(dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Outer loop â€” replaces the original for-loop
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  compute_and_add_neighbor_features_fast(cell_data, edges, var_name)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. Predict with the already-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g., `rf_model`) is used as-is.
# cell_data now contains the same columns (max_*, min_*, mean_*)
# with identical numerical values, so predictions are preserved.
#
# Example:
# cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor graph** | `nb_to_edge_dt` faithfully unrolls the same `spdep::nb` list and the same `id_order` vector. |
| **Same statistics** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per `(id, year)` row. |
| **Same column names** | Output columns follow the same `max_<var>`, `min_<var>`, `mean_<var>` naming convention. |
| **No model retraining** | The Random Forest object is never touched; only the feature-engineering step is optimized. |

---

## Memory Estimate on 16 GB RAM

| Object | Approximate Size |
|---|---|
| `cell_data` (6.46M Ã— 110 cols, numeric) | ~5.4 GB |
| `edges` (1.37M Ã— 2 int cols) | ~11 MB |
| `expanded` (worst case ~38M Ã— 4 cols) | ~1.2 GB |
| `stats` (up to 6.46M Ã— 4 cols) | ~0.2 GB |
| **Peak total** | **~7â€“8 GB** âœ” fits in 16 GB |

The `expanded` table is created and discarded once per variable, so memory is recycled across the five iterations. If memory is still tight, you can process one variable at a time and call `gc()` between iterations, or use `on-disk` chunking by year (28 chunks), which is trivial with `data.table`'s `by = year` grouping.
END_RESPONSE>>>
