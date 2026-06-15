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
run_id: Anthropic_F07_T10_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements by looping row-by-row in R, performing character coercion, `paste`, and named-vector lookups inside each iteration. Specifically:

1. **`build_neighbor_lookup` is O(N Ã— k) with enormous constant factors.** For each of ~6.46M rows, it converts IDs to character, pastes keys, and does named-vector lookups. Named-vector lookup via `[` on a character-keyed vector is hash-table access, but doing it 6.46M times in an R `lapply` with repeated allocations is extremely slow.

2. **`compute_neighbor_stats` is also row-level R.** Another 6.46M-iteration `lapply`, subsetting a numeric vector and computing `max/min/mean` per row. This is comparatively cheaper per iteration but still unnecessarily slow.

3. **Memory pressure.** Storing 6.46M list elements (the neighbor lookup), each an integer vector, creates massive overhead from R's list/vector object headers (~6.46M SEXP allocations).

The fundamental insight: **this is a sparse-matrixâ€“vector product / grouped aggregation problem, not a per-row scripting problem.** The neighbor graph is static across years. We can represent the entire operation as a join + grouped aggregation in `data.table`, which is orders of magnitude faster.

---

## Optimization Strategy

1. **Explode the nb object into an edge table once** â€” a two-column `data.table` of `(id_from, id_to)` directed edges (~1.37M rows).

2. **Join edges to the panel on `(id_to, year)`** to get neighbor values â€” this is a keyed `data.table` equi-join, producing ~1.37M Ã— 28 â‰ˆ 38.5M rows, well within 16 GB RAM (~1â€“2 GB for 5 numeric columns).

3. **Group by `(id_from, year)` and compute `max`, `min`, `mean`** in one pass per variable â€” fully vectorized C-level aggregation inside `data.table`.

4. **Left-join the results back** to the master panel.

This replaces two nested R loops (6.46M iterations each Ã— 5 variables) with a handful of vectorized `data.table` operations. Expected wall time: **minutes, not days.**

5. **The trained Random Forest model and all numerical results are preserved** â€” we are computing the identical `max`, `min`, `mean` of the identical rook-neighbor values; only the execution path changes.

---

## Working R Code

```r
library(data.table)

# â”€â”€ 0. Convert panel to data.table (if not already) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€ 1. Explode the nb object into a directed edge table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : character or integer vector mapping list position â†’ cell id
#
#   Each element rook_neighbors_unique[[i]] contains the *positions* of
#   neighbors of id_order[i].  We map positions back to cell IDs.

edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_pos <- rook_neighbors_unique[[i]]
  # spdep nb lists use 0L to mean "no neighbors"

  nb_pos <- nb_pos[nb_pos > 0L]
  if (length(nb_pos) == 0L) return(NULL)
  data.table(id_from = id_order[i], id_to = id_order[nb_pos])
}))

cat(sprintf("Edge table: %d directed edges\n", nrow(edges)))

# â”€â”€ 2. Function: compute neighbor stats for one variable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_stats_dt <- function(panel, edges, var_name) {

  # Minimal subset for the join: id, year, value
  rhs <- panel[, .(id, year, val = get(var_name))]
  setkey(rhs, id, year)

  # Join: for every (id_from â†’ id_to) edge, attach id_to's value in each year
  #   result has columns: id_from, id_to, year, val
  joined <- rhs[edges, on = .(id = id_to), allow.cartesian = TRUE, nomatch = 0L]
  # After this join the column 'id' holds id_to; 'id_from' holds the focal cell.

  # Aggregate per (id_from, year)
  stats <- joined[!is.na(val),
                  .(nmax  = max(val),
                    nmin  = min(val),
                    nmean = mean(val)),
                  keyby = .(id_from, year)]

  # Rename for merge
  new_names <- paste0("n_", var_name, c("_max", "_min", "_mean"))
  setnames(stats, c("nmax", "nmin", "nmean"), new_names)

  stats
}

# â”€â”€ 3. Loop over the five source variables and left-join results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

setkey(cell_data, id, year)

for (vname in neighbor_source_vars) {
  cat(sprintf("Computing neighbor stats for: %s ...\n", vname))

  stats_dt <- compute_neighbor_stats_dt(cell_data, edges, vname)

  # Left-join onto the master panel
  new_cols <- paste0("n_", vname, c("_max", "_min", "_mean"))

  # Remove old columns if they already exist (idempotent reruns)
  for (nc in new_cols) {
    if (nc %in% names(cell_data)) cell_data[, (nc) := NULL]
  }

  cell_data <- stats_dt[cell_data, on = .(id_from = id, year)]

  # The join puts id_from as the key column; rename back to id

  setnames(cell_data, "id_from", "id")
  setkey(cell_data, id, year)

  cat(sprintf("  done â€“ added %s\n", paste(new_cols, collapse = ", ")))
}

cat("All neighbor features computed.\n")

# â”€â”€ 4. (Optional) Convert back to data.frame if downstream code expects it â”€â”€
# cell_data <- as.data.frame(cell_data)
```

---

### Why the results are identical

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor definition | `rook_neighbors_unique[[i]]` positions â†’ cell IDs | Same positions â†’ same cell IDs via `edges` table |
| Per-variable stats | `max`, `min`, `mean` of non-NA neighbor values | Identical: `joined[!is.na(val), .(max, min, mean)]` |
| NA handling | `neighbor_vals[!is.na(neighbor_vals)]` | `joined[!is.na(val), ...]` â€” same exclusion |
| No-neighbor cells | Return `(NA, NA, NA)` | Left join leaves `NA` for unmatched focal cells |

The trained Random Forest model is not touched; it consumes the same columns with the same values.

### Expected performance

| Step | Estimated time |
|---|---|
| Build edge table (~1.37M rows) | ~1â€“3 seconds |
| Per-variable join + aggregate (Ã—5) | ~20â€“60 seconds each |
| Total | **~2â€“5 minutes** (vs. 86+ hours) |

Peak additional RAM: ~2â€“3 GB for the largest intermediate join table (well within 16 GB).
END_RESPONSE>>>
