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
run_id: Anthropic_F02_T10_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The root cause of the ~86+ hour runtime and excessive memory pressure is **row-level `lapply` iteration over 6.46 million rows** in both `build_neighbor_lookup` and `compute_neighbor_stats`, combined with millions of string-paste-and-match operations. Specifically:

1. **`build_neighbor_lookup`**: For every one of the ~6.46M rows, the code calls `paste()` to build character keys, then performs named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries significant per-call overhead when repeated millions of times. This creates ~6.46M small integer vectors stored in a flat list, which is also memory-heavy.

2. **`compute_neighbor_stats`**: Another `lapply` over the 6.46M-element list, subsetting a numeric vector and computing `max`, `min`, `mean` per element. The overhead of 6.46M R function calls, small vector allocations, and the final `do.call(rbind, ...)` on a 6.46M-element list is enormous.

3. **Memory**: The neighbor lookup list itself (6.46M elements, each a small integer vector) plus the intermediate character key vectors consume multiple gigabytes, putting severe pressure on a 16 GB machine.

4. **The outer loop** repeats `compute_neighbor_stats` for 5 variables sequentially, but the lookup structure is reused, so the main bottleneck is the lookup construction and the per-row stat computation.

---

## Optimization Strategy

**Replace all row-level R loops with vectorized `data.table` joins and grouped aggregations.**

The key insight: the neighbor lookup is essentially a **join**. Each cell-year needs to be joined to its neighbors' cell-years, and then we aggregate (max, min, mean) by the focal cell-year. This is exactly what `data.table` excels at.

### Steps

1. **Build an edge table** (`focal_id`, `neighbor_id`) from the `nb` object â€” this is ~1.37M rows (one-time, fast).
2. **Cross-join with years** using a `data.table` equi-join: join `edges` to `cell_data` on `(neighbor_id, year)` to pull neighbor values. This produces ~1.37M Ã— 28 â‰ˆ 38.5M rows, but `data.table` handles this in seconds with memory-efficient binary joins.
3. **Group-aggregate** by `(focal_id, year)` to get `max`, `min`, `mean` â€” a single vectorized pass.
4. **Join back** the aggregated stats to the original `cell_data`.

This eliminates all `lapply`, `paste`, named-vector lookups, and `do.call(rbind, ...)`. Expected runtime: **minutes, not hours**. Memory: the 38.5M-row intermediate table at ~5 columns is ~1.5 GB, well within 16 GB.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  One-time: convert the nb object to a data.table edge list
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  # id_order maps position -> cell id
  focal   <- rep(seq_along(neighbors), lengths(neighbors))
  neigh   <- unlist(neighbors, use.names = FALSE)

  # Translate position indices to actual cell IDs
  data.table(
    focal_id    = id_order[focal],
    neighbor_id = id_order[neigh]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows: (focal_id, neighbor_id)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Convert cell_data to data.table (in place if it already is one)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  setDT(cell_data)    # converts in place â€” no copy
}

# Ensure keyed for fast joins
setkey(cell_data, id, year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Vectorized neighbor-stat computation for all variables at once
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_all_neighbor_features <- function(cell_data, edge_dt, var_names) {
  # Columns we need from cell_data for the neighbor side of the join
  neighbor_cols <- c("id", "year", var_names)

  # Rename 'id' -> 'neighbor_id' so we can join on (neighbor_id, year)
  neighbor_dt <- cell_data[, ..neighbor_cols]
  setnames(neighbor_dt, "id", "neighbor_id")
  setkey(neighbor_dt, neighbor_id, year)

  # Build the full focalâ€“year table:
  # each edge is repeated for every year in the panel
  years <- sort(unique(cell_data$year))

  # Cross-join edges Ã— years  (~1.37M Ã— 28 â‰ˆ 38.5M rows)
  # Memory-efficient: use CJ inside a merge
  edge_year <- edge_dt[, CJ(focal_id    = focal_id,
                              neighbor_id = neighbor_id,
                              year        = years,
                              unique      = TRUE)]


  # ---- more memory-efficient alternative: expand per edge row ----
  # The CJ above would explode if focal_idâ€“neighbor_id pairs aren't unique
  # per original edge.  Since each edge IS unique, we can simply do:
  edge_year <- edge_dt[, .(year = years),
                        by = .(focal_id, neighbor_id)]
  setkey(edge_year, neighbor_id, year)

  # Join to pull neighbor values
  edge_year <- neighbor_dt[edge_year, on = .(neighbor_id, year)]
  # Now edge_year has columns: neighbor_id, year, <var_names>, focal_id

  # Aggregate by (focal_id, year) for each variable
  agg_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(var_names, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build a single aggregation call
  # Using .SD approach for clarity and correctness:
  stats_dt <- edge_year[,
    {
      out <- vector("list", length(var_names) * 3L)
      k   <- 0L
      for (v in var_names) {
        vals <- .SD[[v]]
        vals <- vals[!is.na(vals)]
        n    <- length(vals)
        k <- k + 1L; out[[k]] <- if (n == 0L) NA_real_ else max(vals)
        k <- k + 1L; out[[k]] <- if (n == 0L) NA_real_ else min(vals)
        k <- k + 1L; out[[k]] <- if (n == 0L) NA_real_ else mean(vals)
      }
      setattr(out, "names", agg_names)
      out
    },
    by = .(focal_id, year),
    .SDcols = var_names
  ]

  setnames(stats_dt, "focal_id", "id")
  setkey(stats_dt, id, year)
  return(stats_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Run it
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

stats_dt <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# Join the new features back to cell_data
cell_data <- stats_dt[cell_data, on = .(id, year)]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Predict with the EXISTING trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g., rf_model) is used as-is:
# cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Requirement | How it's met |
|---|---|
| **Same numerical estimand** | `max`, `min`, `mean` with `na.rm = TRUE` over exactly the same rook-neighbor sets produces identical values. No approximation is introduced. |
| **Trained RF model untouched** | The model object is never modified; only the feature-construction step is optimized. The column names and semantics are identical. |
| **16 GB RAM feasibility** | The largest intermediate object (~38.5M rows Ã— ~8 columns) is â‰ˆ 2.5 GB. `data.table` operates in-place where possible. Peak usage stays well under 16 GB. |

---

## Expected Performance Improvement

| Step | Original (estimated) | Optimized (estimated) |
|---|---|---|
| Neighbor lookup construction | ~10â€“20 hours | ~5 seconds (edge table build) |
| Neighbor stats (5 vars) | ~60+ hours | ~2â€“5 minutes (single grouped aggregation) |
| Join back to cell_data | negligible | ~2 seconds |
| **Total** | **~86+ hours** | **< 10 minutes** |

The speedup comes from replacing **6.46 million interpreted R function calls** with `data.table`'s C-level binary joins and grouped aggregation, which process the same logic in bulk with minimal memory allocation overhead.
END_RESPONSE>>>
