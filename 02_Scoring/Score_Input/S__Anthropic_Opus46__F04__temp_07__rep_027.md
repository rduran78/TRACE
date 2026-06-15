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
run_id: Anthropic_F04_T07_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The computational bottleneck is **not** the Random Forest inference. It resides squarely in the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` â€” Row-level `lapply` over 6.46M rows with per-row string operations

- **`paste()` and named-vector lookup inside a per-row lambda**: For each of ~6.46 million rows, the code calls `paste()` to construct neighbor keys and then does named-vector indexing (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is hash-table access, but doing it millions of times inside `lapply` with repeated string construction and `as.character()` coercions is extremely slow.
- **Redundant recomputation**: Every cell's neighbor set is identical across all 28 years. Yet the function re-derives the neighbor cell IDs and re-pastes keys for every cell-year row. With 344,208 cells Ã— 28 years, the same neighbor list is processed 28 times per cell.

### 2. `compute_neighbor_stats` â€” Row-level `lapply` over 6.46M rows with per-row subsetting

- **Per-row R-level loop**: For each row, it subsets a numeric vector, removes NAs, and computes `max`, `min`, `mean`. This is done ~6.46 million times per variable, and for 5 variables that is ~32.3 million R-level function-call iterations.
- **`do.call(rbind, result)` on a 6.46M-element list**: Binding millions of small vectors into a matrix via `do.call(rbind, ...)` is notoriously slow and memory-hungry.

### 3. Overall scaling

At ~6.46M rows Ã— 5 variables Ã— 3 stats = ~96.9M individual statistics, all computed via interpreted R loops, the 86+ hour estimate is consistent with the overhead.

---

## Optimization Strategy

The key principles are:

1. **Separate the spatial topology (which is year-invariant) from the panel expansion.** Build the neighbor lookup once at the cell level (344K cells), not at the cell-year level (6.46M rows).
2. **Vectorize the neighbor statistics computation using `data.table` grouped operations.** Instead of per-row `lapply`, explode the neighbor relationships into an edge list, join on variable values, and compute grouped `max`/`min`/`mean` in one vectorized pass.
3. **Avoid string key construction entirely.** Use integer-keyed joins throughout.
4. **Process all 5 variables in a single join pass** rather than looping and re-joining 5 times.

### Expected speedup

- The edge list for one year has ~1.37M directed edges; across 28 years that is ~38.5M rows â€” well within `data.table`'s comfort zone.
- Grouped aggregation on ~38.5M rows with integer keys is typically seconds, not hours.
- Estimated total runtime: **2â€“10 minutes** on a standard laptop with 16 GB RAM.

### Preservation guarantees

- The trained Random Forest model is untouched.
- The numerical output (neighbor max, min, mean per variable per cell-year) is identical to the original code's output.

---

## Working R Code

```r
library(data.table)

#' Optimized spatial neighbor feature construction.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and
#'                         all columns named in neighbor_source_vars.
#' @param id_order         integer vector of cell IDs in the order matching the
#'                         spdep::nb object (rook_neighbors_unique).
#' @param rook_neighbors_unique  spdep::nb object (list of integer index vectors).
#' @param neighbor_source_vars   character vector of variable names to summarize.
#' @return data.table with original columns plus neighbor feature columns appended.
build_neighbor_features_fast <- function(cell_data,
                                         id_order,
                                         rook_neighbors_unique,
                                         neighbor_source_vars) {

  # --- Step 0: Convert to data.table if needed (no copy if already dt) --------
  dt <- as.data.table(cell_data)

  # --- Step 1: Build the directed edge list at the CELL level (year-invariant) -
  #     This replaces the entire build_neighbor_lookup function.
  #     Result: a two-column integer data.table (focal_id, neighbor_id).
  edge_list <- rbindlist(lapply(seq_along(id_order), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(focal_id    = id_order[i],
               neighbor_id = id_order[nb_idx])
  }))
  # edge_list has ~1,373,394 rows (one per directed rook relationship).

  # --- Step 2: Expand edge list across years via join -------------------------
  #     We need neighbor values for each (focal_id, year) pair.
  #     Strategy: join edge_list with dt on neighbor_id == id, by year.

  # Subset dt to only the columns we need for the neighbor side.
  neighbor_cols <- c("id", "year", neighbor_source_vars)
  dt_neighbor   <- dt[, ..neighbor_cols]
  setnames(dt_neighbor, "id", "neighbor_id")

  # Key the neighbor data for fast join.
  setkey(dt_neighbor, neighbor_id, year)

  # Add year to edge_list by cross-joining with unique years.
  years <- sort(unique(dt$year))
  edges_by_year <- CJ(edge_idx = seq_len(nrow(edge_list)), year = years)
  edges_by_year[, focal_id    := edge_list$focal_id[edge_idx]]
  edges_by_year[, neighbor_id := edge_list$neighbor_id[edge_idx]]
  edges_by_year[, edge_idx    := NULL]
  # edges_by_year has ~1,373,394 * 28 â‰ˆ 38.5M rows.

  # Join to get neighbor variable values.
  setkey(edges_by_year, neighbor_id, year)
  edges_by_year <- dt_neighbor[edges_by_year, on = .(neighbor_id, year)]
  # Now edges_by_year has columns: neighbor_id, year, <vars>, focal_id

  # --- Step 3: Grouped aggregation â€” vectorized, single pass per variable -----
  setkey(edges_by_year, focal_id, year)

  # Build aggregation expressions dynamically for all variables at once.
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
      bquote(as.numeric(mean(.(as.name(v)),  na.rm = TRUE)))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  # Evaluate grouped aggregation.
  stats_dt <- edges_by_year[,
    eval(as.call(c(as.name("list"), agg_exprs))),
    by = .(focal_id, year)
  ]

  # Replace Inf/-Inf (from max/min on all-NA groups) with NA, matching original.
  for (col in agg_names) {
    set(stats_dt, which(is.infinite(stats_dt[[col]])), col, NA_real_)
  }

  # --- Step 4: Join stats back to the main data.table -------------------------
  setkey(stats_dt, focal_id, year)
  setkey(dt, id, year)
  # Rename focal_id -> id for the join.
  setnames(stats_dt, "focal_id", "id")

  dt <- stats_dt[dt, on = .(id, year)]

  return(dt)
}


# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Proceed directly to prediction:
# preds <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Changes

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | `lapply` over 6.46M rows, string paste + named-vector hash | Integer edge list built once over 344K cells |
| Year expansion | Implicit (re-derived per row) | Explicit `CJ` cross-join (~38.5M edge-year rows) |
| Neighbor value retrieval | Per-row named-vector indexing | Single `data.table` keyed join |
| Statistic computation | Per-row `lapply` with `max`/`min`/`mean` (6.46M Ã— 5 calls) | Single grouped `data.table` aggregation |
| Result assembly | `do.call(rbind, ...)` on 6.46M-element list | Native `data.table` column binding |
| Estimated runtime | 86+ hours | **~2â€“10 minutes** |
| Numerical result | Identical (max, min, mean with NA handling) | Identical |
| RAM peak | Moderate but slow | ~4â€“6 GB peak (fits in 16 GB) |
| Trained RF model | Untouched | Untouched |
END_RESPONSE>>>
