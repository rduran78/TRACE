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
run_id: Anthropic_F07_T03_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It constructs a list of 6.46 million elements, where each element is built by:

1. **`paste()` key construction** â€” 6.46M calls, each generating a character vector of neighbor keys.
2. **Named-vector lookup via `idx_lookup[neighbor_keys]`** â€” This is an O(n) hash-table lookup *per row*, but the hash table itself has 6.46M entries, and the repeated character-key matching is extremely slow in R.
3. **`lapply` over 6.46M rows** â€” Pure R loop overhead is enormous at this scale.

Then `compute_neighbor_stats` does a second `lapply` over 6.46M elements, extracting values and computing `max/min/mean`. This is repeated 5 times (once per variable), but is comparatively cheaper than the lookup build.

**Root cause:** The algorithm is O(N_rows Ã— avg_neighbors) in *interpreted R character operations*. With ~6.46M rows and ~4 neighbors on average, that's ~26M character-paste-and-match operations inside an R `lapply`. Estimated 86+ hours is consistent with this.

**Key insight:** The neighbor structure is *time-invariant*. Cell `i`'s neighbors are the same in every year. The lookup can be built entirely with integer arithmetic using a merge/join, eliminating all character key construction and named-vector lookups. The statistics can then be computed via vectorized `data.table` grouped operations â€” no R-level loop at all.

## Optimization Strategy

1. **Replace the character-key lookup with an integer join.** Create an edge table (a two-column data.table of `(id, neighbor_id)`) from `rook_neighbors_unique`. Cross this with years via a merge on `id` to get `(id, year, neighbor_id)`, then join on `(neighbor_id, year)` to get the row index or value directly. This is a single vectorized `data.table` merge â€” no `lapply`, no `paste`.

2. **Compute all 5 variables' neighbor stats in one grouped aggregation** over the edge table, rather than looping 5 times.

3. **Memory:** The edge table has ~1.37M directed edges Ã— 28 years â‰ˆ 38.5M rows Ã— a few columns â€” well within 16 GB.

**Numerical equivalence:** The `max`, `min`, and `mean` are computed over exactly the same neighbor sets (non-NA rook neighbors in the same year), so the trained Random Forest model's inputs are preserved identically.

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                 "def", "usd_est_n2")) {

  # ---- Step 1: Build directed edge list from the nb object ----
  # rook_neighbors_unique is a list of integer vectors (spdep nb object).
  # Element k contains the indices (into id_order) of neighbors of id_order[k].
  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(k) {
    nb <- rook_neighbors_unique[[k]]
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(id = id_order[k], neighbor_id = id_order[nb])
  }))

  # ---- Step 2: Convert cell_data to data.table and key it ----
  dt <- as.data.table(cell_data)

  # Ensure original row order is preserved for final reassembly
  dt[, .row_order := .I]

  # ---- Step 3: Build the neighbor-value table via join ----
  # For each (id, year) row, we need the values of all neighbors in the same year.
  # Strategy: join edges with dt on neighbor_id == id to get neighbor values.


  # Subset dt to only the columns we need for the neighbor lookup
  value_cols <- intersect(neighbor_source_vars, names(dt))
  dt_vals <- dt[, c("id", "year", value_cols), with = FALSE]

  # Merge edges Ã— dt_vals on (id, year) to expand to (id, year, neighbor_id),
  # then merge again on (neighbor_id, year) to get neighbor values.
  # But more efficiently: merge edges with dt_vals on id = neighbor_id
  # to get neighbor values, keyed by the focal cell.

  # Rename for clarity in the join
  setnames(dt_vals, "id", "cell_id")

  # First: expand edges to all years by joining focal cell's years
  # focal_years: unique (id, year) pairs
  focal_years <- dt[, .(id, year)]

  # Join: for each focal (id, year), attach its neighbor_ids
  # edges has (id, neighbor_id); focal_years has (id, year)
  setkey(edges, id)
  setkey(focal_years, id)
  edge_years <- edges[focal_years, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_years now has columns: id, neighbor_id, year
  # meaning: focal cell `id` in `year` has neighbor `neighbor_id`

  # Now join to get the neighbor's values in that year
  setkey(dt_vals, cell_id, year)
  setkey(edge_years, neighbor_id, year)
  edge_vals <- dt_vals[edge_years, on = c("cell_id==neighbor_id", "year"), nomatch = NA]
  # edge_vals has: cell_id (= neighbor_id), year, <value_cols>, id (= focal id)
  # Rename for clarity
  setnames(edge_vals, "cell_id", "neighbor_id_actual")
  # The focal cell is in column "id", the neighbor is "neighbor_id_actual"

  # ---- Step 4: Grouped aggregation ----
  # For each (id, year), compute max, min, mean of each variable across neighbors
  agg_exprs <- list()
  for (v in value_cols) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("nb_max_", v)]] <- substitute(
      suppressWarnings(max(V, na.rm = TRUE)), list(V = v_sym))
    agg_exprs[[paste0("nb_min_", v)]] <- substitute(
      suppressWarnings(min(V, na.rm = TRUE)), list(V = v_sym))
    agg_exprs[[paste0("nb_mean_", v)]] <- substitute(
      mean(V, na.rm = TRUE), list(V = v_sym))
  }

  # Build the j-expression for data.table
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  nb_stats <- edge_vals[, eval(agg_call), by = .(id, year)]

  # Replace -Inf/Inf from max/min of all-NA groups with NA
  for (col_name in names(nb_stats)) {
    if (grepl("^nb_max_|^nb_min_", col_name)) {
      vals <- nb_stats[[col_name]]
      set(nb_stats, i = which(is.infinite(vals)), j = col_name, value = NA_real_)
    }
  }

  # ---- Step 5: Handle cells with zero neighbors (isolated cells) ----
  # These won't appear in nb_stats. After the merge below they'll get NA, which

  # matches the original code's behavior (returns c(NA, NA, NA)).

  # ---- Step 6: Merge back to original data ----
  setkey(dt, id, year)
  setkey(nb_stats, id, year)

  # Remove any pre-existing neighbor columns to avoid conflicts
  nb_col_names <- names(nb_stats)[!names(nb_stats) %in% c("id", "year")]
  for (cn in nb_col_names) {
    if (cn %in% names(dt)) dt[, (cn) := NULL]
  }

  dt <- nb_stats[dt, on = c("id", "year")]

  # Restore original row order
  setorder(dt, .row_order)
  dt[, .row_order := NULL]

  # ---- Step 7: Return as data.frame to preserve downstream compatibility ----
  as.data.frame(dt)
}

# ============================================================
# Usage â€” drop-in replacement for the original loop
# ============================================================
cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M `paste` + named-vector lookups in `lapply` | One `data.table` keyed join (vectorized C) |
| Stat computation | 5 Ã— `lapply` over 6.46M rows | Single grouped `data.table` aggregation |
| Estimated time | 86+ hours | **~2â€“5 minutes** (dominated by the ~38.5M-row join) |
| Peak memory | ~6.46M-element list of integer vectors + copies | ~38.5M-row edge table (~1â€“2 GB) â€” fits in 16 GB |
| Numerical result | max/min/mean of non-NA rook-neighbor values per cell-year | **Identical** â€” same neighbor sets, same aggregation functions |
| Trained RF model | Unchanged | **Unchanged** â€” input features are numerically equivalent |
END_RESPONSE>>>
