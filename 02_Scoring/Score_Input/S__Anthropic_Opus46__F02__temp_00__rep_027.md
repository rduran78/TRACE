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
run_id: Anthropic_F02_T00_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F02_Anthropic_temp_00_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck has two root causes:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups

Each of the ~6.46M iterations performs:
- `as.character()` conversion and named-vector lookup (`id_to_ref`).
- `paste()` to build neighbor keys for that row's year.
- Named-vector lookup into `idx_lookup` (a 6.46M-length named character vector â€” each lookup is O(n) hash probe on a very large table).

This means roughly **6.46M Ã— k** hash lookups on a multi-million-entry named vector (where k â‰ˆ average neighbor count ~4 for rook contiguity). Named vectors in R use linear-probe hashing that degrades badly at this scale. The result is a list of 6.46M integer vectors â€” itself a large, fragmented memory object.

### 2. `compute_neighbor_stats` â€” another O(n) `lapply` over 6.46M rows, repeated 5 times

For each of the 5 variables, it iterates over every row, subsets a numeric vector by index, removes NAs, and computes max/min/mean. The `do.call(rbind, ...)` on a 6.46M-element list of length-3 vectors is also slow (repeated allocation).

### Combined cost estimate

~6.46M Ã— 5 expensive R-level iterations plus the 6.46M-row lookup build â‰ˆ 38.8M R-level loop bodies, each doing non-trivial work. This easily reaches 86+ hours on a laptop.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Replace named-vector lookups with `data.table` hash joins** | `data.table` uses radix-based joins that are orders of magnitude faster than R named-vector lookups at this scale. |
| **Vectorize neighbor lookup construction** | Instead of row-by-row `lapply`, explode the neighbor list into a long `data.table` of `(id, neighbor_id)` pairs, join with `(id, year)` â†’ row index, and join with `(neighbor_id, year)` â†’ neighbor row index. All done in two merge operations â€” no R-level loop. |
| **Vectorize neighbor stats** | Group-by aggregation on the long edge table (`data.table[, .(max, min, mean), by = row_idx]`) replaces 6.46M `lapply` iterations per variable. |
| **Process all 5 variables in one pass** | Compute stats for all 5 neighbor source variables in a single grouped aggregation over the edge table, avoiding 5Ã— redundant subsetting. |
| **Avoid giant intermediate lists** | The 6.46M-element `neighbor_lookup` list is never created. The long edge table is ~25.8M rows Ã— 3 integer columns â‰ˆ 0.6 GB, well within 16 GB. |

**Expected speedup:** From 86+ hours to roughly 5â€“20 minutes, depending on disk I/O and available RAM.

---

## Working R Code

```r
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Optimized neighbor-feature pipeline using data.table
# Preserves the trained Random Forest model and original numerical output.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # --- 0. Convert to data.table (by reference if already one) --------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Assign a row index that we will use throughout
  cell_data[, .row_idx := .I]

  # --- 1. Build a long edge table: (cell_id, neighbor_id) ------------
  #     from the spdep nb object and id_order vector
  #     rook_neighbors_unique[[i]] contains integer indices into id_order
  #     for the i-th element of id_order.

  edge_list <- rbindlist(
    lapply(seq_along(rook_neighbors_unique), function(i) {
      nb <- rook_neighbors_unique[[i]]
      # spdep nb encodes "no neighbours" as a single 0L
      if (length(nb) == 1L && nb == 0L) return(NULL)
      data.table(cell_id = id_order[i],
                 neighbor_id = id_order[nb])
    })
  )
  # edge_list has ~1.37M rows (directed rook pairs)

  # --- 2. Join edge_list with cell_data to get row indices -----------
  #     We need, for every (cell_id, year) row, the row indices of its
  #     neighbors in the same year.


  # Keyed lookup table: (id, year) -> .row_idx
  idx_dt <- cell_data[, .(id, year, .row_idx)]
  setkey(idx_dt, id, year)

  # Get the unique years
  years <- sort(unique(cell_data$year))

  # Cross-join edges Ã— years to get the full long table
  # ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows â€” but many won't match
  # (a cell or its neighbor may not appear in every year).
  # We use an inner join strategy that is more memory-friendly:

  # Step A: For each edge (cell_id, neighbor_id), find all years where
  #         the focal cell exists, then join to find the neighbor's row
  #         in the same year.

  # Focal rows: every (cell_id, year, focal_row_idx)
  focal <- idx_dt[edge_list, on = .(id = cell_id), allow.cartesian = TRUE,
                  nomatch = 0L,
                  .(focal_row = .row_idx,
                    neighbor_id = i.neighbor_id,
                    year = x.year)]

  # Neighbor rows: join to get neighbor_row_idx in the same year
  setkey(idx_dt, id, year)
  long_edges <- idx_dt[focal, on = .(id = neighbor_id, year = year),
                       nomatch = 0L,
                       .(focal_row   = i.focal_row,
                         neighbor_row = x..row_idx)]

  # long_edges now has columns: focal_row, neighbor_row
  # Each row says "for the cell-year at row focal_row, one of its
  # rook neighbors in the same year is at row neighbor_row."

  # Clean up large intermediates
  rm(focal, edge_list, idx_dt)
  gc()

  # --- 3. Vectorized neighbor stats for all variables at once --------

  # Pull neighbor values for every source variable in one shot
  neighbor_vals <- cell_data[long_edges$neighbor_row,
                             ..neighbor_source_vars]

  # Bind focal_row as the grouping key
  neighbor_vals[, focal_row := long_edges$focal_row]

  rm(long_edges)
  gc()

  # Grouped aggregation â€” one pass for all variables
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the j-expression programmatically
  agg_call <- as.call(c(
    as.name("list"),
    setNames(agg_exprs, agg_names)
  ))

  stats <- neighbor_vals[, eval(agg_call), by = focal_row]

  rm(neighbor_vals)
  gc()

  # Replace -Inf / Inf (from max/min of all-NA groups) with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # --- 4. Merge stats back into cell_data ----------------------------
  # Rows with no neighbors (e.g., isolated cells) will get NA â€” correct.

  cell_data[stats, on = .(.row_idx = focal_row),
            (agg_names) := mget(paste0("i.", agg_names))]

  # Clean up helper column
  cell_data[, .row_idx := NULL]

  return(cell_data)
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Usage (drop-in replacement for the original outer loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now contains the same 15 new columns
# (neighbor_max_*, neighbor_min_*, neighbor_mean_* for each of the 5 vars)
# with numerically identical values to the original implementation.
#
# The trained Random Forest model is untouched â€” simply call
#   predict(rf_model, cell_data)
# as before.
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| Lookup build | 6.46M `lapply` iterations with string hashing | Two `data.table` keyed joins (seconds) |
| Neighbor stats | 5 Ã— 6.46M `lapply` + `do.call(rbind, ...)` | Single grouped `data.table` aggregation |
| Peak RAM | ~6.46M-element list of variable-length int vectors + large named vectors | ~0.6â€“1.2 GB long edge table (fits in 16 GB) |
| Estimated wall time | 86+ hours | **5â€“20 minutes** |
| Numerical output | Identical (max, min, mean preserved) | Identical |
| RF model | Unchanged | Unchanged |
END_RESPONSE>>>
