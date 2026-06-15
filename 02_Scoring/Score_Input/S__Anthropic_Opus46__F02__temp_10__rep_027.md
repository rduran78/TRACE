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
run_id: Anthropic_F02_T10_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup`** function. It calls `lapply` over ~6.46 million rows, and inside each iteration it performs:

1. **Character coercion and named-vector lookups** (`id_to_ref[as.character(...)]`, `idx_lookup[neighbor_keys]`) â€” named vector lookups in R are hash-table operations, but doing them millions of times in an interpreted loop is extremely slow.
2. **`paste()` key construction** for every neighbor of every row â€” this generates millions of small temporary character vectors.
3. **The result is a list of 6.46 million integer vectors**, consuming substantial memory.

Then **`compute_neighbor_stats`** iterates over that 6.46M-element list again, extracting values, filtering NAs, and computing max/min/mean â€” another interpreted loop with per-element allocation.

Multiplied across 5 variables, the estimated 86+ hour runtime is dominated by these two R-level interpreted loops over millions of elements, with heavy per-iteration allocation and hashing overhead.

---

## Optimization Strategy

**Replace interpreted R loops and character-key lookups with vectorized `data.table` joins.**

The key insight: the neighbor lookup is fundamentally a **merge/join** operation. For each `(cell_id, year)` row, we want to find the rows of its rook neighbors in the same year, then aggregate their variable values. This is exactly what `data.table` excels at â€” keyed equi-joins with grouped aggregation â€” and it operates in C, not interpreted R.

### Steps

1. **Expand the neighbor list into an edge table** (`data.table` with columns `id` and `neighbor_id`). This is done once and has ~1.37M rows (times the number of directed edges, but still very manageable).
2. **Join the edge table to the panel data** on `(neighbor_id, year)` to pull neighbor values. This is a single keyed join â€” `data.table` does this in seconds for millions of rows.
3. **Group-aggregate** by `(id, year)` to compute `max`, `min`, `mean` for each neighbor variable.
4. **Merge the aggregated stats back** into the main dataset.

This replaces both `build_neighbor_lookup` and `compute_neighbor_stats` entirely. No 6.46M-element list is ever created. Memory use drops dramatically (the edge table is ~10â€“50 MB; intermediate joins are handled column-wise).

**Expected speedup**: from 86+ hours to roughly **10â€“30 minutes** total for all 5 variables on a 16 GB laptop.

**Preservation guarantees**: The Random Forest model is not touched. The numerical outputs (max, min, mean of neighbor values) are identical to the original code â€” we are computing the same aggregates, just via join rather than loop.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build the edge table (once). Convert the spdep nb object into a
#    two-column data.table of (id, neighbor_id).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_table <- function(id_order, neighbors) {
  # id_order: vector of cell IDs in the same order as the nb object

  # neighbors: spdep nb list (rook_neighbors_unique)
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Convert the main panel to data.table (in-place, no copy needed).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Compute and attach neighbor features for all source variables.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {

  # --- Subset the columns we need (keep memory low) ---
  vals_dt <- cell_data[, .(neighbor_id = id, year, value = get(var))]
  setkey(vals_dt, neighbor_id, year)

  # --- Join edges â†’ values: for each (id, year), get every neighbor's value ---
  #     edge_dt provides (id, neighbor_id);
  #     we join on (neighbor_id, year) to pull the neighbor's value.
  merged <- merge(
    edge_dt,                         # (id, neighbor_id)
    vals_dt,                         # (neighbor_id, year, value)
    by = "neighbor_id",
    allow.cartesian = TRUE           # a neighbor_id appears in many years
  )
  # merged columns: neighbor_id, id, year, value

  # --- Aggregate by (id, year) ---
  agg <- merged[
    !is.na(value),
    .(
      nbr_max  = max(value),
      nbr_min  = min(value),
      nbr_mean = mean(value)
    ),
    keyby = .(id, year)
  ]

  # --- Rename to match the original feature naming convention ---
  new_names <- paste0(var, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  setnames(agg, c("nbr_max", "nbr_min", "nbr_mean"), new_names)

  # --- Left-join back onto the main table ---
  cell_data <- merge(cell_data, agg, by = c("id", "year"), all.x = TRUE)

  # --- Clean up intermediate objects to free memory ---
  rm(vals_dt, merged, agg)
  gc()

  message(sprintf("Done: %s", var))
}

# Convert back to data.frame if downstream code (predict.randomForest, etc.)
# expects one:
# setDF(cell_data)
```

---

## Why This Is Correct and Safe

| Concern | Explanation |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values as the original code. The join `(neighbor_id, year)` replicates the lookup `idx_lookup[neighbor_keys]`. |
| **NA handling** | `merged[!is.na(value), ...]` mirrors `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors get `NA` via the `all.x = TRUE` left join, matching the original `c(NA, NA, NA)` return. |
| **Random Forest model** | Untouched â€” we only prepare the same feature columns the trained model expects. |
| **Memory** | The edge table is ~1.37M rows Ã— 2 integer columns â‰ˆ 11 MB. The largest intermediate (`merged`) is at most ~(1.37M Ã— 28) â‰ˆ 38M rows Ã— 3 columns â‰ˆ 0.9 GB â€” well within 16 GB. Each iteration frees its intermediates with `rm(); gc()`. |
| **Speed** | `data.table` keyed joins and grouped aggregation on ~38M rows typically complete in seconds to low minutes per variable. Total: **~10â€“30 minutes for all 5 variables**. |

This single structural change â€” replacing interpreted per-row loops with vectorized `data.table` joins â€” eliminates the bottleneck without altering any outputs or the trained model.
END_RESPONSE>>>
