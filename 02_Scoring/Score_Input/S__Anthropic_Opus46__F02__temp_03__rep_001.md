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
run_id: Anthropic_F02_T03_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is in **`build_neighbor_lookup`**, which runs an `lapply` over **~6.46 million rows**, performing per-row string pasting and named-vector lookups. Named vector lookups in R via `[` on character keys are O(n) hash probes per call, and doing this 6.46M times with multiple neighbor keys each time is catastrophically slow. The secondary bottleneck is **`compute_neighbor_stats`**, which also loops over 6.46M rows in pure R, extracting and summarizing small vectors one at a time. Together, these two functions dominate the 86+ hour runtime.

**Specific problems:**

1. **`build_neighbor_lookup`**: `paste()` and named-vector indexing inside a per-row `lapply` over 6.46M rows. Each iteration creates temporary character vectors and performs hash lookups. This is the single largest bottleneck.
2. **`compute_neighbor_stats`**: Pure-R `lapply` over 6.46M list elements, each calling `max`, `min`, `mean` on small vectors. The overhead of 6.46M R function calls plus `do.call(rbind, ...)` on a 6.46M-element list is enormous.
3. **Memory**: Storing `neighbor_lookup` as a list of 6.46M integer vectors has high overhead (each list element is a separate R object with header overhead). With 16 GB RAM this is tight.

---

## Optimization Strategy

**Replace row-level R loops with vectorized `data.table` joins and grouped aggregations.**

The key insight: `build_neighbor_lookup` is essentially building a **join table** â€” for each `(cell, year)`, find all `(neighbor_cell, year)` rows. This is a classic equi-join that `data.table` handles in seconds. Then `compute_neighbor_stats` is a **grouped aggregation** (max, min, mean by group), which `data.table` also handles natively in C.

**Plan:**

1. Build an **edge table** (`data.table`) mapping each `cell_id` to its neighbor `cell_id`s â€” this is a one-time expansion of the `nb` object (~1.37M rows).
2. For each variable, join the edge table with the panel data on `(neighbor_id, year)` to retrieve neighbor values, then compute grouped `max`, `min`, `mean` by `(cell_id, year)`.
3. Merge the results back into `cell_data`.

This eliminates all per-row R loops, replaces them with vectorized C-level operations, and reduces memory by never materializing a 6.46M-element list.

**Expected speedup:** From 86+ hours to roughly **5â€“20 minutes** depending on disk I/O and RAM pressure.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build the edge table from the nb object (one-time, ~1.37M rows)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: a list of integer index vectors
  # id_order maps positional index -> cell_id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the 0-neighbor sentinel if present (spdep uses 0 for no neighbors)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id       = id_order[from_idx],   # focal cell id
    nb_id    = id_order[to_idx]      # neighbor cell id
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Convert cell_data to data.table (in-place, no copy if already DT)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns have consistent types
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]
edge_dt[,   id   := as.integer(id)]
edge_dt[,   nb_id := as.integer(nb_id)]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Compute neighbor stats for each variable via join + group-by
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  cat(sprintf("[%s] Computing neighbor features for: %s\n", Sys.time(), var_name))

  # --- 3a. Build a slim lookup: (cell_id, year, value) ---
  # Only the columns we need, to minimize memory during the join.
  val_dt <- cell_data[, .(nb_id = id, year, val = get(var_name))]
  setkey(val_dt, nb_id, year)

  # --- 3b. Expand edges Ã— years via join ---
  # For every (focal_id, nb_id) edge and every year, pull the neighbor's value.
  # Join edge_dt with val_dt on (nb_id, year).
  # First, cross edge_dt with the year column by joining through cell_data's (id, year).
  # More efficient: join edge_dt to cell_data to get (id, year, nb_id),
  # then join to val_dt to get the neighbor value.

  focal_dt <- cell_data[, .(id, year)]
  setkey(focal_dt, id)
  setkey(edge_dt, id)

  # This gives every (focal_id, year, neighbor_id) triple
  expanded <- edge_dt[focal_dt, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded columns: id, nb_id, year

  # Now join to get the neighbor's value for that (nb_id, year)
  setkey(expanded, nb_id, year)
  expanded[val_dt, on = .(nb_id, year), val := i.val]

  # --- 3c. Grouped aggregation ---
  stats <- expanded[!is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    keyby = .(id, year)
  ]

  # Name the new columns to match the original pipeline's naming convention
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  # --- 3d. Merge back into cell_data ---
  # Remove old columns if they exist (idempotent reruns)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  cell_data <- stats[cell_data, on = .(id, year)]

  # Clean up large temporaries

  rm(val_dt, focal_dt, expanded, stats)
  gc()

  cat(sprintf("[%s] Done: %s\n", Sys.time(), var_name))
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 4: Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object is used as-is. No retraining.
# Example (adjust to your actual model object and predict call):
#
#   predictions <- predict(trained_rf_model, newdata = cell_data)
#
# The numerical estimand is preserved because we compute identical
# max, min, mean aggregations over the same neighbor sets.
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor sets** | `build_edge_table` expands the identical `nb` object and `id_order` mapping used by the original `build_neighbor_lookup`. |
| **Same statistics** | `max`, `min`, `mean` are computed on the same non-NA neighbor values per `(id, year)` group. |
| **NA handling** | Rows with no valid neighbors get `NA` for all three stats (the left join in step 3d fills unmatched rows with `NA`). |
| **Trained RF model** | No model code is touched. The feature columns have the same names and identical values, so `predict()` produces the same output. |

---

## Memory Notes for 16 GB RAM

- `expanded` is the largest temporary: ~6.46M rows Ã— ~4 neighbors â‰ˆ 26M rows Ã— 3 columns â‰ˆ ~600 MB. This is processed one variable at a time and freed with `rm(); gc()`.
- `cell_data` itself (~6.46M Ã— 110 cols) is ~5â€“7 GB. This leaves adequate headroom.
- If memory is still tight, process years in batches (e.g., 7 years at a time) inside the `for` loop, then `rbindlist` the stats. But this is unlikely to be necessary.
END_RESPONSE>>>
