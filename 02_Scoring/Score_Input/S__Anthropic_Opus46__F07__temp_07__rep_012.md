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
run_id: Anthropic_F07_T07_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** creates ~6.46 million R closures/list elements via `lapply`, and for each row it performs character key construction, hash lookups, and NA filtering. This alone can take tens of hours because:

1. **Per-row string pasting and hash lookups**: For every one of the ~6.46M rows, `paste(neighbor_cell_ids, data$year[i], sep="_")` constructs character keys, then indexes into a named character vector (`idx_lookup`). Named vector lookup in R is O(n) per query when the vector is large, or at best O(1) with hashing â€” but the overhead of 6.46M calls is enormous.

2. **The neighbor lookup is a list of 6.46M integer vectors**: Building and storing this object is memory-intensive (~6.46M list elements Ã— average ~4 neighbors each).

3. **`compute_neighbor_stats`** then iterates over this 6.46M-element list again *for each of the 5 variables*, doing subsetting and summary stats in pure R loops.

4. **Total work**: ~6.46M Ã— 5 = ~32.3M R-level `lapply` iterations, each with vector subsetting, NA removal, and three summary functions. Combined with the lookup construction, this explains the 86+ hour estimate.

## Optimization Strategy

**Replace the row-level R loop with vectorized operations using `data.table` joins and grouped aggregation.**

The key insight: the neighbor relationship is between *cells* (not cell-years), and the panel is balanced (every cell appears in every year). So we can:

1. **Expand the neighbor list into an edge table** (`from_id`, `to_id`) â€” only ~1.37M rows (the directed rook-neighbor pairs).
2. **Join this edge table to the panel data by `(to_id, year)`** to pull in neighbor values â€” this produces ~1.37M Ã— 28 â‰ˆ ~38.4M rows, but `data.table` handles this in seconds.
3. **Group by `(from_id, year)` and compute `max`, `min`, `mean`** â€” fully vectorized, no R-level row loop.

This reduces the problem from 6.46M R-level iterations to a single vectorized join + grouped aggregation, bringing runtime from 86+ hours to **minutes**.

**The trained Random Forest model is untouched** â€” we only change how the *input features* are computed, and the numerical results are identical (same max/min/mean over the same neighbor sets).

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Convert panel data to data.table (if not already)
# ---------------------------------------------------------------
setDT(cell_data)

# ---------------------------------------------------------------
# 1.  Build a directed edge table from the spdep nb object
#     rook_neighbors_unique: list of integer vectors (indices into id_order)
#     id_order: vector of cell IDs in the order matching the nb object
# ---------------------------------------------------------------
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
    return(data.table(from_id = integer(0), to_id = integer(0)))
  }
  data.table(from_id = id_order[i], to_id = id_order[nb_idx])
}))

# This table has ~1,373,394 rows (one per directed rook-neighbor pair).
# Confirm:
cat("Directed neighbor edges:", nrow(edges), "\n")

# ---------------------------------------------------------------
# 2.  Vectorized neighbor-stat computation for each source variable
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Set keys for fast joins
setkey(cell_data, id, year)

for (var_name in neighbor_source_vars) {

  cat("Processing neighbor stats for:", var_name, "\n")

  # --- 2a. Build a slim table: (id, year, value) for the current variable
  val_dt <- cell_data[, .(id, year, value = get(var_name))]
  setnames(val_dt, "id", "to_id")
  setkey(val_dt, to_id, year)

  # --- 2b. Join edges to values:  for each (from_id, to_id) pair,
  #         expand across all years and attach the neighbor's value.
  #         Because the panel is balanced we can do a keyed join.
  edge_vals <- merge(edges, val_dt, by = "to_id", allow.cartesian = TRUE)
  #   edge_vals columns: to_id, from_id, year, value

  # --- 2c. Compute grouped stats: max, min, mean per (from_id, year)
  stats <- edge_vals[
    !is.na(value),
    .(nb_max  = max(value),
      nb_min  = min(value),
      nb_mean = mean(value)),
    keyby = .(from_id, year)
  ]

  # --- 2d. Rename for merge back into cell_data
  new_names <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), new_names)
  setnames(stats, "from_id", "id")

  # --- 2e. Remove old columns if they exist (idempotent re-runs)
  for (nn in new_names) {
    if (nn %in% names(cell_data)) cell_data[, (nn) := NULL]
  }

  # --- 2f. Merge stats back into the main panel
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)

  # Clean up to keep memory in check

  rm(val_dt, edge_vals, stats)
  gc()

  cat("  Done.\n")
}

# Re-apply key if needed downstream
setkey(cell_data, id, year)

cat("All neighbor features computed. Rows:", nrow(cell_data),
    " Cols:", ncol(cell_data), "\n")
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor sets** | `edges` is built from the identical `rook_neighbors_unique` nb object and `id_order` mapping â€” same directed pairs. |
| **Same summary statistics** | `max`, `min`, `mean` over the same non-NA neighbor values, grouped by `(from_id, year)` â€” algebraically identical to the original `compute_neighbor_stats`. |
| **NA handling** | `edge_vals[!is.na(value), ...]` mirrors `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with zero valid neighbors get `NA` via the `all.x = TRUE` left join â€” same as the original `c(NA, NA, NA)` return. |
| **Trained RF model untouched** | No model code is modified. The feature columns have the same names and identical values. |

## Expected Performance

| Step | Estimated time |
|---|---|
| Build `edges` table (~1.37M rows) | ~1â€“2 seconds |
| Per-variable join + aggregate (Ã—5) | ~30â€“90 seconds each |
| **Total** | **~3â€“8 minutes** |

Peak memory for the largest intermediate (`edge_vals`) is ~38.4M rows Ã— 4 columns â‰ˆ ~1.2 GB, well within 16 GB. Each intermediate is freed before the next variable.
END_RESPONSE>>>
