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
run_id: Anthropic_F02_T03_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_011.md

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

Each call iterates over every row, subsets a numeric vector by index, removes NAs, and computes max/min/mean. The `do.call(rbind, ...)` on a 6.46M-element list of length-3 vectors is also slow (repeated memory allocation).

### Combined cost estimate

| Step | Approximate cost |
|---|---|
| `build_neighbor_lookup` | ~40â€“60 hours (dominated by hash lookups on huge named vector) |
| `compute_neighbor_stats` Ã— 5 vars | ~20â€“30 hours (R-level loop, list allocation) |
| Memory: 6.46M-element list of integer vectors | ~2â€“4 GB just for the lookup; spikes during `rbind` |

---

## Optimization Strategy

### Principle: Replace per-row R-level operations with vectorized joins and matrix operations via `data.table`.

**Key insight:** The neighbor lookup is really a **merge/join** problem. Each row needs to find its neighbors' rows (same year, neighbor cell id). This is a classic equi-join that `data.table` handles in seconds, not hours.

### Step-by-step plan

1. **Flatten the `nb` object into an edge list** (cell_id â†’ neighbor_cell_id). This is done once, producing ~1.37M rows.

2. **Join the edge list to the data twice** â€” once to get the focal row's year, once to get the neighbor row's value â€” using `data.table` keyed joins. This replaces both `build_neighbor_lookup` and `compute_neighbor_stats` in one shot.

3. **Group-by aggregation** (`max`, `min`, `mean`) by focal row, computed inside `data.table` in C-optimized code.

4. **Repeat for each of the 5 variables** (or do all at once).

5. **No list-of-integer-vectors is ever created.** Memory stays flat and tabular.

### Expected improvement

| | Before | After |
|---|---|---|
| `build_neighbor_lookup` | ~50 hours | Eliminated |
| Per-variable stats | ~5 hours each | ~10â€“30 seconds each |
| Peak RAM | >10 GB (list overhead) | ~3â€“5 GB (tabular) |
| **Total wall time** | **86+ hours** | **~2â€“5 minutes** |

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Convert cell_data to data.table (non-destructive; keeps all cols)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# Ensure there is a row index we can use to put results back in order
cell_dt[, .row_id := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Flatten the nb object into an edge-list data.table
#     rook_neighbors_unique is a list of integer vectors (spdep::nb),
#     indexed by position in id_order.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_i <- rook_neighbors_unique[[i]]
  # spdep::nb encodes "no neighbors" as a single 0L

  nb_i <- nb_i[nb_i != 0L]
  if (length(nb_i) == 0L) return(NULL)
  data.table(
    focal_id    = id_order[i],
    neighbor_id = id_order[nb_i]
  )
}))
# edges now has ~1.37 M rows: (focal_id, neighbor_id)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Build a slim keyed table for joining: (id, year, .row_id, <vars>)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Columns we need from cell_dt for the neighbor side
keep_cols <- c("id", "year", neighbor_source_vars)
neighbor_vals <- cell_dt[, ..keep_cols]
setkey(neighbor_vals, id, year)

# Focal side: we need (id, year, .row_id) to identify each focal row
focal_keys <- cell_dt[, .(id, year, .row_id)]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Join:  focal row  â†’  edge list  â†’  neighbor row values
#
#     focal_keys  â‹ˆ  edges       on focal_keys.id = edges.focal_id
#                 â‹ˆ  neighbor_vals on edges.neighbor_id = neighbor_vals.id
#                                    AND focal_keys.year = neighbor_vals.year
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Step 3a: attach edges to every focal row (by id)
setkey(focal_keys, id)
setkey(edges, focal_id)

# This produces one row per (focal_row, neighbor_cell) combination per year
# ~6.46M rows Ã— ~4 neighbors â‰ˆ ~26M rows  (fits in RAM on 16 GB)
focal_edges <- edges[focal_keys,
                     .(neighbor_id, year, .row_id),
                     on = .(focal_id = id),
                     allow.cartesian = TRUE,
                     nomatch = NULL]

# Step 3b: look up the neighbor's variable values in the same year
setkey(focal_edges, neighbor_id, year)
joined <- neighbor_vals[focal_edges, on = .(id = neighbor_id, year = year), nomatch = NA]
# joined now has columns: id (=neighbor_id), year, ntl, ec, ..., .row_id

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Aggregate: for each focal row (.row_id), compute max/min/mean
#     for every neighbor source variable.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

# Build the aggregation call programmatically
agg_call <- parse(text = paste0(
  "joined[, .(",
  paste(
    mapply(function(nm, expr) paste0(nm, " = ", deparse(expr)),
           agg_names, agg_exprs),
    collapse = ", "
  ),
  "), by = .row_id]"
))

stats <- eval(agg_call)

# Replace -Inf/Inf from max/min on all-NA groups with NA
for (col in agg_names) {
  set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Merge aggregated stats back into the main table
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(stats, .row_id)
setkey(cell_dt, .row_id)
cell_dt <- stats[cell_dt, on = ".row_id"]

# Rows with no neighbors will have NA in the new columns (correct behavior,
# matching the original code's c(NA, NA, NA) return).

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6.  Convert back to data.frame if downstream code expects one,
#     and clean up the helper column.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt[, .row_id := NULL]
cell_data <- as.data.frame(cell_dt)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 7.  Predict with the existing (already-trained) Random Forest model
#     (unchanged â€” the feature values are numerically identical)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why the numerical estimand is preserved

Every neighbor statistic (`max`, `min`, `mean`) is computed over exactly the same set of neighbor values as the original code:

| Original code | Optimized code | Equivalence |
|---|---|---|
| `idx_lookup[neighbor_keys]` finds neighbor rows sharing the same `year` and having `id âˆˆ neighbors` | `data.table` equi-join on `(id, year)` restricted to the edge list | Same row set |
| `max(neighbor_vals)`, `min(...)`, `mean(...)` after removing NAs | `max(..., na.rm=TRUE)`, etc., inside `data.table` grouped aggregation | Identical IEEE-754 results |
| Rows with no neighbors â†’ `c(NA, NA, NA)` | No match in join â†’ `NA` after left join | Identical |

The trained Random Forest model is never touched â€” only the feature-construction step is optimized.

---

## Summary

| Concern | Resolution |
|---|---|
| **86+ hour runtime** | Reduced to ~2â€“5 minutes via vectorized `data.table` joins + grouped aggregation |
| **Memory pressure** | Flat tabular intermediates (~3â€“5 GB peak) instead of 6.46M-element R list |
| **Trained RF model** | Untouched; `predict()` call is unchanged |
| **Numerical equivalence** | Guaranteed by identical join logic and same aggregation functions |
END_RESPONSE>>>
