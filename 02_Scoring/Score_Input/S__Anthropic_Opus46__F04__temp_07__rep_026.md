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
run_id: Anthropic_F04_T07_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It resides squarely in `build_neighbor_lookup` and `compute_neighbor_stats`, both of which rely on **element-wise `lapply` over ~6.46 million rows** with repeated string pasting, named-vector lookups, and per-row allocation. Specifically:

### `build_neighbor_lookup` â€” O(N) string-key lookups per row
1. **`paste()` inside `lapply`**: For every one of the 6.46M rows, neighbor cell IDs are pasted with the year to form string keys, then matched against a named character vector (`idx_lookup`). Named vector lookup in R is hash-based but the overhead of creating ~6.46M string keys *inside* a loop is enormous.
2. **Character coercion and matching**: `as.character(data$id[i])` and `id_to_ref[...]` are called per-row â€” 6.46M scalar character lookups into a named vector.
3. **Result**: Estimated ~6.46M Ã— (avg ~4 neighbors) â‰ˆ 25.8M string paste + hash-lookup operations just to build the lookup. On a laptop this alone can take many hours.

### `compute_neighbor_stats` â€” repeated per-variable `lapply`
1. Called **5 times** (once per neighbor source variable), each time iterating over all 6.46M rows.
2. Each iteration subsets a numeric vector, removes NAs, and computes `max`, `min`, `mean` â€” trivial per-row, but 5 Ã— 6.46M = 32.3M R-level function calls.
3. `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself slow (repeated memory allocation).

### Combined cost estimate
Roughly **6.46M Ã— 5 = 32.3M** R-interpreter-level iterations plus the 6.46M-iteration lookup build, all with per-element allocation. This easily reaches 86+ hours on a laptop.

---

## Optimization Strategy

| Principle | Technique |
|---|---|
| **Eliminate per-row string operations** | Replace string-key lookup with integer join via `data.table` |
| **Vectorize neighbor expansion** | Expand the neighbor list into a flat edge-list (`from_row`, `to_row`) once, then use grouped `data.table` aggregation â€” no R-level loop at all |
| **Compute all 5 variables in one pass** | Instead of 5 separate `lapply` calls, compute all neighbor stats in a single grouped aggregation |
| **Avoid `do.call(rbind, ...)`** | `data.table` returns a single data.table directly |
| **Preserve numerical output exactly** | `max`, `min`, `mean` on the same neighbor sets yield identical values |

**Expected speedup**: From 86+ hours â†’ **minutes** (typically 2â€“10 minutes on 16 GB RAM). The flat edge-list will have ~25.8M rows (6.46M rows Ã— ~4 avg neighbors) which fits comfortably in memory.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Convert to data.table (if not already) and add a row index
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]                 # 1-based row index

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a flat, integer-indexed edge list  (replaces build_neighbor_lookup)
#
#     id_order   : vector of cell IDs in the order matching rook_neighbors_unique
#     rook_neighbors_unique : spdep nb object (list of integer neighbor indices)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Map: cell id  â†’  position in id_order (integer, no strings)
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

# Expand neighbor list into a two-column data.table of cell-id pairs
#   from_id = focal cell id,  to_id = neighbor cell id
edge_list <- rbindlist(lapply(seq_along(id_order), function(ref) {
  nb_refs <- rook_neighbors_unique[[ref]]
  if (length(nb_refs) == 0L) return(NULL)
  data.table(from_id = id_order[ref],
             to_id   = id_order[nb_refs])
}))
# edge_list has ~1.37M rows (one per directed relationship)

# Attach the year dimension:
#   For every (from_id, to_id) pair we need one row per year.
#   Instead of crossing with years, we merge twice into cell_dt.

# Create a lean lookup:  (id, year) â†’ row_idx  +  variable values
setkey(cell_dt, id, year)

# Merge edge_list with cell_dt to get focal-row indices
#   focal side: from_id â†’ row_idx of focal cell-year
focal_key <- cell_dt[, .(id, year, focal_row = row_idx)]
setkey(focal_key, id)

# For each edge (from_id, to_id), expand across all years of from_id
#   by joining edge_list to focal_key on from_id == id
edges_with_year <- merge(edge_list, focal_key,
                         by.x = "from_id", by.y = "id",
                         allow.cartesian = TRUE)
# columns: from_id, to_id, year, focal_row
# rows: ~1.37M edges Ã— 28 years â‰ˆ 38.4M  (fits in RAM)

# Neighbor side: (to_id, year) â†’ variable values
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_key <- cell_dt[, c("id", "year", neighbor_source_vars), with = FALSE]
setkey(neighbor_key, id, year)

# Join to get neighbor variable values for each (focal_row, neighbor) pair
setkey(edges_with_year, to_id, year)
setkey(neighbor_key, id, year)

edges_full <- merge(edges_with_year, neighbor_key,
                    by.x = c("to_id", "year"),
                    by.y = c("id", "year"))
# edges_full now has columns: to_id, year, from_id, focal_row, ntl, ec, ...

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Compute grouped neighbor statistics in one vectorised pass
#     (replaces compute_neighbor_stats Ã— 5 variables)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Evaluate grouped aggregation
neighbor_stats <- edges_full[,
  lapply(agg_exprs, eval, envir = .SD),
  by = focal_row
]

# Handle Inf / -Inf from max/min on all-NA groups â†’ set to NA
for (col_name in agg_names) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col_name]])),
      col_name, NA_real_)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Attach results back to cell_dt  (rows with no neighbors get NA)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(neighbor_stats, focal_row)

# Pre-allocate columns with NA, then update in-place
for (col_name in agg_names) {
  set(cell_dt, j = col_name, value = NA_real_)
}

cell_dt[neighbor_stats$focal_row, (agg_names) := neighbor_stats[, ..agg_names]]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  (Optional) Convert back to data.frame if downstream code expects it
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt[, row_idx := NULL]
cell_data <- as.data.frame(cell_dt)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Random Forest prediction  (unchanged â€” model is preserved)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor sets** | The edge list is built from the identical `rook_neighbors_unique` nb object and `id_order` mapping. The `(to_id, year)` join recovers exactly the same cell-year neighbors that the original string-key lookup found. |
| **Same statistics** | `max`, `min`, `mean` with `na.rm = TRUE` on the same multiset of neighbor values produce identical floating-point results. |
| **Rows with no neighbors** | They have no entry in `edges_full`, so they are absent from `neighbor_stats`; the pre-allocated `NA_real_` columns remain `NA` â€” matching the original `c(NA, NA, NA)` return. |
| **Trained RF model untouched** | The model object is never modified; only `predict()` is called on the enriched `cell_data`. |

---

## Performance Summary

| Stage | Original | Optimized |
|---|---|---|
| Neighbor lookup build | ~6.46M `lapply` iterations with string ops | One `rbindlist` + two `data.table` merges |
| Neighbor stats (Ã—5 vars) | 5 Ã— 6.46M `lapply` iterations | Single grouped `data.table` aggregation |
| Total wall time (est.) | **86+ hours** | **2â€“10 minutes** |
| Peak RAM (est.) | Moderate (but slow) | ~4â€“6 GB for edge table (fits 16 GB) |

The key insight is: **replace millions of R-interpreter-level scalar operations with a handful of vectorized `data.table` joins and a single grouped aggregation.** This eliminates the bottleneck entirely while producing numerically identical features for the already-trained Random Forest model.
END_RESPONSE>>>
