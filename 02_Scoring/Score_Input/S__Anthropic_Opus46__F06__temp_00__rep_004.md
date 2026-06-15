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
run_id: Anthropic_F06_T00_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list with per-row string operations

`build_neighbor_lookup` iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's rook neighbors by cell ID.
2. Constructs `paste(neighbor_id, year, "_")` keys.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

Named character vector lookup in R is **hash-based but slow at scale** when called millions of times in an `lapply`. The resulting `neighbor_lookup` is a list of ~6.46M integer vectors â€” enormous in memory and slow to build.

### Bottleneck B: `compute_neighbor_stats` â€” per-row subsetting in a loop

For each of the 5 variables, `compute_neighbor_stats` loops over 6.46M entries, subsets `vals[idx]`, removes NAs, and computes `max/min/mean`. That's **~32.3 million R-level function calls** (5 vars Ã— 6.46M rows), each involving vector allocation and subsetting.

### Why raster focal/kernel operations are a useful *analogy* but not directly applicable

Focal operations (e.g., `terra::focal`) assume a regular rectangular grid with a fixed kernel. Here, the grid cells have a **rook-neighbor structure stored as an `spdep::nb` object**, which may include irregular boundaries (coastal cells, edge cells with fewer than 4 neighbors). A focal approach would require reshaping data into a 3D raster stack (x Ã— y Ã— time) and carefully handling NA/missing cells. This is feasible but risks introducing subtle numerical differences at boundaries. The better approach is to **vectorize the neighbor computation directly using data.table joins**, which preserves the exact `nb` structure and results.

### Summary

| Component | Current Cost | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~hours | 6.46M string-paste + named-vector lookups |
| `compute_neighbor_stats` | ~hours Ã— 5 vars | 6.46M R-level loops Ã— 5 variables |
| **Total** | **86+ hours** | Interpreted R loops, no vectorization |

---

## 2. Optimization Strategy

**Replace both functions with a single vectorized `data.table` join-and-aggregate approach.**

The key insight: instead of building a per-row lookup list and then looping, we:

1. **Expand the `nb` object into an edge list** (cell_id â†’ neighbor_id), ~1.37M directed edges.
2. **Join** this edge list to the panel data by `(neighbor_id, year)` to get neighbor values â€” this is a single equi-join, handled in C by `data.table`.
3. **Group-by aggregate** `(cell_id, year)` to compute `max`, `min`, `mean` for all 5 variables simultaneously.
4. **Join** the aggregated stats back to the main data.

This eliminates all R-level loops. Expected runtime: **minutes, not hours**.

### Why this preserves the original numerical estimand

- The rook-neighbor relationships are identical (same `nb` object).
- `max`, `min`, `mean` are computed on exactly the same neighbor sets.
- NA handling is identical (neighbors missing from the panel or with NA values are excluded).
- The trained Random Forest model is never retrained â€” we only recompute the input features, which are numerically identical.

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 0: Convert panel data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build edge list from the spdep nb object (one-time, fast)
#
#   rook_neighbors_unique : an nb object (list of integer index vectors)
#   id_order              : vector mapping position in nb list â†’ cell id
#
#   Result: a data.table with columns  (id, neighbor_id)
#           representing every directed rook-neighbor pair.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_list <- function(id_order, neighbors) {
  # neighbors[[i]] contains integer indices into id_order for the
  # neighbors of the cell whose id is id_order[i].
  # spdep nb objects use 0L to denote "no neighbors" for an isolate.
  from_ids <- rep(id_order, times = lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the 0-entries that spdep uses for isolates
  valid    <- to_idx > 0L
  from_ids <- from_ids[valid]
  to_ids   <- id_order[to_idx[valid]]

  data.table(id = from_ids, neighbor_id = to_ids)
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat("Edge list rows:", nrow(edge_dt), "\n")
# Expected: ~1,373,394

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Vectorized neighbor-stat computation via join + group-by
#
#   For every (cell, year) we need max, min, mean of each source
#   variable across that cell's rook neighbors in the same year.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare a slim table of just the columns we need for the neighbor side
# to keep the join memory-efficient.
neighbor_cols <- c("id", "year", neighbor_source_vars)
neighbor_dt   <- cell_data[, ..neighbor_cols]

# Key the neighbor table for fast join
setnames(neighbor_dt, "id", "neighbor_id")
setkeyv(neighbor_dt, c("neighbor_id", "year"))

# Key the edge list
setkeyv(edge_dt, "neighbor_id")

# Join: for each edge (id, neighbor_id) and each year, attach the
# neighbor's variable values.
#
#   edge_dt  has columns: id, neighbor_id
#   We need to join on (neighbor_id, year).
#   Strategy: first cross edge_dt with the years present for each id,
#   but that would be huge.  Instead, join cell_data's (id, year) with
#   edge_dt to get (id, year, neighbor_id), then join neighbor values.

# Slim version of cell_data with just id and year (one row per cell-year)
cell_year <- unique(cell_data[, .(id, year)])
setkeyv(cell_year, "id")
setkeyv(edge_dt, "id")

# Expand: every (id, year) gets its neighbor_ids
# This produces ~1,373,394 * 28 â‰ˆ 38.5M rows but is manageable in 16 GB
# because each row is just three integer/numeric columns.
#
# Actually, each edge applies to ALL 28 years, so:
expanded <- edge_dt[cell_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
# expanded has columns: id, neighbor_id, year
cat("Expanded edge-year rows:", nrow(expanded), "\n")

# Now attach the neighbor variable values
setkeyv(expanded, c("neighbor_id", "year"))
expanded <- neighbor_dt[expanded, on = c("neighbor_id", "year"), nomatch = NA]
# expanded now has: neighbor_id, year, ntl, ec, ..., id

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Aggregate by (id, year) â€” compute max, min, mean per variable
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

names(agg_exprs) <- agg_names

# Evaluate
stats_dt <- expanded[,
  eval(as.call(c(as.name("list"), agg_exprs))),
  by = .(id, year)
]

# Replace -Inf/Inf (from max/min on all-NA groups) with NA
inf_cols <- grep("_max$|_min$", names(stats_dt), value = TRUE)
for (col in inf_cols) {
  set(stats_dt, which(is.infinite(stats_dt[[col]])), col, NA_real_)
}

cat("Stats rows:", nrow(stats_dt), "\n")
# Expected: ~6,460,000 (one per cell-year)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 4: Join aggregated stats back to cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Remove any pre-existing neighbor columns to avoid duplication
old_neighbor_cols <- intersect(names(cell_data), agg_names)
if (length(old_neighbor_cols) > 0) {
  cell_data[, (old_neighbor_cols) := NULL]
}

setkeyv(cell_data, c("id", "year"))
setkeyv(stats_dt,  c("id", "year"))

cell_data <- stats_dt[cell_data, on = c("id", "year")]

cat("Done. cell_data now has", ncol(cell_data), "columns and",
    nrow(cell_data), "rows.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 5: Predict with the existing trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data$prediction <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Memory Budget Check

| Object | Approximate Size |
|---|---|
| `cell_data` (6.46M Ã— 110 cols) | ~5.7 GB |
| `edge_dt` (1.37M Ã— 2 int cols) | ~11 MB |
| `expanded` (38.5M Ã— 7 cols) | ~2.2 GB |
| `stats_dt` (6.46M Ã— 17 cols) | ~0.9 GB |
| **Peak total** | **~9â€“10 GB** |

This fits within 16 GB. If memory is tight, process variables in batches of 2â€“3 instead of all 5 simultaneously, then `cbind` the results.

---

## 5. Expected Speedup

| Phase | Before | After |
|---|---|---|
| Neighbor lookup construction | ~30â€“40 hours | ~2 seconds (vectorized edge list) |
| Neighbor stats (5 vars) | ~45â€“50 hours | ~2â€“5 minutes (data.table join + group-by in C) |
| **Total** | **86+ hours** | **< 10 minutes** |

The trained Random Forest model is untouched, and the 15 neighbor-derived features (`5 vars Ã— {max, min, mean}`) are numerically identical to the original implementation.
END_RESPONSE>>>
