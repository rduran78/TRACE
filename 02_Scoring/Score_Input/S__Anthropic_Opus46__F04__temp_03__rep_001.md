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
run_id: Anthropic_F04_T03_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` â€” Row-level `lapply` over 6.46 million rows

Each iteration performs:
- A character coercion and named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
- A `paste()` call to build neighbor keys.
- A named-vector lookup into `idx_lookup` (which is a 6.46M-element named character vector â€” each lookup is O(n) hash probe on a very large vector).
- An `is.na` filter.

Doing this 6.46 million times in an interpreted `lapply` loop is extremely expensive. The named-vector `idx_lookup` with ~6.46M entries makes each key lookup slow, and the per-row `paste` and subsetting compounds the cost.

### 2. `compute_neighbor_stats` â€” Row-level `lapply` over 6.46 million rows

Each iteration subsets a numeric vector, removes NAs, and computes `max`, `min`, `mean`. This is called 5 times (once per variable), so ~32.3 million R-level function calls. The `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also slow.

### 3. Overall: ~6.46M Ã— (key construction + lookup) + 5 Ã— 6.46M Ã— (subset + stats) â‰ˆ 86+ hours

---

## Optimization Strategy

**Replace all row-level R loops with vectorized `data.table` joins and grouped aggregations.**

The key insight: the neighbor lookup is a **join** problem. Each cell-year needs to be joined to its neighbors' cell-years, then grouped statistics are computed. `data.table` performs this with optimized C-level code.

**Steps:**

1. **Build an edge list** (a two-column table of `id â†’ neighbor_id`) from the `nb` object â€” done once, ~1.37M rows.
2. **Join** the edge list to the panel data on `(neighbor_id, year)` to get neighbor variable values â€” this is a keyed `data.table` merge, extremely fast.
3. **Group by** `(id, year)` and compute `max`, `min`, `mean` in one pass per variable.
4. **Merge** the results back to the main table.

This eliminates all row-level R loops, all `paste`-based key construction, and all named-vector lookups. Expected runtime: **minutes, not hours**.

The trained Random Forest model is untouched. The numerical results (max, min, mean of neighbor values) are identical.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 0 â€” Convert the nb object to a data.table edge list (once)
# ---------------------------------------------------------------
build_edge_list_dt <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  # id_order is the vector mapping position -> cell id
  from_ids <- rep(id_order, lengths(neighbors))
  to_ids   <- id_order[unlist(neighbors)]
  data.table(id = from_ids, neighbor_id = to_ids)
}

edge_dt <- build_edge_list_dt(id_order, rook_neighbors_unique)

# ---------------------------------------------------------------
# STEP 1 â€” Convert panel data to data.table and set key
# ---------------------------------------------------------------
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# ---------------------------------------------------------------
# STEP 2 â€” Compute neighbor features for all variables at once
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset to only the columns we need for the neighbor join
# (id, year, and the 5 source variables)
cols_needed <- c("id", "year", neighbor_source_vars)
dt_slim <- dt[, ..cols_needed]
setnames(dt_slim, "id", "neighbor_id")
setkey(dt_slim, neighbor_id, year)

# Join: for every (id, year) pair, look up each neighbor's values
# edge_dt tells us who the neighbors are; we join on (neighbor_id, year)
edge_year <- edge_dt[dt[, .(id, year)], on = "id", allow.cartesian = TRUE, nomatch = 0L]
# edge_year now has columns: id, neighbor_id, year

# Merge in the neighbor values
setkey(edge_year, neighbor_id, year)
edge_vals <- dt_slim[edge_year, on = .(neighbor_id, year), nomatch = NA]
# edge_vals has: neighbor_id, year, ntl, ec, ..., id

# ---------------------------------------------------------------
# STEP 3 â€” Grouped aggregation: max, min, mean per (id, year)
# ---------------------------------------------------------------
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

# Build the aggregation call programmatically
agg_list <- setNames(agg_exprs, agg_names)

neighbor_stats <- edge_vals[,
  lapply(agg_list, eval, envir = .SD),
  by = .(id, year),
  .SDcols = neighbor_source_vars
]

# Handle Inf/-Inf from max/min on all-NA groups (replace with NA)
inf_cols <- grep("_max$|_min$", names(neighbor_stats), value = TRUE)
for (col in inf_cols) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

# ---------------------------------------------------------------
# STEP 4 â€” Merge back to the main data
# ---------------------------------------------------------------
setkey(neighbor_stats, id, year)
setkey(dt, id, year)
dt <- neighbor_stats[dt, on = .(id, year)]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(dt)
```

**If the programmatic `agg_list` evaluation feels fragile**, here is a simpler, equally fast alternative for Step 3:

```r
# ---------------------------------------------------------------
# STEP 3 (alternative) â€” explicit aggregation, still fully vectorized
# ---------------------------------------------------------------
compute_neighbor_features_dt <- function(edge_vals, var_name) {
  v <- var_name
  edge_vals[
    !is.na(get(v)),
    .(
      nb_max  = max(get(v)),
      nb_min  = min(get(v)),
      nb_mean = mean(get(v))
    ),
    by = .(id, year)
  ] |>
    setnames(
      c("nb_max", "nb_min", "nb_mean"),
      paste0("neighbor_", v, c("_max", "_min", "_mean"))
    )
}

# Run for each variable and merge
setkey(dt, id, year)

for (var_name in neighbor_source_vars) {
  stats_dt <- compute_neighbor_features_dt(edge_vals, var_name)
  setkey(stats_dt, id, year)
  dt <- stats_dt[dt, on = .(id, year)]
}

cell_data <- as.data.frame(dt)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained RF model** | Untouched â€” we only rebuild the input features, not the model. |
| **Numerical estimand** | Identical â€” `max`, `min`, `mean` over the same neighbor sets, same NA handling. |
| **Neighbor topology** | The edge list is derived from the same `rook_neighbors_unique` nb object. |
| **Panel alignment** | The join on `(neighbor_id, year)` ensures temporal alignment, identical to the original `paste(id, year)` key logic. |

---

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Neighbor lookup build | ~hours (6.46M `lapply` iterations with `paste` + named vector lookup) | ~seconds (vectorized `rep`/`unlist` to build edge list) |
| Neighbor stats (Ã—5 vars) | ~hours (32.3M R-level loops) | ~minutes (`data.table` grouped aggregation in C) |
| **Total** | **86+ hours** | **~2â€“10 minutes** |

The dominant cost becomes the `data.table` keyed join and grouped aggregation, both of which are implemented in C with radix-sort keys and are orders of magnitude faster than interpreted R loops over millions of rows.
END_RESPONSE>>>
