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
run_id: Anthropic_F07_T10_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over **~6.46 million rows** in a pure-R `lapply`, performing per-row character key construction, hash lookups, and NA filtering. This produces a list of 6.46 million integer vectors â€” a massive overhead in both time and memory. The secondary cost is `compute_neighbor_stats`, which again loops over 6.46 million list entries per variable (Ã—5 variables = ~32.3 million R-level iterations total). At ~48 Âµs per iteration (conservative), this alone accounts for the 86+ hour estimate.

**Root causes:**

1. **Row-level R loop with string operations:** `paste()` and named-vector lookup inside a per-row `lapply` over 6.46M rows is catastrophically slow.
2. **Redundant work across years:** The neighbor topology is *time-invariant* â€” cell A neighbors cell B in every year â€” yet the lookup is rebuilt at the cell-year level, inflating the problem from ~344K cells to ~6.46M rows.
3. **List-of-vectors structure for neighbor indices:** Storing 6.46M small integer vectors has heavy R overhead.
4. **`compute_neighbor_stats` re-traverses the same list 5 times**, once per variable, instead of batching.

---

## Optimization Strategy

1. **Exploit time-invariance:** Build the neighbor lookup once at the *cell* level (344K entries), not the cell-year level (6.46M entries). For each cell, its neighbors are the same in every year.

2. **Use `data.table` for vectorized, keyed joins** instead of character-key hash lookups in a loop. Join the neighbor-pair table (`from_id`, `to_id`) onto the panel by `(to_id, year)` to retrieve neighbor values, then group-aggregate `max/min/mean` by `(from_id, year)` â€” all in compiled C code inside `data.table`.

3. **Batch all 5 variables in a single join-and-aggregate pass** to avoid redundant joins.

4. **Estimated speedup:** The entire operation becomes a single equi-join of ~1.37M neighbor pairs Ã— 28 years â‰ˆ 38.4M rows, followed by a grouped aggregation â€” typically completing in **1â€“3 minutes** on 16 GB RAM, versus 86+ hours.

5. **Numerical equivalence:** The `max`, `min`, and `mean` are computed over exactly the same non-NA neighbor values as before, preserving the original estimand. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# â”€â”€ 1. Build the directed edge list from the nb object (one-time, ~344K cells) â”€

build_edge_dt <- function(id_order, nb_obj) {
  # nb_obj is a list of length |cells|; nb_obj[[i]] gives integer indices

  # of neighbors of the i-th cell in id_order.
  from_id <- rep(id_order, lengths(nb_obj))
  to_id   <- id_order[unlist(nb_obj)]
  data.table(from_id = from_id, to_id = to_id)
}

edges <- build_edge_dt(id_order, rook_neighbors_unique)
# edges has ~1,373,394 rows (directed rook pairs)

# â”€â”€ 2. Convert panel to data.table and set key â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

dt <- as.data.table(cell_data)
# Ensure the id column and year column are properly named "id" and "year".
# Adjust if your columns have different names.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Keep only the columns we need for the join (plus id and year)
value_cols <- intersect(neighbor_source_vars, names(dt))
dt_vals <- dt[, c("id", "year", value_cols), with = FALSE]
setnames(dt_vals, "id", "to_id")       # rename for join
setkey(dt_vals, to_id, year)

# â”€â”€ 3. Expand edges Ã— years and join neighbor values in one pass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Merge edges with the panel on (to_id, year) to get neighbor values
# This is a many-to-many join: each (from_id, year) gets all its neighbors' values.
# We add year via a cross-join with edges, but it's cheaper to join directly.

# Add from_id's year by joining edges onto dt_vals keyed by to_id, year.
# Strategy: for each row in dt (from-cell side), look up its neighbors.

dt_from <- dt[, .(from_id = id, year)]
setkey(dt_from, from_id, year)

# Expand: each (from_id, year) â†’ all to_id neighbors
edges_expanded <- edges[dt_from, on = .(from_id), allow.cartesian = TRUE, nomatch = 0L]
# edges_expanded now has columns: from_id, to_id, year
# ~1.37M pairs Ã— 28 years â‰ˆ 38.4M rows

setkey(edges_expanded, to_id, year)

# Join to get neighbor values
edges_expanded <- dt_vals[edges_expanded, on = .(to_id, year), nomatch = NA]

# â”€â”€ 4. Aggregate neighbor stats per (from_id, year) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(value_cols, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(value_cols, function(v) {
  paste0("nb_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Evaluate the aggregation
neighbor_stats <- edges_expanded[,
  lapply(agg_exprs, eval, envir = .SD),
  by = .(from_id, year)
]

# Replace Inf/-Inf (from max/min on all-NA groups) with NA
inf_to_na <- function(x) { x[is.infinite(x)] <- NA_real_; x }
stat_cols <- names(neighbor_stats)[-(1:2)]
neighbor_stats[, (stat_cols) := lapply(.SD, inf_to_na), .SDcols = stat_cols]

# â”€â”€ 5. Merge back onto the original panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

setnames(neighbor_stats, "from_id", "id")
setkey(neighbor_stats, id, year)
setkey(dt, id, year)

dt <- neighbor_stats[dt, on = .(id, year)]

# â”€â”€ 6. Convert back to data.frame if downstream code expects one â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cell_data <- as.data.frame(dt)

# â”€â”€ Done. The trained Random Forest model is untouched. â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The new columns nb_max_ntl, nb_min_ntl, nb_mean_ntl, ... etc.
# are numerically identical to the original implementation's output
# (same non-NA neighbor sets, same aggregation functions).
```

---

## Performance Notes

| Aspect | Original | Optimized |
|---|---|---|
| **Paradigm** | Row-level R loop + string hashing | Vectorized `data.table` join + grouped aggregation |
| **Iterations in R** | ~6.46M (lookup) + ~32.3M (stats) | 0 (all in C) |
| **Estimated time** | 86+ hours | **1â€“5 minutes** |
| **Peak RAM** | High (6.46M-element list) | ~4â€“6 GB (38.4M-row join table) |
| **Numerical result** | Baseline | Identical (`max`, `min`, `mean` over same non-NA neighbor values) |
| **RF model** | â€” | Untouched; no retraining |

The key insight is that **the neighbor topology is time-invariant**, so the problem decomposes into a spatial edge list (small, ~1.37M rows) crossed with the time dimension, which `data.table` handles as a single compiled equi-join and grouped aggregation â€” eliminating millions of R-interpreter iterations entirely.
END_RESPONSE>>>
