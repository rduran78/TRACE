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
run_id: Anthropic_F07_T00_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Per-row string pasting and named-vector lookup** â€” `paste()` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called ~6.46 million times inside an `lapply`. Each call does string allocation, hashing, and matching.
2. **The `idx_lookup` named vector has ~6.46 million entries**, so every hash-table probe is expensive.
3. **`compute_neighbor_stats`** then loops over the 6.46M-element list again, extracting values one-at-a-time.

Together these two functions produce **O(n Ã— k)** string operations (n â‰ˆ 6.46M rows, k â‰ˆ average neighbor count â‰ˆ 4), all in interpreted R with per-element string allocation. That is the source of the 86+ hour estimate.

### Why naive raster focal operations are unsafe
The document correctly notes that the cell topology may be irregular/masked. A `terra::focal()` or `raster::focal()` call assumes a complete rectangular grid and a uniform kernel â€” cells on mask boundaries or with irregular connectivity would get wrong neighbors. The `spdep::nb` object encodes the *exact* rook-neighbor graph and must be respected.

---

## Optimization Strategy

**Replace the row-level R loop with vectorized joins using `data.table`.**

1. **Explode the neighbor graph into an edge table** â€” a two-column `data.table` with `(id, neighbor_id)` derived directly from the `nb` object. This is done once, ~1.37M rows.
2. **Cross-join with years implicitly via a keyed merge** â€” join `cell_data` to itself on `(neighbor_id, year)` to pull neighbor values. This replaces all string pasting and named-vector lookups with a single indexed equi-join.
3. **Group-by aggregation** â€” compute `max`, `min`, `mean` per `(id, year)` in one grouped operation per variable.

This reduces the work to a handful of `data.table` joins and group-bys â€” all executed in C â€” and should complete in **minutes, not hours**.

The trained Random Forest model is untouched. The numerical results (neighbor max, min, mean) are identical because the same neighbor graph and the same aggregation functions are used.

---

## Working R Code

```r
library(data.table)

# â”€â”€ 1. Convert the nb object to an edge data.table (done once) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

nb_to_edge_dt <- function(nb_obj, id_order) {
  # nb_obj  : spdep nb object (list of integer index vectors)
  # id_order: vector of cell IDs in the same order as nb_obj
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj, use.names = FALSE)
  # Remove the 0-neighbor sentinel that spdep uses (integer(0) is fine,

  # but some nb objects store 0L for no-neighbor entries)
  valid    <- to_idx > 0L
  data.table(
    id          = id_order[from_idx[valid]],
    neighbor_id = id_order[to_idx[valid]]
  )
}

edges <- nb_to_edge_dt(rook_neighbors_unique, id_order)

# â”€â”€ 2. Convert cell_data to data.table (if not already) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# â”€â”€ 3. Compute neighbor stats for all source variables â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset to only the columns we need for the neighbor lookups
# to minimise memory during the join.
join_cols <- c("id", "year", neighbor_source_vars)
nbr_vals  <- cell_data[, ..join_cols]
setnames(nbr_vals, "id", "neighbor_id")          # rename for join
setkey(nbr_vals, neighbor_id, year)

# Merge edges with cell_data to get (focal_id, year, neighbor values)
# edges has (id, neighbor_id); nbr_vals is keyed on (neighbor_id, year).
# We need one row per (focal cell, year, neighbor cell).
# Strategy: join edges to nbr_vals, then group by (id, year).

# Add year dimension: join edges Ã— years via cell_data's own (id, year) pairs
# Rather than a full cross join, we merge through cell_data's existing rows.

# Step A: create a slim focal table with just (id, year, row_position)
focal <- cell_data[, .(id, year)]
focal[, focal_row := .I]

# Step B: attach neighbor_ids to each focal row via the edge list
#         This is an equi-join on 'id'.
setkey(edges, id)
setkey(focal, id)
focal_nbr <- edges[focal, on = "id", allow.cartesian = TRUE, nomatch = 0L]
# Result columns: id, neighbor_id, year, focal_row

# Step C: pull neighbor variable values by joining on (neighbor_id, year)
setkey(focal_nbr, neighbor_id, year)
focal_nbr <- nbr_vals[focal_nbr, on = .(neighbor_id, year), nomatch = NA]
# Now focal_nbr has columns: neighbor_id, year, <vars>, id, focal_row

# Step D: aggregate per (id, year) â€” equivalently per focal_row
for (v in neighbor_source_vars) {
  agg <- focal_nbr[
    !is.na(get(v)),
    .(
      nb_max  = max(get(v)),
      nb_min  = min(get(v)),
      nb_mean = mean(get(v))
    ),
    keyby = .(focal_row)
  ]
  # Name the new columns to match the original pipeline's convention
  max_col  <- paste0("nb_max_",  v)
  min_col  <- paste0("nb_min_",  v)
  mean_col <- paste0("nb_mean_", v)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
                c(max_col,  min_col,  mean_col))

  # Initialise with NA, then fill matched rows
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
  cell_data[agg$focal_row, (max_col)  := agg[[max_col]]]
  cell_data[agg$focal_row, (min_col)  := agg[[min_col]]]
  cell_data[agg$focal_row, (mean_col) := agg[[mean_col]]]
}

# â”€â”€ 4. Clean up large temporaries â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
rm(focal_nbr, focal, nbr_vals, agg, edges)
gc()

# cell_data now has the 15 new neighbor-stat columns (3 Ã— 5 variables),
# numerically identical to the original implementation.
# The trained Random Forest model is unchanged and can be used directly.
```

---

### Why this is safe and equivalent

| Concern | Resolution |
|---|---|
| **Irregular / masked topology** | The exact `spdep::nb` rook-neighbor graph is converted edge-by-edge into `edges`. No rectangular-grid assumption is made. |
| **Numerical equivalence** | The same `max`, `min`, `mean` functions are applied to the same sets of neighbor values. `data.table` aggregation is IEEE-754 compliant; results are identical to the original `lapply` code. |
| **Trained RF model** | No model retraining occurs. Only the feature-engineering (neighbor stats) step is accelerated. The 15 output columns have the same names and values. |
| **Memory (16 GB laptop)** | The largest intermediate (`focal_nbr`) has â‰ˆ 6.46M Ã— 4 neighbors â‰ˆ 26M rows with a few columns â€” roughly 1â€“2 GB. Well within budget, especially since we process one variable at a time in the aggregation and could further chunk if needed. |
| **Speed** | Two keyed `data.table` joins + 5 grouped aggregations, all in C. Expected wall-clock time: **2â€“10 minutes** (down from 86+ hours). |
END_RESPONSE>>>
