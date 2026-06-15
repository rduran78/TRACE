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
run_id: Anthropic_F07_T10_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It constructs a list of 6.46 million elements, where each element is built by:

1. **Character-key hashing per row** â€” `paste(neighbor_cell_ids, data$year[i], sep = "_")` and named-vector lookup (`idx_lookup[neighbor_keys]`) is called 6.46 million times inside an `lapply`. Named-vector lookup in R is O(n) per query against a vector of 6.46M names â€” this is effectively a linear scan repeated millions of times.
2. **Per-row R-level overhead** â€” Each iteration of the `lapply` allocates character vectors, does `paste`, subsets a named vector, and filters `NA`s. With ~6.46M iterations, R's interpreted loop overhead alone is enormous.
3. **Redundant recomputation** â€” The neighbor topology is **time-invariant** (rook neighbors don't change across years). Yet the lookup is built per cell-year row rather than per cell, then broadcast across years. This inflates the work by a factor of 28.

`compute_neighbor_stats` is less severe but still slow: it runs an `lapply` over 6.46M elements, each calling `max`, `min`, `mean` in interpreted R.

**Estimated cost of current approach:** ~6.46M Ã— (string operations + named-vector lookups against 6.46M keys) â‰ˆ 86+ hours.

---

## Optimization Strategy

### 1. Separate spatial topology from temporal indexing
The neighbor structure is purely spatial (344,208 cells). Build a **cell-to-cell** adjacency once (344K entries), then map to rows using vectorized year-matching â€” never build 6.46M string keys.

### 2. Replace named-vector lookup with integer-indexed lookup via `data.table`
Use `data.table` keyed joins to map `(cell_id, year)` â†’ row index in O(1) amortized time.

### 3. Vectorize `compute_neighbor_stats` using a sparse-matrix or a flattened vectorized approach
Expand the neighbor list into an edge list `(row_i, row_j)`, extract values with a single vectorized subscript, then aggregate with `data.table` grouping â€” no R-level loop over 6.46M elements.

### 4. Process all 5 variables in one pass over the edge list
The edge list is the same for all variables; just column-swap the values.

**Expected speedup:** From 86+ hours to **~2â€“5 minutes**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0. Ensure data is a data.table with correct types
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# Preserve original row order so we can write results back in place
cell_dt[, .ROW := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build a CELL-level edge list from the nb object  (done ONCE)
#    rook_neighbors_unique is an nb object of length length(id_order).
#    id_order[k] gives the cell id for the k-th element of the nb list.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_cell_edge_list <- function(id_order, nb_obj) {
  # nb_obj[[k]] contains integer indices into id_order of neighbors of cell k
  from_ref <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_ref   <- unlist(nb_obj)

  # Remove the 0-neighbor sentinel that spdep uses
  valid <- to_ref != 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]

  data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
}

cell_edges <- build_cell_edge_list(id_order, rook_neighbors_unique)
# cell_edges has ~1.37M rows (directed rook-neighbor pairs)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Map cell-level edges to ROW-level edges by joining on year
#    For every year, each (from_id, to_id) pair becomes (from_row, to_row).
#    We do this with two keyed joins â€” fully vectorized.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Create a lookup: (id, year) -> row index
setkey(cell_dt, id, year)
row_lookup <- cell_dt[, .(id, year, .ROW)]
setkey(row_lookup, id, year)

# Expand edges across all years (cross join edges Ã— years)
years <- sort(unique(cell_dt$year))
row_edges <- cell_edges[, CJ_year := NULL]  # safety
row_edges <- cell_edges[
  rep(seq_len(.N), each = length(years))
][, year := rep(years, times = nrow(cell_edges))]

# Join to get from_row
setnames(row_lookup, c("id", "year", ".ROW"), c("from_id", "year", "from_row"))
setkey(row_lookup, from_id, year)
setkey(row_edges, from_id, year)
row_edges <- row_lookup[row_edges, nomatch = 0L]

# Join to get to_row
setnames(row_lookup, c("from_id", "year", "from_row"), c("to_id", "year", "to_row"))
setkey(row_lookup, to_id, year)
setkey(row_edges, to_id, year)
row_edges <- row_lookup[row_edges, nomatch = 0L]

# Restore lookup names
setnames(row_lookup, c("to_id", "year", "to_row"), c("id", "year", ".ROW"))

# row_edges now has columns: to_row, from_row, from_id, to_id, year
# "from_row" is the focal cell-year; "to_row" is its neighbor cell-year.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Compute neighbor max, min, mean for each variable â€” vectorised
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {

  # Pull neighbor values via integer indexing (single vectorised subscript)
  row_edges[, nval := cell_dt[[var]][to_row]]

  # Aggregate: group by focal row (from_row), dropping NAs
  stats <- row_edges[!is.na(nval),
    .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ),
    keyby = from_row
  ]

  # Initialise new columns to NA (handles cells with 0 valid neighbors)
  max_col  <- paste0(var, "_neighbor_max")
  min_col  <- paste0(var, "_neighbor_min")
  mean_col <- paste0(var, "_neighbor_mean")

  set(cell_dt, j = max_col,  value = NA_real_)
  set(cell_dt, j = min_col,  value = NA_real_)
  set(cell_dt, j = mean_col, value = NA_real_)

  # Write results back by integer row index
  set(cell_dt, i = stats$from_row, j = max_col,  value = stats$nb_max)
  set(cell_dt, i = stats$from_row, j = min_col,  value = stats$nb_min)
  set(cell_dt, i = stats$from_row, j = mean_col, value = stats$nb_mean)
}

row_edges[, nval := NULL]  # clean up temp column

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Convert back to data.frame if downstream code expects one
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt[, .ROW := NULL]
cell_data <- as.data.frame(cell_dt)
```

---

## Memory Note

The `row_edges` table has ~1.37M Ã— 28 â‰ˆ **38.4M rows** with a few integer/numeric columns â€” roughly **~1â€“1.5 GB**. This fits in 16 GB alongside the 6.46M-row dataset and the trained Random Forest object. If memory is tight, you can process years in batches:

```r
# Memory-conservative variant: process one year at a time
for (yr in years) {
  yr_rows  <- cell_dt[year == yr, .ROW]
  yr_edges <- cell_edges[, .(from_id, to_id, year = yr)]
  # ... join & aggregate as above, restricted to this year ...
}
```

---

## What is preserved

| Requirement | Status |
|---|---|
| **Trained Random Forest model** | Untouched â€” no retraining; we only compute the same 15 neighbor features the model expects. |
| **Original numerical estimand** | Identical â€” `max`, `min`, `mean` over exactly the same rook-neighbor sets, with the same `NA` handling (`NA` when no valid neighbors exist). |
| **Column naming convention** | Same `{var}_neighbor_max/min/mean` names as the original `compute_and_add_neighbor_features` would produce. Adjust the `paste0` pattern if your wrapper used different suffixes. |
END_RESPONSE>>>
