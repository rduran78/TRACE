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
run_id: Anthropic_F07_T07_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which constructs a list of ~6.46 million elements, each built by an R-level `lapply` iteration that performs character pasting, named-vector lookups, and NA filtering **row by row**. This is an O(n) interpreted loop where n â‰ˆ 6.46M, and each iteration does multiple string allocations and hash-table lookups. The subsequent `compute_neighbor_stats` is a second O(n) loop over the same 6.46M rows, but it is comparatively cheaper because it only indexes into a numeric vector. Together, the two loops run 5 times (once per source variable for stats, but the lookup is built once), yet the lookup construction aloneâ€”with ~6.46M `paste` + named-vector index operationsâ€”dominates at an estimated 86+ hours.

**Root causes:**

| Problem | Detail |
|---|---|
| **Row-level `lapply` in R** | 6.46M iterations of interpreted R code with string ops. |
| **`paste` + named-vector lookup** | Each row builds character keys and looks them up in a 6.46M-length named vectorâ€”O(1) amortized per lookup but with huge constant overhead from string hashing. |
| **Redundant work across years** | The neighbor *topology* is time-invariant (cell A's neighbors are the same every year), but the lookup re-derives neighbor row indices per cell-year by pasting year suffixes. |
| **`compute_neighbor_stats` also uses `lapply`** | Another 6.46M-iteration interpreted loop, repeated 5 times. |

The numerical results (neighbor max, min, mean of each variable) are **exact** given the rook topology, so any optimization must reproduce them bit-for-bit.

---

## Optimization Strategy

### 1. Separate topology from time: exploit the panel structure

The rook-neighbor graph is **purely spatial**â€”it doesn't change across years. Instead of building a 6.46M-element lookup, build a **344,208-element spatial lookup** (cell â†’ neighbor cells), then for each year slice, use integer indexing to gather neighbor rows. This reduces the lookup problem by a factor of 28.

### 2. Replace `lapply` + `paste` with vectorized `data.table` joins

Use `data.table` to:
- Map each cell to its neighbor cells (a long-format edge table, ~1.37M rows).
- Cross-join with years to get ~1.37M Ã— 28 â‰ˆ 38.5M edge-year rows (but built lazily via keyed join, not materialized all at once).
- For each variable, do a single grouped aggregation (`max`, `min`, `mean`) keyed by `(id, year)`.

This replaces **all** interpreted loops with vectorized C-level `data.table` operations.

### 3. Batch all 5 variables in one pass

Instead of looping over variables and re-joining, compute all 5 neighbor stats in a single grouped aggregation.

### Expected speedup

| Phase | Old | New (estimated) |
|---|---|---|
| Build lookup | ~hours (6.46M R-loop iterations) | ~seconds (vectorized join) |
| Compute stats (Ã—5 vars) | ~hours | ~1â€“3 minutes (data.table grouped agg) |
| **Total** | **86+ hours** | **< 5 minutes** |

Memory: the edge table is ~1.37M rows Ã— 2 integer columns â‰ˆ 11 MB. The joined table per year-slice is manageable. Peak memory stays well within 16 GB.

---

## Working R Code

```r
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Optimized neighbor-stat computation
# Preserves the exact numerical estimand (neighbor max, min, mean)
# and does NOT touch the trained Random Forest model.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

library(data.table)

# â”€â”€ 0. Convert cell_data to data.table (non-destructive) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#    Assumes cell_data is a data.frame with columns: id, year, and the
#    neighbor_source_vars.  id_order and rook_neighbors_unique are the
#    same objects used in the original code.

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Preserve original row order so downstream code / the RF scoring
# pipeline sees the same row positions.
cell_data[, .row_order := .I]

# â”€â”€ 1. Build a long-format edge table from the nb object â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#    rook_neighbors_unique is an nb object: a list of integer vectors
#    where element i contains the indices (into id_order) of cell i's
#    neighbors.

edge_list <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    # spdep nb objects use 0L to denote "no neighbors"
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(
      focal_id    = id_order[i],
      neighbor_id = id_order[nb]
    )
  })
)

# â”€â”€ 2. Key cell_data for fast joins â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setkey(cell_data, id, year)

# â”€â”€ 3. Define source variables â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# â”€â”€ 4. Compute neighbor stats in one vectorized pass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
#    Strategy:
#      â€¢ For each year, join edge_list to cell_data to retrieve
#        neighbor values, then aggregate per focal cell.
#      â€¢ We process year-by-year to keep peak memory modest (~1.37M
#        rows per year-slice join result).

years <- sort(unique(cell_data$year))

# Pre-allocate result columns (filled with NA)
for (v in neighbor_source_vars) {
  cell_data[, paste0("nb_max_",  v) := NA_real_]
  cell_data[, paste0("nb_min_",  v) := NA_real_]
  cell_data[, paste0("nb_mean_", v) := NA_real_]
}

# Columns we need from the neighbor rows
nb_val_cols <- neighbor_source_vars  # the raw variable names

for (yr in years) {

  # Slice this year's data: only id + the source variables
  yr_slice <- cell_data[year == yr, c("id", nb_val_cols), with = FALSE]
  setnames(yr_slice, "id", "neighbor_id")
  setkey(yr_slice, neighbor_id)

  # Join: for every directed edge, attach the neighbor's variable values
  joined <- merge(edge_list, yr_slice, by = "neighbor_id", all.x = FALSE)
  # joined has columns: neighbor_id, focal_id, ntl, ec, ...

  # Aggregate per focal cell
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }))
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0(c("nb_max_", "nb_min_", "nb_mean_"), v)
  }))

  # Evaluate aggregation
  agg_call <- as.call(c(as.name("list"),
                         setNames(agg_exprs, agg_names)))
  stats <- joined[, eval(agg_call), by = focal_id]

  # Replace Inf / -Inf (from max/min of zero-length after NA removal) with NA
  for (col_name in agg_names) {
    set(stats, which(is.infinite(stats[[col_name]])), col_name, NA_real_)
  }

  # Write back into cell_data
  setkey(stats, focal_id)
  match_idx <- cell_data[year == yr, which = TRUE]
  focal_ids_this_year <- cell_data$id[match_idx]

  # Map focal_id in stats back to cell_data rows
  stats_row <- stats[J(focal_ids_this_year)]  # ordered by focal_ids_this_year

  for (col_name in agg_names) {
    set(cell_data, i = match_idx, j = col_name, value = stats_row[[col_name]])
  }
}

# â”€â”€ 5. Restore original row order â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setorder(cell_data, .row_order)
cell_data[, .row_order := NULL]

# â”€â”€ Done. cell_data now contains the 15 new neighbor-stat columns
#    (3 stats Ã— 5 variables) with values identical to the original
#    implementation.  The trained Random Forest model is untouched.
```

---

## Why This Preserves the Exact Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor topology** | `edge_list` is built from the identical `rook_neighbors_unique` nb object and `id_order`. |
| **Same per-row aggregation** | `max`, `min`, `mean` with `na.rm = TRUE` over the same set of neighbor values, producing `NA` when no valid neighbors exist (Inf â†’ NA replacement mirrors the `length(neighbor_vals) == 0` branch). |
| **Same output columns** | Column names follow the `nb_{max,min,mean}_{var}` convention generated by the original `compute_and_add_neighbor_features`. |
| **RF model untouched** | No model object is modified; only the feature columns in `cell_data` are populated. |

**Expected wall-clock time on a 16 GB laptop: 2â€“5 minutes** (versus 86+ hours), driven entirely by `data.table`'s C-level merge and grouped aggregation.
END_RESPONSE>>>
