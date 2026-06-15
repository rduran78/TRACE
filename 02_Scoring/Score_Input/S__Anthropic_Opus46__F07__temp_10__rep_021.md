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
run_id: Anthropic_F07_T10_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### 1. `build_neighbor_lookup` â€” O(nÂ²)-like overhead
- **`lapply` over ~6.46 million rows**, each doing string pasting, named-vector lookups (`idx_lookup[neighbor_keys]`), and character matching.
- `idx_lookup` is a **named character vector with 6.46M entries**. Each lookup by name is O(n) in base R (hash lookup on names, but construction and repeated access on a 6.46M-element named vector is very expensive).
- String concatenation (`paste(id, year, sep="_")`) is done millions of times inside the loop.
- The result is a **list of 6.46 million integer vectors** â€” huge memory overhead.

### 2. `compute_neighbor_stats` â€” Repeated per variable but the real cost is the lookup structure
- Called 5 times (once per variable), each iterating over 6.46M rows.
- Each iteration subsets a numeric vector by index, computes max/min/mean â€” this part is actually fast per call, but 5 Ã— 6.46M `lapply` iterations in R is still slow.
- The neighbor index vectors are reused, which is good, but the R-level loop is the bottleneck.

### Combined estimate
- ~6.46M R-level iterations for building the lookup (with expensive string ops) + 5 Ã— 6.46M iterations for stats = **~38.8 million R-level loop iterations** with non-trivial work each. This easily explains 86+ hours.

---

## Optimization Strategy

### Key Insight: Vectorize everything using `data.table` joins and sparse-matrix / grouped operations.

1. **Replace the named-vector lookup with a `data.table` keyed join.** Map `(cell_id, year)` â†’ row index using a hash join instead of named-vector indexing.

2. **Build the neighbor edge list as a data.table.** Convert the `nb` object into a flat edge table `(from_id, to_id)`. Then join with the panel to get `(from_row, to_row)` pairs. This replaces the entire `build_neighbor_lookup` function and eliminates 6.46M R-level iterations.

3. **Compute neighbor stats via grouped aggregation.** With the edge table expressed as `(from_row, to_row)`, extract `vals[to_row]` for each variable, group by `from_row`, and compute `max`, `min`, `mean` â€” all in vectorized `data.table` operations.

4. **Handle cells with no neighbors** by left-joining back to the full row set and filling with `NA`.

5. **Process all 5 variables in one pass** over the edge table (or 5 fast grouped aggregations on the same grouped structure).

**Expected speedup:** From 86+ hours to **minutes** (likely 2â€“10 minutes depending on disk I/O and RAM pressure). All operations become vectorized C-level data.table operations. Memory footprint of the edge table is modest (~1.37M edges Ã— 28 years â‰ˆ ~38M edge-rows, each with two integer columns â‰ˆ ~300 MB).

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Convert cell_data to data.table (if not already) and add row index
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)
cell_data[, row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build flat edge list from the nb object
#     rook_neighbors_unique is a list of integer vectors (spdep nb);
#     id_order maps position â†’ cell id.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_i <- rook_neighbors_unique[[i]]
  # nb objects use 0L to mean "no neighbors"

  nb_i <- nb_i[nb_i != 0L]
  if (length(nb_i) == 0L) return(NULL)
  data.table(from_id = id_order[i], to_id = id_order[nb_i])
}))

cat("Edge list rows (directed):", nrow(edges), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Map (cell_id, year) â†’ row_idx via keyed join
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Create a small lookup: id â†’ row indices per year
id_year_lookup <- cell_data[, .(id, year, row_idx)]
setkey(id_year_lookup, id, year)

# Get the unique years
years <- sort(unique(cell_data$year))

# Cross-join edges Ã— years so every edge exists in every year
edge_year <- CJ_dt <- edges[, .(from_id, to_id)]
# Use a cross join that is memory-efficient:
edge_year <- edge_year[, .(year = years), by = .(from_id, to_id)]

cat("Edge-year rows:", nrow(edge_year), "\n")
# Expected: ~1.37M * 28 â‰ˆ 38.4M rows

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Attach row indices for "from" and "to"
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# from row index
setnames(id_year_lookup, "id", "from_id")
setkey(id_year_lookup, from_id, year)
edge_year <- id_year_lookup[edge_year, on = .(from_id, year), nomatch = 0L]
setnames(edge_year, "row_idx", "from_row")

# to row index
setnames(id_year_lookup, "from_id", "to_id")
setkey(id_year_lookup, to_id, year)
edge_year <- id_year_lookup[edge_year, on = .(to_id, year), nomatch = 0L]
setnames(edge_year, "row_idx", "to_row")

# Restore name
setnames(id_year_lookup, "to_id", "id")

# Now edge_year has columns: from_row, to_row  (plus from_id, to_id, year)
# Keep only what we need to save memory
edge_year <- edge_year[, .(from_row, to_row)]
setkey(edge_year, from_row)

cat("Final edge-year rows (matched):", nrow(edge_year), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Compute neighbor stats for each source variable â€” fully vectorized
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  cat("Processing neighbor stats for:", var_name, "\n")

  # Pull the variable values into the edge table
  edge_year[, val := cell_data[[var_name]][to_row]]

  # Drop NAs in the variable before aggregation
  agg <- edge_year[!is.na(val),
    .(nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)),
    by = from_row
  ]

  # Initialise columns with NA (handles cells with no valid neighbors)
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)

  # Write results back by row index
  set(cell_data, i = agg$from_row, j = max_col,  value = agg$nb_max)
  set(cell_data, i = agg$from_row, j = min_col,  value = agg$nb_min)
  set(cell_data, i = agg$from_row, j = mean_col, value = agg$nb_mean)

  # Clean up the temporary column
  edge_year[, val := NULL]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Clean up helper column
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data[, row_idx := NULL]

cat("Done. Neighbor features added for",
    length(neighbor_source_vars), "variables.\n")
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | The edge list is derived from the same `rook_neighbors_unique` nb object and `id_order` mapping. Every `(from, to, year)` triple is identical to the original. |
| **Same statistics** | `max`, `min`, `mean` are computed on exactly the same non-NA neighbor values per cell-year as the original code. The `data.table` grouped aggregation is algebraically identical. |
| **Trained Random Forest unchanged** | No model retraining occurs. Only the feature-engineering step is optimized. The output columns have the same names and values, so downstream `predict()` calls produce identical results. |
| **Numerical estimand preserved** | All operations use base R `max`/`min`/`mean` on `double` vectors â€” identical floating-point results. |

---

## Memory Considerations (16 GB laptop)

| Object | Approximate Size |
|---|---|
| `cell_data` (~6.46M Ã— 110 cols) | ~5.7 GB |
| `edge_year` (~38.4M Ã— 2 int cols) | ~0.3 GB |
| Temporary `val` column + `agg` | ~0.5 GB peak |
| **Total peak** | **~6.5 GB** âœ“ fits in 16 GB |

**Expected runtime: 3â€“10 minutes** (down from 86+ hours), dominated by the grouped aggregations on ~38M rows, which `data.table` handles in seconds per variable.
END_RESPONSE>>>
