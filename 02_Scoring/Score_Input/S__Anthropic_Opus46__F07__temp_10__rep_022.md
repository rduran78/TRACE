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
run_id: Anthropic_F07_T10_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** creates a list of ~6.46 million elements, each constructed via an `lapply` call that performs per-row string pasting, hash lookups, and NA filtering. This is O(n) in rows Ã— O(k) in neighbors, but with enormous R-interpreter overhead per iteration (string allocation, named-vector lookup, subsetting). The 86+ hour estimate comes almost entirely from:

1. **`build_neighbor_lookup`**: ~6.46M iterations of `lapply`, each doing `paste`, named-vector indexing (`id_to_ref`, `idx_lookup`), and NA removal. Named-vector lookup in R is O(n) in the worst case for long vectors, and `paste` allocates a new string each time. This alone can take tens of hours.

2. **`compute_neighbor_stats`**: Another 6.46M-iteration `lapply` per variable (Ã—5 variables = ~32.3M iterations), each subsetting a numeric vector and computing max/min/mean. The per-call overhead is smaller but still substantial.

3. **Memory pressure**: Storing 6.46M list elements, each an integer vector, plus intermediate string vectors, can push past 16 GB and trigger garbage collection thrashing.

**Root cause**: The code solves a spatial-temporal join problem using R-level loops over millions of rows with string-key lookups, when this is fundamentally a vectorized join + grouped aggregation problem.

---

## Optimization Strategy

### Key Insight
The neighbor lookup can be expressed as a **join table** (an edge list in cell-year space), and the statistics as a **grouped aggregation** over that join â€” both of which `data.table` handles in vectorized C code.

**Steps:**

1. **Build a directed edge list** from the `nb` object: `(from_id, to_id)` for all rook-neighbor pairs. This is ~1.37M rows (spatial edges only, time-invariant).

2. **Cross-join with years** to get a cell-year edge list: `(from_id, year, to_id)`. This is ~1.37M Ã— 28 â‰ˆ ~38.5M rows â€” large but manageable in RAM as a 3-column integer `data.table` (~900 MB).

3. **Join** the edge list to the data to attach each neighbor's variable values.

4. **Group by `(from_id, year)`** and compute `max`, `min`, `mean` â€” a single vectorized `data.table` aggregation.

5. **Join results back** to the main data table.

This replaces ~86 hours of R-level looping with a few vectorized operations that should complete in **minutes**.

### Memory Management
- Process one variable at a time to avoid duplicating the full edge-list with all 5 variable columns simultaneously.
- The cross-joined edge list (~38.5M rows Ã— 3 integer columns) uses ~900 MB. Each variable join adds one double column (~308 MB). Total peak is well under 16 GB.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Prepare data  (assumed available: cell_data, id_order,
#     rook_neighbors_unique, the trained RF model)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table if not already (non-destructive; keeps all columns)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure there is a row key for final merge-back
cell_data[, .rowid := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build the spatial edge list from the nb object  (~1.37M rows)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_list <- function(id_order, nb_obj) {
  # nb_obj is a list of integer vectors (indices into id_order)
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj, use.names = FALSE)

  # Remove the 0-neighbor sentinel that spdep uses (integer(0) â†’ nothing,
  # but some nb objects store a 0L for islands)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

edge_spatial <- build_edge_list(id_order, rook_neighbors_unique)
# ~ 1.37M rows, two integer (or numeric) columns

cat(sprintf("Spatial edge list: %s rows\n", format(nrow(edge_spatial), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Cross-join with years to get the cell-year edge list  (~38.5M rows)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

years <- sort(unique(cell_data$year))   # 1992:2019, 28 values

edge_cy <- edge_spatial[, .(year = years), by = .(from_id, to_id)]
# This is a cross join: each spatial edge Ã— each year
# Result: ~1.37M Ã— 28 â‰ˆ 38.5M rows

# Set keys for fast joins
setkey(edge_cy, to_id, year)

cat(sprintf("Cell-year edge list: %s rows\n", format(nrow(edge_cy), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  For each neighbor source variable, compute max/min/mean and attach
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-build a minimal keyed lookup table: id, year, + all 5 vars
# (avoids re-keying inside the loop)
lookup_cols <- c("id", "year", neighbor_source_vars)
val_lookup  <- cell_data[, ..lookup_cols]
setnames(val_lookup, "id", "to_id")
setkey(val_lookup, to_id, year)

for (var_name in neighbor_source_vars) {

  cat(sprintf("Processing neighbor stats for: %s\n", var_name))

  # Join: attach the neighbor's value to each edge
  # (only the column we need, to save memory)
  edges_with_val <- val_lookup[edge_cy, .(from_id, year, val = get(var_name)), on = .(to_id, year)]

  # Remove NAs before aggregation (matches original logic)
  edges_with_val <- edges_with_val[!is.na(val)]

  # Grouped aggregation
  stats <- edges_with_val[,
    .(
      nbr_max  = max(val),
      nbr_min  = min(val),
      nbr_mean = mean(val)
    ),
    by = .(from_id, year)
  ]

  # Rename columns to match the original naming convention
  new_names <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  setnames(stats, c("nbr_max", "nbr_min", "nbr_mean"), new_names)

  # Merge back to cell_data (left join so cells with no neighbors get NA)
  setnames(stats, "from_id", "id")
  setkeyv(stats, c("id", "year"))
  setkeyv(cell_data, c("id", "year"))

  cell_data <- stats[cell_data, on = .(id, year)]

  # Clean up to free memory
  rm(edges_with_val, stats)
  gc()

  cat(sprintf("  Done. Columns added: %s\n", paste(new_names, collapse = ", ")))
}

# Restore original row order
setorder(cell_data, .rowid)
cell_data[, .rowid := NULL]

cat("All neighbor features computed.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Prediction with the existing trained Random Forest
#     (No retraining â€” just call predict with the enriched data)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Example (adapt to your actual model object name):
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations with string hashing | Vectorized edge-list construction + `data.table` cross-join |
| `compute_neighbor_stats` | 6.46M Ã— 5 = 32.3M `lapply` iterations | 5 vectorized `data.table` grouped aggregations |
| Estimated runtime | 86+ hours | **~5â€“15 minutes** |
| Peak RAM | Potentially > 16 GB (list + strings + GC) | ~4â€“6 GB (edge list + one variable column at a time) |
| Numerical equivalence | baseline | **Identical** `max`, `min`, `mean` over non-NA rook neighbors |
| RF model | â€” | **Untouched; no retraining** |

The numerical estimand is preserved exactly: for each `(cell, year)`, the max, min, and mean are computed over the same set of non-NA rook-neighbor values as in the original code. The trained Random Forest model is never modified or retrained.
END_RESPONSE>>>
