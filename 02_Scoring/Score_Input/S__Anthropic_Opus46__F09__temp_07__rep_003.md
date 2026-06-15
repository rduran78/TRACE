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
run_id: Anthropic_F09_T07_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. Row-level `lapply` over 6.46 million rows in `build_neighbor_lookup`
For every one of the ~6.46 million cell-year rows, the function:
- Looks up the cell's reference index in the neighbor list.
- Retrieves neighbor cell IDs.
- Pastes together character keys (`id_year`) for each neighbor.
- Performs named-vector lookups on a 6.46-million-element named character vector (`idx_lookup`).

This means ~6.46 million iterations, each doing string concatenation and named-vector search (which is O(n) or O(log n) per lookup in base R). The `idx_lookup` named vector with 6.46M entries is particularly punishing because R's named vector lookup is hash-based but still slow at this scale when called millions of times from an `lapply`.

### 2. Redundant recomputation of spatial structure per cell-year
The neighbor topology is **time-invariant** â€” cell A's rook neighbors are the same in 1992 as in 2019. Yet `build_neighbor_lookup` rebuilds the full neighbor mapping at the **cell-year** level (6.46M entries) rather than at the **cell** level (344,208 entries) and then joining by year. This inflates the work by a factor of 28Ã—.

### 3. Row-level `lapply` in `compute_neighbor_stats`
After the lookup is built, `compute_neighbor_stats` iterates over 6.46M list elements, extracting values and computing `max`/`min`/`mean` one row at a time. This prevents any vectorized or batch operation.

### Summary of bottlenecks

| Bottleneck | Scale | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | 6.46M iterations Ã— string ops | Neighbor topology rebuilt per cell-year instead of per cell |
| `idx_lookup` named vector | 6.46M-element named vector queried 6.46M times | Base R named lookup is slow at this scale |
| `compute_neighbor_stats` | 6.46M `lapply` iterations per variable Ã— 5 variables | Row-level R loop instead of vectorized join |

---

## Optimization Strategy

**Core insight:** Build the neighbor table once at the **cell level** (344,208 cells Ã— ~4 neighbors each â‰ˆ 1.37M directed edges), then use a vectorized `data.table` join to attach yearly attributes and compute grouped statistics.

### Steps:

1. **Build a cell-level edge table once** â€” a two-column `data.table` with `(cell_id, neighbor_id)` from the `spdep::nb` object. This is ~1.37M rows and is time-invariant.

2. **For each year**, join the cell attributes onto both sides of the edge table (the focal cell's year and the neighbor cell's year are the same), then compute `max`, `min`, `mean` of each neighbor variable grouped by `(cell_id, year)`.

3. **Merge** the resulting neighbor statistics back onto the main dataset.

This replaces 6.46M-element `lapply` calls with vectorized `data.table` keyed joins and grouped aggregations, which are orders of magnitude faster.

### Expected speedup:
- Edge table construction: seconds (one-time, 1.37M rows).
- Per-variable join + aggregation: the join expands to ~1.37M Ã— 28 â‰ˆ 38.4M rows, then groups back down to 6.46M. With `data.table`, this takes seconds to low minutes per variable.
- Total for 5 variables: **~1â€“5 minutes** instead of 86+ hours.

### Preserving the trained model and numerical estimand:
- The neighbor statistics (`max`, `min`, `mean`) are computed identically â€” same neighbor definitions, same variable values, same aggregation functions.
- The trained Random Forest model is not retrained; only the input feature table is rebuilt faster.
- Column names are preserved to match the model's expected feature names.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Ensure cell_data is a data.table with columns: id, year, and
#         the 5 neighbor source variables.
#         id_order and rook_neighbors_unique come from the existing pipeline.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build the time-invariant cell-level edge table ONCE
#         from the spdep::nb object (rook_neighbors_unique).
#         This replaces build_neighbor_lookup entirely.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj is a list of length = number of cells.
  # nb_obj[[i]] contains integer indices into id_order for neighbors of cell i.
  # An entry of 0L means no neighbors.
  edges <- rbindlist(lapply(seq_along(nb_obj), function(i) {
    nb_idx <- nb_obj[[i]]
    # spdep::nb encodes "no neighbors" as a single 0
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(
      focal_id    = id_order[i],
      neighbor_id = id_order[nb_idx]
    )
  }))
  return(edges)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows: (focal_id, neighbor_id)
# This is built ONCE and reused for all variables and all years.

cat("Edge table rows:", nrow(edge_dt), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: For each neighbor source variable, join yearly attributes
#         onto the edge table and compute grouped neighbor stats.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key cell_dt for fast joins
setkey(cell_dt, id, year)

for (var_name in neighbor_source_vars) {

  cat("Computing neighbor features for:", var_name, "\n")

  # Extract only the columns we need for the join: id, year, variable
  # This keeps memory usage low.
  attr_dt <- cell_dt[, .(id, year, value = get(var_name))]
  setnames(attr_dt, "id", "neighbor_id")
  setkey(attr_dt, neighbor_id, year)

  # Cross-join edge table with all 28 years, then join neighbor values.
  # More memory-efficient: join edge_dt to the attribute table directly.
  # We need to expand edges by year. Instead of a full cross join,
  # we join edges onto the data by (neighbor_id, year).

  # First, get the unique years
  years <- sort(unique(cell_dt$year))

  # Expand edge table by year: ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows
  # This fits comfortably in 16 GB RAM (3 integer/numeric columns).
  edge_year_dt <- CJ_dt_year(edge_dt, years)

  # Helper: expand edges by year efficiently
  # We use a cross join approach
  edge_year_dt <- edge_dt[, .(year = years), by = .(focal_id, neighbor_id)]

  # Join neighbor attribute values onto the expanded edge table
  setkey(edge_year_dt, neighbor_id, year)
  edge_year_dt[attr_dt, neighbor_value := i.value, on = .(neighbor_id, year)]

  # Remove rows where the neighbor had no data for that year (NA)
  # (matches the original !is.na filter in compute_neighbor_stats)
  valid_edges <- edge_year_dt[!is.na(neighbor_value)]

  # Compute grouped statistics: max, min, mean per (focal_id, year)
  stats_dt <- valid_edges[,
    .(
      nb_max  = max(neighbor_value),
      nb_min  = min(neighbor_value),
      nb_mean = mean(neighbor_value)
    ),
    by = .(focal_id, year)
  ]

  # Name columns to match the existing pipeline's naming convention.
  # The original compute_and_add_neighbor_features presumably creates
  # columns like: <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  setnames(stats_dt, c("nb_max", "nb_min", "nb_mean"),
                      c(max_col,  min_col,  mean_col))
  setnames(stats_dt, "focal_id", "id")

  # Merge back onto cell_dt
  # First remove old columns if they exist (idempotent reruns)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  setkey(stats_dt, id, year)
  cell_dt <- stats_dt[cell_dt, on = .(id, year)]

  # Clean up intermediate objects to conserve RAM

rm(attr_dt, edge_year_dt, valid_edges, stats_dt)
  gc()

  cat("  Done:", var_name, "\n")
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Convert back to data.frame if the downstream RF predict
#         expects one, then run prediction with the existing model.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model (e.g., rf_model) is unchanged.
# Predict exactly as before:
# cell_data$predicted <- predict(rf_model, newdata = cell_data)
```

### Handling cells with zero neighbors

The original code returns `c(NA, NA, NA)` for cells with no valid neighbors. The optimized code handles this automatically: cells that have no entry in `stats_dt` will receive `NA` for the neighbor columns after the right join (`stats_dt[cell_dt, ...]`), preserving identical behavior.

---

## Performance Comparison

| Metric | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M `lapply` iterations with string ops | One-time edge table: ~344K iterations, seconds |
| Per-variable computation | ~6.46M `lapply` iterations | Vectorized `data.table` join + `groupby` on ~38.4M rows |
| Total estimated time (5 vars) | **86+ hours** | **~1â€“5 minutes** |
| RAM peak | Moderate (6.46M-element list) | ~38.4M rows Ã— few columns â‰ˆ ~1â€“2 GB peak per variable |
| Numerical results | Baseline | **Identical** (same neighbors, same aggregations) |
| Trained RF model | Preserved | **Preserved (no retraining)** |

The key transformation is: **separate spatial topology (time-invariant) from temporal attributes (year-varying), build the topology once as a flat edge table, and use vectorized keyed joins instead of row-level R loops.**
END_RESPONSE>>>
