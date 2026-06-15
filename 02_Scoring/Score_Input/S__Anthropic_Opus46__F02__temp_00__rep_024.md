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
run_id: Anthropic_F02_T00_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F02_Anthropic_temp_00_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and triggers repeated hash lookups on a named vector of 6.46 million entries â€” an O(nÂ²)-like workload in practice. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again per variable, subsetting numeric vectors by index lists. With 5 variables, that's ~32.3 million list iterations total.

**Specific problems:**

1. **`paste()` + named-vector lookup per row in `build_neighbor_lookup`:** Creating ~6.46M string keys per row-batch and looking them up in a 6.46M-entry named vector is extremely slow. Named vector lookup in R is O(n) per query in the worst case (hashing collisions at scale).
2. **`lapply` over 6.46M elements:** Each closure call has R interpreter overhead. The neighbor lookup returns a list of 6.46M integer vectors â€” high memory fragmentation.
3. **`do.call(rbind, result)` on a 6.46M-element list:** This is a known slow pattern in R; it incrementally builds a matrix.
4. **No vectorization or use of data.table/matrix operations:** Everything is scalar/list-based R.

---

## Optimization Strategy

### Key Insight: Replace row-level iteration with vectorized joins and grouped aggregations using `data.table`.

1. **Replace `build_neighbor_lookup` entirely.** Instead of building a 6.46M-element list of index vectors, build a **flat edge table** (`data.table`) of `(row_i, neighbor_row_j)` pairs. This is a one-time vectorized merge.

2. **Replace `compute_neighbor_stats` with a grouped `data.table` aggregation.** Join the edge table to the variable column, then compute `max`, `min`, `mean` grouped by the focal row index â€” fully vectorized in C via `data.table`.

3. **Memory estimate:** The edge table will have ~1.37M neighbor pairs Ã— 28 years â‰ˆ ~38.5M rows Ã— 2 integer columns â‰ˆ ~308 MB. The full dataset of 6.46M rows Ã— 110 columns â‰ˆ ~5.7 GB at 8 bytes/double. This is tight on 16 GB but feasible if we avoid duplication and process variables one at a time.

4. **Preserve the trained RF model and original numerical estimand:** We only change how features are computed, not what is computed. The `max`, `min`, `mean` aggregations are identical, so predictions are numerically identical.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 1: Build a flat edge table (replaces build_neighbor_lookup)
# ---------------------------------------------------------------
build_neighbor_edges <- function(cell_data_dt, id_order, neighbors) {
  # Map each cell id to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Build a flat data.table of (focal_id, neighbor_id) from the nb object
  # neighbors[[k]] gives the indices in id_order that are neighbors of id_order[k]
  focal_refs <- rep(seq_along(neighbors), lengths(neighbors))
  neighbor_refs <- unlist(neighbors, use.names = FALSE)

  edge_ids <- data.table(
    focal_id    = id_order[focal_refs],
    neighbor_id = id_order[neighbor_refs]
  )

  # Get the unique years
  years <- sort(unique(cell_data_dt$year))

  # Cross-join edges with years to get (focal_id, year, neighbor_id) triples
  # This is the panel-expanded edge list
  edge_panel <- edge_ids[, .(year = years), by = .(focal_id, neighbor_id)]

  # Now map (focal_id, year) -> row index in cell_data_dt
  # and (neighbor_id, year) -> row index in cell_data_dt
  # We add a row index column to cell_data_dt
  cell_data_dt[, .row_idx := .I]

  # Create keyed lookup: (id, year) -> row_idx
  id_year_lookup <- cell_data_dt[, .(id, year, .row_idx)]
  setkey(id_year_lookup, id, year)

  # Map focal
  setnames(id_year_lookup, c("id", "year", ".row_idx"),
           c("focal_id", "year", "focal_row"))
  setkey(id_year_lookup, focal_id, year)
  edge_panel <- id_year_lookup[edge_panel, on = .(focal_id, year), nomatch = 0L]

  # Map neighbor
  setnames(id_year_lookup, c("focal_id", "year", "focal_row"),
           c("neighbor_id", "year", "neighbor_row"))
  setkey(id_year_lookup, neighbor_id, year)
  edge_panel <- id_year_lookup[edge_panel, on = .(neighbor_id, year), nomatch = 0L]

  # Clean up: return only the integer row indices
  edge_panel[, .(focal_row, neighbor_row)]
}

# ---------------------------------------------------------------
# STEP 2: Compute neighbor stats via grouped aggregation
#         (replaces compute_neighbor_stats)
# ---------------------------------------------------------------
compute_neighbor_stats_dt <- function(cell_data_dt, edge_dt, var_name) {
  n <- nrow(cell_data_dt)

  # Extract neighbor values via the edge table
  vals <- cell_data_dt[[var_name]]
  work <- edge_dt[, .(focal_row, nval = vals[neighbor_row])]

  # Remove NAs in neighbor values
  work <- work[!is.na(nval)]

  # Grouped aggregation
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row]

  # Initialize result columns with NA
  res_max  <- rep(NA_real_, n)
  res_min  <- rep(NA_real_, n)
  res_mean <- rep(NA_real_, n)

  # Fill in computed values
  res_max[agg$focal_row]  <- agg$nb_max
  res_min[agg$focal_row]  <- agg$nb_min
  res_mean[agg$focal_row] <- agg$nb_mean

  data.table(nb_max = res_max, nb_min = res_min, nb_mean = res_mean)
}

# ---------------------------------------------------------------
# STEP 3: Main pipeline (replaces outer loop)
# ---------------------------------------------------------------
run_neighbor_feature_pipeline <- function(cell_data, id_order,
                                          rook_neighbors_unique) {
  cell_data_dt <- as.data.table(cell_data)

  message("Building edge table...")
  edge_dt <- build_neighbor_edges(cell_data_dt, id_order, rook_neighbors_unique)
  setkey(edge_dt, focal_row)
  message(sprintf("Edge table: %s rows", format(nrow(edge_dt), big.mark = ",")))

  # Remove temporary column
  cell_data_dt[, .row_idx := NULL]

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_neighbor_stats_dt(cell_data_dt, edge_dt, var_name)

    # Add columns with the same naming convention as the original code
    set(cell_data_dt, j = paste0(var_name, "_nb_max"),  value = stats$nb_max)
    set(cell_data_dt, j = paste0(var_name, "_nb_min"),  value = stats$nb_min)
    set(cell_data_dt, j = paste0(var_name, "_nb_mean"), value = stats$nb_mean)
  }

  # Return as data.frame if the original was a data.frame
  if (inherits(cell_data, "data.frame") && !inherits(cell_data, "data.table")) {
    return(as.data.frame(cell_data_dt))
  }
  cell_data_dt
}

# ---------------------------------------------------------------
# USAGE (drop-in replacement for the original outer loop)
# ---------------------------------------------------------------
cell_data <- run_neighbor_feature_pipeline(
  cell_data, id_order, rook_neighbors_unique
)

# The trained Random Forest model is unchanged.
# Predictions use the same feature columns with identical numerical values.
```

---

## Expected Improvement

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M `lapply` iterations with string ops | One vectorized `data.table` merge (~minutes) |
| `compute_neighbor_stats` (per var) | ~6.46M `lapply` iterations | One `data.table` grouped aggregation (~seconds) |
| Total estimated time | 86+ hours | **~10â€“30 minutes** |
| Peak RAM | Unbounded list fragmentation | ~4â€“8 GB (controlled) |
| Numerical output | max/min/mean per cell-year | **Identical** max/min/mean per cell-year |

The trained Random Forest model requires no changes â€” the feature columns produced are numerically identical to the originals, just computed orders of magnitude faster.
END_RESPONSE>>>
