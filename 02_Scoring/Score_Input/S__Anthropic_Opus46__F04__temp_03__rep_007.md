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
run_id: Anthropic_F04_T03_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Performs a character-based hash lookup (`id_to_ref`),
2. Indexes into the `neighbors` list,
3. Constructs character key strings via `paste()` for every neighbor,
4. Looks up those keys in `idx_lookup` (a named character vector).

With ~6.46M rows and an average of ~4 rook neighbors per cell, this creates and searches **~25.8 million paste/key-lookup operations** inside an R-level loop â€” all interpreted, no vectorization. `compute_neighbor_stats` then loops again over 6.46M entries, but is lighter. The combined cost explains the 86+ hour estimate.

**Root causes:**
- Row-level `lapply` in pure R over millions of rows.
- Repeated `paste()` string construction and named-vector lookup (O(n) or hash overhead per call).
- `compute_neighbor_stats` uses another `lapply` + `do.call(rbind, ...)` over 6.46M small vectors.

## Optimization Strategy

**Replace the row-level loop with a vectorized, year-grouped merge + `data.table` aggregation.**

Key insight: For a given year, every cell's neighbors are the same set of cell IDs (from the static `rook_neighbors_unique` nb object). So we can:

1. Build an **edge list** from the nb object once (source_id â†’ neighbor_id), ~1.37M edges.
2. For each year, join the edge list to the data to retrieve neighbor variable values â€” this is a vectorized merge.
3. Aggregate (max, min, mean) by (source_id, year) using `data.table` grouped operations.

This eliminates all per-row R-level loops and string key construction. Expected speedup: **~100â€“500x** (minutes instead of days).

## Optimized R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Build a static edge list from the nb object (run once)
# ---------------------------------------------------------------
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  # id_order is the vector mapping position -> cell id
  n <- length(neighbors)
  # Pre-calculate total edges for pre-allocation
  n_edges <- sum(vapply(neighbors, length, integer(1)))
  src <- integer(n_edges)
  dst <- integer(n_edges)
  pos <- 1L
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    len <- length(nb_i)
    if (len > 0L) {
      src[pos:(pos + len - 1L)] <- id_order[i]
      dst[pos:(pos + len - 1L)] <- id_order[nb_i]
      pos <- pos + len
    }
  }
  data.table(source_id = src, neighbor_id = dst)
}

# ---------------------------------------------------------------
# 2. Compute neighbor features for one variable (vectorized)
# ---------------------------------------------------------------
compute_neighbor_features_fast <- function(dt, edge_dt, var_name) {
  # dt is a data.table with columns: id, year, <var_name>
  # edge_dt is data.table(source_id, neighbor_id)

  # Subset to needed columns for the join
  vals_dt <- dt[, .(neighbor_id = id, year, val = get(var_name))]

  # Join: for each (source_id, year), look up each neighbor's value
  # Keyed join on (neighbor_id, year)
  setkey(vals_dt, neighbor_id, year)
  # Expand edges by year via join
  joined <- edge_dt[vals_dt, on = .(neighbor_id), allow.cartesian = TRUE, nomatch = 0L]
  # joined now has columns: source_id, neighbor_id, year, val

  # Remove NAs in val before aggregation
  joined <- joined[!is.na(val)]

  # Aggregate by (source_id, year)
  agg <- joined[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = .(source_id, year)]

  # Rename to match variable-specific column names
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(agg, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  agg
}

# ---------------------------------------------------------------
# 3. Main pipeline (replaces the original outer loop)
# ---------------------------------------------------------------
# Convert to data.table if not already
cell_dt <- as.data.table(cell_data)

# Build edge list once (~1.37M rows)
edge_dt <- build_edge_list(id_order, rook_neighbors_unique)

# Source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute and merge neighbor features for each variable
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")
  agg <- compute_neighbor_features_fast(cell_dt, edge_dt, var_name)
  # Merge back onto main table
  cell_dt <- merge(cell_dt, agg, by.x = c("id", "year"), by.y = c("source_id", "year"), all.x = TRUE)
  rm(agg)
}

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | The aggregation computes the identical `max`, `min`, `mean` of non-NA neighbor values per (cell, year). No approximation. |
| **Trained RF model** | No model retraining; we only reconstruct the same input features the model expects. Column names follow the same `nb_max_<var>`, `nb_min_<var>`, `nb_mean_<var>` convention (adjust `setnames` if the original used different names). |
| **Missing data handling** | `!is.na(val)` filtering and `all.x = TRUE` merge produce `NA` for cells with zero valid neighbors, matching the original `c(NA, NA, NA)` return. |

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build lookup / edge list | ~hours (6.46M `paste` + lookup) | **~2 seconds** (vectorized pre-allocation) |
| Neighbor stats (Ã—5 vars) | ~days (6.46M Ã— 5 `lapply`) | **~2â€“5 minutes** (`data.table` keyed join + grouped agg) |
| **Total** | **~86+ hours** | **~5â€“10 minutes** |

The `data.table` keyed join on `(neighbor_id, year)` is O(n log n) and cache-friendly, and the grouped aggregation is internally parallelized in C. Peak memory for the largest intermediate (`joined`) is roughly 1.37M edges Ã— 28 years Ã— 3 columns â‰ˆ ~300 MB, well within 16 GB RAM.
END_RESPONSE>>>
