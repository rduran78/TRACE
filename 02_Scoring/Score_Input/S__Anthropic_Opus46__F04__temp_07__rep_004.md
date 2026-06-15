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
run_id: Anthropic_F04_T07_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Looks up the cell's reference index via character coercion and named-vector lookup.
2. Retrieves neighbor cell IDs.
3. Constructs `paste()`-based string keys for each neighborâ€“year pair.
4. Matches those keys against a named character vector (`idx_lookup`).

**String construction and named-vector matching inside a per-row `lapply` over 6.46M rows is the dominant cost.** Each iteration does character coercion, `paste()`, and `[` on a 6.46M-length named vector (effectively a linear-ish hash lookup repeated for every neighbor of every row). This is O(N Ã— k) string operations where N â‰ˆ 6.46M and k â‰ˆ average neighbor count (~4 for rook). That's ~26M string constructions and lookups, all in an interpreted R loop.

`compute_neighbor_stats` is a secondary bottleneck: another `lapply` over 6.46M rows doing subsetting and summary stats per row, repeated 5 times (once per variable).

## Optimization Strategy

1. **Replace string-key lookups with integer-arithmetic indexing.** Since years are a contiguous sequence (1992â€“2019, 28 years) and cell IDs can be mapped to integers 1â€“344,208, every (cell, year) pair maps to a unique row via `(cell_index - 1) * 28 + (year - 1992 + 1)` â€” no strings needed. This eliminates all `paste()` and named-vector lookups.

2. **Vectorize `build_neighbor_lookup` entirely** by pre-expanding the neighbor list into a flat edge table, then computing target row indices with vectorized integer arithmetic.

3. **Vectorize `compute_neighbor_stats`** using the flat edge table with `data.table` grouped aggregation â€” replacing the per-row `lapply` with a single grouped operation per variable.

4. **Process all 5 variables in one pass** over the edge table rather than 5 separate `lapply` calls.

These changes reduce estimated runtime from 86+ hours to **minutes**, stay well within 16 GB RAM, and produce numerically identical output.

## Optimized R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Ensure cell_data is a data.table sorted by (id, year)
#     so that row position can be computed by integer arithmetic.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# Build integer cell index (1-based, matching id_order)
id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

# Guarantee row ordering: cell index fastest-varying within year,
# or equivalently year fastest-varying within cell.  We choose the
# latter: rows are ordered (id, year) so that for cell index c and
# year index t the row number is  (c - 1) * n_years + t.
years      <- sort(unique(cell_dt$year))
n_years    <- length(years)                       # 28
year_to_t  <- setNames(seq_along(years), as.character(years))

cell_dt[, cell_idx := id_to_idx[as.character(id)]]
cell_dt[, year_idx := year_to_t[as.character(year)]]
setorder(cell_dt, cell_idx, year_idx)             # deterministic order
# Now row number = (cell_idx - 1) * n_years + year_idx
# Verify:
stopifnot(all(cell_dt$cell_idx == rep(seq_along(id_order), each = n_years)))
stopifnot(all(cell_dt$year_idx == rep(seq_len(n_years), times = length(id_order))))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build flat edge table (vectorised, no per-row loop)
#     Each row: (source_cell_idx, neighbor_cell_idx)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
n_cells <- length(id_order)

# Expand the nb object into a two-column integer matrix
source_idx <- rep(seq_len(n_cells),
                  times = lengths(rook_neighbors_unique))
neighbor_idx <- unlist(rook_neighbors_unique, use.names = FALSE)

# Remove any 0-length or self-referencing entries produced by spdep
valid <- neighbor_idx > 0L & neighbor_idx <= n_cells
source_idx   <- source_idx[valid]
neighbor_idx <- neighbor_idx[valid]

# Now expand across all 28 years: for every year t, every directed
# edge (s, n) maps to  source_row -> neighbor_row.
# source_row   = (source_idx   - 1) * n_years + t
# neighbor_row = (neighbor_idx - 1) * n_years + t
edges_per_year <- length(source_idx)              # ~1.37 M

t_vec <- rep(seq_len(n_years), each = edges_per_year)   # year indices
s_vec <- rep(source_idx,       times = n_years)
n_vec <- rep(neighbor_idx,     times = n_years)

edge_dt <- data.table(
  source_row   = (s_vec - 1L) * n_years + t_vec,
  neighbor_row = (n_vec - 1L) * n_years + t_vec
)
rm(t_vec, s_vec, n_vec, valid, source_idx, neighbor_idx)  # free memory

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Compute neighbor stats for all variables at once
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Attach neighbor values for every variable to the edge table
for (v in neighbor_source_vars) {
  set(edge_dt, j = v, value = cell_dt[[v]][edge_dt$neighbor_row])
}

# Grouped aggregation: max, min, mean per source_row per variable
# This replaces the 6.46 M-iteration lapply, executed once for all vars.
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

stats_dt <- edge_dt[, lapply(agg_exprs, eval), by = source_row]

# Replace Inf / -Inf (from max/min on all-NA groups) with NA
for (col in agg_names) {
  vals <- stats_dt[[col]]
  set(stats_dt, i = which(is.infinite(vals)), j = col, value = NA_real_)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Left-join back to cell_dt, preserving rows that had no neighbors
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt[, source_row := .I]
cell_dt <- merge(cell_dt, stats_dt, by = "source_row", all.x = TRUE)

# Clean up helper columns
cell_dt[, c("cell_idx", "year_idx", "source_row") := NULL]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Convert back to data.frame if the downstream RF predict() expects one
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.frame(cell_dt)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Predict with the already-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

## Why This Is Equivalent

| Original | Optimized | Equivalence |
|---|---|---|
| `paste(id, year)` string key â†’ named vector lookup | `(cell_idx - 1) * n_years + year_idx` integer arithmetic | Bijection between (id, year) and row number; same rows retrieved |
| `lapply` over 6.46M rows computing `max`, `min`, `mean` of neighbor values | `data.table` grouped `max`, `min`, `mean` by `source_row` | Identical aggregation functions, identical NA handling |
| Rows with no neighbors â†’ `c(NA, NA, NA)` | `all.x = TRUE` merge leaves them as `NA` | Identical |

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M string-based R-loop iterations â†’ **hours** | Vectorized integer expansion â†’ **~10â€“30 seconds** |
| `compute_neighbor_stats` (Ã—5 vars) | 5 Ã— 6.46M `lapply` iterations â†’ **tens of hours** | Single `data.table` grouped aggregation â†’ **~1â€“3 minutes** |
| Peak RAM (edge table, 5 float cols + 2 int cols) | Moderate (list of 6.46M integer vectors) | ~38.4M edges Ã— 7 cols Ã— 8 bytes â‰ˆ **~2.1 GB** (fits in 16 GB) |
| **Total** | **86+ hours** | **~2â€“5 minutes** |

The trained Random Forest model is untouched; only the feature-construction step is replaced. The numerical output (max, min, mean of each neighbor variable per cell-year) is identical.
END_RESPONSE>>>
