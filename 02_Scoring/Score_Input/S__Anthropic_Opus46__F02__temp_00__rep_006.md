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
run_id: Anthropic_F02_T00_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F02_Anthropic_temp_00_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and named-vector lookups, which are extremely slow in interpreted R. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again, subsetting a numeric vector with variable-length index lists â€” repeated for each of the 5 neighbor source variables.

**Specific problems:**

1. **String-key lookups in `build_neighbor_lookup`:** For every row, `paste()` constructs neighbor keys and `idx_lookup[neighbor_keys]` performs named-vector lookup. Over 6.46M rows Ã— ~4 neighbors each, this is ~25M string constructions and hash lookups â€” all in an interpreted `lapply` loop.

2. **Redundant per-year work:** The neighbor *topology* is identical across all 28 years. Yet the lookup is rebuilt at the cell-year level, inflating the problem 28Ã—.

3. **`lapply` + `do.call(rbind, ...)`:** Allocating millions of tiny 3-element vectors and then row-binding them is memory-inefficient and slow.

4. **No vectorization or compiled-code path:** Everything runs in base R's interpreter with no use of `data.table`, matrix operations, or C++-backed routines.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Separate topology from time** | Build a cell-to-cell neighbor edge list *once* (344K cells), then join to panel by year using `data.table` equi-joins. |
| **Replace string keys with integer joins** | Use integer cell-ID and year columns directly; avoid all `paste()`/named-vector lookups. |
| **Vectorized grouped aggregation** | Explode the neighbor list into an edge table `(row_i, neighbor_row_j)`, join the variable values, and compute `max/min/mean` with `data.table`'s `by=` grouping â€” fully vectorized in C. |
| **Process all 5 variables in one pass** | Instead of looping `compute_neighbor_stats` 5 times (each scanning 6.46M rows), compute all 15 summary columns in a single grouped aggregation. |
| **Memory control** | The edge table has ~6.46M Ã— 4 â‰ˆ 25.8M rows of two integer columns (~200 MB), plus the joined numeric columns. Peak RAM stays well under 16 GB. |

**Expected speedup:** From 86+ hours to roughly **5â€“15 minutes** on the same laptop.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Convert panel data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a cell-level edge list from the nb object (once)
#     rook_neighbors_unique is a list of length = # cells,
#     indexed in the same order as id_order.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # neighbors[[k]] gives integer indices into id_order for cell k
  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells),
                  times = lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the spdep "0L" sentinel for cells with no neighbors

  valid    <- to_idx != 0L
  data.table(
    focal_id    = id_order[from_idx[valid]],
    neighbor_id = id_order[to_idx[valid]]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Give every cell-year row a fast integer key
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data[, row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Expand edges Ã— years  â†’  (focal_row, neighbor_row)
#     by joining on (id, year) for both sides
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Slim lookup: id, year â†’ row_idx
id_year_key <- cell_data[, .(id, year, row_idx)]
setkey(id_year_key, id, year)

# All unique years in the panel
years <- sort(unique(cell_data$year))

# Cross-join edges with years, then map to row indices
edge_yr <- CJ_dt <- edge_dt[, .(focal_id, neighbor_id)]
# Replicate for every year (memory: ~25.8M Ã— 28 â‰ˆ but we only
# need one year at a time conceptually; however data.table handles
# the full cross efficiently).
edge_yr <- edge_dt[, CJ(year = years), by = .(focal_id, neighbor_id)]

# Map focal  â†’ row_idx
setkey(edge_yr, focal_id, year)
edge_yr[id_year_key, focal_row := i.row_idx, on = .(focal_id = id, year)]

# Map neighbor â†’ row_idx
setkey(edge_yr, neighbor_id, year)
edge_yr[id_year_key, neighbor_row := i.row_idx, on = .(neighbor_id = id, year)]

# Drop edges where either side has no matching row
edge_yr <- edge_yr[!is.na(focal_row) & !is.na(neighbor_row)]

# Keep only what we need
edge_yr <- edge_yr[, .(focal_row, neighbor_row)]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Vectorised neighbor statistics for ALL variables at once
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Attach neighbor values to every edge
neighbor_vals <- cell_data[edge_yr$neighbor_row, ..neighbor_source_vars]
edge_yr <- cbind(edge_yr, neighbor_vals)

# Grouped aggregation â€” one pass, all 15 output columns
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

# Build the j-expression programmatically
agg_call <- as.call(c(
  as.name("list"),
  setNames(agg_exprs, agg_names)
))

stats_dt <- edge_yr[, eval(agg_call), by = .(focal_row)]

# Replace Inf / -Inf (from max/min on all-NA groups) with NA
for (col in agg_names) {
  set(stats_dt, which(is.infinite(stats_dt[[col]])), col, NA_real_)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Merge back into cell_data, preserving row order
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Rows with no neighbors will remain NA after the join
cell_data[stats_dt, (agg_names) := mget(paste0("i.", agg_names)),
          on = .(row_idx = focal_row)]

# Clean up helper column
cell_data[, row_idx := NULL]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6.  Predict with the existing Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g. `rf_model`) is used as-is.
# cell_data now contains the same 15 neighbor feature columns
# (neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, â€¦)
# with numerically identical values, so predictions are preserved.
#
# Example:
# cell_data$predicted_gdp <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Estimand

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The same `max`, `min`, `mean` aggregations are computed over exactly the same neighbor sets (rook contiguity, same year). No approximation or sampling is introduced. |
| **Trained RF model untouched** | The model object is never modified; only the feature-construction step is optimized. Column names and semantics are identical. |
| **No topology change** | The edge table is a faithful, lossless expansion of the original `nb` object. |

---

## Summary of Gains

| Metric | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M interpreted `lapply` iterations with string ops | One `data.table` equi-join (C-level hash join) |
| Stat computation | 5 Ã— 6.46M `lapply` loops | Single grouped `by=` aggregation over all 5 vars |
| Estimated wall time | 86+ hours | ~5â€“15 minutes |
| Peak RAM | Millions of small R list elements | ~2â€“4 GB contiguous `data.table` columns |
| Code complexity | Comparable | Comparable (fewer lines) |
END_RESPONSE>>>
