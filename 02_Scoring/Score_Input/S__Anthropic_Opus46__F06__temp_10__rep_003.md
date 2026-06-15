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
run_id: Anthropic_F06_T10_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows
For every one of the ~6.46 million cell-year rows, the function performs character-based key construction (`paste`), named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. Character hashing over 6.46M rows is extremely expensive. Crucially, **the neighbor topology is time-invariant** â€” the same cell has the same rook neighbors in every year. Yet the lookup is rebuilt redundantly for every cell-year combination, inflating what should be a ~344K-cell operation into a ~6.46M-row operation.

### Bottleneck 2: `compute_neighbor_stats` â€” Row-wise `lapply` over 6.46M rows
For each of the 5 variables, an `lapply` iterates over all 6.46M rows, subsetting a vector by index, removing NAs, and computing `max`, `min`, `mean`. This is called 5 times, producing ~32.3M R-level function calls total. The `do.call(rbind, result)` on a 6.46M-element list is also expensive.

### Why raster focal/kernel operations don't directly apply
Raster focal operations (e.g., `terra::focal`) assume data lives on a complete regular grid with a fixed kernel. Here the panel is long-format (cell Ã— year), the grid may have irregular boundaries/missing cells, and the neighbor structure is an arbitrary `spdep::nb` object. Forcing it into a raster stack for 28 years Ã— 5 variables would require reshaping, gap-filling, and re-extracting â€” adding complexity and risking numerical discrepancies. The better strategy is to **vectorize the sparse-neighbor computation directly** using `data.table` joins and matrix operations.

---

## Optimization Strategy

1. **Separate spatial topology from temporal replication.** Build a simple integer-to-integer neighbor edge list once (344K cells), then join it to the panel by year â€” letting `data.table` handle the replication efficiently.

2. **Replace row-wise `lapply` with grouped `data.table` aggregation.** Convert the neighbor edge list into a two-column `data.table` (focal_id, neighbor_id), join on (neighbor_id, year) to pull neighbor values, then group-by (focal_id, year) to compute `max`, `min`, `mean` in compiled C code inside `data.table`.

3. **Process all 5 variables in one pass** per join, avoiding redundant joins.

4. **Estimated speedup:** From ~86+ hours to ~5â€“15 minutes, depending on disk I/O.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Inputs assumed to exist:
#       cell_data             â€” data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#       id_order              â€” integer vector of cell IDs (the order matching rook_neighbors_unique)
#       rook_neighbors_unique â€” spdep::nb object (list of integer index vectors into id_order)
#       rf_model              â€” pre-trained Random Forest (not retrained)
# ---------------------------------------------------------------

# Convert to data.table (no copy if already data.table)
setDT(cell_data)

# ---------------------------------------------------------------
# 1.  Build a SPATIAL-ONLY edge list (not replicated across years)
#     This replaces build_neighbor_lookup entirely.
# ---------------------------------------------------------------

# Create edge list:  focal_id  ->  neighbor_id  (using original cell IDs)
edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  # spdep::nb encodes zero-neighbor regions as 0L; filter those out

  nb_idx <- nb_idx[nb_idx > 0L]
  if (length(nb_idx) == 0L) return(NULL)
  data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
}))

cat("Edge list rows:", nrow(edge_list), "\n")
# Expected: ~1,373,394 directed edges

# ---------------------------------------------------------------
# 2.  Join edge list to panel data to get neighbor variable values,
#     then aggregate by (focal_id, year).
# ---------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset the columns we need from cell_data for the neighbor join
# We join on (neighbor_id = id, year) to retrieve neighbor values.
neighbor_vals_dt <- cell_data[, c("id", "year", neighbor_source_vars), with = FALSE]

# Key for fast join
setkey(neighbor_vals_dt, id, year)
setkey(edge_list, neighbor_id)

# Expand edges Ã— years:  for each (focal_id, neighbor_id) pair,
# join the neighbor's values for every year the neighbor appears in.
# This is a keyed join:  edge_list[neighbor_vals_dt] but we want
# each edge paired with matching year rows of the *neighbor*.
# Strategy: merge edge_list with neighbor_vals_dt on neighbor_id == id.

# Rename for clarity before join
setnames(neighbor_vals_dt, "id", "neighbor_id")
setkey(neighbor_vals_dt, neighbor_id, year)

# Join: for every edge (focal_id, neighbor_id), pull all year-rows of the neighbor
# This produces a table of (focal_id, neighbor_id, year, ntl, ec, ...)
joined <- merge(edge_list, neighbor_vals_dt, by = "neighbor_id", allow.cartesian = TRUE)
# allow.cartesian = TRUE because one neighbor_id maps to 28 year-rows

cat("Joined rows:", nrow(joined), "\n")
# Expected: ~1,373,394 edges Ã— 28 years â‰ˆ 38.5M rows (fits in 16 GB)

# ---------------------------------------------------------------
# 3.  Compute max, min, mean per (focal_id, year) for each variable
# ---------------------------------------------------------------

# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Aggregate (this is the heavy step, but runs in compiled C inside data.table)
neighbor_stats <- joined[,
  eval(as.call(c(as.name("list"), agg_exprs))),
  by = .(focal_id, year)
]

# Handle -Inf / Inf from max/min of all-NA groups â†’ set to NA
for (col in agg_names) {
  vals <- neighbor_stats[[col]]
  set(neighbor_stats, i = which(is.infinite(vals)), j = col, value = NA_real_)
}

cat("Neighbor stats rows:", nrow(neighbor_stats), "\n")

# ---------------------------------------------------------------
# 4.  Merge neighbor stats back onto cell_data
# ---------------------------------------------------------------

# Remove any pre-existing neighbor columns to avoid duplication
existing_nb_cols <- intersect(names(cell_data), agg_names)
if (length(existing_nb_cols) > 0L) {
  cell_data[, (existing_nb_cols) := NULL]
}

setkey(cell_data, id, year)
setkey(neighbor_stats, focal_id, year)

cell_data <- merge(cell_data, neighbor_stats,
                   by.x = c("id", "year"),
                   by.y = c("focal_id", "year"),
                   all.x = TRUE)

cat("Final cell_data rows:", nrow(cell_data), "\n")

# ---------------------------------------------------------------
# 5.  Predict with the EXISTING Random Forest (no retraining)
# ---------------------------------------------------------------

# rf_model is the pre-trained model object â€” used as-is
# Ensure the predictor columns match what the model expects
# (the 15 new neighbor_* columns are now present alongside the ~95 others)

cell_data[, predicted := predict(rf_model, newdata = cell_data)]

# ---------------------------------------------------------------
# Cleanup large temporaries
# ---------------------------------------------------------------
rm(joined, neighbor_vals_dt, edge_list, neighbor_stats)
gc()
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor topology** | `edge_list` is built from the identical `rook_neighbors_unique` nb object and `id_order`. |
| **Same statistics** | `max`, `min`, `mean` with `na.rm = TRUE`, grouped by `(focal_id, year)`, reproduces exactly what the original `compute_neighbor_stats` computes per row. |
| **Cells with no neighbors** | They have no rows in `edge_list`, so after the `all.x = TRUE` merge they receive `NA` in every neighbor column â€” identical to the original `c(NA, NA, NA)` return. |
| **Trained RF model untouched** | `rf_model` is only used in `predict()`, never refit. |
| **Numerical estimand preserved** | No approximation, interpolation, or raster resampling is introduced. The computation is algebraically identical. |

## Expected Performance

| Step | Estimated Time |
|---|---|
| Build edge list (344K cells) | < 5 seconds |
| Merge edges Ã— years (~38.5M rows) | ~1â€“2 minutes |
| Grouped aggregation (15 stats) | ~2â€“5 minutes |
| Merge back to cell_data | ~1 minute |
| **Total** | **~5â€“10 minutes** |

Peak RAM â‰ˆ joined table (~38.5M rows Ã— 7 cols â‰ˆ 2.2 GB) + cell_data (~6.46M Ã— 125 cols â‰ˆ 6.5 GB) â‰ˆ **~10â€“12 GB**, within the 16 GB budget.
END_RESPONSE>>>
