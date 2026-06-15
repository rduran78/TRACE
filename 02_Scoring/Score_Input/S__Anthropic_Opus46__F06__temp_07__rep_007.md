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
run_id: Anthropic_F06_T07_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) per-row `lapply` over 6.46M rows

For every single cell-year row (~6.46M), the function:
1. Looks up the spatial cell's rook neighbors (fine).
2. Constructs character key strings via `paste()` for every neighbor Ã— every row.
3. Indexes into a named character vector (`idx_lookup`) â€” named vector lookup in R is **O(n)** in the worst case because it uses linear hashing over strings.

This means ~6.46M iterations, each doing string construction and named-vector lookups. The `idx_lookup` vector itself has 6.46M entries, so each named lookup is expensive.

### Bottleneck 2: `compute_neighbor_stats` â€” `lapply` over 6.46M rows returning lists, then `do.call(rbind, ...)`

- The `lapply` returns a list of 6.46M 3-element vectors.
- `do.call(rbind, result)` on a list of 6.46M elements is extremely slow â€” it repeatedly allocates and copies memory.
- This is called **5 times** (once per source variable), compounding the cost.

### Why raster focal/kernel operations don't directly apply

Raster focal operations (e.g., `terra::focal`) assume a **complete regular grid with a fixed kernel window**. This panel dataset has:
- Potentially irregular spatial coverage (not all cells present in all years).
- A precomputed `spdep::nb` neighbor structure that may not map to a simple rectangular kernel.
- The need to operate within-year only (neighbors in the same year).

If the grid **is** complete and regular, focal operations could work, but they would change the pipeline semantics (e.g., boundary handling, NA handling). The safest approach that **preserves the original numerical estimand exactly** is to vectorize the current logic using `data.table` joins.

---

## Optimization Strategy

| Step | Current | Proposed | Speedup source |
|---|---|---|---|
| Neighbor lookup | Per-row `paste` + named vector lookup | Pre-build a `data.table` edge list of `(row_i, row_j)` using integer joins â€” **no character keys** | Eliminate 6.46M string ops; use hash joins |
| Neighbor stats | `lapply` over 6.46M rows + `do.call(rbind,...)` | Single vectorized `data.table` grouped aggregation: join edge list to values, group by `row_i`, compute `max/min/mean` | Vectorized C-level grouping |
| Repeat Ã—5 vars | 5 separate passes rebuilding lists | One join brings all 5 variables; compute all 15 features in one grouped operation | 5Ã— fewer passes |

**Expected runtime: ~1â€“3 minutes** instead of 86+ hours.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Inputs assumed to exist:
#       cell_data            â€” data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#       id_order             â€” integer/character vector of cell IDs in the order matching rook_neighbors_unique
#       rook_neighbors_unique â€” spdep::nb object (list of integer index vectors into id_order)
#       rf_model             â€” pre-trained Random Forest model (not retrained)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table (by reference if already a data.table)
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build a spatial directed edge list:  (from_id, to_id)
#     This encodes "to_id is a rook neighbor of from_id".
#     Done once; purely spatial, no year dimension yet.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
    return(data.table(from_id = integer(0), to_id = integer(0)))
  }
  data.table(from_id = id_order[i], to_id = id_order[nb_idx])
}))

cat("Edge list rows:", nrow(edge_list), "\n")
# Should be ~1,373,394

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Add a row index to cell_data so we can map results back.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

cell_data[, .row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Build the full (row_i  â†’  row_j) edge list in row-index space
#     by joining on (id, year).  This is the key step that replaces
#     build_neighbor_lookup entirely.
#
#     For each row i with (from_id, year), we find all rows j with
#     (to_id, same year).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Slim lookup tables â€” only what we need for the join
lookup_from <- cell_data[, .(row_i = .row_idx, from_id = id, year)]
lookup_to   <- cell_data[, .(row_j = .row_idx, to_id   = id, year)]

# Key for fast join
setkey(edge_list, from_id)
setkey(lookup_from, from_id)

# Step A: attach row_i and year to each edge via from_id
#   result: (from_id, to_id, row_i, year)
edges_with_i <- edge_list[lookup_from, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]

# Step B: attach row_j via (to_id, year)
setkey(edges_with_i, to_id, year)
setkey(lookup_to, to_id, year)

row_edges <- lookup_to[edges_with_i, on = c("to_id", "year"), nomatch = 0L]
# result columns: row_i, row_j  (plus from_id, to_id, year â€” can drop)

# Keep only what we need
row_edges <- row_edges[, .(row_i, row_j)]

cat("Row-level edge pairs:", nrow(row_edges), "\n")

# Free intermediate objects
rm(lookup_from, lookup_to, edges_with_i)
gc()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Compute all neighbor stats in one vectorized pass.
#     For each of the 5 source variables, compute max, min, mean
#     of the neighbor values, grouped by row_i.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Extract neighbor values: attach all 5 variable columns from the neighbor rows
neighbor_vals <- cell_data[row_edges$row_j, ..neighbor_source_vars]
neighbor_vals[, row_i := row_edges$row_i]

# Grouped aggregation â€” one pass for all 5 variables
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Build and evaluate the aggregation call
stats <- neighbor_vals[, lapply(agg_exprs, eval, envir = .SD), by = row_i]

# â”€â”€â”€ Alternative cleaner approach (equivalent, avoids bquote complexity) â”€â”€â”€
# Compute stats per variable in a simple loop â€” still fully vectorized inside
stats_list <- vector("list", length(neighbor_source_vars))

for (k in seq_along(neighbor_source_vars)) {
  v <- neighbor_source_vars[k]
  col_vals <- cell_data[[v]][row_edges$row_j]

  tmp <- data.table(row_i = row_edges$row_i, val = col_vals)
  # Remove NAs before aggregation to match original logic
  tmp <- tmp[!is.na(val)]

  agg_k <- tmp[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = row_i]

  setnames(agg_k, c("nb_max", "nb_min", "nb_mean"),
           paste0("neighbor_", c("max_", "min_", "mean_"), v))

  stats_list[[k]] <- agg_k
}

# Merge all stats together by row_i
all_stats <- stats_list[[1]]
for (k in 2:length(stats_list)) {
  all_stats <- merge(all_stats, stats_list[[k]], by = "row_i", all = TRUE)
}

rm(neighbor_vals, stats_list, tmp, row_edges)
gc()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Join the 15 neighbor features back to cell_data.
#     Rows with no neighbors (or all-NA neighbors) get NA â€” matching
#     the original implementation.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Ensure any infinite values from max/min on empty sets become NA
inf_cols <- setdiff(names(all_stats), "row_i")
for (col in inf_cols) {
  set(all_stats, which(is.infinite(all_stats[[col]])), col, NA_real_)
}

# Left join back to cell_data
cell_data <- merge(cell_data, all_stats, by.x = ".row_idx", by.y = "row_i", all.x = TRUE)

# Clean up helper column
cell_data[, .row_idx := NULL]

# Restore original row order (merge may reorder)
setorder(cell_data, .row_idx)  
# If .row_idx was already removed, re-sort by original key, e.g.:
# setkey(cell_data, id, year)

rm(all_stats)
gc()

cat("Neighbor features added. Columns:", ncol(cell_data), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6.  Predict with the pre-trained Random Forest (unchanged).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# The RF model expects a data.frame; convert only the predictor columns
# (adjust predictor_names to match your model's expected features)
# predictor_names <- rf_model$forest$independent.variable.names  # if ranger
# predictor_names <- names(rf_model$forest$xlevels)              # if randomForest

cell_data$prediction <- predict(rf_model, newdata = as.data.frame(cell_data))$predictions
# Adjust the predict call syntax to match your RF package (ranger, randomForest, etc.)
```

---

## Summary of Changes and Guarantees

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The `max`, `min`, `mean` computations use the same NA-removal logic. No approximation, no kernel substitution. Results are identical to the original. |
| **RF model untouched** | The pre-trained model is only used for `predict()` â€” no retraining. |
| **Memory** | The largest object is `row_edges` (~1.37M edges Ã— 28 years â‰ˆ 38M rows Ã— 2 integer cols â‰ˆ 0.6 GB). Combined with `cell_data` (~6.46M Ã— 110 cols), total stays well within 16 GB. Intermediate objects are freed with `rm(); gc()`. |
| **Runtime** | Edge list construction: ~10 sec. Row-edge join: ~30 sec. Grouped aggregation (5 vars): ~2â€“3 min. **Total: ~3â€“5 minutes** vs. 86+ hours. |
| **No raster focal substitution** | Focal/kernel operations were considered but rejected because they cannot guarantee identical results for irregular boundaries, missing cells, or the specific `spdep::nb` topology. The `data.table` join approach is both faster and exact. |
END_RESPONSE>>>
