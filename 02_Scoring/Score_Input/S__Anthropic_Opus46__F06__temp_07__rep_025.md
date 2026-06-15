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
run_id: Anthropic_F06_T07_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's reference index.
2. Retrieves that cell's rook neighbors (spatial, ~4 per cell).
3. Pastes together `neighbor_id_year` keys (character concatenation over millions of rows).
4. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) â€” each lookup is O(n) hash probe on a 6.46M-element named vector, repeated ~4 times per row.

This produces a **list of 6.46 million integer vectors**. The `paste`-and-lookup pattern is executed ~25.8 million times (6.46M rows Ã— ~4 neighbors). This is the dominant cost, likely accounting for 70+ hours of the 86-hour estimate.

### Bottleneck B: `compute_neighbor_stats` â€” repeated `lapply` over 6.46M rows

For each of 5 variables, another `lapply` iterates over the 6.46M-element lookup list, subsetting a numeric vector and computing `max`, `min`, `mean`. That's 5 Ã— 6.46M = 32.3M small R function calls. This is slow but secondary (~15 hours estimated).

### Why raster focal/kernel operations don't directly apply

Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed kernel applied uniformly. While the data *is* on a grid, the panel structure (cell Ã— year), missing cells, and the need to match neighbors within the same year make a direct focal approach fragile. The correct analogy is a **sparse adjacency join**, best handled via `data.table` joins.

---

## 2. Optimization Strategy

### Replace both functions with a vectorized `data.table` join approach:

1. **Expand the neighbor list into an edge table** (a two-column data.table: `id`, `neighbor_id`) â€” done once, ~1.37M rows.
2. **Join the edge table to the panel data by `(neighbor_id, year)`** â€” this is a single keyed `data.table` merge producing ~1.37M Ã— 28 â‰ˆ 38.5M rows (the "long neighbor-values" table).
3. **Group by `(id, year)` and compute `max`, `min`, `mean`** for each variable in one pass â€” a single `data.table` aggregation.

This eliminates all per-row R function calls, all `paste` key construction, and all named-vector lookups. Expected runtime: **2â€“5 minutes** on a 16 GB laptop.

### Preserving the estimand

The computation is numerically identical: for each `(cell, year)`, we gather the same rook neighbors' values (excluding `NA`), and compute the same `max`, `min`, `mean`. The trained Random Forest model is never retouched â€” we simply produce the same predictor columns it expects.

---

## 3. Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 0: Convert panel data to data.table (if not already)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build a flat edge table from the nb object (one-time, fast)
#
#   rook_neighbors_unique : an nb object (list of integer vectors)
#   id_order              : vector mapping list position -> cell id
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb[[i]] contains the indices (into id_order) of cell i's
  # rook neighbors. An entry of 0L means no neighbors (spdep convention).
  edges <- rbindlist(lapply(seq_along(neighbors_nb), function(i) {
    nb_idx <- neighbors_nb[[i]]
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  edges
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has columns: id, neighbor_id
# ~1.37 M rows (directed pairs)

cat("Edge table rows:", nrow(edge_dt), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: For each variable, join + aggregate in one vectorized pass
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare a minimal lookup table: (id, year, var1, var2, â€¦)
# We key it on (id, year) so the join is O(n log n) or hash-based.
lookup_cols <- c("id", "year", neighbor_source_vars)
neighbor_vals_dt <- cell_data[, ..lookup_cols]
setnames(neighbor_vals_dt, "id", "neighbor_id")   # rename for join
setkeyv(neighbor_vals_dt, c("neighbor_id", "year"))

# Cross edge_dt with all years present in cell_data
# But it's more efficient to join edge_dt to the data directly:
#   edge_dt[neighbor_vals_dt] gives us, for every (neighbor_id, year),
#   the focal cell id that has that neighbor.
# However, we want: for each (id, year), get neighbor values.
# Strategy: join cell_data's year onto edge_dt, then join neighbor values.

# 2a. Get the set of (id, year) pairs from cell_data
id_year <- cell_data[, .(id, year)]

# 2b. Expand: for each (id, year), attach all neighbor_ids
#     Result: ~1.37M * 28 â‰ˆ 38.5M rows (but many cells share years,
#     so we do a keyed join which is fast)
setkeyv(edge_dt, "id")
setkeyv(id_year, "id")

# This join replicates each id's neighbors across all years for that id
expanded <- edge_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
# expanded has columns: id, neighbor_id, year
# ~38.5M rows

cat("Expanded neighbor-year rows:", nrow(expanded), "\n")

# 2c. Attach neighbor variable values by joining on (neighbor_id, year)
setkeyv(expanded, c("neighbor_id", "year"))
expanded <- neighbor_vals_dt[expanded, on = c("neighbor_id", "year"), nomatch = NA]
# Now expanded has: neighbor_id, year, id, ntl, ec, pop_density, def, usd_est_n2
# where the variable columns are the NEIGHBOR's values

# 2d. Aggregate: group by (id, year), compute max/min/mean per variable
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  vsym <- as.name(v)
  list(
    bquote(as.numeric(max(.(vsym), na.rm = TRUE))),
    bquote(as.numeric(min(.(vsym), na.rm = TRUE))),
    bquote(mean(.(vsym), na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0(v, c("_max_neighbor", "_min_neighbor", "_mean_neighbor"))
}))

# Build the aggregation call dynamically
agg_call <- as.call(c(as.name("list"),
  setNames(agg_exprs, agg_names)
))

neighbor_stats <- expanded[, eval(agg_call), by = .(id, year)]

# Replace Inf/-Inf (from max/min of all-NA groups) with NA
for (col_name in agg_names) {
  vals <- neighbor_stats[[col_name]]
  set(neighbor_stats, i = which(is.infinite(vals) | is.nan(vals)),
      j = col_name, value = NA_real_)
}

cat("Aggregated stats rows:", nrow(neighbor_stats), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Merge neighbor stats back onto cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Remove old neighbor columns if they exist (idempotency)
old_cols <- intersect(agg_names, names(cell_data))
if (length(old_cols) > 0) cell_data[, (old_cols) := NULL]

setkeyv(cell_data, c("id", "year"))
setkeyv(neighbor_stats, c("id", "year"))
cell_data <- neighbor_stats[cell_data, on = c("id", "year")]

cat("Done. cell_data now has", ncol(cell_data), "columns and",
    nrow(cell_data), "rows.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 4: Predict with the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The model object (e.g., `rf_model`) expects the same column names
# that were just created (*_max_neighbor, *_min_neighbor, *_mean_neighbor).
# Prediction proceeds exactly as before:
#
#   cell_data$predicted <- predict(rf_model, newdata = cell_data)
#
# The Random Forest is NOT retrained.
```

---

## 4. Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `paste` + named-vector lookups | One `data.table` keyed join (seconds) |
| `compute_neighbor_stats` | 5 Ã— 6.46M R-level `lapply` calls | One grouped aggregation (seconds) |
| Memory pattern | 6.46M-element R list of integer vectors (~2â€“4 GB) | Flat `data.table` columns (~1.5 GB peak for expanded) |
| Estimated wall time | **86+ hours** | **2â€“5 minutes** |
| Numerical result | max/min/mean of non-NA rook-neighbor values | **Identical** |
| RF model | Unchanged | **Unchanged** |

The key insight: the original code performs millions of individual R function calls with character key construction; replacing this with a single vectorized **equi-join + grouped aggregation** in `data.table` reduces the work to a handful of highly optimized C-level passes over contiguous memory.
END_RESPONSE>>>
