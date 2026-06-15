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
run_id: Anthropic_F11_T10_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence clearly points to a different bottleneck: **the spatial neighbor feature engineering pipeline**, specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a `character` coercion + hash lookup (`id_to_ref`), a subsetting of the neighbor list, a `paste()` to build keys, and a named-vector lookup (`idx_lookup[neighbor_keys]`). That's ~6.46 million iterations of character-based hash lookups, string concatenation, and named vector indexing â€” all in interpreted R. With ~1.37M neighbor relationships spread across those rows, each row touches on average ~4 neighbors, meaning tens of millions of `paste()` and name-matching operations.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over ~6.46 million elements via `lapply`, subsetting a numeric vector, removing `NA`s, and computing `max/min/mean`. That's 5 Ã— 6.46M â‰ˆ **32.3 million** interpreted R function calls, each involving allocation and subsetting.

3. The final `do.call(rbind, result)` inside `compute_neighbor_stats` binds ~6.46 million 3-element vectors into a matrix â€” another expensive operation done 5 times.

4. **Random Forest `predict()`** by contrast is a single vectorized C/C++ call on a pre-trained model. Even with 6.46M rows Ã— 110 predictors, a single `predict()` call on a `ranger` or `randomForest` object typically completes in seconds to a few minutes. Loading/writing is I/O-bound and also comparatively trivial.

**Conclusion:** The bottleneck is the row-level, interpreted-R, string-based spatial neighbor feature construction â€” not the Random Forest inference. The estimated 86+ hours runtime is dominated by tens of millions of `lapply` iterations with `paste()`, named-vector lookups, and per-element subsetting.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a fully vectorized `data.table`-based join approach. Instead of iterating over every row, explode the neighbor list into an edge table (cell_id â†’ neighbor_id), join it with the panel data on (neighbor_id, year) to get neighbor row indices, and group.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation per variable â€” no `lapply`, no per-row R function calls.

3. **Avoid all `paste()`-based key construction** â€” use integer-keyed joins with multi-column keys `(id, year)`.

4. These operations become vectorized C-level `data.table` merges and grouped aggregations, reducing the workload from hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert cell_data to data.table; add a row index
# ============================================================
setDT(cell_data)
cell_data[, .row_idx := .I]

# ============================================================
# STEP 1: Build an edge table from the nb object (vectorized)
#         This replaces build_neighbor_lookup entirely.
# ============================================================
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj is a list of integer vectors (indices into id_order)
  # id_order maps positional index -> cell id
  lens <- lengths(nb_obj)
  from_pos <- rep(seq_along(nb_obj), lens)
  to_pos   <- unlist(nb_obj, use.names = FALSE)

  # Remove 0-entries that spdep uses for "no neighbors"
  valid <- to_pos > 0L
  from_pos <- from_pos[valid]
  to_pos   <- to_pos[valid]

  data.table(
    focal_id    = id_order[from_pos],
    neighbor_id = id_order[to_pos]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has columns: focal_id, neighbor_id
# ~ 1,373,394 rows (directed rook edges)

# ============================================================
# STEP 2: Join edges with panel data to create the full
#         (focal_row, neighbor_row, year) mapping
# ============================================================

# We need: for each (focal_id, year), all neighbor rows that share
# the same year and whose id is a rook neighbor.

# Subset of cell_data for joining: just id, year, row_idx, and the
# neighbor source variable columns.
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

join_cols <- c("id", "year", ".row_idx", neighbor_source_vars)
cd_slim <- cell_data[, ..join_cols]

# Merge edges with focal-side data to get years for each focal cell
# Then merge with neighbor-side data on (neighbor_id, year)

# Focal side: attach year to every edge
setkey(cd_slim, id)
focal_years <- cd_slim[, .(id, year, .row_idx)]
setnames(focal_years, c("id", "year", ".row_idx"),
                      c("focal_id", "year", "focal_row"))
setkey(focal_years, focal_id)
setkey(edge_dt, focal_id)

# This join replicates each edge across all years the focal cell appears in
edge_year <- edge_dt[focal_years, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
# Columns: focal_id, neighbor_id, year, focal_row

# Neighbor side: attach variable values
neighbor_data <- cd_slim[, c("id", "year", neighbor_source_vars), with = FALSE]
setnames(neighbor_data, "id", "neighbor_id")
setkey(neighbor_data, neighbor_id, year)
setkey(edge_year, neighbor_id, year)

# Main join: each focal-row gets its neighbor's variable values in the same year
joined <- neighbor_data[edge_year, on = c("neighbor_id", "year"), nomatch = NA]
# Columns: neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2,
#           focal_id, focal_row

# ============================================================
# STEP 3: Grouped aggregation â€” replaces compute_neighbor_stats
# ============================================================
# For each focal_row and each variable, compute max, min, mean
# across all neighbor rows (ignoring NAs).

agg_exprs <- list()
for (v in neighbor_source_vars) {
  v_sym <- as.name(v)
  agg_exprs[[paste0("neighbor_max_", v)]]  <- bquote(as.numeric(max(.(v_sym), na.rm = TRUE)))
  agg_exprs[[paste0("neighbor_min_", v)]]  <- bquote(as.numeric(min(.(v_sym), na.rm = TRUE)))
  agg_exprs[[paste0("neighbor_mean_", v)]] <- bquote(mean(.(v_sym), na.rm = TRUE))
}

# Handle the edge case where all neighbors are NA â†’ returns Inf/-Inf/NaN;
# we will fix those after aggregation.

agg_result <- joined[, eval(as.call(c(as.name("list"), agg_exprs))),
                      by = focal_row]

# Replace Inf/-Inf with NA (from max/min on all-NA groups)
inf_cols <- grep("^neighbor_(max|min)_", names(agg_result), value = TRUE)
for (col in inf_cols) {
  set(agg_result, which(is.infinite(agg_result[[col]])), col, NA_real_)
}
# Replace NaN with NA (from mean on all-NA groups)
mean_cols <- grep("^neighbor_mean_", names(agg_result), value = TRUE)
for (col in mean_cols) {
  set(agg_result, which(is.nan(agg_result[[col]])), col, NA_real_)
}

# ============================================================
# STEP 4: Merge aggregated neighbor features back into cell_data
# ============================================================
setkey(agg_result, focal_row)
feature_cols <- setdiff(names(agg_result), "focal_row")

# Pre-allocate columns in cell_data
for (col in feature_cols) {
  set(cell_data, j = col, value = NA_real_)
}

# Assign by row index
matched_rows <- agg_result[["focal_row"]]
for (col in feature_cols) {
  set(cell_data, i = matched_rows, j = col, value = agg_result[[col]])
}

# Clean up helper column
cell_data[, .row_idx := NULL]

# ============================================================
# STEP 5: Random Forest prediction (unchanged â€” not the bottleneck)
# ============================================================
# library(ranger)  # or randomForest â€” whichever was used
# trained_model <- readRDS("path/to/trained_rf_model.rds")
# predictions <- predict(trained_model, data = cell_data)$predictions
# cell_data[, prediction := predictions]
# fwrite(cell_data, "output.csv")  # or saveRDS(...)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste()` + named vector lookups | One vectorized edge-table construction + `data.table` keyed join |
| `compute_neighbor_stats` | 5 Ã— 6.46M `lapply` iterations (32.3M R-level calls) | One grouped `data.table` aggregation (C-level) |
| `do.call(rbind, ...)` | 5 Ã— binding 6.46M small vectors | Eliminated entirely |
| Estimated runtime | 86+ hours | **Minutes** (typically 5â€“15 min on a 16 GB laptop) |
| RF model | Preserved (untouched) | Preserved (untouched) |
| Numerical estimand | Identical max/min/mean per (focal cell, year) | Identical â€” same arithmetic, same NA handling |

The Random Forest inference step (load â†’ predict â†’ write) for ~6.46M rows is a **negligible fraction** of the total runtime and requires no optimization. The real bottleneck is the interpreted-R, row-by-row, string-based neighbor feature engineering, which the `data.table` rewrite eliminates.
END_RESPONSE>>>
