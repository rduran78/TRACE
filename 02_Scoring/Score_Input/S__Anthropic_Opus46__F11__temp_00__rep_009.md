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
run_id: Anthropic_F11_T00_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates roughly **25.8 million string-paste and hash-lookup operations** inside a sequential `lapply`.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's another **~32.3 million** `lapply` iterations total, each allocating small vectors.

3. The combined cost is dominated by **millions of R-level interpreted loop iterations with per-element memory allocation, string operations, and hash lookups**. This is the classic R anti-pattern that produces multi-day runtimes.

4. By contrast, Random Forest **prediction** on a pre-trained model with ~6.46M rows and ~110 predictors is a single vectorized call to `predict()` that typically completes in seconds to minutes, even for large forests. Loading a serialized model with `readRDS()` and writing predictions are also fast I/O operations. There is no evidence in the code that inference is iterated, looped, or otherwise inefficiently implemented.

**Conclusion:** The bottleneck is the row-level `lapply` loops over 6.46M rows in `build_neighbor_lookup()` and `compute_neighbor_stats()`, not Random Forest inference.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a fully vectorized `data.table` merge/join approach. Instead of looping row-by-row to find neighbor indices, we expand the neighbor list into an edge table (`cell_id`, `neighbor_id`), join it with the year dimension to get (`cell_id`, `year`, `neighbor_id`, `year`), and then join against the data to retrieve row indices â€” all using `data.table` keyed joins with zero R-level loops.

2. **Replace `compute_neighbor_stats()`** with a single `data.table` grouped aggregation. Once we have an edge table mapping each row to its neighbor rows, we simply join in the variable values and compute `max`, `min`, `mean` by group â€” a fully vectorized, C-level operation in `data.table`.

3. **Process all 5 variables in one pass** over the edge table rather than 5 separate `lapply` calls over 6.46M rows.

This reduces the estimated runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Assume these objects already exist in the environment:
#       cell_data              â€” data.frame / data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, â€¦
#       id_order               â€” integer/character vector of cell IDs in the order matching rook_neighbors_unique
#       rook_neighbors_unique  â€” spdep nb object (list of integer index vectors)
#       rf_model               â€” the pre-trained Random Forest model (untouched)
# ---------------------------------------------------------------

# Convert to data.table if not already
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)

# Preserve original row order for downstream prediction alignment
cell_data[, .row_id := .I]

# ---------------------------------------------------------------
# 1.  Build a vectorized edge table from the nb object
#     Each entry: (cell_id, neighbor_id)
# ---------------------------------------------------------------
edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  # spdep nb: 0-neighbor cells are encoded as integer(0) or 0L

  nb_idx <- nb_idx[nb_idx > 0L]
  if (length(nb_idx) == 0L) return(NULL)
  data.table(cell_id = id_order[i], neighbor_id = id_order[nb_idx])
}))
# This loop is over 344,208 cells (NOT 6.46M rows) â€” fast.

# ---------------------------------------------------------------
# 2.  Create a keyed lookup:  (id, year) -> row_id
# ---------------------------------------------------------------
setkey(cell_data, id, year)
row_lookup <- cell_data[, .(id, year, .row_id)]
setkey(row_lookup, id, year)

# ---------------------------------------------------------------
# 3.  Expand edges Ã— years to get the full neighbor-row mapping
#     For every (focal_row, neighbor_row) pair sharing the same year
# ---------------------------------------------------------------

# Get unique years
years <- unique(cell_data$year)

# Cross join edges with years
edge_year <- CJ_dt_edges <- edge_list[, .(cell_id, neighbor_id)]
edge_year <- edge_year[, .(year = years), by = .(cell_id, neighbor_id)]

# Map focal (cell_id, year) -> focal_row_id
setnames(row_lookup, c("id", "year", ".row_id"), c("cell_id", "year", "focal_row_id"))
setkey(row_lookup, cell_id, year)
setkey(edge_year, cell_id, year)
edge_year <- row_lookup[edge_year, nomatch = 0L]

# Map neighbor (neighbor_id, year) -> neighbor_row_id
neighbor_lookup_dt <- cell_data[, .(neighbor_id = id, year, neighbor_row_id = .row_id)]
setkey(neighbor_lookup_dt, neighbor_id, year)
setkey(edge_year, neighbor_id, year)
edge_year <- neighbor_lookup_dt[edge_year, nomatch = 0L]

# Restore row_lookup column names
setnames(row_lookup, c("cell_id", "year", "focal_row_id"), c("id", "year", ".row_id"))

# edge_year now has columns: focal_row_id, neighbor_row_id (and cell_id, neighbor_id, year)
# Keep only what we need
edge_year <- edge_year[, .(focal_row_id, neighbor_row_id)]

# ---------------------------------------------------------------
# 4.  Compute neighbor stats for all 5 variables in one pass
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Attach neighbor variable values
neighbor_vals <- cell_data[edge_year$neighbor_row_id, ..neighbor_source_vars]
neighbor_vals[, focal_row_id := edge_year$focal_row_id]

# Grouped aggregation â€” fully vectorized in data.table's C backend
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
}))

# Build the aggregation call
agg_stats <- neighbor_vals[,
  setNames(lapply(neighbor_source_vars, function(v) {
    vals <- get(v)
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) list(NA_real_, NA_real_, NA_real_)
    else list(max(vals), min(vals), mean(vals))
  }), neighbor_source_vars),
  by = focal_row_id
]

# More efficient: compute all 15 stats in one grouped operation
agg_stats <- neighbor_vals[, {
  out <- vector("list", length(neighbor_source_vars) * 3L)
  k <- 1L
  for (v in neighbor_source_vars) {
    vals <- get(v)
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) {
      out[[k]] <- NA_real_; out[[k+1L]] <- NA_real_; out[[k+2L]] <- NA_real_
    } else {
      out[[k]] <- max(vals); out[[k+1L]] <- min(vals); out[[k+2L]] <- mean(vals)
    }
    k <- k + 3L
  }
  names(out) <- agg_names
  out
}, by = focal_row_id]

# ---------------------------------------------------------------
# 5.  Join aggregated neighbor features back to cell_data
# ---------------------------------------------------------------
setkey(agg_stats, focal_row_id)
setkey(cell_data, .row_id)

# Remove old neighbor columns if they exist (idempotency)
old_cols <- intersect(agg_names, names(cell_data))
if (length(old_cols)) cell_data[, (old_cols) := NULL]

cell_data <- agg_stats[cell_data, on = .(focal_row_id = .row_id)]

# Rename focal_row_id back
setnames(cell_data, "focal_row_id", ".row_id")

# Restore original row order
setorder(cell_data, .row_id)

# ---------------------------------------------------------------
# 6.  Random Forest prediction (unchanged â€” NOT the bottleneck)
# ---------------------------------------------------------------
# Ensure predictor columns match the trained model's expectations
predictions <- predict(rf_model, newdata = cell_data)

cell_data[, predicted := predictions]

# Clean up helper column
cell_data[, .row_id := NULL]

# ---------------------------------------------------------------
# Done. The trained RF model is preserved; the numerical estimand
# (predicted values from the same model on the same features) is
# identical to the original pipeline's output.
# ---------------------------------------------------------------
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | `lapply` over 6.46M rows with string ops & hash lookups | Vectorized `data.table` keyed join on 344K edges Ã— 28 years |
| `compute_neighbor_stats` | 5 Ã— `lapply` over 6.46M rows (32.3M iterations) | Single grouped `data.table` aggregation (C-level) |
| RF inference | Single `predict()` call â€” already fast | Unchanged |
| Estimated runtime | 86+ hours | ~5â€“15 minutes on the same laptop |
| Trained RF model | Preserved | Preserved |
| Numerical output | Original estimand | Identical â€” same features, same model, same `predict()` |
END_RESPONSE>>>
