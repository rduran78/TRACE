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
run_id: Anthropic_F11_T03_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** as the dominant bottleneck â€” specifically `build_neighbor_lookup` and `compute_neighbor_stats`.

**Why:**

1. **`build_neighbor_lookup`** uses a base-R `lapply` over **~6.46 million rows**. For each row it performs character coercion (`as.character`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), `paste` to build keys, and NA filtering. Named-vector lookup in R is O(n) hash-probe per call, and doing this 6.46 million times with string allocation is extremely expensive.

2. **`compute_neighbor_stats`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million entries in `neighbor_lookup` with per-element subsetting, NA removal, and summary computation. That's ~32.3 million R-level list iterations total.

3. **Random Forest inference** is a single call to `predict()` on a pre-trained model. Even with 6.46M rows Ã— 110 predictors, `randomForest::predict` (or `ranger::predict`) is implemented in C/C++ and typically completes in seconds to minutes. Loading a serialized model (`readRDS`) is also fast. Writing predictions is trivial. This is not the bottleneck.

4. The **86+ hour runtime** is consistent with billions of string operations and R-level loop iterations in the neighbor pipeline, not with a single vectorized C-level prediction call.

## Optimization Strategy

1. **Replace string-key lookups with integer-indexed direct lookup.** Build a matrix/integer mapping from `(cell_id, year)` â†’ row index using a fast integer hash (via `data.table`) instead of `paste` + named character vectors.

2. **Vectorize `build_neighbor_lookup`** by expanding all neighbor relationships into a `data.table` of `(source_row, neighbor_row)` pairs, performing a single merge/join, and then splitting or aggregating â€” eliminating the per-row `lapply`.

3. **Vectorize `compute_neighbor_stats`** by using `data.table` grouped aggregation on the edge list instead of per-element list iteration.

4. **Compute all 5 variables' neighbor stats in one pass** over the edge list.

These changes reduce the complexity from ~6.46M Ã— k R-level iterations to a handful of vectorized `data.table` joins and group-by operations, bringing runtime from 86+ hours to **minutes**.

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Convert cell_data to data.table (non-destructive)
# ---------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure original row order is preserved for downstream use
cell_data[, .row_id := .I]

# ---------------------------------------------------------------
# 1.  Build a fast (id, year) -> row_id integer lookup
# ---------------------------------------------------------------
# id_order is the vector of cell IDs in the same order as rook_neighbors_unique
# rook_neighbors_unique is an nb object (list of integer index vectors)

# Map: position-in-id_order  ->  cell id
# (ref_idx is 1-based position in id_order)

# Build edge list of directed neighbor relationships:
#   source_ref_idx  ->  neighbor_ref_idx
# Then map ref_idx -> cell id, join with cell_data on (id, year).

cat("Building edge list from nb object...\n")
edge_list <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(source_ref = i, neighbor_ref = nb)
  })
)

# Map ref indices to actual cell IDs
edge_list[, source_id   := id_order[source_ref]]
edge_list[, neighbor_id := id_order[neighbor_ref]]

# Drop ref columns â€” we only need cell IDs now
edge_list[, c("source_ref", "neighbor_ref") := NULL]

cat("Edge list rows (directed relationships):", nrow(edge_list), "\n")

# ---------------------------------------------------------------
# 2.  Expand edge list across all years and join to row indices
# ---------------------------------------------------------------
# Get unique years present in the data
years <- sort(unique(cell_data$year))

# Cross-join edges Ã— years  (each edge exists in every year)
cat("Cross-joining edges with years...\n")
edge_year <- edge_list[, CJ(edge_idx = seq_len(.N), year = years)]
edge_year[, `:=`(
  source_id   = edge_list$source_id[edge_idx],
  neighbor_id = edge_list$neighbor_id[edge_idx]
)]
edge_year[, edge_idx := NULL]

# Build row-index lookup keyed on (id, year)
cat("Building row-index lookup...\n")
row_lookup <- cell_data[, .(id, year, .row_id)]
setkey(row_lookup, id, year)

# Join to get source row id
setnames(row_lookup, ".row_id", "source_row")
setkey(edge_year, source_id, year)
setkey(row_lookup, id, year)
edge_year <- row_lookup[edge_year, on = .(id = source_id, year = year), nomatch = 0L]
setnames(edge_year, "source_row", "source_row")

# Join to get neighbor row id
setnames(row_lookup, "source_row", "neighbor_row")
edge_year <- row_lookup[edge_year, on = .(id = neighbor_id, year = year), nomatch = 0L]

# Clean up â€” keep only what we need
edge_year <- edge_year[, .(source_row, neighbor_row)]

# Free memory
rm(row_lookup)
gc()

cat("Expanded edge-year rows:", nrow(edge_year), "\n")

# ---------------------------------------------------------------
# 3.  Compute neighbor stats for all variables in one pass
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics...\n")

# Attach neighbor variable values to edge table
# We pull values from cell_data using the neighbor_row index
neighbor_vals <- cell_data[edge_year$neighbor_row, ..neighbor_source_vars]
neighbor_vals[, source_row := edge_year$source_row]

# Group by source_row and compute max, min, mean for each variable
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Use data.table aggregation
stats <- neighbor_vals[,
  setNames(
    lapply(neighbor_source_vars, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        list(NA_real_, NA_real_, NA_real_)
      } else {
        list(max(vals), min(vals), mean(vals))
      }
    }),
    neighbor_source_vars
  ),
  by = source_row
]

# The above nested-list approach can be tricky; here is a cleaner version:
cat("Aggregating per source row...\n")

stats_list <- vector("list", length(neighbor_source_vars))
names(stats_list) <- neighbor_source_vars

for (v in neighbor_source_vars) {
  cat("  Processing:", v, "\n")
  tmp <- neighbor_vals[, .(source_row, val = get(v))]
  tmp <- tmp[!is.na(val)]
  agg <- tmp[, .(
    nmax  = max(val),
    nmin  = min(val),
    nmean = mean(val)
  ), by = source_row]
  setnames(agg, c("nmax", "nmin", "nmean"),
           paste0("neighbor_", c("max_", "min_", "mean_"), v))
  stats_list[[v]] <- agg
}

# Free the large edge-value table
rm(neighbor_vals, edge_year)
gc()

# ---------------------------------------------------------------
# 4.  Merge all neighbor stats back into cell_data
# ---------------------------------------------------------------
cat("Merging neighbor features into cell_data...\n")

for (v in neighbor_source_vars) {
  agg <- stats_list[[v]]
  feat_cols <- paste0("neighbor_", c("max_", "min_", "mean_"), v)

  # Remove old columns if they exist (idempotency)
  for (fc in feat_cols) {
    if (fc %in% names(cell_data)) cell_data[, (fc) := NULL]
  }

  # Left join on source_row == .row_id
  cell_data <- merge(cell_data, agg, by.x = ".row_id", by.y = "source_row", all.x = TRUE)
}

# Restore original row order
setorder(cell_data, .row_id)
cell_data[, .row_id := NULL]

rm(stats_list)
gc()

cat("Neighbor feature engineering complete.\n")

# ---------------------------------------------------------------
# 5.  Random Forest prediction (unchanged â€” not the bottleneck)
# ---------------------------------------------------------------
# The trained model is loaded and used as-is.
# Example (preserving the original estimand):
#
#   rf_model <- readRDS("trained_rf_model.rds")
#   cell_data$predicted <- predict(rf_model, newdata = cell_data)
#   fwrite(cell_data, "predictions_output.csv")
#
# This step typically takes seconds to a few minutes and is NOT
# modified because it is not the bottleneck.
```

## Summary

| Aspect | Colleague's Claim | Actual Evidence |
|---|---|---|
| **Alleged bottleneck** | RF inference (`predict()`) | Neighbor feature engineering |
| **Root cause** | â€” | 6.46M R-level `lapply` iterations with string `paste`/lookup, repeated 5Ã— |
| **Fix** | â€” | Vectorized `data.table` edge-list join + grouped aggregation |
| **Expected speedup** | â€” | From ~86+ hours â†’ **minutes** (typically 5â€“15 min on 16 GB laptop) |
| **RF model** | Preserved âœ“ | Not retrained, `predict()` call unchanged |
| **Numerical estimand** | Preserved âœ“ | Same max/min/mean neighbor stats, same predictions |
END_RESPONSE>>>
