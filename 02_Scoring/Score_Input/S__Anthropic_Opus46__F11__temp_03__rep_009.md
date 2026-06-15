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
run_id: Anthropic_F11_T03_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** stage â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries significant per-call overhead when done 6.46 million times. With ~1.37 million directed neighbor relationships spread across 344,208 cells Ã— 28 years, this function alone generates tens of millions of string operations.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows with `lapply`, subsetting vectors, removing NAs, and computing `max/min/mean`. That's ~32.3 million R-level function invocations just for the stats, each with allocation overhead from anonymous function closures and `c()` calls.

3. The final `do.call(rbind, result)` on a list of 6.46 million 3-element vectors is itself a well-known R performance anti-pattern.

4. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46 million rows and 110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in C/C++ and typically completes in seconds to a few minutes. Loading a serialized model from disk (`readRDS`) is also fast. Writing predictions is a single vectorized write. None of these operations involve millions of R-level loop iterations.

**Conclusion:** The bottleneck is the O(n) R-level loop over 6.46 million rows done 6 times (1 build + 5 stats), totaling ~38.8 million interpreted R iterations with heavy string and list operations. This is what produces the 86+ hour runtime.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace the row-by-row `lapply` with a fully vectorized approach using `data.table` joins. Pre-expand the neighbor relationships into an edge table (cellâ†’neighbor, by year) and join to get row indices, eliminating all per-row string operations.

2. **Vectorize `compute_neighbor_stats()`**: Use `data.table` grouped aggregation (`max`, `min`, `mean`) over the edge table instead of `lapply` over millions of rows.

3. **Preserve the trained Random Forest model**: No changes to the model or the predict step.

4. **Preserve the original numerical estimand**: The computed neighbor max, min, and mean values are numerically identical.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Assume these objects already exist in the environment:
#       cell_data              â€” data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, â€¦
#       id_order               â€” integer/character vector of cell IDs (index = position in nb object)
#       rook_neighbors_unique  â€” spdep::nb list (length = length(id_order))
#       rf_model               â€” the pre-trained Random Forest model
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# 1.  Build a vectorized edge table of directed neighbor pairs
#     (focal_id -> neighbor_id) from the nb object â€” done ONCE.
# ---------------------------------------------------------------

build_edge_table <- function(id_order, nb_obj) {
  # For each cell position, get its neighbor positions
  from_pos <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_pos   <- unlist(nb_obj)
  
  # Remove 0-entries that spdep uses for cells with no neighbors
  valid    <- to_pos != 0L
  from_pos <- from_pos[valid]
  to_pos   <- to_pos[valid]
  
  data.table(
    focal_id    = id_order[from_pos],
    neighbor_id = id_order[to_pos]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edge_dt), "\n")

# ---------------------------------------------------------------
# 2.  Convert cell_data to data.table and add a row-index column
# ---------------------------------------------------------------

dt <- as.data.table(cell_data)
dt[, row_idx := .I]

# ---------------------------------------------------------------
# 3.  Build the full focalâ€“neighborâ€“year edge table by joining
#     on year.  This replaces build_neighbor_lookup entirely.
#
#     For every (focal_id, year) row, we find all neighbor rows
#     that share the same (neighbor_id, year).
# ---------------------------------------------------------------

# Keyed subsets for fast joins
focal_key   <- dt[, .(focal_row = row_idx, focal_id = id, year)]
neighbor_key <- dt[, .(neighbor_row = row_idx, neighbor_id = id, year)]

setkey(edge_dt, focal_id, neighbor_id)

# Expand edges across years:
#   focal_key  â‹ˆ  edge_dt  on focal_id  â†’  (focal_row, year, neighbor_id)
#   then       â‹ˆ  neighbor_key on (neighbor_id, year) â†’ (focal_row, neighbor_row)

# Step A: attach year to each edge via focal cell
setkey(focal_key, focal_id)
setkey(edge_dt, focal_id)

edges_with_year <- edge_dt[focal_key,
  .(focal_row, neighbor_id, year),
  on = "focal_id",
  allow.cartesian = TRUE,
  nomatch = NULL
]

cat("Edges Ã— years rows:", nrow(edges_with_year), "\n")

# Step B: resolve neighbor_id + year â†’ neighbor_row
setkey(edges_with_year, neighbor_id, year)
setkey(neighbor_key, neighbor_id, year)

edges_resolved <- neighbor_key[edges_with_year,
  .(focal_row, neighbor_row),
  on = c("neighbor_id", "year"),
  nomatch = NULL
]

cat("Resolved edge-year rows:", nrow(edges_resolved), "\n")

# Clean up intermediates
rm(focal_key, neighbor_key, edges_with_year)
gc()

# ---------------------------------------------------------------
# 4.  Compute neighbor stats for all 5 variables in one pass
#     using data.table grouped aggregation.  This replaces
#     compute_neighbor_stats + the outer for-loop.
# ---------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Attach neighbor values to the edge table (only the columns we need)
edges_resolved[, (neighbor_source_vars) :=
  dt[neighbor_row, ..neighbor_source_vars]
]

# Group by focal_row and compute max / min / mean for each variable
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

# Build the aggregation call
stats_dt <- edges_resolved[,
  setNames(lapply(agg_exprs, eval, envir = .SD), agg_names),
  by = focal_row
]

# Handle Inf/-Inf from max/min on all-NA groups â†’ convert to NA
for (col in agg_names) {
  set(stats_dt, which(is.infinite(stats_dt[[col]])), col, NA_real_)
}

# ---------------------------------------------------------------
# 5.  Join the aggregated neighbor features back to the main table
# ---------------------------------------------------------------

# Initialise new columns to NA (covers rows with no neighbors)
for (col in agg_names) {
  set(dt, j = col, value = NA_real_)
}

# Fast update join
dt[stats_dt, (agg_names) := mget(paste0("i.", agg_names)), on = "focal_row"]

# Drop helper column
dt[, row_idx := NULL]

# ---------------------------------------------------------------
# 6.  Random Forest prediction (UNCHANGED â€” not the bottleneck)
# ---------------------------------------------------------------

# Convert back if predict method expects a data.frame
cell_data <- as.data.frame(dt)

predictions <- predict(rf_model, newdata = cell_data)

cell_data$predicted <- predictions

# Write output
# e.g. fwrite(dt, "predictions_output.csv")

cat("Done. Predictions added.\n")
```

---

## Why This Is Fast

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M `lapply` iterations with `paste`/string matching | Single vectorized `data.table` join (~seconds) |
| Neighbor stats | 5 Ã— 6.46M `lapply` iterations with per-row `max/min/mean` | One grouped `data.table` aggregation over the edge table |
| String operations | ~6.46M `paste()` + named-vector lookups per call | Zero â€” integer joins only |
| Memory pattern | 6.46M-element list of small vectors â†’ `do.call(rbind, ...)` | Columnar `data.table` â€” cache-friendly, no list overhead |
| Expected runtime | 86+ hours | **Minutes** (edge expansion + grouped aggregation on ~38M rows is routine for `data.table`) |

The Random Forest `predict()` call remains untouched and was never the bottleneck. The numerical estimand (neighbor max, min, mean per variable per cell-year) is preserved exactly.
END_RESPONSE>>>
