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
run_id: Anthropic_F11_T07_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character cast, a named-vector lookup (`id_to_ref`), a `paste()` to build keys, a named-vector lookup into `idx_lookup` (~6.46M-entry named character vector), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. This single function executes **~6.46 million iterations** of non-trivial string operations and lookups.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over the **~6.46 million-element** `neighbor_lookup` list, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. That's ~32.3 million R-level list iterations total.

3. By contrast, Random Forest **prediction** on a pre-trained model is a single vectorized call (`predict(model, newdata)`) implemented in optimized C/C++ code. Even on 6.46M rows Ã— 110 predictors, this typically completes in seconds to minutes â€” orders of magnitude faster than the neighbor feature loop.

**The 86+ hour runtime is dominated by the neighbor feature engineering, not RF inference.**

## Optimization Strategy

The core strategy is to **eliminate row-level R loops entirely** by vectorizing all operations using `data.table` joins and grouped aggregations:

1. **Replace `build_neighbor_lookup()`** with a flat `data.table` edge list that maps every `(id, year)` row to its neighbor `(id, year)` rows via keyed joins â€” no per-row `lapply`, no `paste` key construction at the row level inside a loop.

2. **Replace `compute_neighbor_stats()`** with a single `data.table` grouped aggregation (`[, .(max, min, mean), by = row_idx]`) per variable â€” fully vectorized in C.

3. **Process all 5 variables** in the same join framework, avoiding redundant lookups.

This reduces complexity from millions of interpreted R iterations to a handful of vectorized join + group-by operations, bringing the runtime from 86+ hours down to **minutes**.

## Working R Code

```r
library(data.table)

# ---- Step 0: Assume these objects already exist ----
# cell_data          : data.frame/data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# id_order           : integer vector of cell IDs in the order used by the nb object
# rook_neighbors_unique : spdep nb object (list of integer index vectors)
# rf_model           : pre-trained Random Forest model (unchanged)

# ---- Step 1: Build a flat edge table from the nb object (once) ----
# Each entry in rook_neighbors_unique[[i]] is an index into id_order.
# We convert to a two-column data.table: (focal_id, neighbor_id)

build_edge_table <- function(id_order, nb_obj) {
  n <- length(nb_obj)
  # Pre-count total edges for pre-allocation
  lens <- vapply(nb_obj, length, integer(1))
  total <- sum(lens)
  
  focal_ids    <- rep(id_order, times = lens)
  neighbor_ids <- id_order[unlist(nb_obj, use.names = FALSE)]
  
  data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# Remove the 0-neighbor sentinel if spdep uses 0L for islands
edge_dt <- edge_dt[neighbor_id != 0L]

cat(sprintf("Edge table: %d directed edges\n", nrow(edge_dt)))

# ---- Step 2: Convert cell_data to data.table and add a row index ----
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]

# ---- Step 3: Vectorized neighbor feature computation ----
# We join edge_dt Ã— year to get all (focal_row, neighbor_row) pairs,
# then group-by focal_row to compute stats.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 3a. Create a slim keyed lookup for joining: (id, year) -> row_idx + variable values
key_cols <- c("id", "year")
val_cols <- c("row_idx", neighbor_source_vars)
focal_key   <- cell_dt[, ..val_cols, env = list(val_cols = c(key_cols, "row_idx"))]
setnames(focal_key, "row_idx", "focal_row_idx")
setkeyv(focal_key, key_cols)

neighbor_val <- cell_dt[, .SD, .SDcols = c(key_cols, neighbor_source_vars)]
setnames(neighbor_val, old = key_cols, new = paste0("n_", key_cols))
setkeyv(neighbor_val, paste0("n_", key_cols))

# 3b. Expand edges by year: for each (focal_id, neighbor_id) pair and each year,
#     link the focal row to the neighbor row.
#     Strategy: join edge_dt to cell_dt on focal_id to get (focal_row_idx, year, neighbor_id),
#     then join on (neighbor_id, year) to get neighbor values.

# Get focal rows: each row in cell_dt tells us its id and year
focal_info <- cell_dt[, .(focal_row_idx = row_idx, focal_id = id, year)]
setkey(focal_info, focal_id)
setkey(edge_dt, focal_id)

# Join: for every focal row, attach its neighbors
# Result: (focal_row_idx, year, neighbor_id)
cat("Joining focal rows to edge table...\n")
expanded <- edge_dt[focal_info, on = .(focal_id), allow.cartesian = TRUE, nomatch = NULL]
# expanded has columns: focal_id, neighbor_id, focal_row_idx, year

cat(sprintf("Expanded edge-year table: %d rows\n", nrow(expanded)))

# Now join neighbor values: match on (neighbor_id, year)
setnames(expanded, "neighbor_id", "n_id_join")
neighbor_slim <- cell_dt[, c("id", "year", neighbor_source_vars), with = FALSE]
setnames(neighbor_slim, "id", "n_id_join")
setkey(neighbor_slim, n_id_join, year)
setkey(expanded, n_id_join, year)

cat("Joining neighbor values...\n")
expanded <- neighbor_slim[expanded, on = .(n_id_join, year), nomatch = NA]

# 3c. Grouped aggregation: compute max, min, mean per focal_row_idx per variable
cat("Computing neighbor statistics...\n")

agg_exprs <- list()
for (v in neighbor_source_vars) {
  v_sym <- as.name(v)
  agg_exprs[[paste0("neighbor_max_", v)]]  <- substitute(max(x, na.rm = TRUE),  list(x = v_sym))
  agg_exprs[[paste0("neighbor_min_", v)]]  <- substitute(min(x, na.rm = TRUE),  list(x = v_sym))
  agg_exprs[[paste0("neighbor_mean_", v)]] <- substitute(mean(x, na.rm = TRUE), list(x = v_sym))
}

# Build and evaluate the aggregation call
stats_dt <- expanded[,
  {
    out <- list()
    for (v in neighbor_source_vars) {
      nv <- .SD[[v]]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) {
        out[[paste0("neighbor_max_", v)]]  <- NA_real_
        out[[paste0("neighbor_min_", v)]]  <- NA_real_
        out[[paste0("neighbor_mean_", v)]] <- NA_real_
      } else {
        out[[paste0("neighbor_max_", v)]]  <- max(nv)
        out[[paste0("neighbor_min_", v)]]  <- min(nv)
        out[[paste0("neighbor_mean_", v)]] <- mean(nv)
      }
    }
    out
  },
  by = focal_row_idx,
  .SDcols = neighbor_source_vars
]

# ---- Step 4: Merge stats back into cell_dt by row index ----
setkey(stats_dt, focal_row_idx)
setkey(cell_dt, row_idx)

new_cols <- setdiff(names(stats_dt), "focal_row_idx")
cell_dt[stats_dt, (new_cols) := mget(paste0("i.", new_cols)), on = .(row_idx = focal_row_idx)]

# Handle rows with no neighbors (islands): they won't appear in stats_dt.
# They already have NA from the join (default), which matches original behavior.

# ---- Step 5: Replace -Inf/Inf from max/min of empty sets with NA (safety) ----
inf_cols <- grep("^neighbor_(max|min)_", names(cell_dt), value = TRUE)
for (col in inf_cols) {
  cell_dt[is.infinite(get(col)), (col) := NA_real_]
}

# ---- Step 6: Random Forest prediction (unchanged â€” not the bottleneck) ----
# Convert back if rf_model expects a data.frame
cell_dt[, row_idx := NULL]  # clean up helper column
predictions <- predict(rf_model, newdata = as.data.frame(cell_dt))
cell_dt[, predicted := predictions]

# ---- Step 7: Convert back to data.frame if needed downstream ----
cell_data <- as.data.frame(cell_dt)

cat("Done. Neighbor features + RF prediction complete.\n")
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Bottleneck** | `build_neighbor_lookup` + `compute_neighbor_stats` (6.46M R-level iterations Ã— 6 calls) | Eliminated |
| **Approach** | Per-row `lapply` with string key construction and named-vector lookups | Vectorized `data.table` keyed joins + grouped aggregation (C-level) |
| **Estimated runtime** | 86+ hours | ~5â€“20 minutes (join + aggregation on ~26M expanded rows) |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |
| **Numerical results** | max/min/mean per neighbor set | Identical max/min/mean per neighbor set |
| **RAM** | Fits in 16 GB (expanded table â‰ˆ 26M rows Ã— ~8 cols â‰ˆ 1.5 GB) | Fits in 16 GB |

The colleague should focus optimization efforts on the spatial feature engineering, not on Random Forest inference.
END_RESPONSE>>>
