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
run_id: Anthropic_F11_T10_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup` and `compute_neighbor_stats` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup`** iterates over **~6.46 million rows** one at a time via `lapply`, performing per-row string pasting (`paste()`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and `NA` filtering. Named-vector lookup in R is effectively a hash-table probe, but doing it millions of times inside an R-level loop with repeated string allocation is extremely slow. For each of the 6.46M rows, it looks up ~4 neighbors (rook), creating and indexing character keys each time. That's **~25+ million string operations** inside a single-threaded R loop.

2. **`compute_neighbor_stats`** is then called **5 times** (once per neighbor source variable), each time iterating over 6.46M rows via `lapply`, subsetting a numeric vector, removing `NA`s, and computing `max/min/mean`. That's **5 Ã— 6.46M = ~32.3 million R-level function calls** with per-element allocation.

3. By contrast, Random Forest **prediction** on a pre-trained model is a single call to `predict()` over a matrix/data.frame â€” highly optimized in C/C++ internally (whether `randomForest`, `ranger`, or similar). Even on 6.46M rows Ã— 110 predictors, this typically completes in seconds to minutes, not hours.

**The 86+ hour runtime is dominated by the O(n) R-level loops with millions of string allocations and named-vector lookups in the neighbor engineering step.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup` with a vectorized `data.table` equi-join.** Instead of looping row-by-row with string keys, expand the neighbor relationships into an edge table `(id, neighbor_id)`, merge with the data on `(neighbor_id, year)` via `data.table` keyed joins, and compute grouped statistics with `data.table` aggregation â€” all in C-level vectorized operations.

2. **Replace `compute_neighbor_stats` (called 5 times) with a single grouped `data.table` aggregation** that computes max/min/mean for all 5 variables at once per `(id, year)` group.

3. **Eliminate all per-row `lapply`, `paste`, and named-vector lookups entirely.**

Expected speedup: from 86+ hours to **minutes** (typically 2â€“10 minutes depending on hardware).

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# INPUTS (assumed already in the environment):
#   cell_data              â€” data.frame/data.table, ~6.46M rows
#                            with columns: id, year, ntl, ec,
#                            pop_density, def, usd_est_n2, ...
#   id_order               â€” integer vector of grid-cell IDs
#                            (length 344,208) matching the nb object
#   rook_neighbors_unique  â€” nb object (list of length 344,208),
#                            each element is an integer vector of
#                            neighbor indices into id_order
#   rf_model               â€” pre-trained Random Forest model
# ---------------------------------------------------------------

# ========================
# STEP 1: Build edge table from the nb object (vectorized)
# ========================
# Convert spdep nb list â†’ two-column data.table of (id, neighbor_id)

edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  # Exclude zero-length (no-neighbor) entries

  if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
    return(NULL)
  }
  data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
}))
# edge_list has ~1,373,394 rows: one row per directed neighbor pair

# ========================
# STEP 2: Vectorized neighbor feature computation via data.table join
# ========================

# Ensure cell_data is a data.table
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset the columns we need for the neighbor join
# We need (id, year) + the 5 source variables from the neighbor rows
neighbor_vals_dt <- cell_data[, c("id", "year", neighbor_source_vars), with = FALSE]

# Key for fast join
setkey(neighbor_vals_dt, id, year)

# Merge: for every (id, year) row, find its neighbors' variable values
# Join edge_list with neighbor_vals_dt on neighbor_id == id, carrying year from
# the focal cell.
#
# Strategy:
#   1. Join cell_data with edge_list on id â†’ gives (id, year, neighbor_id) for every cell-year
#   2. Join result with neighbor_vals_dt on (neighbor_id, year) â†’ gives neighbor variable values
#   3. Aggregate by (id, year) â†’ max, min, mean per variable

# Step 2a: Expand cell-year rows to cell-year-neighbor rows
# We only need (id, year) from the focal cell plus neighbor_id
focal_keys <- cell_data[, .(id, year)]
setkey(edge_list, id)
setkey(focal_keys, id)

# This join replicates each (id, year) row for every neighbor of that id
# Result: ~6.46M * ~4 neighbors â‰ˆ ~26M rows (manageable in 16 GB)
expanded <- edge_list[focal_keys, on = "id", allow.cartesian = TRUE, nomatch = 0L]
# expanded columns: id, neighbor_id, year

# Step 2b: Pull in neighbor variable values
setnames(neighbor_vals_dt, "id", "neighbor_id")
setkey(neighbor_vals_dt, neighbor_id, year)
setkey(expanded, neighbor_id, year)

expanded_with_vals <- neighbor_vals_dt[expanded, on = c("neighbor_id", "year"), nomatch = NA]
# expanded_with_vals columns: neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, id

# Step 2c: Aggregate by (id, year) â€” compute max, min, mean for each variable
# Build the aggregation expression dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

# Construct a single j-expression for data.table
agg_call <- as.call(c(
  as.name("list"),
  setNames(agg_exprs, agg_names)
))

neighbor_stats <- expanded_with_vals[, eval(agg_call), by = .(id, year)]

# Replace any -Inf/Inf from max/min on all-NA groups with NA
for (col_name in agg_names) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col_name]])), col_name, NA_real_)
}

# ========================
# STEP 3: Merge neighbor features back to cell_data
# ========================
setkey(neighbor_stats, id, year)
setkey(cell_data, id, year)

# Remove any pre-existing neighbor columns to avoid duplication
existing_neighbor_cols <- intersect(names(cell_data), agg_names)
if (length(existing_neighbor_cols) > 0L) {
  cell_data[, (existing_neighbor_cols) := NULL]
}

cell_data <- neighbor_stats[cell_data, on = c("id", "year")]

# ========================
# STEP 4: Random Forest prediction (unchanged â€” already fast)
# ========================
# The RF model is pre-trained; we only call predict().
# This is a single vectorized C-level call and is NOT the bottleneck.

# Ensure predictor columns match what the model expects
# (This depends on your specific model object; adjust column selection as needed)
pred_features <- cell_data[, .SD, .SDcols = setdiff(names(cell_data),
                           c("id", "year", "target_variable"))]

cell_data[, prediction := predict(rf_model, newdata = pred_features)]
```

---

## Summary

| Component | Time Complexity (Original) | Time Complexity (Optimized) |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level iterations with string ops | **Eliminated** (vectorized join) |
| `compute_neighbor_stats` (Ã—5) | ~32.3M R-level iterations | **Single** `data.table` grouped aggregation |
| RF `predict()` | Single vectorized C call | **Unchanged** (not the bottleneck) |

**Verdict:** The bottleneck is the R-level row-by-row neighbor feature engineering, not Random Forest inference. The optimized code replaces all per-row loops and string-key lookups with `data.table` keyed joins and grouped aggregation, reducing the estimated runtime from **86+ hours to minutes** while preserving the trained RF model and the original numerical estimand (identical max/min/mean neighbor statistics â†’ identical predictions).
END_RESPONSE>>>
