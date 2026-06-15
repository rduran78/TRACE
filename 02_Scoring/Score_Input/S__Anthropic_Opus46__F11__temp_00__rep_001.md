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
run_id: Anthropic_F11_T00_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates roughly **25.8 million string-paste and hash-lookup operations** inside a sequential `lapply`.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's another **~32.3 million** R-level function invocations total.

3. **`do.call(rbind, result)`** on a list of ~6.46M small vectors is notoriously slow in R â€” it creates a massive argument list and binds row-by-row.

4. By contrast, Random Forest **prediction** on a pre-trained model is a single vectorized call (`predict(model, newdata)`) that runs in compiled C/C++ code. Even with 110 predictors and 6.46M rows, this typically completes in seconds to a few minutes. Loading a serialized model (`readRDS`) is also fast. Writing predictions is a single `fwrite` or `write.csv` call.

**Conclusion:** The 86+ hour runtime is dominated by the R-level row-wise iteration in neighbor lookup construction and neighbor statistics computation, not by Random Forest inference.

---

## Optimization Strategy

1. **Replace named-vector lookups with `data.table` hash joins** â€” `data.table` performs equi-joins in compiled C code, eliminating millions of R-level string operations.
2. **Vectorize neighbor lookup construction** â€” Expand the `nb` object into an edge list (cell_id, neighbor_id), join with year to get all (cell-year, neighbor-cell-year) pairs, then join against the row index. No per-row `lapply`.
3. **Vectorize neighbor statistics** â€” Use `data.table` grouped aggregation (`[, .(max, min, mean), by = row_idx]`) over the expanded edge table instead of per-row `lapply`.
4. **Compute all 5 variables' stats in one pass** over the joined edge table, or at minimum use vectorized grouped operations per variable.
5. **Avoid `do.call(rbind, ...)`** â€” `data.table` returns results as a data.table directly.

Expected speedup: from 86+ hours to **minutes** (the entire pipeline), because all hot loops move from interpreted R to compiled C.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Assume these objects already exist:
#     - cell_data        : data.frame/data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#     - id_order          : integer/character vector of unique cell IDs (same order as nb object)
#     - rook_neighbors_unique : spdep nb object (list of integer index vectors)
#     - rf_model          : pre-trained Random Forest model (do NOT retrain)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table if not already
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)

# Ensure a deterministic row order for later re-attachment
cell_data[, .row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Build the edge list from the nb object (vectorized, one-time cost)
#     Each entry rook_neighbors_unique[[i]] contains integer indices into
#     id_order for the neighbors of id_order[i].
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  # spdep nb objects use 0L for cells with no neighbors

  nb_idx <- nb_idx[nb_idx != 0L]
  if (length(nb_idx) == 0L) return(NULL)
  data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
}))

cat("Edge list rows:", nrow(edge_list), "\n")
# Expected: ~1,373,394 directed edges

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Build a vectorized neighbor lookup via data.table joins
#     For every (focal_id, year) row, find all (neighbor_id, year) rows.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Minimal index table: maps (id, year) -> .row_idx
idx_table <- cell_data[, .(id, year, .row_idx)]
setkey(idx_table, id, year)

# Expand edges Ã— years:  for each focal row, get its neighbor rows
# Step A: join focal rows to edge list to get (focal .row_idx, neighbor_id, year)
focal_edges <- merge(
  cell_data[, .(focal_row = .row_idx, focal_id = id, year)],
  edge_list,
  by.x = "focal_id",
  by.y = "focal_id",
  allow.cartesian = TRUE
)
# focal_edges columns: focal_id, focal_row, year, neighbor_id

# Step B: join to find the neighbor's row index for the same year
setkey(focal_edges, neighbor_id, year)
setkey(idx_table, id, year)

focal_edges[idx_table, neighbor_row := i..row_idx,
            on = .(neighbor_id = id, year = year)]

# Drop edges where the neighbor-year row doesn't exist
focal_edges <- focal_edges[!is.na(neighbor_row)]

cat("Expanded focal-neighbor-year pairs:", nrow(focal_edges), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Compute neighbor statistics for all 5 variables (vectorized)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Attach neighbor values for all source variables at once
# (only the columns we need, indexed by neighbor_row)
neighbor_vals <- cell_data[focal_edges$neighbor_row, ..neighbor_source_vars]
focal_edges <- cbind(focal_edges[, .(focal_row)], neighbor_vals)

# Grouped aggregation: max, min, mean per focal_row per variable
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

names(agg_exprs) <- agg_names

# Evaluate the aggregation
neighbor_stats <- focal_edges[,
  lapply(agg_exprs, eval, envir = .SD),
  by = focal_row
]

# Replace -Inf/Inf (from max/min on all-NA) with NA
for (col in agg_names) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3b. Alternative cleaner aggregation (if bquote approach is unwieldy)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Uncomment the block below as a simpler alternative to step 3 above:
#
# neighbor_stats <- focal_edges[, {
#   out <- list(focal_row = focal_row[1])
#   for (v in neighbor_source_vars) {
#     vals <- get(v)
#     vals <- vals[!is.na(vals)]
#     if (length(vals) == 0L) {
#       out[[paste0("neighbor_", v, "_max")]]  <- NA_real_
#       out[[paste0("neighbor_", v, "_min")]]  <- NA_real_
#       out[[paste0("neighbor_", v, "_mean")]] <- NA_real_
#     } else {
#       out[[paste0("neighbor_", v, "_max")]]  <- max(vals)
#       out[[paste0("neighbor_", v, "_min")]]  <- min(vals)
#       out[[paste0("neighbor_", v, "_mean")]] <- mean(vals)
#     }
#   }
#   out
# }, by = focal_row]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Merge neighbor stats back into cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

setkey(neighbor_stats, focal_row)
setkey(cell_data, .row_idx)

# Remove old neighbor columns if they exist (idempotency)
old_cols <- intersect(agg_names, names(cell_data))
if (length(old_cols)) cell_data[, (old_cols) := NULL]

cell_data <- merge(cell_data, neighbor_stats, by.x = ".row_idx", by.y = "focal_row", all.x = TRUE)

# Rows with no neighbors will have NA for all neighbor stats (correct behavior)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Random Forest prediction (NOT the bottleneck â€” keep as-is)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# rf_model <- readRDS("path/to/trained_rf_model.rds")   # if not already loaded
# predictions <- predict(rf_model, newdata = cell_data)
# cell_data[, prediction := predictions]
# fwrite(cell_data, "output_predictions.csv")

# Clean up helper column
cell_data[, .row_idx := NULL]

cat("Done. Neighbor feature engineering complete.\n")
```

---

## Summary

| Aspect | Colleague's Claim (RF Inference) | Actual Bottleneck (Neighbor Features) |
|---|---|---|
| **Operation type** | Vectorized C/C++ (`predict()`) | Row-wise R `lapply` over 6.46M rows Ã— 5 vars |
| **String operations** | None | ~25.8M `paste()` + named-vector hash lookups |
| **Binding cost** | Single output vector | `do.call(rbind, ...)` on 6.46M 3-element vectors |
| **Estimated time** | Minutes | 86+ hours |
| **Fix** | None needed | Replace with `data.table` joins + grouped aggregation |

The optimized code replaces all per-row R-level iteration with compiled `data.table` joins and grouped aggregations, reducing the expected runtime from **86+ hours to minutes** while preserving the trained Random Forest model and the original numerical estimand (identical max/min/mean neighbor statistics, identical predictions).
END_RESPONSE>>>
