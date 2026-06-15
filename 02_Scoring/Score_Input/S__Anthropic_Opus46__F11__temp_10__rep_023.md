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
run_id: Anthropic_F11_T10_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points overwhelmingly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup` and `compute_neighbor_stats` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup`** runs an `lapply` over **~6.46 million rows**. For each row, it performs character coercion (`as.character`), hash-table lookups into `id_to_ref`, constructs `paste(..., sep="_")` key strings for every neighbor, and indexes into `idx_lookup`. With ~1.37 million directed neighbor relationships spread across 28 years, this creates enormous repeated string construction and named-vector lookup overhead â€” all in a sequential, interpreted R loop.

2. **`compute_neighbor_stats`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows with an `lapply`, subsetting values, removing NAs, and computing `max`/`min`/`mean`. The `do.call(rbind, result)` on a 6.46-million-element list is itself expensive.

3. **Combined cost**: `build_neighbor_lookup` is O(6.46M Ã— avg_neighbors) string operations. `compute_neighbor_stats` is called 5 times Ã— 6.46M iterations. Together, these dominate runtime by orders of magnitude.

4. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model â€” a vectorized C/C++ operation in packages like `randomForest` or `ranger`. Even with 6.46M rows and 110 predictors, this typically takes minutes, not hours.

**Verdict**: The 86+ hour runtime is caused by row-level interpreted R loops with expensive string operations over millions of rows, not by RF inference.

---

## Optimization Strategy

1. **Replace the row-level `lapply` in `build_neighbor_lookup` with a vectorized `data.table` merge/join.** Instead of building a lookup per row, expand the neighbor list into an edge table (cell_id â†’ neighbor_id), merge with year to get (cell_id, year) â†’ (neighbor_id, year), and join against the data to obtain neighbor row indices. This turns millions of sequential string operations into a single indexed join.

2. **Replace the row-level `lapply` in `compute_neighbor_stats` with a grouped `data.table` aggregation** over the edge table. Compute max, min, and mean of neighbor values using `data.table`'s optimized grouped operations (`by=`), which run in C and avoid R-level iteration entirely.

3. **Compute all 5 variables' neighbor stats in one pass** (or with minimal passes) over the same edge structure.

This should reduce the 86+ hour runtime to **minutes**.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Assume these objects already exist in the environment:
#     - cell_data        : data.frame / data.table with columns id, year, 
#                          ntl, ec, pop_density, def, usd_est_n2, â€¦
#     - id_order         : integer vector of cell IDs in the order used by
#                          the nb object (i.e., id_order[i] is the cell ID
#                          for the i-th element of rook_neighbors_unique)
#     - rook_neighbors_unique : an nb object (list of integer vectors)
#     - rf_model         : the pre-trained Random Forest model
# ---------------------------------------------------------------

# Convert cell_data to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ---------------------------------------------------------------
# 1.  Build a spatial edge table from the nb object (one-time cost)
#     This maps each cell to its rook neighbors using integer cell IDs.
# ---------------------------------------------------------------
build_edge_table <- function(id_order, nb_obj) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(nb_obj, length, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_along(nb_obj)) {
    nb_i <- nb_obj[[i]]
    n_i  <- length(nb_i)
    if (n_i == 0L) next
    # Filter out the "no-neighbor" sentinel (0) used by spdep
    nb_i <- nb_i[nb_i != 0L]
    n_i  <- length(nb_i)
    if (n_i == 0L) next
    
    from_id[pos:(pos + n_i - 1L)] <- id_order[i]
    to_id[pos:(pos + n_i - 1L)]   <- id_order[nb_i]
    pos <- pos + n_i
  }
  
  # Trim if any sentinels were removed
  data.table(from_id = from_id[1:(pos - 1L)],
             to_id   = to_id[1:(pos - 1L)])
}

edges <- build_edge_table(id_order, rook_neighbors_unique)

# ---------------------------------------------------------------
# 2.  Vectorised neighbor-feature computation via data.table join
#     For every (from_id, year) we look up every (to_id, year) row
#     and aggregate the neighbor values.
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare a lean table with only the columns we need for the join
# Keying on (id, year) makes the join O(n log n) or better
neighbor_val_cols <- c("id", "year", neighbor_source_vars)
neighbor_data <- cell_data[, ..neighbor_val_cols]
setnames(neighbor_data, "id", "to_id")
setkey(neighbor_data, to_id, year)

# Expand edges Ã— years:  for each (from_id, year), get all (to_id, year)
# Step A â€” attach year from the focal cell to the edge
focal_years <- unique(cell_data[, .(from_id = id, year)])
setkey(focal_years, from_id)
setkey(edges, from_id)

# Cross-join edges with the years each focal cell appears in
edge_year <- edges[focal_years, on = "from_id", allow.cartesian = TRUE, nomatch = NULL]
# edge_year now has columns: from_id, to_id, year

# Step B â€” join to get neighbor variable values
setkey(edge_year, to_id, year)
edge_year <- neighbor_data[edge_year, on = .(to_id, year), nomatch = NA]
# edge_year now has: to_id, year, ntl, ec, â€¦, from_id

# Step C â€” aggregate by (from_id, year) to get max, min, mean for each variable
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

names(agg_exprs) <- agg_names

neighbor_stats <- edge_year[,
  lapply(agg_exprs, eval, envir = .SD),
  by = .(from_id, year)
]

# Replace Inf/-Inf (from max/min on all-NA groups) with NA
for (col in agg_names) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

# ---------------------------------------------------------------
#  Alternative cleaner aggregation (if bquote approach is tricky
#  in your R version):
# ---------------------------------------------------------------
# neighbor_stats <- edge_year[, {
#   out <- list(from_id = from_id[1L])
#   for (v in neighbor_source_vars) {
#     vals <- .SD[[v]]
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
# }, by = .(from_id, year)]

# ---------------------------------------------------------------
# >>> ROBUST ALTERNATIVE (preferred â€” avoids bquote complexity) <<<
# ---------------------------------------------------------------
neighbor_stats <- edge_year[, {
  res <- vector("list", length(neighbor_source_vars) * 3L)
  nm  <- character(length(res))
  k   <- 0L
  for (v in neighbor_source_vars) {
    vals <- get(v)
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) {
      mx <- NA_real_; mn <- NA_real_; mu <- NA_real_
    } else {
      mx <- max(vals); mn <- min(vals); mu <- mean(vals)
    }
    k <- k + 1L; res[[k]] <- mx;  nm[k] <- paste0("neighbor_", v, "_max")
    k <- k + 1L; res[[k]] <- mn;  nm[k] <- paste0("neighbor_", v, "_min")
    k <- k + 1L; res[[k]] <- mu;  nm[k] <- paste0("neighbor_", v, "_mean")
  }
  setNames(res, nm)
}, by = .(from_id, year)]

# ---------------------------------------------------------------
# 3.  Merge neighbor stats back onto cell_data
# ---------------------------------------------------------------
setnames(neighbor_stats, "from_id", "id")
setkey(neighbor_stats, id, year)
setkey(cell_data, id, year)

cell_data <- neighbor_stats[cell_data, on = .(id, year)]

# ---------------------------------------------------------------
# 4.  Random Forest prediction (unchanged â€” this was never the
#     bottleneck). The trained model is preserved as-is.
# ---------------------------------------------------------------
# Ensure column order / names match what rf_model expects.
# predict() on a ranger or randomForest object is vectorized C code.
pred_vars <- names(rf_model$forest$xlevels)
# â€” or for ranger: pred_vars <- rf_model$forest$independent.variable.names

predictions <- predict(rf_model, newdata = cell_data[, ..pred_vars])

# For randomForest objects:
cell_data[, prediction := predictions]
# For ranger objects:
# cell_data[, prediction := predictions$predictions]

# ---------------------------------------------------------------
# Done. The original numerical estimand is preserved because we
# (a) use the exact same trained RF model with no retraining,
# (b) compute identical neighbor max/min/mean statistics, and
# (c) only changed the computational strategy, not the logic.
# ---------------------------------------------------------------
```

---

## Summary

| Aspect | Colleague's Claim (RF Inference) | Actual Bottleneck (Neighbor Features) |
|---|---|---|
| **Operation type** | Vectorized C/C++ `predict()` | Interpreted R `lapply` over 6.46M rows |
| **String ops** | None | Millions of `paste()`, `as.character()`, named-vector lookups |
| **Iterations** | 1 call | 6.46M Ã— (1 build + 5 stat passes) â‰ˆ 38.8M R-level iterations |
| **Expected time** | Minutes | 86+ hours |
| **Fix** | N/A | `data.table` keyed join + grouped aggregation |
| **Expected speedup** | â€” | ~100â€“500Ã— (minutes instead of days) |

The trained Random Forest model is preserved untouched; only the upstream feature engineering is optimized. The numerical results (and thus the original estimand) remain identical.
END_RESPONSE>>>
