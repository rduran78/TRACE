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
run_id: Anthropic_F11_T10_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** as the dominant bottleneck â€” specifically `build_neighbor_lookup` and `compute_neighbor_stats`.

**Evidence from the code:**

1. **`build_neighbor_lookup`** uses a top-level `lapply` over **~6.46 million rows**. Inside each iteration, it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and NA filtering â€” all in pure interpreted R. That is ~6.46 million iterations of non-vectorized, allocation-heavy string operations.

2. **`compute_neighbor_stats`** then iterates over the resulting 6.46-million-element list, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean` per element. This is called **5 times** (once per neighbor source variable), yielding ~32.3 million R-level loop iterations total.

3. By contrast, Random Forest prediction (`predict()`) on a pre-trained model with ~6.46M rows and ~110 columns is a single vectorized C/C++ call (in `randomForest` or `ranger`). It is typically minutes, not hours, on a laptop. Loading and writing are also fast I/O operations relative to the neighbor computation.

**The bottleneck is the O(n) pure-R loop with per-row string allocation and named-vector lookup across 6.46M rows, repeated for 5 variables â€” totaling the estimated 86+ hours.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup` with a vectorized `data.table` equi-join.** Instead of building a per-row list via `lapply` with string key lookups, we expand all neighbor pairs into a two-column table `(focal_row, neighbor_row)` using integer joins. This eliminates millions of `paste()` and named-vector lookups.

2. **Replace `compute_neighbor_stats` with a grouped `data.table` aggregation.** Once we have a long-form `(focal_row, neighbor_value)` table, computing `max`, `min`, and `mean` per focal row is a single `data.table` grouped operation â€” fully vectorized in C.

3. **Do all 5 variables in one pass** over the neighbor edge table to avoid redundant joins.

This reduces runtime from ~86+ hours to an estimated **minutes** (dominated by the join and group-by on ~8.9 billion-ish? No â€” ~1.37M directed edges Ã— 28 years of matching, but actually the expansion is bounded by the edge list size, which we compute below).

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Prepare: convert cell_data to data.table (non-destructive)
# ---------------------------------------------------------------
# cell_data must have columns: id, year, and the neighbor_source_vars
# id_order: vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: spdep nb object (list of integer index vectors)

cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]            # preserve original row order

# ---------------------------------------------------------------
# 1.  Build the directed edge list (focal_id -> neighbor_id)
#     This replaces build_neighbor_lookup entirely.
# ---------------------------------------------------------------
# Expand the nb object into a two-column integer edge table
#   focal_pos:    position in id_order
#   neighbor_pos: position in id_order

edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_pos = i, neighbor_pos = nb)
}))

# Map positions back to cell IDs
edge_list[, focal_id    := id_order[focal_pos]]
edge_list[, neighbor_id := id_order[neighbor_pos]]
edge_list[, c("focal_pos", "neighbor_pos") := NULL]

# ---------------------------------------------------------------
# 2.  Join edges with the panel on year to get (focal_row, neighbor_row)
#     For every year, each directed edge becomes a row-pair.
# ---------------------------------------------------------------
# Key the cell data for fast joins
setkey(cell_dt, id, year)

# Get unique years
years <- sort(unique(cell_dt$year))

# Cross the edge list with years
edges_by_year <- CJ_dt <- edge_list[, .(focal_id, neighbor_id)]
edges_by_year <- edges_by_year[, .(year = years), by = .(focal_id, neighbor_id)]

# Join to get focal row index
edges_by_year <- merge(
  edges_by_year,
  cell_dt[, .(focal_id = id, year, focal_row = row_idx)],
  by.x = c("focal_id", "year"),
  by.y = c("focal_id", "year"),
  all.x = TRUE,
  allow.cartesian = FALSE
)

# Join to get neighbor row index and neighbor variable values
neighbor_cols <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_subset <- cell_dt[, c("id", "year", "row_idx", neighbor_cols), with = FALSE]
setnames(neighbor_subset, "id", "neighbor_id")
setnames(neighbor_subset, "row_idx", "neighbor_row")

edges_by_year <- merge(
  edges_by_year,
  neighbor_subset,
  by.x = c("neighbor_id", "year"),
  by.y = c("neighbor_id", "year"),
  all.x = TRUE,
  allow.cartesian = FALSE
)

# Remove edges where either focal or neighbor row was missing
edges_by_year <- edges_by_year[!is.na(focal_row) & !is.na(neighbor_row)]

# ---------------------------------------------------------------
# 3.  Compute neighbor stats: max, min, mean per focal_row per variable
#     This replaces compute_neighbor_stats (called 5Ã—) with one pass.
# ---------------------------------------------------------------
stat_exprs <- unlist(lapply(neighbor_cols, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}), recursive = FALSE)

stat_names <- unlist(lapply(neighbor_cols, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Efficient grouped aggregation in data.table
neighbor_stats <- edges_by_year[,
  setNames(lapply(neighbor_cols, function(v) {
    vals <- get(v)
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) list(NA_real_, NA_real_, NA_real_)
    else list(max(vals), min(vals), mean(vals))
  }), neighbor_cols),
  by = focal_row
]

# The above nested-list approach can be tricky; here is the
# clearer and robust version:

neighbor_stats <- edges_by_year[, {
  out <- vector("list", length(neighbor_cols) * 3L)
  k <- 0L
  for (v in neighbor_cols) {
    vals <- get(v)
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) {
      out[[k + 1L]] <- NA_real_
      out[[k + 2L]] <- NA_real_
      out[[k + 3L]] <- NA_real_
    } else {
      out[[k + 1L]] <- max(vals)
      out[[k + 2L]] <- min(vals)
      out[[k + 3L]] <- mean(vals)
    }
    k <- k + 3L
  }
  names(out) <- stat_names
  out
}, by = focal_row]

# ---------------------------------------------------------------
# 4.  Join the computed stats back onto cell_dt by row index
# ---------------------------------------------------------------
setkey(neighbor_stats, focal_row)
setkey(cell_dt, row_idx)

cell_dt <- merge(cell_dt, neighbor_stats, by.x = "row_idx", by.y = "focal_row", all.x = TRUE)

# Restore original row order
setorder(cell_dt, row_idx)

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)
cell_data$row_idx <- NULL

# ---------------------------------------------------------------
# 5.  Random Forest prediction â€” unchanged (not the bottleneck)
# ---------------------------------------------------------------
# library(ranger)  # or library(randomForest)
# rf_model  <- readRDS("trained_rf_model.rds")
# preds     <- predict(rf_model, data = cell_data)
# cell_data$prediction <- preds$predictions   # ranger
# saveRDS(cell_data, "final_predictions.rds")
```

---

### Memory-Optimized Variant (if the year-expanded edge table is too large for 16 GB)

The full `edges_by_year` table could be large (~1.37M edges Ã— 28 years â‰ˆ 38.4M rows Ã— several columns). If memory is tight, process **one year at a time**:

```r
library(data.table)

cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]

# Build edge list (same as above, ~1.37M rows, tiny)
edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) return(NULL)
  data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
}))

neighbor_cols <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
stat_names    <- unlist(lapply(neighbor_cols, function(v)
  paste0("neighbor_", c("max_", "min_", "mean_"), v)))

years <- sort(unique(cell_dt$year))

# Pre-allocate result matrix
result_mat <- matrix(NA_real_, nrow = nrow(cell_dt), ncol = length(stat_names))
colnames(result_mat) <- stat_names

setkey(cell_dt, id, year)

for (yr in years) {
  cat("Processing year:", yr, "\n")
  
  yr_dt <- cell_dt[year == yr, c("id", "row_idx", neighbor_cols), with = FALSE]
  setkey(yr_dt, id)
  
  # Focal join
  edges_yr <- merge(edge_list, yr_dt[, .(id, focal_row = row_idx)],
                    by.x = "focal_id", by.y = "id", all.x = FALSE)
  
  # Neighbor join
  edges_yr <- merge(edges_yr,
                    yr_dt[, c("id", "row_idx", neighbor_cols), with = FALSE],
                    by.x = "neighbor_id", by.y = "id", all.x = FALSE)
  
  # Grouped aggregation
  stats_yr <- edges_yr[, {
    out <- vector("list", length(neighbor_cols) * 3L)
    k <- 0L
    for (v in neighbor_cols) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[k + 1L]] <- NA_real_; out[[k + 2L]] <- NA_real_; out[[k + 3L]] <- NA_real_
      } else {
        out[[k + 1L]] <- max(vals); out[[k + 2L]] <- min(vals); out[[k + 3L]] <- mean(vals)
      }
      k <- k + 3L
    }
    names(out) <- stat_names
    out
  }, by = focal_row]
  
  # Write into result matrix
  result_mat[stats_yr$focal_row, ] <- as.matrix(stats_yr[, ..stat_names])
}

# Bind results back
for (j in seq_along(stat_names)) {
  cell_dt[, (stat_names[j]) := result_mat[, j]]
}

cell_dt[, row_idx := NULL]
setorder(cell_dt, id, year)
cell_data <- as.data.frame(cell_dt)

# --- Random Forest prediction unchanged ---
# rf_model <- readRDS("trained_rf_model.rds")
# cell_data$prediction <- predict(rf_model, data = cell_data)$predictions
```

---

## Summary

| Aspect | Colleague's claim (RF inference) | Actual bottleneck (neighbor features) |
|---|---|---|
| **Operation** | `predict()` â€” single vectorized C call | 6.46M `lapply` iterations with `paste()` + named lookups Ã— 5 vars |
| **Complexity** | O(n Ã— trees) in compiled C | O(n Ã— k) in interpreted R with per-row string alloc |
| **Est. time** | Minutes | 86+ hours |
| **Fix** | N/A (already fast) | Vectorized `data.table` join + grouped aggregation |

The trained Random Forest model is **preserved unchanged** â€” no retraining. The numerical estimand (max, min, mean of neighbor values) is **preserved exactly**. The optimization targets only the data preparation stage that the code proves is the true bottleneck.
END_RESPONSE>>>
