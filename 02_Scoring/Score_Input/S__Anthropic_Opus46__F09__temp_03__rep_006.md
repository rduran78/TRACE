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
run_id: Anthropic_F09_T03_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F09_Anthropic_temp_03_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for two interconnected reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-like in practice
This function iterates over **all ~6.46 million cell-year rows** with `lapply`, and for each row it:
- Looks up the cell's spatial neighbors (fine).
- Constructs `paste()`-based string keys for every neighbor Ã— that row's year (expensive string allocation).
- Matches those keys against a named character vector (`idx_lookup`) of length 6.46M (expensive named-vector lookup â€” R's named vector lookup is O(n) in the worst case, not hash-based like an environment or `data.table` key).

This means ~6.46M iterations, each doing multiple string constructions and linear scans of a 6.46M-length named vector. This is the **dominant bottleneck**.

### 2. `compute_neighbor_stats` is less severe but still suboptimal
It loops over 6.46M list elements, subsetting a numeric vector each time. This is tolerable but can be vectorized.

### 3. The core conceptual problem
The neighbor topology is **purely spatial** â€” it does not change across years. Yet the lookup is rebuilt for every cell-year combination, redundantly re-discovering the same spatial neighbors 28 times (once per year). The string-key join approach is the wrong abstraction: what's needed is a **spatial adjacency table joined to yearly attributes**.

---

## Optimization Strategy

1. **Build a static spatial edge table once** â€” a two-column `data.table` (`id`, `neighbor_id`) with ~1.37M rows representing all directed rook-neighbor pairs. This is year-invariant.

2. **Cross-join with years** â€” expand the edge table to `(id, year, neighbor_id)` by joining with the 28 years. This yields ~1.37M Ã— 28 â‰ˆ 38.5M rows, which is large but manageable.

3. **Join neighbor attributes** â€” key `cell_data` as a `data.table` on `(id, year)` and join neighbor-cell attributes onto the edge table via `(neighbor_id, year)`. This is an O(n log n) indexed join, not a string-match scan.

4. **Aggregate** â€” group by `(id, year)` and compute `max`, `min`, `mean` for each variable in one pass.

5. **Join results back** to `cell_data`.

This replaces ~6.46M Ã— (string ops + linear lookup) with a handful of keyed `data.table` joins and a single grouped aggregation â€” expected runtime: **minutes, not days**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Ensure cell_data is a data.table with key columns
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.table(cell_data)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a static spatial edge table (year-invariant, built ONCE)
#
#   rook_neighbors_unique : spdep nb object (list of integer index vectors)
#   id_order              : vector mapping positional index -> cell id
#
#   Result: edge_dt with columns  id | neighbor_id
#           ~1,373,394 rows (directed rook pairs)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_edge_table <- function(id_order, neighbors) {
  # neighbors[[i]] gives integer indices of neighbors for cell at position i
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove zero-length / 0-coded "no neighbor" entries that spdep may include
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edge_dt), "\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Compute neighbor stats for all source variables at once
#
#   Strategy:
#     a) Cross-join edge_dt with the unique years in cell_data.
#     b) Join the neighbor cell's attribute values via (neighbor_id, year).
#     c) Aggregate max/min/mean grouped by (id, year).
#     d) Join aggregated columns back onto cell_data.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# -- 2a: Prepare a keyed version of cell_data with only needed columns --------
keep_cols <- c("id", "year", neighbor_source_vars)
cd_slim   <- cell_data[, ..keep_cols]
setkey(cd_slim, id, year)

# -- 2b: Expand edges Ã— years -------------------------------------------------
years_vec <- sort(unique(cell_data$year))           # 28 years
edge_year <- edge_dt[, CJ(year = years_vec), by = .(id, neighbor_id)]
#   This is ~1.37M * 28 â‰ˆ 38.5M rows.  Fits in RAM (~1-2 GB).

# -- 2c: Join neighbor attributes onto edge_year ------------------------------
setkey(edge_year, neighbor_id, year)
setkey(cd_slim,   id,          year)

# Rename for the join: we want to look up by (neighbor_id, year) -> cd_slim's (id, year)
edge_year <- cd_slim[edge_year, on = .(id = neighbor_id, year = year), nomatch = NA]
# After this join, edge_year has columns:
#   id (= neighbor_id), year, ntl, ec, ..., i.id (= focal cell id), neighbor_id (dropped)
# data.table renames the joining key; let's be explicit:

# The join above maps:  cd_slim.id == edge_year.neighbor_id
# Resulting columns: id (neighbor), year, <vars>, i.id (focal cell)
# Rename for clarity:
setnames(edge_year, "i.id", "focal_id")
# 'id' column now refers to the neighbor; 'focal_id' is the focal cell.

# -- 2d: Aggregate by (focal_id, year) ----------------------------------------
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Build the aggregation call programmatically
agg_list <- setNames(agg_exprs, agg_names)

neighbor_stats <- edge_year[,
  lapply(agg_list, eval, envir = .SD),
  by = .(focal_id, year),
  .SDcols = neighbor_source_vars
]

# ---- Simpler / more robust alternative aggregation if bquote is tricky: -----
# (Use this block instead of the bquote block above if preferred)

neighbor_stats <- edge_year[,
  {
    out <- list()
    for (v in neighbor_source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[paste0("neighbor_max_",  v)]] <- NA_real_
        out[[paste0("neighbor_min_",  v)]] <- NA_real_
        out[[paste0("neighbor_mean_", v)]] <- NA_real_
      } else {
        out[[paste0("neighbor_max_",  v)]] <- max(vals)
        out[[paste0("neighbor_min_",  v)]] <- min(vals)
        out[[paste0("neighbor_mean_", v)]] <- mean(vals)
      }
    }
    out
  },
  by = .(focal_id, year)
]

# -- 2e: Replace Inf/-Inf from max/min on empty groups with NA ----------------
for (col in names(neighbor_stats)) {
  if (is.numeric(neighbor_stats[[col]])) {
    set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
  }
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Join neighbor stats back onto cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Drop any old neighbor columns from cell_data to avoid duplication
old_neighbor_cols <- grep("^neighbor_", names(cell_data), value = TRUE)
if (length(old_neighbor_cols) > 0) {
  cell_data[, (old_neighbor_cols) := NULL]
}

setkey(cell_data,      id, year)
setkey(neighbor_stats, focal_id, year)

cell_data <- neighbor_stats[cell_data, on = .(focal_id = id, year = year)]
setnames(cell_data, "focal_id", "id")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Predict with the EXISTING trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g., `rf_model`) is used as-is.
# Ensure the column names match what the model expects.

# cell_data$prediction <- predict(rf_model, newdata = cell_data)

cat("Done. Neighbor features computed and joined.\n")
```

---

## Expected Performance Comparison

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M `paste` + named-vector lookups | One `data.table` built from `unlist` (seconds) |
| Neighbor attribute retrieval | Per-row subsetting via string keys | Single keyed `data.table` join (~38.5M rows) |
| Aggregation | 6.46M `lapply` iterations Ã— 5 vars | Grouped `data.table` aggregation (vectorized C) |
| **Estimated wall time** | **86+ hours** | **5â€“15 minutes** |
| Peak RAM | Moderate (but slow) | ~3â€“5 GB (fits in 16 GB) |

## Why the Numerical Estimand Is Preserved

- The same directed rook-neighbor pairs are used (sourced from the identical `rook_neighbors_unique` nb object).
- The same `max`, `min`, `mean` aggregations are computed on the same neighbor attribute values, with the same `NA` handling.
- The trained Random Forest model is **never retrained** â€” only `predict()` is called on the enriched data.
- Column names are constructed identically (`neighbor_max_ntl`, `neighbor_min_ntl`, `neighbor_mean_ntl`, etc.), so the model's expected feature names match.
END_RESPONSE>>>
