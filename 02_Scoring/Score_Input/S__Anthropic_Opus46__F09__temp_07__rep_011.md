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
run_id: Anthropic_F09_T07_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F09_Anthropic_temp_07_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of **two compounding bottlenecks**:

### 1. `build_neighbor_lookup` â€” O(nÂ²)-like overhead via per-row string hashing
- It creates a named character vector `idx_lookup` of length ~6.46 million, keyed by `paste(id, year)`.
- Then, **for each of the 6.46 million rows**, it does string-paste and named-vector lookups (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is O(n) per query in the worst case (linear scan of names), not O(1). With ~6.46M rows Ã— ~4 neighbors each â‰ˆ 25.8M string lookups against a 6.46M-length named vector, this is catastrophically slow.
- The `lapply` over 6.46M rows also creates 6.46M small integer vectors, which is GC-heavy.

### 2. `compute_neighbor_stats` â€” repeated per-variable `lapply` over 6.46M rows
- For each of the 5 neighbor source variables, another `lapply` iterates over all 6.46M rows, extracting small vectors, computing `max/min/mean`, and returning 3-element vectors that are then `rbind`-ed (another slow operation at scale).
- Total: 5 Ã— 6.46M = 32.3M R-level function calls.

### Core insight
The **cell-neighbor topology is time-invariant**. There are only ~344K cells with ~1.37M directed rook-neighbor edges. This adjacency structure is the same for every year. The current code **re-expands** this to the cell-year level (6.46M rows), which is wasteful. The correct approach is:

1. **Build the adjacency table once** at the cell level (~1.37M edge rows).
2. **Join yearly attributes** onto both sides of each edge.
3. **Group-by aggregate** (max, min, mean) using vectorized `data.table` operations.

This replaces all `lapply` loops and string lookups with hash-joined, vectorized columnar operations.

---

## Optimization Strategy

| Step | What | Why |
|------|------|-----|
| **A** | Build a `data.table` of directed edges: `(cell_id, neighbor_id)` from the `nb` object â€” ~1.37M rows, built once. | Time-invariant topology, reusable. |
| **B** | For each year, join cell attributes onto the neighbor side of the edge table via keyed `data.table` join. | Replaces string-paste + named-vector lookup with O(1) hash join. |
| **C** | Group by `(cell_id, year)` and compute `max`, `min`, `mean` of each variable in one vectorized pass. | Replaces 6.46M Ã— 5 R-level `lapply` calls with a single grouped aggregation. |
| **D** | Join the aggregated neighbor stats back onto `cell_data`. | Produces the same columns the Random Forest model expects. |

**Expected speedup**: From ~86 hours to **~2â€“10 minutes** on a 16 GB laptop.

**Numerical equivalence**: The operations `max`, `min`, `mean` over the same neighbor sets produce identical results. The trained Random Forest model is never retrained â€” we only produce the same predictor columns.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP A: Build the time-invariant cell-neighbor edge table ONCE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Inputs:
#   id_order            â€” integer/numeric vector of cell IDs (length 344,208),
#                          ordered to match the nb object indices.
#   rook_neighbors_unique â€” an nb object (list of length 344,208), where each
#                          element is an integer vector of neighbor indices
#                          (referencing positions in id_order), with 0L
#                          indicating no neighbors.

build_edge_table <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total edges
  n_cells <- length(id_order)
  edge_list <- vector("list", n_cells)

  for (i in seq_len(n_cells)) {
    nb_idx <- neighbors_nb[[i]]
    # nb objects use 0L to denote no neighbors
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) > 0L) {
      edge_list[[i]] <- data.table(
        cell_id     = id_order[i],
        neighbor_id = id_order[nb_idx]
      )
    }
  }

  rbindlist(edge_list)
}

cat("Building time-invariant edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP B & C: Compute neighbor stats for all variables at once
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Inputs:
#   cell_data â€” data.frame/data.table with columns: id, year, and the
#               neighbor_source_vars columns.
#   edge_dt   â€” from Step A.
#   neighbor_source_vars â€” character vector of variable names.

compute_all_neighbor_features <- function(cell_data, edge_dt, neighbor_source_vars) {
  # Convert to data.table if needed (by reference if already)
  if (!is.data.table(cell_data)) {
    cell_dt <- as.data.table(cell_data)
  } else {
    cell_dt <- copy(cell_data)
  }

  # Columns we need from the neighbor rows
  join_cols <- c("id", "year", neighbor_source_vars)

  # Subset to only what we need for the join (keep memory down)
  neighbor_attrs <- cell_dt[, ..join_cols]
  setnames(neighbor_attrs, "id", "neighbor_id")

  # Key for fast join
  setkey(neighbor_attrs, neighbor_id, year)

  # Cross the edge table with all years present in the data
  years <- sort(unique(cell_dt$year))
  cat(sprintf("  Expanding edge table across %d years...\n", length(years)))

  # Expand edges Ã— years: ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows
  # This is manageable in 16 GB RAM
  edges_by_year <- CJ_dt_edges(edge_dt, years)

  # Join neighbor attributes onto the edge-year table
  cat("  Joining neighbor attributes...\n")
  setkey(edges_by_year, neighbor_id, year)
  edges_by_year <- neighbor_attrs[edges_by_year, on = .(neighbor_id, year), nomatch = NA]

  # Aggregate: group by (cell_id, year), compute max/min/mean for each var
  cat("  Aggregating neighbor stats...\n")

  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("n_", v, c("_max", "_min", "_mean"))
  }))

  names(agg_exprs) <- agg_names

  # Perform the grouped aggregation
  stats_dt <- edges_by_year[,
    lapply(agg_exprs, eval),
    by = .(cell_id, year)
  ]

  # Replace -Inf/Inf from max/min on all-NA groups with NA
  for (col_name in agg_names) {
    vals <- stats_dt[[col_name]]
    set(stats_dt, i = which(is.infinite(vals)), j = col_name, value = NA_real_)
  }

  return(stats_dt)
}

# Helper: expand edge table across all years
CJ_dt_edges <- function(edge_dt, years) {
  years_dt <- data.table(year = years)
  # Cross join: every edge Ã— every year
  result <- edge_dt[, .(year = years), by = .(cell_id, neighbor_id)]
  return(result)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP D: Run it and join back onto cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (vectorized)...\n")
t0 <- proc.time()

neighbor_stats <- compute_all_neighbor_features(
  cell_data, edge_dt, neighbor_source_vars
)

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("  Done in %.1f seconds.\n", elapsed))

# Merge back onto the main dataset
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Remove old neighbor columns if they exist (to avoid duplicates)
old_cols <- intersect(
  names(cell_data),
  unlist(lapply(neighbor_source_vars, function(v) paste0("n_", v, c("_max", "_min", "_mean"))))
)
if (length(old_cols) > 0) {
  cell_data[, (old_cols) := NULL]
}

# Keyed merge
setkey(cell_data, id, year)
setnames(neighbor_stats, "cell_id", "id")
setkey(neighbor_stats, id, year)

cell_data <- neighbor_stats[cell_data, on = .(id, year)]

cat("Neighbor features merged. Ready for Random Forest prediction.\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP E: Predict with the existing trained model (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained Random Forest model object (e.g., rf_model) is preserved
# exactly as-is. We only call predict():
#
#   cell_data$prediction <- predict(rf_model, newdata = cell_data)
#
# The column names and numerical values are identical to the original
# pipeline, so predictions are numerically equivalent.
```

---

## Why This Is Equivalent and Safe

| Concern | Guarantee |
|---------|-----------|
| **Same neighbor sets** | `edge_dt` is built from the same `rook_neighbors_unique` nb object and `id_order`. Each `(cell_id, neighbor_id)` pair is identical. |
| **Same aggregation** | `max`, `min`, `mean` with `na.rm = TRUE` over the same values produces identical results. The `Inf`â†’`NA` cleanup matches the original `length(neighbor_vals) == 0` â†’ `NA` logic. |
| **Same column names** | Output columns are `n_{var}_max`, `n_{var}_min`, `n_{var}_mean` â€” adjust the naming pattern if the original used a different convention (e.g., `{var}_neighbor_max`). |
| **RF model untouched** | The model object is never modified or retrained. Only `predict()` is called. |
| **RAM feasibility** | The largest intermediate object is the expanded edge-year table: ~38.5M rows Ã— 3 key cols + 5 value cols â‰ˆ ~2.5 GB. Well within 16 GB. |

---

## Summary

| Metric | Before | After |
|--------|--------|-------|
| **Runtime** | ~86 hours | **~2â€“10 minutes** |
| **Bottleneck** | 6.46M string lookups in named vectors + 32.3M `lapply` calls | Eliminated â€” replaced by hash-keyed `data.table` joins and vectorized grouped aggregation |
| **Topology reuse** | Rebuilt implicitly per row | Built once as `edge_dt`, joined per year |
| **Numerical result** | Original | Identical |
| **RF model** | Preserved | Preserved |
END_RESPONSE>>>
