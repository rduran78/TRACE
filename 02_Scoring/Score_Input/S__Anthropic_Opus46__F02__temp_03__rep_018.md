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
run_id: Anthropic_F02_T03_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting, hash-table lookups, and NA filtering in pure R. This creates millions of small character vectors and triggers repeated hash lookups on a named vector of 6.46 million entries â€” an O(nÂ²)-like workload in practice. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again for each of the 5 variables, subsetting numeric vectors by index lists. The combination of these two stages, repeated for 5 variables, produces the estimated 86+ hour runtime.

**Specific problems:**

1. **String-key lookups at scale:** `idx_lookup` is a named vector with ~6.46M entries. Named-vector lookup in R uses linear hashing that degrades at this scale. Each of the 6.46M rows performs multiple lookups into it.
2. **Per-row `lapply` with allocations:** Each iteration creates temporary character vectors (`paste`), subsets a named vector, and filters NAs â€” millions of small allocations that thrash the garbage collector.
3. **Redundant computation:** The neighbor *structure* is time-invariant (same grid, same rook neighbors every year), but the lookup is rebuilt as if it varies per row. The neighbor graph is only ~344K cells; the time dimension simply replicates it.
4. **`do.call(rbind, ...)` on a 6.46M-element list:** This is a known slow pattern in R.

---

## Optimization Strategy

### Key Insight: Separate Space from Time

The neighbor graph is **purely spatial** â€” cell A's neighbors are the same in every year. So we should:

1. **Build the neighbor lookup once at the cell level (344K cells), not the cell-year level (6.46M rows).**
2. **Use `data.table` for fast indexed joins** instead of named-vector lookups.
3. **Vectorize the stats computation** using `data.table` grouped operations â€” join each row to its neighbors' values and compute `max`, `min`, `mean` in bulk.

This replaces millions of R-level iterations with a single large equi-join + grouped aggregation, which `data.table` handles in seconds.

### Complexity Reduction

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | 6.46M R-level iterations with string ops | 344K-cell edge list built once (vectorized) |
| Stats computation (per variable) | 6.46M `lapply` iterations | One `data.table` join + group-by aggregation |
| Total R-level loop iterations | ~6.46M Ã— (1 + 5) â‰ˆ 38.8M | 0 (fully vectorized) |

### Memory Estimate

The edge list (directed rook neighbors) has ~1.37M rows Ã— 2 integer columns â‰ˆ 11 MB. The main `data.table` is ~6.46M rows Ã— 110 columns â‰ˆ 5.7 GB for doubles, which fits in 16 GB. The join temporarily expands to ~6.46M Ã— (avg ~4 neighbors) â‰ˆ 25.8M rows but only for a few columns at a time â€” manageable.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build a spatial edge list ONCE (cell-level, not row-level)
# ============================================================
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer index vectors
  # id_order maps position -> cell id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)

  # Remove any 0-entries (spdep uses 0 for "no neighbors")
  valid    <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id       = id_order[from_idx],
    nb_id    = id_order[to_idx]
  )
}

# ============================================================
# STEP 2: Compute neighbor stats for one variable (vectorized)
# ============================================================
compute_neighbor_stats_dt <- function(dt, edge_dt, var_name) {
  # dt must have columns: id, year, and <var_name>
  # edge_dt must have columns: id, nb_id

  # Subset to only needed columns for the join
  vals_dt <- dt[, .(id, year, val = get(var_name))]

  # Join: for each (id, year), find all neighbors' values

  # First, attach neighbor ids
  joined <- edge_dt[vals_dt, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # joined now has columns: id, nb_id, year, val
  # We need the NEIGHBOR's value, not the focal cell's value
  # So join again to get nb's val in that year
  setnames(vals_dt, c("id", "year", "val"), c("nb_id", "year", "nb_val"))
  joined <- vals_dt[joined, on = c("nb_id", "year"), nomatch = NA]
  # joined now has: nb_id, year, nb_val, id, val

  # Compute grouped stats: group by (id, year)
  stats <- joined[!is.na(nb_val),
                   .(nb_max  = max(nb_val),
                     nb_min  = min(nb_val),
                     nb_mean = mean(nb_val)),
                   by = .(id, year)]

  # Build output column names
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  stats
}

# ============================================================
# STEP 3: Main pipeline
# ============================================================
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  # Convert to data.table (by reference if already, otherwise copy)
  if (!is.data.table(cell_data)) {
    dt <- as.data.table(cell_data)
  } else {
    dt <- copy(cell_data)
  }

  # Ensure key columns exist
  stopifnot(all(c("id", "year") %in% names(dt)))

  # Step 1: Build edge list once
  message("Building spatial edge list...")
  edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
  setkey(edge_dt, id)
  message(sprintf("  Edge list: %s directed edges", format(nrow(edge_dt), big.mark = ",")))

  # Step 2: For each variable, compute and merge neighbor stats
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))

    stats_dt <- compute_neighbor_stats_dt(dt, edge_dt, var_name)

    # Merge back into main table
    dt <- merge(dt, stats_dt, by = c("id", "year"), all.x = TRUE)

    # Free memory
    rm(stats_dt)
    gc()
  }

  message("Done. All neighbor features added.")
  return(dt)
}

# ============================================================
# USAGE
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- run_neighbor_feature_pipeline(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets and the same variable values. The join logic replicates the original's "find all rook neighbors of cell *i* that exist in year *t*, get their values, compute stats." |
| **Trained RF model** | No model retraining. The output columns have the same names and values, so `predict(rf_model, newdata = ...)` works identically. |
| **NA handling** | `!is.na(nb_val)` in the grouped aggregation mirrors the original `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`. Cells with no valid neighbors get `NA` via `all.x = TRUE` in the merge. |

---

## Expected Performance

| Metric | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | ~6.46M R iterations, string ops | Vectorized edge list build (~0.5 sec) |
| **Stats per variable** | ~6.46M `lapply` calls | One `data.table` join + grouped agg (~30â€“90 sec) |
| **Total wall time** | 86+ hours (estimated) | **~5â€“10 minutes** |
| **Peak RAM** | Uncontrolled (millions of small allocations) | ~8â€“12 GB (controlled, fits in 16 GB) |
END_RESPONSE>>>
