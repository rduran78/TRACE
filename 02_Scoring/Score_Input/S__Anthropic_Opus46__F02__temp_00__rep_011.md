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
run_id: Anthropic_F02_T00_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F02_Anthropic_temp_00_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R via character keys are O(n) hash lookups, but doing this millions of times with millions of keys creates enormous overhead. The `compute_neighbor_stats` function then iterates over the same 6.46 million entries again, once per variable. Combined:

1. **`build_neighbor_lookup`**: ~6.46M iterations, each doing character coercion, `paste`, and multiple named-vector lookups. The `idx_lookup` named vector has ~6.46M entries, so each `[` lookup is expensive. This alone accounts for the vast majority of the 86+ hour estimate.

2. **`compute_neighbor_stats`**: Uses `lapply` over 6.46M elements, each calling `max`, `min`, `mean` on small vectors. The `do.call(rbind, ...)` on a 6.46M-element list of 3-element vectors is also slow (repeated memory allocation).

3. **Memory**: Storing `neighbor_lookup` as a list of 6.46M integer vectors is memory-heavy. Each list element has R object overhead (~128 bytes minimum), so 6.46M elements Ã— 128 bytes â‰ˆ 800 MB just in overhead, plus the actual index data.

**Root cause summary**: The design expands a *cell-level* neighbor graph into a *cell-year-level* lookup (inflating by 28Ã—), using slow R-level string operations and named vector indexing in a loop.

---

## Optimization Strategy

### Key Insight: Separate the spatial and temporal dimensions

The neighbor structure is **purely spatial** â€” it doesn't change across years. There are only 344,208 cells, not 6.46M cell-years. We should:

1. **Build the neighbor lookup at the cell level (344K entries), not the cell-year level (6.46M entries).** The year dimension is handled by a merge/join, not by replicating the graph 28 times.

2. **Replace `lapply` + string keys with `data.table` equi-joins.** Instead of looking up neighbors row-by-row, we "explode" the neighbor list into an edge table `(cell_id, neighbor_id)`, join it to the data on `(neighbor_id, year)`, and compute grouped aggregates â€” all vectorized.

3. **Compute all 5 variables' stats in a single grouped operation** rather than looping over variables with separate passes.

4. **Use `data.table` throughout** for memory-efficient, in-place column addition and fast grouped aggregation.

This reduces the problem from ~6.46M R-level iterations with string operations to a single vectorized join + grouped aggregation, bringing runtime from 86+ hours to **minutes**.

### Why this preserves correctness
- The neighbor relationships are identical (same `rook_neighbors_unique` nb object).
- The statistics computed (max, min, mean of neighbor values per cell-year) are numerically identical.
- No model retraining is needed; we are only producing the same feature columns faster.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build a spatial edge table from the nb object (once)
# ============================================================
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer index vectors
  # id_order maps position -> cell_id
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)

  data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has ~1.37M rows (directed rook edges)

# ============================================================
# STEP 2: Convert cell_data to data.table (if not already)
# ============================================================
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns exist and are properly typed
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ============================================================
# STEP 3: Compute all neighbor features in one vectorized pass
# ============================================================
compute_all_neighbor_features <- function(cell_data, edge_dt, source_vars) {
  # Subset to only the columns we need for the join
  join_cols <- c("id", "year", source_vars)
  neighbor_data <- cell_data[, ..join_cols]

  # Rename 'id' to 'neighbor_id' for joining
  setnames(neighbor_data, "id", "neighbor_id")

  # Join: for each edge (cell_id, neighbor_id), attach the neighbor's

  # year-specific variable values.
  # First, add year to edges by cross-joining with cell_data's (id, year).
  # Actually, more efficient: join edge_dt to neighbor_data on neighbor_id & year.

  # We need (cell_id, year, neighbor's values).
  # Strategy:
  #   1. Take cell_data's (id, year) as the "anchor".
  #   2. For each (id, year), find neighbors via edge_dt.
  #   3. Look up neighbor values from cell_data.

  # Efficient approach: merge edge_dt with neighbor_data on neighbor_id,
  # which gives (cell_id, neighbor_id, year, var1, var2, ...).
  # Then group by (cell_id, year) to get stats.

  setkey(neighbor_data, neighbor_id, year)
  setkey(edge_dt, neighbor_id)

  # This join replicates each edge across all 28 years via the neighbor_data rows.
  # Result: ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows (manageable).
  merged <- edge_dt[neighbor_data, on = "neighbor_id", allow.cartesian = TRUE, nomatch = NULL]
  # merged columns: cell_id, neighbor_id, year, <source_vars>

  # Group by (cell_id, year) and compute max, min, mean for each variable
  agg_exprs <- list()
  for (v in source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]]  <- bquote(
      as.numeric(max(.(v_sym), na.rm = TRUE))
    )
    agg_exprs[[paste0("neighbor_min_", v)]]  <- bquote(
      as.numeric(min(.(v_sym), na.rm = TRUE))
    )
    agg_exprs[[paste0("neighbor_mean_", v)]] <- bquote(
      mean(.(v_sym), na.rm = TRUE)
    )
  }

  # Build the aggregation call
  agg_list <- as.call(c(as.name("list"), agg_exprs))
  stats_dt <- merged[, eval(agg_list), by = .(cell_id, year)]

  # Replace -Inf/Inf from max/min of all-NA groups with NA
  stat_cols <- setdiff(names(stats_dt), c("cell_id", "year"))
  for (col in stat_cols) {
    set(stats_dt, which(is.infinite(stats_dt[[col]])), col, NA_real_)
  }

  return(stats_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

stats_dt <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# ============================================================
# STEP 4: Join the neighbor features back to cell_data
# ============================================================
# Merge on (id = cell_id, year)
setkey(stats_dt, cell_id, year)
setkey(cell_data, id, year)

cell_data <- stats_dt[cell_data, on = .(cell_id = id, year = year)]

# Restore the 'id' column name (the join puts cell_id as the key)
setnames(cell_data, "cell_id", "id")

# ============================================================
# STEP 5: Handle cells with no neighbors (islands / boundary)
# ============================================================
# Cells not present in edge_dt will have NA for all neighbor stats,
# which matches the original code's behavior (return c(NA, NA, NA)).
# No additional action needed.

# ============================================================
# STEP 6: Predict with the existing trained Random Forest
# ============================================================
# The trained RF model is unchanged. Use it directly:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Expected Performance

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level iterations with string ops | Eliminated; replaced by 1.37M-row edge table built in <1 sec |
| Neighbor stats computation | 5 passes Ã— 6.46M `lapply` iterations | Single vectorized `data.table` grouped aggregation on ~38.4M rows |
| Peak memory for lookup | ~800 MB+ (list overhead) | ~600 MB (flat edge table + merged table, then freed) |
| Estimated runtime | 86+ hours | **5â€“15 minutes** on a 16 GB laptop |
| Numerical results | Identical | Identical (same max/min/mean over same neighbor sets) |
| Trained RF model | Preserved | Preserved (no retraining) |

The key transformation is moving from a **row-level R loop with string-key lookups** to a **vectorized relational join and grouped aggregation** â€” the canonical `data.table` pattern for spatial lag features.
END_RESPONSE>>>
