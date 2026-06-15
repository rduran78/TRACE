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
run_id: Anthropic_F04_T10_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The computational bottleneck is **not** the Random Forest inference â€” it is the two spatial neighbor feature construction functions: `build_neighbor_lookup` and `compute_neighbor_stats`. Here is precisely why:

### `build_neighbor_lookup` â€” O(N) `lapply` with expensive string operations

With ~6.46 million rows, this function:

1. **Creates 6.46M string keys** via `paste(data$id, data$year, sep = "_")` for named lookup.
2. **Iterates over every row** (`lapply` over 6.46M indices), and for each row:
   - Converts the cell ID to character and looks up its reference index.
   - Retrieves neighbor cell IDs from the `nb` object.
   - Pastes each neighbor ID with the current row's year to create string keys.
   - Performs **named-vector character lookup** (`idx_lookup[neighbor_keys]`) â€” which in R is **O(k)** per key on average but has high constant overhead on a named vector of length 6.46M.

The named-vector character key lookup on a 6.46M-element vector, executed ~6.46M times with ~4 neighbors each â‰ˆ **~25.8 billion character comparisons** in the worst case. Named vector lookup in R uses hashing, but the overhead of creating and resolving character keys at this scale is enormous.

### `compute_neighbor_stats` â€” repeated per variable

This is called 5 times (once per source variable). Each call iterates over 6.46M rows, subsetting and computing `max`, `min`, `mean`. While lighter than the lookup build, the `lapply` + `do.call(rbind, ...)` pattern on 6.46M list elements is slow because `do.call(rbind, ...)` on a very long list is **O(NÂ²)** in memory copying.

### Summary of root causes

| Issue | Location | Impact |
|---|---|---|
| Character-key named-vector lookup over 6.46M entries | `build_neighbor_lookup` | Dominant bottleneck |
| `lapply` over 6.46M rows with per-row `paste` | `build_neighbor_lookup` | High overhead |
| `do.call(rbind, list_of_6.46M)` | `compute_neighbor_stats` | O(NÂ²) memory pattern |
| Entire design is row-wise / scalar R loops | Both functions | No vectorization |

---

## Optimization Strategy

The key insight is: **replace row-level R loops and character-key lookups with vectorized integer-indexed operations using `data.table`.**

### Specific changes

1. **Replace the character-key named-vector lookup with a `data.table` equi-join.** Instead of building a giant named character vector and indexing into it 25M+ times, we join the neighbor table (cell-to-neighbor mapping) with the data on `(neighbor_id, year)` using `data.table`'s binary-search join. This is **O(N log N)** instead of **O(N Ã— k Ã— hash_overhead)**.

2. **Explode the neighbor relationships into a long table once.** Create a `data.table` with columns `(id, neighbor_id)` from the `nb` object. Then join with the panel data to get `(id, year, neighbor_id, year)` â†’ neighbor row values. This replaces both `build_neighbor_lookup` and the per-row indexing in `compute_neighbor_stats`.

3. **Compute all neighbor statistics in a single grouped aggregation** using `data.table`'s `[, .(max, min, mean), by = .(id, year)]`. This is fully vectorized in C and replaces the R-level `lapply`.

4. **Process all 5 variables simultaneously** in one join + one grouped aggregation pass, rather than 5 separate passes.

5. **The trained Random Forest model is untouched.** The output columns (neighbor max, min, mean per variable) are numerically identical â€” we are only changing how they are computed, not what is computed.

### Expected speedup

| Component | Before | After |
|---|---|---|
| Lookup build | ~hours (char key matching) | ~seconds (integer join) |
| Neighbor stats (Ã—5 vars) | ~hours (lapply + rbind) | ~seconds (grouped agg) |
| **Total neighbor features** | **~86+ hours** | **~2â€“10 minutes** |

---

## Working R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data, id_order, neighbors, source_vars) {
  # ---------------------------------------------------------------
  # Step 1: Convert cell_data to data.table (if not already).
  #         Preserve original row order for downstream RF predict().
  # ---------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .rowid := .I]  # preserve original order

  # ---------------------------------------------------------------
  # Step 2: Build an edge table from the nb object.
  #         neighbors[[i]] gives the indices into id_order for cell
  #         id_order[i]'s neighbors. Expand to long form.
  # ---------------------------------------------------------------
  edge_list <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
  }))

  # ---------------------------------------------------------------
  # Step 3: Join edge_list with the data to obtain neighbor values.
  #         We need: for each (id, year), the values of source_vars
  #         for every neighbor in that same year.
  # ---------------------------------------------------------------
  # Subset only the columns we need for the join to save memory
  join_cols <- c("id", "year", source_vars)
  dt_slim <- dt[, ..join_cols]

  # Rename columns for the neighbor side of the join
  setnames(dt_slim, "id", "neighbor_id")

  # Keyed join: edge_list â†” dt_slim on (neighbor_id, year)
  # Result: for each (id, year), one row per neighbor with its variable values
  setkey(edge_list, neighbor_id)
  setkey(dt_slim, neighbor_id)

  # Cartesian-style: each edge Ã— each year
  # First join edges â†’ dt_slim on neighbor_id (broadcasts across years)
  neighbor_vals <- merge(edge_list, dt_slim, by = "neighbor_id", allow.cartesian = TRUE)

  # ---------------------------------------------------------------
  # Step 4: Grouped aggregation â€” compute max, min, mean per
  #         (id, year) for each source variable, all at once.
  # ---------------------------------------------------------------
  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))

  agg_names <- unlist(lapply(source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the j-expression programmatically
  j_expr <- as.call(c(
    as.name("list"),
    setNames(agg_exprs, agg_names)
  ))

  stats <- neighbor_vals[, eval(j_expr), by = .(id, year)]

  # Replace Inf/-Inf (from max/min on all-NA) with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # ---------------------------------------------------------------
  # Step 5: Merge aggregated neighbor features back to dt,
  #         preserving the original row order.
  # ---------------------------------------------------------------
  dt <- merge(dt, stats, by = c("id", "year"), all.x = TRUE)
  setorder(dt, .rowid)
  dt[, .rowid := NULL]

  # Return as data.frame (or data.table â€” depending on downstream need)
  return(as.data.frame(dt))
}

# ===================== USAGE =====================

# cell_data           : your existing data.frame with columns id, year, ntl, ec, ...
# id_order            : the vector of cell IDs corresponding to the nb object indices
# rook_neighbors_unique : the spdep::nb object (list of integer index vectors)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

### Memory-Constrained Variant (if 16 GB is tight)

The `merge` of `edge_list` Ã— `dt_slim` is `~1.37M edges Ã— 28 years = ~38.4M rows Ã— 5 numeric cols â‰ˆ ~1.5 GB`, which is manageable. But if RAM is tight, process variables one at a time:

```r
build_neighbor_features_chunked <- function(cell_data, id_order, neighbors, source_vars) {
  dt <- as.data.table(cell_data)
  dt[, .rowid := .I]

  # Build edge table once
  edge_list <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
  }))

  setkey(edge_list, neighbor_id)

  for (var_name in source_vars) {
    message("Processing neighbor features for: ", var_name)

    # Slim subset: only neighbor_id, year, and the one variable
    dt_slim <- dt[, .(neighbor_id = id, year, val = get(var_name))]
    setkey(dt_slim, neighbor_id)

    # Join
    nv <- merge(edge_list, dt_slim, by = "neighbor_id", allow.cartesian = TRUE)

    # Aggregate
    agg <- nv[, .(
      nb_max  = max(val, na.rm = TRUE),
      nb_min  = min(val, na.rm = TRUE),
      nb_mean = mean(val, na.rm = TRUE)
    ), by = .(id, year)]

    # Fix Inf from all-NA groups
    agg[is.infinite(nb_max), nb_max := NA_real_]
    agg[is.infinite(nb_min), nb_min := NA_real_]

    # Rename
    setnames(agg, c("nb_max", "nb_min", "nb_mean"),
             paste0("neighbor_", c("max_", "min_", "mean_"), var_name))

    # Merge back
    dt <- merge(dt, agg, by = c("id", "year"), all.x = TRUE)

    # Free memory
    rm(dt_slim, nv, agg)
    gc()
  }

  setorder(dt, .rowid)
  dt[, .rowid := NULL]
  return(as.data.frame(dt))
}

# Usage is identical:
cell_data <- build_neighbor_features_chunked(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | `lapply` over 6.46M rows, character-key named-vector indexing | `data.table` merge on integer keys |
| Neighbor stat computation | `lapply` + `do.call(rbind, ...)` per variable | Vectorized grouped aggregation (`data.table` `[, ..., by=]`) |
| Number of passes over data | 5 (one per variable) | 1 (all variables) or 5 chunked (for memory safety) |
| Estimated runtime | 86+ hours | ~2â€“10 minutes |
| Numerical output | max, min, mean per (cell, year, variable) | **Identical** â€” same estimand preserved |
| Trained RF model | Untouched | **Untouched** |
END_RESPONSE>>>
