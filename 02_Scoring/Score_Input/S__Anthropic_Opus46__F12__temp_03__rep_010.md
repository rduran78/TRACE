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
run_id: Anthropic_F12_T03_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized bind operation per variable (only 5 calls total). Each inner function returns a fixed-length-3 numeric vector, so there is no "repeated list binding" inside `compute_neighbor_stats()` â€” it's a straightforward `lapply` returning pre-allocated length-3 vectors.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** Inside the `lapply`, for every single row `i`, the code:
   - Calls `as.character(data$id[i])` â€” 6.46M character coercions.
   - Looks up `id_to_ref[as.character(...)]` â€” 6.46M named-vector hash lookups against a 344K-element named vector.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” for each row, pastes ~4 neighbor IDs (rook neighbors average ~4) with the year, creating ~25.8M paste operations total.
   - Looks up `idx_lookup[neighbor_keys]` â€” 25.8M named-vector lookups against a **6.46M-element** named vector. Named vector lookup in R is O(n) per probe in the worst case and uses hashing that becomes very slow at this scale.

2. **The `lapply` over 6.46M rows is inherently serial in R**, and each iteration does multiple allocations, string operations, and hash lookups. This is the dominant cost â€” likely 85+ of the 86+ hours.

3. `compute_neighbor_stats()` by contrast does only numeric subsetting (`vals[idx]`) and three simple aggregates per row. Even with 6.46M iterations Ã— 5 variables, this is comparatively fast because it avoids string operations entirely.

**Root cause summary:** The pipeline builds 6.46 million per-row neighbor index lists using expensive per-row string concatenation and named-vector lookups against a 6.46M-key lookup table. This is an O(N Ã— k Ã— lookup_cost) operation where N = 6.46M, k â‰ˆ 4, and lookup_cost is high due to R's named-vector hashing at scale.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()` entirely** â€” eliminate the per-row `lapply`. Instead, expand the neighbor relationships into a flat edge table (cell-year-row â†’ neighbor-cell-year-row) using `data.table` joins, which use radix-based indexing rather than string hashing.

2. **Vectorize `compute_neighbor_stats()`** â€” use the flat edge table with `data.table` grouped aggregation (`[, .(max, min, mean), by = row_id]`) to compute all neighbor stats in one pass per variable, eliminating both the `lapply` and `do.call(rbind, ...)`.

3. **Preserve the trained Random Forest model** â€” we only change the feature-engineering pipeline, producing numerically identical columns. The RF model object is untouched.

4. **Preserve the original numerical estimand** â€” the optimized code computes the same `max`, `min`, and `mean` of non-NA neighbor values, yielding identical results.

Expected speedup: from 86+ hours to **minutes** (the dominant cost becomes a few `data.table` equi-joins on integer keys over ~26M edge rows).

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED build_neighbor_lookup (returns a data.table edge list, not a list)
# ==============================================================================
build_neighbor_edge_table <- function(data, id_order, neighbors) {
  # data must have columns: id, year (and be ordered as the original data.frame)
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  dt <- as.data.table(data[, c("id", "year")])
  dt[, row_idx := .I]  # preserve original row position

  # --- Step 1: Build a flat neighbor table at the cell level ---------------
  #   For each cell index j in id_order, expand its rook neighbors.
  nb_from <- rep(seq_along(neighbors), lengths(neighbors))
  nb_to   <- unlist(neighbors, use.names = FALSE)

  # Map cell indices back to cell IDs
  cell_edges <- data.table(
    from_id = id_order[nb_from],
    to_id   = id_order[nb_to]
  )
  # ~1.37M rows (directed rook-neighbor relationships)

  # --- Step 2: Cross with years to get row-level edges ---------------------
  #   For every (from_id, year) row in the data, find the row index of each
  #   (to_id, year) neighbor row.

  # Create keyed lookup: (id, year) -> row_idx
  setkey(dt, id, year)

  # Expand cell_edges Ã— years via join on the "from" side to get the focal row_idx
  #   and on the "to" side to get the neighbor row_idx.

  # Join from side: attach focal row index
  edges <- cell_edges[dt, on = .(from_id = id), allow.cartesian = TRUE,
                      nomatch = NULL,
                      .(focal_row = i.row_idx,
                        to_id     = x.to_id,
                        year      = i.year)]

  # Join to side: attach neighbor row index
  edges <- dt[edges, on = .(id = to_id, year = year), nomatch = NA,
              .(focal_row    = i.focal_row,
                neighbor_row = x.row_idx)]

  # Drop edges where the neighbor cell-year doesn't exist in the data
  edges <- edges[!is.na(neighbor_row)]

  # Key for fast grouped operations later
  setkey(edges, focal_row)

  return(edges)
}

# ==============================================================================
# OPTIMIZED compute_neighbor_stats (vectorized via data.table grouping)
# ==============================================================================
compute_neighbor_stats_dt <- function(data_dt, edges, var_name) {
  # data_dt: data.table with a row_idx column (1:nrow)
  # edges:   data.table with (focal_row, neighbor_row)
  # var_name: character, column name to aggregate

  vals <- data_dt[[var_name]]

  # Attach neighbor values
  edge_vals <- edges[, .(focal_row, nval = vals[neighbor_row])]

  # Drop NAs in the variable (matches original: neighbor_vals[!is.na(...)])
  edge_vals <- edge_vals[!is.na(nval)]

  # Grouped aggregation
  stats <- edge_vals[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row]

  # Allocate full-length result (NA for rows with no valid neighbors)
  n <- nrow(data_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  out_max[stats$focal_row]  <- stats$nb_max
  out_min[stats$focal_row]  <- stats$nb_min
  out_mean[stats$focal_row] <- stats$nb_mean

  list(nb_max = out_max, nb_min = out_min, nb_mean = out_mean)
}

# ==============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# ==============================================================================
compute_and_add_neighbor_features_dt <- function(data_dt, var_name, edges) {
  stats <- compute_neighbor_stats_dt(data_dt, edges, var_name)

  # Column names must match what the trained RF model expects.
  # Typical naming convention: <var>_nb_max, <var>_nb_min, <var>_nb_mean
  # Adjust suffixes below if the original code used different names.
  set(data_dt, j = paste0(var_name, "_nb_max"),  value = stats$nb_max)
  set(data_dt, j = paste0(var_name, "_nb_min"),  value = stats$nb_min)
  set(data_dt, j = paste0(var_name, "_nb_mean"), value = stats$nb_mean)

  invisible(data_dt)
}

# ==============================================================================
# MAIN PIPELINE (drop-in replacement for the outer loop)
# ==============================================================================

# Convert to data.table (in-place if already); add row index
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]

# Build the edge table ONCE (replaces build_neighbor_lookup)
# This is the expensive step, but now vectorized: ~minutes, not days.
edges <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  compute_and_add_neighbor_features_dt(cell_dt, var_name, edges)
}

# Convert back to data.frame if downstream code / the RF predict() expects one
cell_data <- as.data.frame(cell_dt)
cell_data$row_idx <- NULL

# The trained Random Forest model is untouched.
# Predictions proceed exactly as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Colleague's Diagnosis | Correct Diagnosis |
|---|---|---|
| Claimed bottleneck | `do.call(rbind, ...)` and list binding in `compute_neighbor_stats()` | `build_neighbor_lookup()`: per-row `lapply` with string pasting and named-vector lookups over 6.46M rows against a 6.46M-key hash |
| Actual cost of claimed bottleneck | Minor (~5 calls, each binding 6.46M Ã— 3 matrix) | N/A |
| True dominant cost | â€” | ~25.8M string-paste + named-vector hash probes inside a serial `lapply` |
| Fix | â€” | Replace both functions with vectorized `data.table` joins and grouped aggregation |
| Expected runtime | â€” | Minutes instead of 86+ hours |
| RF model impact | â€” | None; numerically identical features produced |
END_RESPONSE>>>
