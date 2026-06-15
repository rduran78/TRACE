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
run_id: Anthropic_F12_T03_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized operation that binds pre-allocated 3-element vectors â€” this is relatively efficient and takes seconds, not hours.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **`paste()` and named-vector lookup executed ~6.46 million times inside `lapply`**: For each of the ~6.46 million rows, the function calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` to construct string keys, then does named-vector indexing into `idx_lookup` (a named vector of ~6.46 million elements). Named-vector lookup in R uses linear hashing that degrades at scale, and constructing millions of temporary paste strings is expensive.

2. **Per-row character coercion**: `as.character(data$id[i])` is called 6.46 million times individually rather than once vectorially.

3. **The neighbor lookup is built once but costs O(N Ã— k) string operations** where N â‰ˆ 6.46M and k â‰ˆ average number of neighbors (~4 for rook contiguity). That's ~25.8 million `paste` + hash-lookup operations against a 6.46M-entry named vector.

4. **`compute_neighbor_stats()` by contrast** does only integer indexing (`vals[idx]`) and simple numeric operations â€” these are fast. Even `do.call(rbind, result)` on 6.46M three-element vectors completes in a few seconds.

**Quantitative estimate**: The `build_neighbor_lookup` step alone, doing ~6.46M iterations of paste + named-vector lookup against a 6.46M-key table, accounts for the vast majority of the 86+ hour runtime. The `compute_neighbor_stats` function (called 5 times) is comparatively negligible.

## Optimization Strategy

1. **Replace string-key lookups with integer arithmetic**: Instead of pasting `id_year` strings and looking them up in a named vector, compute row indices directly using integer math. If data is sorted by `(id, year)` or we build an integer-keyed hash (via `data.table` or `match()`), we eliminate all string operations.

2. **Vectorize `build_neighbor_lookup` entirely**: Expand the neighbor relationships into a full edge list (source_row â†’ neighbor_row) using `data.table` joins, avoiding the per-row `lapply` entirely.

3. **Vectorize `compute_neighbor_stats`**: Use `data.table` grouped aggregation on the edge list to compute max/min/mean in one pass per variable â€” no `lapply`, no `do.call(rbind, ...)`.

4. **Preserve the trained Random Forest model**: We only change feature engineering; the output columns have identical names and identical numerical values (same estimand).

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED PIPELINE â€” replaces build_neighbor_lookup + compute_neighbor_stats
# ==============================================================================

build_neighbor_edge_list <- function(data_dt, id_order, neighbors) {
 # -------------------------------------------------------------------------
 # Instead of per-row string lookups, we:
 #   1. Build a full directed edge list of (source_id, neighbor_id) from the
 #      nb object â€” this is only ~1.37M edges, independent of years.
 #   2. Cross-join with years to get (source_id, year, neighbor_id, year)
 #      â€” but since source and neighbor share the same year, we just join
 #      on (neighbor_id, year) to get the neighbor's row index.
 #   3. Result: an edge list of (source_row, neighbor_row) with ~25.8M edges
 #      built entirely via vectorized data.table joins.
 # -------------------------------------------------------------------------

 # Step 1: Build spatial edge list from nb object
 #   neighbors[[i]] gives the indices (into id_order) of neighbors of id_order[i]
 source_ref <- rep(seq_along(neighbors), lengths(neighbors))
 target_ref <- unlist(neighbors)

 spatial_edges <- data.table(
   source_id = id_order[source_ref],
   neighbor_id = id_order[target_ref]
 )

 # Step 2: Ensure data_dt has a row index
 data_dt[, .row_idx := .I]

 # Step 3: Get all unique years
 years <- unique(data_dt$year)

 # Step 4: Expand spatial edges across all years
 #   CJ-like expansion: each spatial edge exists for every year
 edge_year <- spatial_edges[, .(year = years), by = .(source_id, neighbor_id)]

 # Step 5: Join to get source row index
 setkey(data_dt, id, year)
 setnames(edge_year, "source_id", "id")
 edge_year <- data_dt[edge_year, on = .(id, year), nomatch = 0L,
                       .(source_row = .row_idx,
                         neighbor_id = i.neighbor_id,
                         year = i.year)]

 # Step 6: Join to get neighbor row index
 #   We need to join on neighbor_id = id AND same year
 edge_year[, id := neighbor_id]
 edge_year <- data_dt[edge_year, on = .(id, year), nomatch = 0L,
                       .(source_row = i.source_row,
                         neighbor_row = .row_idx)]

 # Clean up
 data_dt[, .row_idx := NULL]

 return(edge_year)
}


compute_all_neighbor_features <- function(cell_data, id_order, neighbors,
                                          neighbor_source_vars) {
 # -------------------------------------------------------------------------
 # Computes max, min, mean of each variable across rook neighbors,
 # for every cell-year row. Produces identical columns to the original code.
 # -------------------------------------------------------------------------

 data_dt <- as.data.table(cell_data)
 data_dt[, .row_idx := .I]

 cat("Building vectorized edge list...\n")
 t0 <- proc.time()

 # --- Build spatial edge list from nb object (year-independent) ---
 source_ref <- rep(seq_along(neighbors), lengths(neighbors))
 target_ref <- unlist(neighbors)

 spatial_edges <- data.table(
   source_id = id_order[source_ref],
   neighbor_id = id_order[target_ref]
 )

 cat(sprintf("  Spatial edges: %s\n", format(nrow(spatial_edges), big.mark = ",")))

 # --- Map (id, year) -> row_idx via data.table keyed join ---
 row_map <- data_dt[, .(id, year, .row_idx)]
 setkey(row_map, id, year)

 # --- Expand edges across years and resolve row indices ---
 years_vec <- sort(unique(data_dt$year))
 n_years <- length(years_vec)

 # Expand: each spatial edge Ã— each year
 full_edges <- spatial_edges[rep(seq_len(.N), each = n_years)]
 full_edges[, year := rep(years_vec, times = nrow(spatial_edges))]

 cat(sprintf("  Full edges (before join): %s\n",
             format(nrow(full_edges), big.mark = ",")))

 # Join to get source_row
 setnames(full_edges, "source_id", "id")
 full_edges <- row_map[full_edges, on = .(id, year), nomatch = 0L]
 setnames(full_edges, c("id", ".row_idx"), c("source_id", "source_row"))

 # Join to get neighbor_row
 setnames(full_edges, "neighbor_id", "id")
 full_edges <- row_map[full_edges, on = .(id, year), nomatch = 0L]
 setnames(full_edges, c("id", ".row_idx"), c("neighbor_id", "neighbor_row"))

 # Keep only what we need
 edges <- full_edges[, .(source_row, neighbor_row)]

 cat(sprintf("  Resolved edges: %s\n", format(nrow(edges), big.mark = ",")))
 cat(sprintf("  Edge list built in %.1f seconds.\n",
             (proc.time() - t0)[3]))

 # --- Compute neighbor stats for each variable ---
 for (var_name in neighbor_source_vars) {
   cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
   t1 <- proc.time()

   # Attach neighbor values
   edges[, neighbor_val := data_dt[[var_name]][neighbor_row]]

   # Remove NAs in neighbor values for aggregation
   valid_edges <- edges[!is.na(neighbor_val)]

   # Aggregate: max, min, mean grouped by source_row
   agg <- valid_edges[, .(
     nb_max  = max(neighbor_val),
     nb_min  = min(neighbor_val),
     nb_mean = mean(neighbor_val)
   ), by = source_row]

   # Initialize result columns with NA
   max_col  <- paste0("neighbor_max_", var_name)
   min_col  <- paste0("neighbor_min_", var_name)
   mean_col <- paste0("neighbor_mean_", var_name)

   data_dt[, (max_col)  := NA_real_]
   data_dt[, (min_col)  := NA_real_]
   data_dt[, (mean_col) := NA_real_]

   # Fill in computed values
   data_dt[agg$source_row, (max_col)  := agg$nb_max]
   data_dt[agg$source_row, (min_col)  := agg$nb_min]
   data_dt[agg$source_row, (mean_col) := agg$nb_mean]

   cat(sprintf("  Done in %.1f seconds.\n", (proc.time() - t1)[3]))
 }

 # Clean up helper column
 data_dt[, .row_idx := NULL]

 # Return as data.frame to preserve downstream compatibility
 return(as.data.frame(data_dt))
}

# ==============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
 cell_data,
 id_order,
 rook_neighbors_unique,
 neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names and numerical values are identical to the original pipeline.
# Proceed directly to prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Expected Performance Improvement

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` (string paste + named-vector lookup, 6.46M iterations) | **~80+ hours** (dominant bottleneck) | **~30â€“90 seconds** (vectorized `data.table` keyed joins) |
| `compute_neighbor_stats` (5 variables Ã— 6.46M rows) | ~minutes | **~10â€“30 seconds** (grouped `data.table` aggregation) |
| `do.call(rbind, ...)` | ~seconds | Eliminated entirely |
| **Total** | **86+ hours** | **~2â€“5 minutes** |

## Summary

The colleague misidentified the bottleneck. `do.call(rbind, ...)` and the `lapply` in `compute_neighbor_stats` are minor costs. The true bottleneck is `build_neighbor_lookup()`, which performs **6.46 million iterations** of string construction (`paste`) and named-vector hash lookups against a 6.46M-entry table. The fix is to replace all string-keyed lookups with vectorized integer-keyed `data.table` joins, building the full edge list in one pass and computing all neighbor statistics via grouped aggregation.
END_RESPONSE>>>
