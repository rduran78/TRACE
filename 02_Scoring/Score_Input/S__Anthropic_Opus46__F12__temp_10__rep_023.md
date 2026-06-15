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
run_id: Anthropic_F12_T10_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, result)` on a list of 6.46 million small 3-element vectors is not free, it runs in seconds â€” it's binding fixed-width rows (3 columns), and `rbind` on simple numeric vectors is relatively efficient.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **`paste()` and character key creation at massive scale:** `idx_lookup` is built by pasting `data$id` and `data$year` for all ~6.46 million rows. Then, *inside the `lapply` over all 6.46 million rows*, the function calls `paste()` again to build `neighbor_keys` for each row's neighbors. With ~1.37M directed neighbor relationships spread across 6.46M cell-years, the average cell has ~4 neighbors, meaning the inner `paste()` is called ~6.46M times, each producing ~4 strings, and each time performing a named-vector lookup (`idx_lookup[neighbor_keys]`) against a 6.46-million-element named character vector.

2. **Named vector lookup is O(n) per probe in the worst case:** R named vectors use a linear-hashed internal lookup. Looking up keys in a 6.46M-element named vector ~6.46M Ã— 4 times â‰ˆ **25.8 billion character comparisons** in the hashing layer. This is the dominant cost, dwarfing everything in `compute_neighbor_stats()`.

3. **`as.character()` and `id_to_ref` lookups** add further per-row overhead inside the same `lapply`.

4. **`compute_neighbor_stats()`**, by contrast, does only integer indexing (`vals[idx]`) and simple numeric operations â€” this is near-instantaneous by comparison.

**Conclusion:** The bottleneck is the O(N Ã— k) character key construction and named-vector lookup inside `build_neighbor_lookup()`, not the `do.call(rbind, ...)` in `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Replace the character-key named vector with an integer-keyed lookup using a `data.table` hash join or a pre-built integer matrix.** Instead of `paste(id, year)` â†’ character key â†’ named vector probe, create a 2D integer index: a matrix or `data.table` keyed on `(id, year)` that returns the row number in O(1) via `data.table`'s binary-search join.

2. **Vectorize `build_neighbor_lookup()`** by expanding all neighbor relationships into a single `data.table` join operation, eliminating the `lapply` over 6.46M rows entirely.

3. **In `compute_neighbor_stats()`, replace the `lapply` + `do.call(rbind, ...)` with a vectorized grouped aggregation** using `data.table`, computing max/min/mean in one pass.

4. **Preserve the trained Random Forest model** â€” we only change the feature-engineering pipeline, not the model or its input column schema.

5. **Preserve the original numerical estimand** â€” the optimized code computes the identical max, min, and mean of neighbor values.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 1: Build a vectorized neighbor-row mapping (replaces
#         build_neighbor_lookup entirely)
# ---------------------------------------------------------------
build_neighbor_edges <- function(data_dt, id_order, neighbors) {
 # data_dt: a data.table with columns id, year, and a .row_idx column
 # id_order: vector mapping reference index -> cell id
 # neighbors: spdep nb object (list of integer neighbor indices)

 # Map each cell id to its position in id_order
 id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

 # Build a data.table of directed edges: (focal_id, neighbor_id)
 edges <- rbindlist(lapply(seq_along(neighbors), function(ref_idx) {
   nb_ref_indices <- neighbors[[ref_idx]]
   if (length(nb_ref_indices) == 0L) return(NULL)
   data.table(
     focal_id    = id_order[ref_idx],
     neighbor_id = id_order[nb_ref_indices]
   )
 }))
 # edges has ~1.37M rows (one per directed rook-neighbor relationship)

 # Create a keyed lookup: (id, year) -> row index in data_dt
 row_lookup <- data_dt[, .(id, year, focal_row = .row_idx)]
 setkey(row_lookup, id, year)

 # For each (focal_id, year) combination, find the focal row index
 # Expand edges across all 28 years
 years <- sort(unique(data_dt$year))

 # Cross join edges Ã— years to get ~1.37M Ã— 28 â‰ˆ 38.5M rows
 # (only the subset that actually exists in data)
 edge_years <- CJ_dt(edges, years)

 # --- Join to get focal_row (row index of the focal cell-year) ---
 setnames(row_lookup, c("id", "year", "focal_row"))
 setkey(row_lookup, id, year)
 edge_years <- row_lookup[edge_years, on = .(id = focal_id, year), nomatch = 0L]
 # Now edge_years has columns: id, year, focal_row, neighbor_id

 # --- Join to get neighbor_row (row index of the neighbor cell-year) ---
 neighbor_lookup_dt <- data_dt[, .(neighbor_id = id, year, neighbor_row = .row_idx)]
 setkey(neighbor_lookup_dt, neighbor_id, year)
 edge_years <- neighbor_lookup_dt[edge_years,
   on = .(neighbor_id, year), nomatch = 0L]
 # Now edge_years has: neighbor_id, year, neighbor_row, id (focal), focal_row

 edge_years[, .(focal_row, neighbor_row)]
}

# Helper: cross join a data.table of edges with a vector of years
CJ_dt <- function(edges, years) {
 edges[, .row_e := .I]
 yr_dt <- data.table(year = years)
 out <- edges[rep(seq_len(.N), each = length(years))]
 out[, year := rep(years, times = nrow(edges))]
 out[, .row_e := NULL]
 out
}


# ---------------------------------------------------------------
# STEP 2: Vectorized neighbor stats (replaces compute_neighbor_stats)
# ---------------------------------------------------------------
compute_neighbor_stats_vec <- function(data_dt, edge_map, var_name) {
 # edge_map: data.table with (focal_row, neighbor_row)
 # Compute grouped max, min, mean of the neighbor values

 vals <- data_dt[[var_name]]

 # Attach neighbor values
 work <- edge_map[, .(focal_row, nval = vals[neighbor_row])]

 # Drop NAs in neighbor values
 work <- work[!is.na(nval)]

 # Grouped aggregation â€” single vectorized pass
 agg <- work[, .(
   nb_max  = max(nval),
   nb_min  = min(nval),
   nb_mean = mean(nval)
 ), keyby = focal_row]

 # Initialize output columns with NA
 n <- nrow(data_dt)
 out_max  <- rep(NA_real_, n)
 out_min  <- rep(NA_real_, n)
 out_mean <- rep(NA_real_, n)

 out_max[agg$focal_row]  <- agg$nb_max
 out_min[agg$focal_row]  <- agg$nb_min
 out_mean[agg$focal_row] <- agg$nb_mean

 list(nb_max = out_max, nb_min = out_min, nb_mean = out_mean)
}


# ---------------------------------------------------------------
# STEP 3: Main pipeline (replaces the outer loop)
# ---------------------------------------------------------------
run_neighbor_feature_pipeline <- function(cell_data, id_order,
                                          rook_neighbors_unique) {

 # Convert to data.table if needed (no copy if already data.table)
 if (!is.data.table(cell_data)) {
   cell_data <- as.data.table(cell_data)
 }

 # Add row index
 cell_data[, .row_idx := .I]

 # --- STEP 1: build the edge map ONCE (the expensive part, now fast) ---
 message("Building vectorized neighbor edge map...")
 edge_map <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)
 setkey(edge_map, focal_row)
 message(sprintf("Edge map: %s cell-year-neighbor rows", format(nrow(edge_map),
   big.mark = ",")))

 # --- STEP 2: compute neighbor features for each variable ---
 neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

 for (var_name in neighbor_source_vars) {
   message(sprintf("Computing neighbor stats for: %s", var_name))
   stats <- compute_neighbor_stats_vec(cell_data, edge_map, var_name)

   # Add columns with identical naming convention as the original code
   col_max  <- paste0(var_name, "_neighbor_max")
   col_min  <- paste0(var_name, "_neighbor_min")
   col_mean <- paste0(var_name, "_neighbor_mean")

   set(cell_data, j = col_max,  value = stats$nb_max)
   set(cell_data, j = col_min,  value = stats$nb_min)
   set(cell_data, j = col_mean, value = stats$nb_mean)
 }

 # Clean up helper column
 cell_data[, .row_idx := NULL]

 cell_data
}


# ---------------------------------------------------------------
# USAGE (preserves existing trained RF model)
# ---------------------------------------------------------------
# cell_data <- run_neighbor_feature_pipeline(
#   cell_data, id_order, rook_neighbors_unique
# )
#
# # The Random Forest model is untouched; use it for prediction as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup()`: 6.46M `lapply` iterations, each doing `paste()` + named-vector lookup against a 6.46M-element vector | Eliminated entirely |
| **Key construction** | ~25.8B character hash probes | Integer-keyed `data.table` binary-search joins â€” O(n log n) once |
| **Neighbor stats** | `lapply` over 6.46M rows + `do.call(rbind, ...)` | Single vectorized `data.table` grouped aggregation |
| **Edge map** | Rebuilt implicitly every call (was inside lookup) | Built **once**, reused for all 5 variables |
| **Expected runtime** | ~86+ hours | **Minutes** (edge map build ~1-3 min; each variable's stats ~10-30 sec) |
| **Numerical result** | max, min, mean of non-NA neighbor values | **Identical** â€” same max, min, mean logic |
| **RF model** | â€” | **Preserved untouched** |
| **RAM (16 GB)** | Marginal | Edge map ~300 MB + data.table overhead fits comfortably |

The colleague was looking at the wrong function. The real cost was millions of character-key constructions and named-vector lookups in `build_neighbor_lookup()`, not the `rbind` in `compute_neighbor_stats()`.
END_RESPONSE>>>
