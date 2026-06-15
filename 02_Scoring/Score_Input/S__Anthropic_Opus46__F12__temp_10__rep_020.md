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
run_id: Anthropic_F12_T10_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized bind operation per variable (only 5 calls total). This is a minor cost.

**The true bottleneck is `build_neighbor_lookup()`.**

Here is the evidence from the code:

1. **`paste()` and named-vector lookup over 6.46M rows**: `idx_lookup` is a named vector of length ~6.46M. For every single row `i`, the function does `idx_lookup[neighbor_keys]` â€” a character-based name lookup into a 6.46M-length named vector. R's named vector lookup is O(n) per query in the worst case (hash-based, but with collision overhead and repeated hashing of long key strings). This is called **6.46 million times**, each time for multiple neighbors.

2. **`paste()` inside the per-row `lapply`**: For each of the 6.46M rows, `paste(neighbor_cell_ids, data$year[i], sep = "_")` constructs character keys. That's ~6.46M calls to `paste()` with variable-length output, creating enormous garbage-collection pressure.

3. **`as.character()` and `id_to_ref[]` per row**: Another named-vector lookup per row, 6.46 million times.

4. **The `lapply` over 6.46M rows is inherently serial in R** â€” each iteration does string construction, named lookup, and subsetting. The total number of key lookups is on the order of the number of directed neighbor-year pairs: ~1.37M neighbors Ã— 28 years â‰ˆ **38.5 million** string-match lookups into a 6.46M-entry named vector.

In contrast, `compute_neighbor_stats()` does simple numeric indexing (`vals[idx]`) which is O(1) per element, and the `do.call(rbind, result)` on a list of 6.46M length-3 vectors is a single operation taking seconds at most.

**Conclusion**: The bottleneck is the O(tens-of-millions) character-key construction and named-vector lookup in `build_neighbor_lookup()`, not the `rbind` in `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Replace character-key lookups with integer arithmetic.** Instead of `paste(id, year, sep="_")` â†’ named vector lookup, build a direct integer index. If we map each `(id, year)` pair to a row number using a fast integer-keyed hash (via `data.table`) or direct arithmetic indexing, we eliminate all string operations.

2. **Vectorize `build_neighbor_lookup()`** â€” eliminate the per-row `lapply` by expanding the neighbor relationships into a full edge table, joining with year in a vectorized/batch manner using `data.table`.

3. **Vectorize `compute_neighbor_stats()`** â€” once we have an edge table mapping each row to its neighbor rows, compute grouped statistics using `data.table` aggregation, eliminating the per-row `lapply` and the `do.call(rbind, ...)` entirely.

4. **Preserve the trained Random Forest model** â€” we only change the feature-engineering pipeline; the resulting columns are numerically identical, so the model remains valid.

Estimated speedup: from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build neighbor lookup as an edge table (vectorized)
# ============================================================
build_neighbor_edge_table <- function(data_dt, id_order, neighbors) {
 # data_dt: a data.table with columns 'id', 'year', and a '.row_idx' column
 #          (.row_idx = seq_len(nrow(data_dt)))
 # id_order: vector of cell IDs in the order matching the nb object
 # neighbors: spdep nb object (list of integer neighbor indices)

 # --- Map each cell ID to its position in id_order ---
 id_to_ref <- data.table(
   id  = id_order,
   ref = seq_along(id_order)
 )

 # --- Expand neighbor list into a directed edge list (cell-level) ---
 # Each element neighbors[[i]] gives the neighbor indices for id_order[i]
 from_ref <- rep(seq_along(neighbors), lengths(neighbors))
 to_ref   <- unlist(neighbors, use.names = FALSE)

 cell_edges <- data.table(
   from_id = id_order[from_ref],
   to_id   = id_order[to_ref]
 )
 # cell_edges now has ~1,373,394 rows (directed rook-neighbor pairs)

 # --- Create a fast (id, year) -> row_idx lookup ---
 # Ensure data_dt has .row_idx
 setkey(data_dt, id, year)
 row_lookup <- data_dt[, .(id, year, .row_idx)]
 setkey(row_lookup, id, year)

 # --- Get all unique years ---
 years <- sort(unique(data_dt$year))

 # --- Cross-join cell_edges with years to get row-level edges ---
 # This creates the full edge table: for every (from_id, year) row,
 # which (to_id, year) rows are its neighbors.
 # Use CJ-like expansion but efficiently:
 edge_year <- CJ_dt_edges(cell_edges, years, row_lookup)

 return(edge_year)
}

CJ_dt_edges <- function(cell_edges, years, row_lookup) {
 # Expand cell_edges Ã— years
 # For memory efficiency, process in chunks if needed, but with ~1.37M edges
 # and 28 years, the result is ~38.5M rows â€” fits in 16GB easily.

 year_dt <- data.table(year = years)
 # Cross join: each cell edge paired with each year
 edge_year <- cell_edges[, .(from_id, to_id)]
 edge_year <- edge_year[rep(seq_len(.N), each = length(years))]
 edge_year[, year := rep(years, times = nrow(cell_edges))]

 # Now join to get row indices for 'from' side
 setkey(row_lookup, id, year)

 # from_row_idx
 edge_year[row_lookup, from_row := i..row_idx, on = .(from_id = id, year = year)]

 # to_row_idx
 edge_year[row_lookup, to_row := i..row_idx, on = .(to_id = id, year = year)]

 # Drop edges where either side has no matching row (boundary/missing)
 edge_year <- edge_year[!is.na(from_row) & !is.na(to_row)]

 # We only need the row-index mapping
 edge_year <- edge_year[, .(from_row, to_row)]

 return(edge_year)
}


# ============================================================
# STEP 2: Compute neighbor stats vectorized via data.table
# ============================================================
compute_neighbor_stats_fast <- function(data_dt, edge_table, var_name) {
 # edge_table has columns: from_row, to_row
 # For each from_row, gather var_name values at all to_row neighbors,
 # compute max, min, mean.

 vals <- data_dt[[var_name]]

 # Build a temporary table with the neighbor values
 stat_dt <- edge_table[, .(neighbor_val = vals[to_row]), by = from_row]

 # Remove NAs in neighbor values
 stat_dt <- stat_dt[!is.na(neighbor_val)]

 # Aggregate
 agg <- stat_dt[, .(
   nb_max  = max(neighbor_val),
   nb_min  = min(neighbor_val),
   nb_mean = mean(neighbor_val)
 ), by = from_row]

 # Prepare output columns â€” initialize with NA for all rows
 n <- nrow(data_dt)
 out_max  <- rep(NA_real_, n)
 out_min  <- rep(NA_real_, n)
 out_mean <- rep(NA_real_, n)

 out_max[agg$from_row]  <- agg$nb_max
 out_min[agg$from_row]  <- agg$nb_min
 out_mean[agg$from_row] <- agg$nb_mean

 # Name columns to match original pipeline output
 max_col  <- paste0(var_name, "_neighbor_max")
 min_col  <- paste0(var_name, "_neighbor_min")
 mean_col <- paste0(var_name, "_neighbor_mean")

 data_dt[, (max_col)  := out_max]
 data_dt[, (min_col)  := out_min]
 data_dt[, (mean_col) := out_mean]

 return(data_dt)
}


# ============================================================
# STEP 3: Full optimized pipeline (drop-in replacement)
# ============================================================
run_optimized_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

 # Convert to data.table if needed (non-destructive to original)
 if (!is.data.table(cell_data)) {
   cell_data <- as.data.table(cell_data)
 }

 # Add row index
 cell_data[, .row_idx := seq_len(.N)]

 message("Building vectorized edge table...")
 t0 <- Sys.time()
 edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
 message("  Edge table: ", format(nrow(edge_table), big.mark = ","), " edges built in ",
         round(difftime(Sys.time(), t0, units = "secs"), 1), "s")

 # Compute neighbor features for each source variable
 neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

 for (var_name in neighbor_source_vars) {
   message("Computing neighbor stats for: ", var_name)
   t1 <- Sys.time()
   cell_data <- compute_neighbor_stats_fast(cell_data, edge_table, var_name)
   message("  Done in ", round(difftime(Sys.time(), t1, units = "secs"), 1), "s")
 }

 # Clean up helper column
 cell_data[, .row_idx := NULL]

 message("Neighbor feature pipeline complete.")
 return(cell_data)
}


# ============================================================
# USAGE (preserves trained RF model â€” no retraining needed)
# ============================================================
# cell_data <- run_optimized_neighbor_pipeline(cell_data, id_order, rook_neighbors_unique)
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| **Alleged bottleneck** | `do.call(rbind, ...)` and list binding in `compute_neighbor_stats()` | **Wrong.** The `rbind` of 6.46M length-3 vectors takes seconds. |
| **True bottleneck** | â€” | `build_neighbor_lookup()`: 6.46M iterations of `paste()` + character-key named-vector lookup into a 6.46M-entry vector (~38.5M string-match lookups total). |
| **Optimization** | â€” | Replace all character-key operations with a vectorized integer edge table built via `data.table` joins; replace per-row `lapply` aggregation with `data.table` grouped aggregation. |
| **Expected speedup** | â€” | From **86+ hours â†’ minutes** (primarily by eliminating ~38.5M string constructions and hash lookups). |
| **Model preservation** | â€” | Numerical outputs are identical; the trained Random Forest is used as-is with no retraining. |
END_RESPONSE>>>
