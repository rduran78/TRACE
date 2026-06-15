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
run_id: Anthropic_F04_T10_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Looks up the cell's reference index via character conversion and named-vector lookup.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Pastes neighbor IDs with the current row's year to form string keys.
4. Matches those keys against a named vector (`idx_lookup`) of ~6.46M entries.

**String operations (`paste`, named-vector character matching) on 6.46M rows Ã— ~4 neighbors each â‰ˆ 25.8M string constructions and hash lookups.** Named vector lookup in R is O(n) in the worst case per query due to hashing overhead at scale. The `lapply` is also not vectorized â€” it's a pure R loop with per-element allocations.

`compute_neighbor_stats` then loops over the 6.46M-element lookup list again for each of 5 variables (32.3M list iterations), calling `max`/`min`/`mean` individually.

**Estimated cost:** ~6.46M iterations Ã— (string paste + hash lookup) Ã— overhead â‰ˆ 86+ hours on a laptop.

## Optimization Strategy

1. **Replace string-key lookups with integer-arithmetic indexing.** Since `year` is contiguous (1992â€“2019, 28 values), encode each `(cell_id, year)` pair as a unique integer and use `match()` or direct array indexing instead of named character vectors.

2. **Vectorize `build_neighbor_lookup` using `data.table` joins** â€” expand all neighbor relationships into a flat edge table, join on `(neighbor_id, year)` to get row indices, then split by source row. This replaces 6.46M R-level iterations with a single vectorized join.

3. **Vectorize `compute_neighbor_stats` using `data.table` grouped aggregation** â€” instead of `lapply` over a list, perform grouped `max`/`min`/`mean` in one pass per variable.

4. **Compute all 5 variables' neighbor stats in a single grouped pass** to avoid 5 separate iterations.

## Optimized Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build neighbor lookup as a flat edge table (vectorized)
# ============================================================
build_neighbor_edges <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns id, year, and a row index
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors)

  # Map each cell ID to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Expand nb object into a flat edge list: (source_ref, neighbor_ref)
  # This is ~1.37M directed edges
  src_refs <- rep(seq_along(neighbors), lengths(neighbors))
  nbr_refs <- unlist(neighbors, use.names = FALSE)

  edge_dt <- data.table(
    src_cell_id = id_order[src_refs],
    nbr_cell_id = id_order[nbr_refs]
  )

  # Create a row-index lookup: (id, year) -> row_index in data_dt
  data_dt[, row_idx := .I]

  # Cross-join edges with all years present in the data
  years <- sort(unique(data_dt$year))

  # Expand edges Ã— years: each spatial edge exists for every year
  # ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” fits in 16 GB RAM
  edge_year <- edge_dt[, CJ(edge_row = seq_len(.N), year = years)]
  edge_year[, `:=`(
    src_cell_id = edge_dt$src_cell_id[edge_row],
    nbr_cell_id = edge_dt$nbr_cell_id[edge_row]
  )]
  edge_year[, edge_row := NULL]

  # Join to get source row index
  setkey(data_dt, id, year)
  edge_year <- merge(
    edge_year,
    data_dt[, .(id, year, src_row_idx = row_idx)],
    by.x = c("src_cell_id", "year"),
    by.y = c("id", "year"),
    all.x = FALSE
  )

 # Join to get neighbor row index
  edge_year <- merge(
    edge_year,
    data_dt[, .(id, year, nbr_row_idx = row_idx)],
    by.x = c("nbr_cell_id", "year"),
    by.y = c("id", "year"),
    all.x = FALSE
  )

  return(edge_year)
}

# ============================================================
# STEP 2: Compute all neighbor stats in one vectorized pass
# ============================================================
compute_all_neighbor_stats <- function(data_dt, edge_year, var_names) {
  # Attach neighbor variable values to the edge table
  # We pull only needed columns to keep memory in check

  nbr_vals <- data_dt[edge_year$nbr_row_idx, ..var_names]
  work <- data.table(
    src_row_idx = edge_year$src_row_idx
  )
  work <- cbind(work, nbr_vals)

  # Grouped aggregation: max, min, mean per source row, for all vars at once
  agg_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(var_names, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  stats <- work[, lapply(agg_exprs, eval, envir = .SD), by = src_row_idx]

  # Replace -Inf/Inf (from max/min of empty after na.rm) with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  return(stats)
}

# ============================================================
# STEP 3: Main execution â€” drop-in replacement for outer loop
# ============================================================
optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {

  cell_dt <- as.data.table(cell_data)
  cell_dt[, row_idx := .I]

  message("Building vectorized neighbor edge table...")
  t0 <- Sys.time()

  # --- memory-efficient edge expansion (no CJ of full edge table) ---
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  src_refs  <- rep(seq_along(rook_neighbors_unique),
                   lengths(rook_neighbors_unique))
  nbr_refs  <- unlist(rook_neighbors_unique, use.names = FALSE)

  edge_dt <- data.table(
    src_cell_id = id_order[src_refs],
    nbr_cell_id = id_order[nbr_refs]
  )

  # Key the data for fast binary-search joins
  setkey(cell_dt, id, year)

  # Instead of expanding all edges Ã— all years at once (38.5M rows),
  # we join edges against existing (id, year) pairs directly.

  # For each edge, the valid years are those where BOTH src and nbr exist.
  # Since this is a balanced panel (344,208 cells Ã— 28 years), every cell
  # appears in every year. We can expand safely.

  years <- sort(unique(cell_dt$year))
  n_edges <- nrow(edge_dt)
  n_years <- length(years)

  # Expand: repeat each edge for each year
  edge_year <- data.table(
    src_cell_id = rep(edge_dt$src_cell_id, each = n_years),
    nbr_cell_id = rep(edge_dt$nbr_cell_id, each = n_years),
    year        = rep(years, times = n_edges)
  )

  message(sprintf("  Edge-year table: %s rows (%.1f MB)",
                  format(nrow(edge_year), big.mark = ","),
                  object.size(edge_year) / 1e6))

  # Join to get src and nbr row indices
  src_lookup <- cell_dt[, .(src_cell_id = id, year, src_row_idx = row_idx)]
  setkey(src_lookup, src_cell_id, year)
  edge_year <- src_lookup[edge_year, on = .(src_cell_id, year), nomatch = 0L]

  nbr_lookup <- cell_dt[, .(nbr_cell_id = id, year, nbr_row_idx = row_idx)]
  setkey(nbr_lookup, nbr_cell_id, year)
  edge_year <- nbr_lookup[edge_year, on = .(nbr_cell_id, year), nomatch = 0L]

  message(sprintf("  Edge table built in %.1f seconds.", difftime(Sys.time(), t0, units = "secs")))

  # --- Compute neighbor statistics for all variables at once ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  message("Computing neighbor statistics (vectorized)...")
  t1 <- Sys.time()

  # Pull neighbor values for all vars at once
  nbr_vals <- cell_dt[edge_year$nbr_row_idx, ..neighbor_source_vars]
  nbr_vals[, src_row_idx := edge_year$src_row_idx]

  # Build aggregation expressions programmatically
  agg_list <- list()
  for (v in neighbor_source_vars) {
    agg_list[[paste0("neighbor_max_", v)]]  <-
      substitute(fifelse(all(is.na(x)), NA_real_, max(x, na.rm = TRUE)), list(x = as.name(v)))
    agg_list[[paste0("neighbor_min_", v)]]  <-
      substitute(fifelse(all(is.na(x)), NA_real_, min(x, na.rm = TRUE)), list(x = as.name(v)))
    agg_list[[paste0("neighbor_mean_", v)]] <-
      substitute(mean(x, na.rm = TRUE), list(x = as.name(v)))
  }

  stats <- nbr_vals[, lapply(agg_list, eval), by = src_row_idx]

  message(sprintf("  Stats computed in %.1f seconds.", difftime(Sys.time(), t1, units = "secs")))

  # --- Merge back into cell_dt by row index ---
  # Rows with no neighbors won't appear in stats; those get NA (correct).
  new_cols <- setdiff(names(stats), "src_row_idx")

  # Remove any pre-existing neighbor columns to avoid conflicts
  for (col in intersect(new_cols, names(cell_dt))) {
    cell_dt[, (col) := NULL]
  }

  cell_dt <- merge(cell_dt, stats, by.x = "row_idx", by.y = "src_row_idx", all.x = TRUE)

  # Restore original row order
  setorder(cell_dt, row_idx)
  cell_dt[, row_idx := NULL]

  # Convert back to data.frame to preserve downstream compatibility
  cell_data <- as.data.frame(cell_dt)

  message("Done. Neighbor features added.")
  return(cell_data)
}

# ============================================================
# Usage (drop-in replacement for the original outer loop):
# ============================================================
# cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique)
#
# Then proceed with predict(trained_rf_model, cell_data) as before.
# The trained Random Forest model is untouched.
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M R-level iterations with `paste`+named-vector lookup | Single vectorized `data.table` keyed join |
| **Stat computation** | 5 variables Ã— 6.46M `lapply` calls (32.3M R iterations) | One grouped `data.table` aggregation pass |
| **String operations** | ~25.8M `paste()` calls + character hash lookups | Zero â€” integer keys only |
| **Estimated time** | 86+ hours | **~2â€“10 minutes** on same laptop |
| **Peak RAM** | Moderate (list of 6.46M vectors) | ~4â€“6 GB (38.5M-row edge table + aggregation) |
| **Numerical output** | max / min / mean per neighbor set | Identical max / min / mean (preserves estimand) |
| **RF model** | â€” | Untouched; no retraining |

The core insight: **replace 6.46M sequential R-level iterations (each doing string manipulation and hash lookups) with a single flat join-and-group operation on integer keys using `data.table`'s radix-based binary search.** This converts an O(NÃ—k) interpreted-R loop into a compiled-C vectorized operation, yielding roughly a 500â€“2500Ã— speedup.
END_RESPONSE>>>
