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
run_id: Anthropic_F12_T07_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, result)` on ~6.46 million rows is not free, each inner function returns a fixed-length numeric vector of length 3, so `rbind`-ing 6.46M such vectors is a single matrix assembly â€” costly but not catastrophic (order of seconds, not hours).

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row `paste()` and character-keyed lookup for 6.46 million rows**: The `lapply` inside `build_neighbor_lookup` iterates over every row (~6.46M), and for each row it:
   - Converts `data$id[i]` to character and looks it up in a named vector (`id_to_ref`).
   - Extracts neighbor cell IDs, then calls `paste()` to build composite keys.
   - Performs named-vector lookup (`idx_lookup[neighbor_keys]`) against a named vector of length 6.46M.

2. **Named vector lookup is O(n) per probe in R**: R's named vectors use linear hashing that degrades with size. With ~6.46M names in `idx_lookup`, each lookup is expensive. Each row has ~4 rook neighbors on average (from ~1.37M directed relationships / 344K cells â‰ˆ 4), so across 6.46M rows that's ~25.8M individual key lookups into a 6.46M-entry named vector. This is the dominant cost â€” easily tens of hours on a laptop.

3. **The lookup is rebuilt once but is astronomically expensive at this scale**: The entire 86+ hour estimate is dominated by this single `lapply` call.

`compute_neighbor_stats()`, by contrast, does simple numeric indexing (`vals[idx]`) which is O(1) per element, and the `do.call(rbind, ...)` on fixed-width rows is a one-time matrix construction.

## Optimization Strategy

1. **Replace the per-row neighbor lookup with a vectorized, year-broadcast approach**: Since the neighbor graph is spatial (same for every year), compute spatial neighbors once for the 344K cells, then broadcast across all 28 years using vectorized integer arithmetic â€” no `paste`, no named-vector lookup.

2. **Replace `do.call(rbind, lapply(...))` in `compute_neighbor_stats` with a single vectorized grouped aggregation** using `data.table` or pre-allocated matrix operations with the CSR (compressed sparse row) representation of the neighbor graph.

3. **Use `data.table` for fast keyed joins** instead of named-vector lookups.

These changes reduce complexity from ~O(N_rows Ã— neighbors Ã— name_lookup_cost) to ~O(N_rows Ã— avg_neighbors) with tiny constants.

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build neighbor lookup via vectorized integer mapping
#         (replaces build_neighbor_lookup)
# ============================================================
build_neighbor_lookup_fast <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id' and 'year' (and others)
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices)
  
  # --- Spatial neighbor edge list (done once, 344K cells) ---
  n_cells <- length(id_order)
  from <- rep(seq_len(n_cells), lengths(neighbors))
  to   <- unlist(neighbors, use.names = FALSE)
  
  # Map positional index -> cell ID
  from_id <- id_order[from]
  to_id   <- id_order[to]
  
  # Spatial edge table
  edges <- data.table(from_id = from_id, to_id = to_id)
  

  # --- Map (id, year) -> row index in data_dt ---
  # Ensure data_dt has a row-index column
  data_dt[, .row_idx := .I]
  
  # Create keyed lookup: cell id -> which rows in data_dt
  # We need to join edges Ã— years efficiently.
  
  # Unique years
  years <- sort(unique(data_dt$year))
  
  # For each edge (from_id, to_id), and each year, we need:
  #   source_row = row where id == from_id & year == y
  #   neighbor_row = row where id == to_id & year == y
  #
  # Instead of doing this per-row, we do a massive vectorized join.
  
  # Key the data for fast join
  setkey(data_dt, id, year)
  
  # Build the full edge-year table by joining edges with the source rows
  # Source rows: every (from_id, year) pair that exists in data
  source_rows <- data_dt[, .(id, year, source_row_idx = .row_idx)]
  setnames(source_rows, "id", "from_id")
  
  # Join edges to source rows to get (from_id, to_id, year, source_row_idx)
  # This broadcasts each edge across all years for that from_id
  setkey(edges, from_id)
  setkey(source_rows, from_id)
  edge_year <- edges[source_rows, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_year now has: from_id, to_id, year, source_row_idx
  
  # Now join to get the neighbor's row index
  neighbor_rows <- data_dt[, .(id, year, neighbor_row_idx = .row_idx)]
  setnames(neighbor_rows, "id", "to_id")
  setkey(neighbor_rows, to_id, year)
  setkey(edge_year, to_id, year)
  
  edge_year <- neighbor_rows[edge_year, on = c("to_id", "year"), nomatch = NA]
  # edge_year now has: to_id, year, neighbor_row_idx, from_id, source_row_idx
  
  # Drop edges where the neighbor row doesn't exist
  edge_year <- edge_year[!is.na(neighbor_row_idx)]
  
  # Sort by source_row_idx for grouped operations
  setkey(edge_year, source_row_idx)
  
  # Return the edge_year table â€” this IS the neighbor lookup
  # Also return total number of rows for downstream use
  list(
    edge_year  = edge_year,
    n_rows     = nrow(data_dt)
  )
}


# ============================================================
# STEP 2: Vectorized neighbor stats (replaces compute_neighbor_stats)
# ============================================================
compute_neighbor_stats_fast <- function(data_dt, lookup, var_name) {
  edge_year <- lookup$edge_year
  n_rows    <- lookup$n_rows
  
  # Pull the variable values for all neighbor rows at once
  vals <- data_dt[[var_name]]
  edge_year[, nval := vals[neighbor_row_idx]]
  
  # Drop NAs in neighbor values
  valid <- edge_year[!is.na(nval)]
  
  # Grouped aggregation â€” fully vectorized
  agg <- valid[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = source_row_idx]
  
  # Allocate full-length result columns (NA default)
  res_max  <- rep(NA_real_, n_rows)
  res_min  <- rep(NA_real_, n_rows)
  res_mean <- rep(NA_real_, n_rows)
  
  res_max[agg$source_row_idx]  <- agg$nb_max
  res_min[agg$source_row_idx]  <- agg$nb_min
  res_mean[agg$source_row_idx] <- agg$nb_mean
  
  # Clean up temporary column
  edge_year[, nval := NULL]
  
  list(nb_max = res_max, nb_min = res_min, nb_mean = res_mean)
}


# ============================================================
# STEP 3: Wrapper that adds columns to the data.table
# ============================================================
compute_and_add_neighbor_features_fast <- function(data_dt, var_name, lookup) {
  stats <- compute_neighbor_stats_fast(data_dt, lookup, var_name)
  
  set(data_dt, j = paste0(var_name, "_neighbor_max"),  value = stats$nb_max)
  set(data_dt, j = paste0(var_name, "_neighbor_min"),  value = stats$nb_min)
  set(data_dt, j = paste0(var_name, "_neighbor_mean"), value = stats$nb_mean)
  
  invisible(data_dt)
}


# ============================================================
# STEP 4: Main pipeline (replaces the outer loop)
# ============================================================
run_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  # Convert to data.table if not already (in-place)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  message("Building fast neighbor lookup...")
  t0 <- Sys.time()
  lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
  message("  Lookup built in ", round(difftime(Sys.time(), t0, units = "mins"), 1), " min")
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor stats for: ", var_name)
    t1 <- Sys.time()
    compute_and_add_neighbor_features_fast(cell_data, var_name, lookup)
    message("  Done in ", round(difftime(Sys.time(), t1, units = "secs"), 1), " sec")
  }
  
  # Remove helper column
  cell_data[, .row_idx := NULL]
  
  return(cell_data)
}

# ============================================================
# USAGE (drop-in replacement)
# ============================================================
# cell_data <- run_neighbor_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# Then predict with the already-trained Random Forest as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
#
# The numerical output is identical: same max, min, mean per 
# (cell, year) over the same rook-neighbor set.
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M iterations, each doing `paste()` + named-vector probe into a 6.46M-key vector | Eliminated entirely |
| **Lookup construction** | O(N_rows Ã— avg_neighbors Ã— name_lookup) â‰ˆ 86+ hrs | Two `data.table` keyed joins â‰ˆ **2â€“5 minutes** |
| **Stat computation** | `lapply` + `do.call(rbind, ...)` over 6.46M rows (minutes, not hours) | Vectorized `data.table` grouped aggregation â‰ˆ **seconds per variable** |
| **Total estimated time** | 86+ hours | **~5â€“10 minutes** |
| **RF model** | Preserved (untouched) | Preserved (untouched) |
| **Numerical output** | max/min/mean of non-NA neighbor values | Identical max/min/mean of non-NA neighbor values |

The colleague misidentified the symptom (`do.call(rbind, ...)`) as the disease. The real pathology is the O(NÂ²)-behaving named-vector lookups inside `build_neighbor_lookup()`, which is where virtually all 86+ hours are spent.
END_RESPONSE>>>
