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
run_id: Anthropic_F12_T07_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of ~6.46 million small 3-element vectors is not free, it executes only 5 times (once per variable) and each call binds vectors of fixed length 3 â€” this is a well-known, manageable pattern and takes seconds at most.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Massive per-row string operations**: For each of ~6.46 million rows, the function calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` and then performs named-vector lookups via `idx_lookup[neighbor_keys]`. With ~1.37 million directed neighbor relationships spread across 28 years, this means roughly **38+ million string constructions and named-vector hash lookups** inside a single `lapply`.

2. **Redundant character coercion**: `as.character(data$id[i])` is called 6.46 million times individually instead of vectorized once.

3. **Named vector lookup scales poorly**: `idx_lookup` is a named vector with ~6.46 million entries. Repeated partial lookups into a named vector of this size are far slower than hash-table (environment) or `data.table` join approaches.

4. **The lookup is spatially redundant across years**: Every cell has the same neighbors in every year. The function recomputes the neighbor *identity* for each cell-year row, when it only needs to compute it once per cell and then replicate across 28 years.

In summary, `build_neighbor_lookup()` dominates runtime (estimated at many hours) because it performs **~6.46M iterations of string construction + named-vector lookup**, each touching multiple neighbors. `compute_neighbor_stats()` is comparatively cheap â€” it's just integer indexing into a numeric vector.

## Optimization Strategy

1. **Separate spatial logic from temporal replication**: Compute each cell's neighbor cell IDs once (344,208 cells), then expand to cell-years via a fast integer join.

2. **Replace named-vector lookups with `data.table` keyed joins**: `data.table` binary-search joins are orders of magnitude faster than named-vector character lookups at this scale.

3. **Vectorize `compute_neighbor_stats()`**: Instead of `lapply` + `do.call(rbind, ...)`, use a `data.table` grouped aggregation over a pre-built edge table to compute max/min/mean in one vectorized pass per variable.

4. **Preserve the trained Random Forest model and the original numerical estimand**: The output columns are identical in name, meaning, and numerical value â€” only the computational path changes.

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build a fast neighbor-row lookup using data.table
# ============================================================
build_neighbor_lookup_fast <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns 'id' and 'year' (and others)
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices)

  # --- 1a. Build cell-level neighbor edge list (spatial only, done once) ---
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  edges <- rbindlist(lapply(seq_along(id_order), function(ref_idx) {
    nb_idx <- neighbors[[ref_idx]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[ref_idx],
               neighbor_id = id_order[nb_idx])
  }))
  # edges has ~1.37M rows (one per directed neighbor pair)

  # --- 1b. Add row indices to data_dt ---
  if (!"..row_idx.." %in% names(data_dt)) {
    data_dt[, `..row_idx..` := .I]
  }

  # --- 1c. Create a keyed lookup: (id, year) -> row_idx ---
  row_key <- data_dt[, .(id, year, `..row_idx..`)]
  setkey(row_key, id, year)

  # --- 1d. Expand edges across all years ---
  #     Instead of replicating the full edge table Ã— 28 years in memory,
  #     we join edges to focal rows, then join neighbor rows.

  # Focal rows: every (focal_id, year) combination that exists in data
  focal_rows <- row_key[, .(focal_id = id, year, focal_row = `..row_idx..`)]
  setkey(focal_rows, focal_id)

  # Join edges to focal rows (keyed on focal_id)
  # Result: for each focal row, all its neighbor cell IDs + the year
  edge_year <- edges[focal_rows, on = .(focal_id), allow.cartesian = TRUE, nomatch = NULL]
  # edge_year columns: focal_id, neighbor_id, year, focal_row

  # Now join to get neighbor row indices
  setkey(edge_year, neighbor_id, year)
  edge_year[row_key, neighbor_row := i.`..row_idx..`, on = .(neighbor_id = id, year)]

  # Drop rows where neighbor doesn't exist in that year
  edge_year <- edge_year[!is.na(neighbor_row)]

  # Return the edge table and total number of rows
  list(
    edge_dt = edge_year[, .(focal_row, neighbor_row)],
    n_rows  = nrow(data_dt)
  )
}

# ============================================================
# STEP 2: Vectorized neighbor stats via data.table grouping
# ============================================================
compute_neighbor_stats_fast <- function(data_dt, edge_dt, n_rows, var_name) {
  vals <- data_dt[[var_name]]

  # Build a working table with neighbor values
  work <- edge_dt[, .(focal_row, nval = vals[neighbor_row])]
  work <- work[!is.na(nval)]

  # Grouped aggregation â€” one vectorized pass
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row]

  # Allocate result columns (NA for rows with no valid neighbors)
  max_col  <- rep(NA_real_, n_rows)
  min_col  <- rep(NA_real_, n_rows)
  mean_col <- rep(NA_real_, n_rows)

  max_col[agg$focal_row]  <- agg$nb_max
  min_col[agg$focal_row]  <- agg$nb_min
  mean_col[agg$focal_row] <- agg$nb_mean

  list(max = max_col, min = min_col, mean = mean_col)
}

# ============================================================
# STEP 3: Compute and add features (drop-in replacement)
# ============================================================
compute_and_add_neighbor_features_fast <- function(data_dt, var_name, edge_dt, n_rows) {
  stats <- compute_neighbor_stats_fast(data_dt, edge_dt, n_rows, var_name)

  # Use the same column naming convention as the original pipeline
  set(data_dt, j = paste0(var_name, "_nb_max"),  value = stats$max)
  set(data_dt, j = paste0(var_name, "_nb_min"),  value = stats$min)
  set(data_dt, j = paste0(var_name, "_nb_mean"), value = stats$mean)

  data_dt
}

# ============================================================
# STEP 4: Full optimized pipeline
# ============================================================
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  # Convert to data.table if not already (by reference if possible)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  cat("Building fast neighbor lookup...\n")
  t0 <- proc.time()
  lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
  edge_dt <- lookup$edge_dt
  n_rows  <- lookup$n_rows
  cat("  Lookup built in", (proc.time() - t0)[3], "seconds\n")

  # Pre-key the edge table for fast grouping
  setkey(edge_dt, focal_row)

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor stats for:", var_name, "...\n")
    t1 <- proc.time()
    cell_data <- compute_and_add_neighbor_features_fast(
      cell_data, var_name, edge_dt, n_rows
    )
    cat("  Done in", (proc.time() - t1)[3], "seconds\n")
  }

  # Clean up temporary column
  if ("..row_idx.." %in% names(cell_data)) {
    cell_data[, `..row_idx..` := NULL]
  }

  cell_data
}

# ============================================================
# Usage (preserves trained RF model â€” no retraining needed):
# ============================================================
# cell_data <- run_optimized_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# # The existing trained Random Forest model is used as-is:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Expected Performance Improvement

| Component | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | ~80+ hours (6.46M Ã— string ops + named-vector lookup) | ~1â€“3 minutes (data.table keyed joins) |
| **Stats per variable** | ~1 hour (lapply + integer indexing + do.call rbind) | ~10â€“30 seconds (data.table grouped agg) |
| **Total (5 vars)** | ~86+ hours | **~5â€“10 minutes** |

**Key insight**: The bottleneck was never `do.call(rbind, ...)` â€” it was the O(6.46M) loop in `build_neighbor_lookup()` performing millions of string paste + named-vector hash lookups. The fix separates spatial neighbor logic (344K cells, done once) from temporal expansion (done via a single data.table join), and replaces per-row `lapply` aggregation with vectorized `data.table` grouped operations. The trained Random Forest model and all numerical outputs are preserved exactly.
END_RESPONSE>>>
