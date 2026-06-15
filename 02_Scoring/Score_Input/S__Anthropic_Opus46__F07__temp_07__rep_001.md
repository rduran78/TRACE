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
run_id: Anthropic_F07_T07_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates ~6.46 million R list entries, each built by an anonymous `lapply` function that performs per-row character key construction, hash lookups, and NA filtering. This is an O(N) loop in pure R over 6.46M rows, each doing string operations (`paste`, named-vector lookup). The `compute_neighbor_stats` function then loops again over 6.46M entries, subsetting a numeric vector each time.

**Specific problems:**

1. **`build_neighbor_lookup` is O(NÂ·k) in interpreted R** with expensive string-key hashing. With N â‰ˆ 6.46M and k â‰ˆ 4 (average rook neighbors), this produces ~25.8M string operations plus named-vector lookups. The named-vector lookup `idx_lookup[neighbor_keys]` is O(k) per call but with R's string hashing overhead, across 6.46M rows this is extremely slow.

2. **`compute_neighbor_stats` uses `lapply` over 6.46M entries** â€” each call creates small vectors, computes max/min/mean, and returns a length-3 vector. The overhead per iteration is small but 6.46M iterations of interpreted R adds up to hours.

3. **The lookup is rebuilt once but is a ~6.46M-element list of integer vectors** â€” this alone consumes substantial memory and time to construct.

4. **The outer loop recomputes stats 5 times**, each time iterating over the full 6.46M-element lookup. This is unavoidable in structure but the per-iteration cost can be drastically reduced.

## Optimization Strategy

**Replace the row-level R loops with vectorized operations using `data.table` and integer-indexed sparse neighbor matrices.**

1. **Replace string-key lookup with integer join.** Map `(id, year)` â†’ row index using `data.table` keyed joins instead of named character vectors. This eliminates all `paste` and string hashing.

2. **Build an edge list, not a per-row list.** Expand the neighbor structure into a two-column integer edge list `(from_row, to_row)` that covers all cell-years. This is a single vectorized merge operation.

3. **Compute neighbor stats via `data.table` grouped aggregation** on the edge list: group by `from_row`, compute `max`, `min`, `mean` of the neighbor values. This replaces 6.46M `lapply` iterations with a single C-level grouped operation.

4. **Process all 5 variables in one pass** over the edge list (or 5 fast grouped aggregations reusing the same edge list).

**Expected speedup:** From ~86+ hours to **minutes** (typically 2â€“10 minutes depending on disk I/O and RAM pressure).

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build the global edge list (once)
# ============================================================
build_edge_list <- function(cell_data_dt, id_order, rook_neighbors_unique) {
  # cell_data_dt must be a data.table with columns: id, year
  # id_order: vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique: spdep nb object (list of integer neighbor indices)

  # --- A. Expand spatial neighbor pairs into a cell-ID edge list ---
  n_cells <- length(id_order)
  from_id <- rep(id_order, times = lengths(rook_neighbors_unique))
  to_id   <- id_order[unlist(rook_neighbors_unique)]

  spatial_edges <- data.table(id_from = from_id, id_to = to_id)
  # Remove entries from cells with 0 neighbors (spdep uses integer(0))
  spatial_edges <- spatial_edges[!is.na(id_to)]

  # --- B. Add row-index mapping: (id, year) -> row_idx ---
  cell_data_dt[, row_idx := .I]

  # Key for fast join
  idx_map <- cell_data_dt[, .(id, year, row_idx)]
  setkey(idx_map, id)

  # --- C. Get unique years ---
  years <- sort(unique(cell_data_dt$year))

  # --- D. Cross-join spatial edges Ã— years, then map to row indices ---
  # Use CJ inside a merge chain for memory efficiency.
  # First, attach from_row_idx by joining on (id_from, year)
  setnames(idx_map, c("id", "year", "row_idx"), c("id_from", "year", "from_row"))
  setkey(idx_map, id_from, year)

  # Expand: each spatial edge exists in every year
  spatial_edges_yr <- spatial_edges[, .(year = years), by = .(id_from, id_to)]
  # This is ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” fits comfortably in RAM

  # Join to get from_row
  setkey(spatial_edges_yr, id_from, year)
  spatial_edges_yr <- idx_map[spatial_edges_yr, nomatch = 0L]

  # Now join to get to_row
  setnames(idx_map, c("id_from", "year", "from_row"), c("id_to", "year", "to_row"))
  setkey(idx_map, id_to, year)
  setkey(spatial_edges_yr, id_to, year)
  spatial_edges_yr <- idx_map[spatial_edges_yr, nomatch = 0L]

  # Result columns: from_row, to_row (and possibly id_from, id_to, year)
  # Keep only what we need
  edge_list <- spatial_edges_yr[, .(from_row, to_row)]

  # Clean up temporary column
  cell_data_dt[, row_idx := NULL]

  return(edge_list)
}

# ============================================================
# STEP 2: Compute neighbor stats for one variable (vectorized)
# ============================================================
compute_neighbor_stats_fast <- function(cell_data_dt, edge_list, var_name) {
  n <- nrow(cell_data_dt)

  # Attach the neighbor's value to each edge
  vals <- cell_data_dt[[var_name]]
  edge_vals <- data.table(
    from_row = edge_list$from_row,
    val      = vals[edge_list$to_row]
  )

  # Drop edges where the neighbor value is NA
  edge_vals <- edge_vals[!is.na(val)]

  # Grouped aggregation â€” single pass in C
  stats <- edge_vals[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = from_row]

  # Allocate full-length result columns (NA for cells with no valid neighbors)
  max_col  <- rep(NA_real_, n)
  min_col  <- rep(NA_real_, n)
  mean_col <- rep(NA_real_, n)

  max_col[stats$from_row]  <- stats$nb_max
  min_col[stats$from_row]  <- stats$nb_min
  mean_col[stats$from_row] <- stats$nb_mean

  # Name columns to match original pipeline expectations
  prefix <- var_name
  out <- data.table(
    x1 = max_col,
    x2 = min_col,
    x3 = mean_col
  )
  setnames(out, c(
    paste0(prefix, "_neighbor_max"),
    paste0(prefix, "_neighbor_min"),
    paste0(prefix, "_neighbor_mean")
  ))
  return(out)
}

# ============================================================
# STEP 3: Full pipeline
# ============================================================
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  # Convert to data.table if needed (by reference if already one)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  cat("Building edge list...\n")
  edge_list <- build_edge_list(cell_data, id_order, rook_neighbors_unique)
  cat(sprintf("  Edge list: %s rows\n", format(nrow(edge_list), big.mark = ",")))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
    stats_dt <- compute_neighbor_stats_fast(cell_data, edge_list, var_name)
    # Bind new columns to cell_data by reference
    for (col_name in names(stats_dt)) {
      set(cell_data, j = col_name, value = stats_dt[[col_name]])
    }
  }

  cat("Done. 15 neighbor feature columns added.\n")
  return(cell_data)
}

# ============================================================
# USAGE (drop-in replacement for the original outer loop)
# ============================================================
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# The trained Random Forest model is untouched.
# The 15 new columns (5 vars Ã— {max, min, mean}) are numerically
# identical to the original implementation's output.
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The same `max`, `min`, `mean` functions are applied to exactly the same sets of neighbor values. The edge list encodes the identical rook-neighbor relationships from `rook_neighbors_unique`. NA handling is identical (NAs excluded before aggregation; rows with no valid neighbors get NA). |
| **Trained RF model** | No model code is touched. Only the feature-engineering step is accelerated. The output columns have the same names and values. |
| **Irregular topology / masked cells** | The edge list is built from the actual `spdep::nb` object, not from any raster assumption. Missing cell-years (if any) are handled by `nomatch = 0L` in the join, exactly mirroring the original `!is.na(result)` filter. |

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build lookup / edge list | ~hours (6.46M `lapply` + string ops) | ~30â€“90 seconds (vectorized `data.table` joins) |
| Stats per variable | ~15+ hours (6.46M `lapply`) | ~10â€“30 seconds (grouped aggregation in C) |
| **Total (5 variables)** | **~86+ hours** | **~3â€“5 minutes** |
| Peak RAM | ~6â€“10 GB (6.46M-element list) | ~4â€“6 GB (38.5M-row edge table) |
END_RESPONSE>>>
