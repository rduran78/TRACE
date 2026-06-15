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
run_id: Anthropic_F05_T07_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Local inefficiency
- `paste(data$id, data$year, sep = "_")` constructs ~6.46 million strings to build `idx_lookup`. This happens once, but then inside the `lapply` over all 6.46M rows, another `paste()` call constructs neighbor keys and performs named-vector lookup (hash-table probe) for every neighbor of every row. With ~1.37M directed neighbor relationships replicated across 28 years, that's ~38.4 million string constructions and hash lookups inside the loop.

### Broader algorithmic problem
The fundamental issue is that **the neighbor topology is year-invariant**. Every cell has the same rook neighbors in every year. The `build_neighbor_lookup` function re-discovers this fact row by row by joining on `(neighbor_id, same_year)`, but the mapping from a cell-year row to its neighbor-cell-year rows is structurally just: *"find the rows for the same year for my spatial neighbors."* This can be computed entirely with integer arithmetic â€” no strings, no hashing, no per-row `lapply`.

Additionally, `compute_neighbor_stats` loops over 6.46M entries in R-level `lapply`, extracting and summarizing small vectors one at a time. This is a second bottleneck that can be replaced with a vectorized/matrix approach.

### Cost accounting (current)

| Step | Operations | Bottleneck |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations, each with `paste` + hash lookup | ~38M string ops, R-loop overhead |
| `compute_neighbor_stats` | Called 5Ã— over 6.46M rows | 32.3M R-level iterations |
| **Total** | ~70M+ R-level small-vector operations | **~86+ hours** |

## Optimization Strategy

### Key insight
If the data is sorted by `(year, id)` â€” or we can build an integer index by year â€” then for every cell `c` with spatial neighbors `{n1, n2, ...}`, the row indices of those neighbors in year `t` are deterministic integer offsets. No strings needed.

**Step 1 â€” Build a year-offset table.** For each year, record where that year's block of rows starts and build a within-year cell-id â†’ position map. This is O(N) and done once.

**Step 2 â€” Expand the neighbor adjacency into a two-column integer edge list (row_index, neighbor_row_index) across all years.** This is a vectorized outer-join of the spatial adjacency with the year blocks. Result: ~38.4M integer pairs, built in seconds.

**Step 3 â€” Compute all neighbor statistics in one vectorized pass per variable** using this edge list, via `data.table` grouped aggregation or matrix indexing. No R-level row loop.

This reduces the entire pipeline from ~86 hours to **minutes**.

## Working R Code

```r
# =============================================================================
# Optimized neighbor-feature construction
# Drop-in replacement for build_neighbor_lookup + compute_neighbor_stats loop
# Preserves the exact numerical estimand (max, min, mean of non-NA neighbor vals)
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  # ------------------------------------------------------------------
  # 0. Convert to data.table (by reference if already one; copy if not)
  # ------------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    dt <- as.data.table(cell_data)
  } else {
    dt <- copy(cell_data)
  }

  # Preserve original row order for final output
  dt[, orig_row_idx__ := .I]

  # ------------------------------------------------------------------
  # 1. Build integer mapping: for each (year, id) -> row index
  #    Sort by year and id so we can do everything with integer math.
  # ------------------------------------------------------------------
  # Create a cell-id factor with levels in id_order for consistent indexing
  id_order_char <- as.character(id_order)
  n_cells <- length(id_order)

  # Map each id to its position in id_order (1-based spatial index)
  id_to_spatial_idx <- setNames(seq_along(id_order), id_order_char)
  dt[, spatial_idx__ := id_to_spatial_idx[as.character(id)]]

  # ------------------------------------------------------------------
  # 2. Build the spatial edge list from the nb object (year-invariant)
  #    rook_neighbors_unique[[i]] gives the neighbor indices (into id_order)
  #    for the i-th element of id_order.
  # ------------------------------------------------------------------
  # Expand nb object to edge list: (from_spatial_idx, to_spatial_idx)
  from_idx <- rep(
    seq_len(n_cells),
    times = lengths(rook_neighbors_unique)
  )
  to_idx <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove any 0-entries (spdep uses 0 for "no neighbors")
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  spatial_edges <- data.table(
    from_spatial = as.integer(from_idx),
    to_spatial   = as.integer(to_idx)
  )
  rm(from_idx, to_idx, valid)

  cat(sprintf("Spatial edge list: %d directed edges\n", nrow(spatial_edges)))

  # ------------------------------------------------------------------
  # 3. Build a lookup: (spatial_idx, year) -> row index in dt
  #    This replaces all the string-key hashing.
  # ------------------------------------------------------------------
  row_lookup <- dt[, .(row_idx = orig_row_idx__, spatial_idx__, year)]
  setkey(row_lookup, spatial_idx__, year)

  # ------------------------------------------------------------------
  # 4. Expand spatial edges across all years to get the full
  #    (focal_row, neighbor_row) edge list.
  #
  #    For each year t and each spatial edge (a -> b), we need:
  #      focal_row   = row where spatial_idx == a & year == t
  #      neighbor_row = row where spatial_idx == b & year == t
  # ------------------------------------------------------------------
  years <- sort(unique(dt$year))

  # Cross join spatial_edges Ã— years, then look up row indices
  # To keep memory manageable, process in year chunks
  cat("Building full (focal_row, neighbor_row) edge list by year...\n")

  edge_list_parts <- vector("list", length(years))

  for (yi in seq_along(years)) {
    yr <- years[yi]
    # Rows for this year
    yr_rows <- row_lookup[year == yr]
    setkey(yr_rows, spatial_idx__)

    # Map from spatial_idx to row_idx for this year
    sp_to_row <- yr_rows$row_idx
    names(sp_to_row) <- as.character(yr_rows$spatial_idx__)

    focal_rows    <- sp_to_row[as.character(spatial_edges$from_spatial)]
    neighbor_rows <- sp_to_row[as.character(spatial_edges$to_spatial)]

    # Keep only pairs where both exist
    both_valid <- !is.na(focal_rows) & !is.na(neighbor_rows)

    edge_list_parts[[yi]] <- data.table(
      focal_row    = as.integer(focal_rows[both_valid]),
      neighbor_row = as.integer(neighbor_rows[both_valid])
    )
  }

  full_edges <- rbindlist(edge_list_parts)
  rm(edge_list_parts, row_lookup)
  gc()

  cat(sprintf("Full edge list: %d directed cell-year edges\n", nrow(full_edges)))

  # ------------------------------------------------------------------
  # 5. For each source variable, compute neighbor max, min, mean
  #    using vectorized data.table grouped aggregation.
  # ------------------------------------------------------------------
  setkey(full_edges, focal_row)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))

    # Attach neighbor values
    full_edges[, nval__ := dt[[var_name]][neighbor_row]]

    # Grouped aggregation â€” drop NAs within groups
    stats <- full_edges[!is.na(nval__),
      .(
        nb_max  = max(nval__),
        nb_min  = min(nval__),
        nb_mean = mean(nval__)
      ),
      by = focal_row
    ]

    # Initialize result columns with NA
    max_col  <- paste0(var_name, "_max_neighbor")
    min_col  <- paste0(var_name, "_min_neighbor")
    mean_col <- paste0(var_name, "_mean_neighbor")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Fill in computed values
    dt[stats$focal_row, (max_col)  := stats$nb_max]
    dt[stats$focal_row, (min_col)  := stats$nb_min]
    dt[stats$focal_row, (mean_col) := stats$nb_mean]

    cat(sprintf("  Done: %s â€” %d rows with neighbor data\n", var_name, nrow(stats)))
  }

  # Clean up helper columns
  full_edges[, nval__ := NULL]

  # ------------------------------------------------------------------
  # 6. Restore original row order and return
  # ------------------------------------------------------------------
  setorder(dt, orig_row_idx__)
  dt[, c("orig_row_idx__", "spatial_idx__") := NULL]

  # Return as data.frame if the input was a data.frame
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# =============================================================================
# Usage â€” drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Column names match the original pattern: {var}_max_neighbor, {var}_min_neighbor, {var}_mean_neighbor
# Numerical results are identical (same max/min/mean of non-NA neighbor values).
```

## Why This Works and Performance Estimate

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M R-level `lapply` iterations with string paste + hash | One-time integer join per year (28 vectorized passes) |
| **Stat computation** | 5 Ã— 6.46M R-level `lapply` iterations | 5 Ã— one `data.table` grouped aggregation over ~38.4M edges |
| **String operations** | ~44.8M `paste()` calls | **Zero** |
| **Estimated runtime** | ~86+ hours | **~5â€“15 minutes** on 16 GB laptop |
| **Peak memory** | Low per-row but massive wall-clock | ~38.4M-row edge table â‰ˆ ~600 MB (fits in 16 GB) |
| **Numerical result** | max/min/mean of non-NA neighbor values | **Identical** |
| **RF model** | â€” | **Untouched** |

### Summary of the root cause

The string-key construction was not just a local inefficiency â€” it was the visible symptom of an **O(N Ã— K) R-interpreted loop** (N = 6.46M rows, K = avg neighbors) performing work that has an **O(1)-per-edge vectorized solution**. The topology is year-invariant, so the entire neighbor-row mapping can be precomputed as an integer edge list and all statistics computed via a single grouped aggregation per variable. This eliminates both the string-hashing bottleneck and the R-level row-iteration bottleneck simultaneously.
END_RESPONSE>>>
