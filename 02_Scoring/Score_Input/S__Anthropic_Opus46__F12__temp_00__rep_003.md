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
run_id: Anthropic_F12_T00_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized concatenation step per variable â€” only 5 calls total. Each inner function returns a fixed-length-3 numeric vector, so there is no repeated list *growth* (no quadratic copy pattern). This is O(n) and takes seconds, not hours.

**The true bottleneck is `build_neighbor_lookup()`.**

Here is why:

1. **`paste()` key construction and named-vector lookup for 6.46 million rows:** `idx_lookup` is a named integer vector of length ~6.46M. For every single row `i`, the function calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` to build string keys, then does `idx_lookup[neighbor_keys]` â€” a named character lookup into a 6.46M-element named vector. Named vector lookup in R is **O(n)** per query (linear scan or hash with overhead), and this is done ~6.46M times, each with ~4 neighbors on average (rook adjacency on a grid). That is roughly **25.8 million string-match lookups into a 6.46M-length named vector**.

2. **`as.character()` and `paste()` allocations inside the per-row `lapply`:** Each of the 6.46M iterations allocates new character vectors. The cumulative allocation and garbage-collection pressure is enormous.

3. **The `id_to_ref` lookup is also a named-vector character lookup**, called 6.46M times.

In total, `build_neighbor_lookup()` is doing tens of millions of expensive string operations and named-vector lookups. On a 16 GB laptop, this easily accounts for the 86+ hour estimate. `compute_neighbor_stats()`, by contrast, is doing simple numeric indexing and arithmetic â€” fast by comparison.

## Optimization Strategy

1. **Replace all string-key lookups with integer-arithmetic direct indexing.** Since the panel is balanced (344,208 cells Ã— 28 years = 9,637,824 potential slots, with ~6.46M populated), we can build a fast integer mapping from `(cell_id, year)` â†’ row number using a hash table (`data.table` or `fastmatch`) or, if IDs are dense, a matrix.

2. **Vectorize `build_neighbor_lookup()` entirely** using `data.table` joins instead of per-row `lapply`. Expand the neighbor list into an edge table, join on `(neighbor_id, year)` to get row indices, then split by source row. This replaces 6.46M R-level iterations with a single vectorized join.

3. **Vectorize `compute_neighbor_stats()`** using `data.table` grouped aggregation on the edge table, eliminating the per-row `lapply` and the `do.call(rbind, ...)` entirely.

4. **Preserve the trained Random Forest model** â€” we only change feature-engineering code, producing numerically identical columns.

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build a vectorized neighbor edge table (replaces
#         build_neighbor_lookup entirely)
# ============================================================

build_neighbor_edges <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns id, year, and a row index .row_id
  # id_order: vector mapping reference index -> cell id
  # neighbors: spdep nb list (neighbors[[ref_idx]] gives ref indices of neighbors)

  # Map cell id -> reference index
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # --- Build directed edge list from the nb object ---
  # Each entry neighbors[[j]] is an integer vector of neighbor ref indices for ref j
  n_lengths <- lengths(neighbors)
  from_ref  <- rep(seq_along(neighbors), times = n_lengths)
  to_ref    <- unlist(neighbors, use.names = FALSE)

  # Remove the spdep "no neighbors" sentinel (0)
  valid     <- to_ref != 0L
  from_ref  <- from_ref[valid]
  to_ref    <- to_ref[valid]

  # Convert ref indices to cell ids
  from_id <- id_order[from_ref]
  to_id   <- id_order[to_ref]

  edge_dt <- data.table(from_id = from_id, to_id = to_id)

  # --- Cross with years present in data ---
  # Get unique years
  years <- sort(unique(data_dt$year))

  # Expand edges Ã— years  (CJ-like expansion)
  # ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” fits in 16 GB easily
  edge_year <- edge_dt[, .(from_id, to_id, year = rep(list(years), .N))]
  edge_year <- edge_year[, .(year = unlist(year)), by = .(from_id, to_id)]

  # --- Map (from_id, year) -> source row index ---
  data_dt[, .row_id := .I]
  setkey(data_dt, id, year)

  # Join to get source row
  edge_year[data_dt, on = .(from_id = id, year = year), src_row := i..row_id]

  # Join to get neighbor row
  edge_year[data_dt, on = .(to_id = id, year = year), nbr_row := i..row_id]

  # Keep only edges where both source and neighbor exist in the data
  edge_year <- edge_year[!is.na(src_row) & !is.na(nbr_row)]

  edge_year
}

# ============================================================
# STEP 2: Compute neighbor stats vectorized via data.table
#         grouped aggregation (replaces compute_neighbor_stats)
# ============================================================

compute_neighbor_stats_fast <- function(data_dt, edge_year, var_name) {
  # Extract neighbor values via direct integer indexing (very fast)
  edge_year[, nbr_val := data_dt[[var_name]][nbr_row]]

  # Grouped aggregation â€” one pass
  stats <- edge_year[!is.na(nbr_val),
    .(
      nb_max  = max(nbr_val),
      nb_min  = min(nbr_val),
      nb_mean = mean(nbr_val)
    ),
    by = src_row
  ]

  # Build full-length result aligned to data_dt rows
  n <- nrow(data_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  out_max[stats$src_row]  <- stats$nb_max
  out_min[stats$src_row]  <- stats$nb_min
  out_mean[stats$src_row] <- stats$nb_mean

  # Column names matching original pipeline
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  data_dt[, (max_col)  := out_max]
  data_dt[, (min_col)  := out_min]
  data_dt[, (mean_col) := out_mean]

  # Clean up temporary column
  edge_year[, nbr_val := NULL]

  invisible(data_dt)
}

# ============================================================
# STEP 3: Full pipeline (replaces outer loop)
# ============================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  # Convert to data.table if needed (in-place, no deep copy)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  cat("Building vectorized neighbor edge table...\n")
  edge_year <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)
  cat(sprintf("  Edge-year table: %s rows\n", format(nrow(edge_year), big.mark = ",")))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
    compute_neighbor_stats_fast(cell_data, edge_year, var_name)
  }

  cat("Done.\n")
  cell_data
}

# ============================================================
# USAGE (drop-in replacement):
# ============================================================
# cell_data <- run_neighbor_feature_pipeline(
#   cell_data, id_order, rook_neighbors_unique
# )
#
# # The trained Random Forest model is untouched.
# # Column names and numerical values are identical to the original.
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Expected Performance Improvement

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level iterations with string ops â†’ **~80+ hours** | Single vectorized `data.table` join â†’ **~2â€“5 minutes** |
| `compute_neighbor_stats` (Ã—5 vars) | ~6.46M R-level iterations per var â†’ **~5 hours total** | Grouped `data.table` aggregation â†’ **~1â€“3 minutes total** |
| `do.call(rbind, ...)` | ~seconds per call | Eliminated entirely |
| **Total** | **~86+ hours** | **~5â€“10 minutes** |

The optimization converts the entire pipeline from O(n Ã— k) R-interpreter-level iterations with expensive string hashing into O(1) vectorized `data.table` joins and group-by operations executed in C. Memory usage for the edge-year table (~38.5M rows Ã— 4 integer columns â‰ˆ ~600 MB) fits comfortably within 16 GB. The trained Random Forest model is never touched, and all output columns are numerically identical to the original implementation.
END_RESPONSE>>>
