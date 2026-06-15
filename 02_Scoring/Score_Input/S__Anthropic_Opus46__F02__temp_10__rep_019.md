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
run_id: Anthropic_F02_T10_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is twofold:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over ~6.46 million rows
Each iteration does **character coercion, `paste` key construction, and named-vector lookups** (which are hash-table probes but with per-call overhead). With 6.46M rows, this single `lapply` creates millions of temporary character vectors and performs millions of hash lookups. On a 16 GB laptop this is both slow (~hours) and memory-heavy (millions of small character allocations).

### 2. `compute_neighbor_stats` â€” called 5 times, each iterating over 6.46M rows
Each call does an R-level loop with subsetting, `is.na` filtering, and summary stats. The `do.call(rbind, â€¦)` on a 6.46M-element list of 3-vectors is also expensive (creates a temporary list then copies everything into a matrix).

**In total:** ~6.46M Ã— (1 key-build + 5 stat-passes) = ~38.8M R-level loop iterations, each with non-trivial overhead. This explains the 86+ hour estimate.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Per-row `paste`/character key lookups in `build_neighbor_lookup` | Replace with **integer arithmetic keys** (`id * 10000L + year`) and use `data.table` or `match()` on integer vectors â€” orders of magnitude faster than named character lookups. |
| `lapply` over 6.46M rows in R | Vectorize the neighbor expansion using `data.table` joins: explode the neighbor list once into an edge table `(row_i, neighbor_cell_id)`, join to get `neighbor_row_j`, then compute grouped statistics with `data.table` â€” no R-level row loop at all. |
| `compute_neighbor_stats` called 5 times independently | Compute **all 5 variables' stats in a single grouped aggregation pass** over the edge table. |
| `do.call(rbind, â€¦)` on millions of small vectors | Eliminated entirely; `data.table` returns a matrix/data.frame directly. |
| Memory: millions of temporary character vectors | Integer keys and `data.table` in-place operations dramatically reduce peak memory. |

**Expected speedup:** From 86+ hours to roughly **5â€“20 minutes** on the same laptop, depending on disk I/O. Memory peak should stay well within 16 GB.

**Preservation guarantees:**
- The trained Random Forest model is untouched (we only change feature-engineering code).
- The numerical output (max, min, mean of neighbor values per variable) is identical to the original.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# Step 1: Build an edge table (row_i -> neighbor_row_j) ONCE
#         Replaces build_neighbor_lookup entirely.
# =============================================================================
build_neighbor_edge_table <- function(cell_dt, id_order, neighbors) {

  # cell_dt must be a data.table with columns: id, year (and all feature cols)
  # id_order: integer vector of cell IDs (index position â†” nb list position)
  # neighbors: spdep nb object (list of integer index vectors into id_order)

  # --- Map each cell ID to its position in id_order (1-based ref index) ------
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # --- Build integer key for every row: unique per (id, year) ----------------
  #     Assumes year âˆˆ [1992, 2019] so id * 10000L + year is collision-free
  cell_dt[, row_idx := .I]
  cell_dt[, int_key := id * 10000L + year]

  # Fast lookup: int_key -> row_idx
  key_to_row <- cell_dt$row_idx
  names(key_to_row) <- as.character(cell_dt$int_key)

  # --- Expand neighbor list into an edge list --------------------------------
  #     For each cell ID, get its neighbor cell IDs from the nb object.
  n_ids <- length(id_order)
  from_id <- rep(id_order, times = lengths(neighbors))
  to_ref  <- unlist(neighbors, use.names = FALSE)
  to_id   <- id_order[to_ref]

  edge_cells <- data.table(from_id = from_id, to_id = to_id)

  # --- Cross-join with years to get (from_id, year) -> (to_id, year) ---------
  years <- sort(unique(cell_dt$year))

  # Instead of a huge cross join (edges Ã— years), we join via int_key.
  # For every row i in cell_dt, find its neighbor rows.

  # Map from_id -> ref index (same for all years)
  cell_dt[, ref_idx := id_to_ref[as.character(id)]]

  # For each row, the neighbor cell IDs are: id_order[neighbors[[ref_idx]]]
  # We vectorize: expand by row using the precomputed edge_cells table.

  # Step A: for each unique cell id, store its neighbor cell ids
  #         (already in edge_cells)
  # Step B: join cell_dt to edge_cells on from_id == id to get all
  #         (row_idx, to_id, year) triples
  # Step C: compute to_int_key = to_id * 10000L + year, then look up to_row_idx

  # Keyed join: cell_dt[, .(row_idx, id, year)] â‹ˆ edge_cells on id == from_id
  setkey(edge_cells, from_id)
  row_info <- cell_dt[, .(row_idx, id, year)]
  setkey(row_info, id)

  # This is the big join â€” produces one row per (row_i, neighbor_cell_id) pair
  # Approximate size: 6.46M rows Ã— avg ~4 rook neighbors â‰ˆ 26M rows (manageable)
  edges <- edge_cells[row_info, on = .(from_id = id),
                      .(row_i = i.row_idx,
                        to_id = x.to_id,
                        year  = i.year),
                      allow.cartesian = TRUE,
                      nomatch = 0L]

  # Compute integer key for the neighbor and look up its row index
  edges[, to_key := to_id * 10000L + year]
  edges[, row_j  := key_to_row[as.character(to_key)]]

  # Drop edges where the neighbor-year combination doesn't exist in the data
  edges <- edges[!is.na(row_j)]

  # Clean up temp columns from cell_dt
  cell_dt[, c("row_idx", "int_key", "ref_idx") := NULL]

  # Return only what we need
  edges[, .(row_i, row_j)]
}


# =============================================================================
# Step 2: Compute neighbor stats for ALL variables in one pass
#         Replaces compute_neighbor_stats + the outer for-loop.
# =============================================================================
compute_all_neighbor_features <- function(cell_dt, edge_dt, var_names) {
  # cell_dt : data.table with all feature columns
  # edge_dt : data.table with columns row_i, row_j
  # var_names : character vector of source variable names

  # Pre-allocate result columns in cell_dt (filled with NA)
  for (v in var_names) {
    cell_dt[, paste0("neighbor_max_",  v) := NA_real_]
    cell_dt[, paste0("neighbor_min_",  v) := NA_real_]
    cell_dt[, paste0("neighbor_mean_", v) := NA_real_]
  }

  n_rows <- nrow(cell_dt)

  # Process one variable at a time to limit peak memory

  for (v in var_names) {
    message("  Computing neighbor stats for: ", v)

    vals <- cell_dt[[v]]

    # Attach the neighbor's value to each edge
    work <- edge_dt[, .(row_i, nval = vals[row_j])]

    # Drop NA neighbor values (matches original behaviour)
    work <- work[!is.na(nval)]

    # Grouped aggregation â€” extremely fast in data.table
    agg <- work[, .(nb_max  = max(nval),
                     nb_min  = min(nval),
                     nb_mean = mean(nval)),
                 keyby = row_i]

    # Write results back into cell_dt by row index
    max_col  <- paste0("neighbor_max_",  v)
    min_col  <- paste0("neighbor_min_",  v)
    mean_col <- paste0("neighbor_mean_", v)

    set(cell_dt, i = agg$row_i, j = max_col,  value = agg$nb_max)
    set(cell_dt, i = agg$row_i, j = min_col,  value = agg$nb_min)
    set(cell_dt, i = agg$row_i, j = mean_col, value = agg$nb_mean)

    rm(work, agg)
    gc()
  }

  invisible(cell_dt)
}


# =============================================================================
# Step 3: Drop-in replacement for the original outer loop
# =============================================================================

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

message("Building neighbor edge table ...")
edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
message("  Edge table rows: ", format(nrow(edge_table), big.mark = ","))

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor features ...")
compute_all_neighbor_features(cell_data, edge_table, neighbor_source_vars)

message("Done.")

# Clean up the large edge table
rm(edge_table)
gc()

# cell_data now contains 15 new columns:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... (same pattern for pop_density, def, usd_est_n2)
#
# These are numerically identical to the original implementation's output.
# The trained Random Forest model can be used as-is with predict().
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| Data structure | `data.frame` + named character vectors | `data.table` + integer keys |
| Neighbor lookup | 6.46M-iteration `lapply` with `paste`/hash | Single vectorized `data.table` join â†’ edge table (~26M rows) |
| Stat computation | 5 Ã— 6.46M-iteration `lapply` + `do.call(rbind,â€¦)` | 5 Ã— single `data.table` grouped aggregation (`keyby`) |
| Estimated runtime | 86+ hours | **5â€“20 minutes** |
| Peak RAM | Excessive (character vector bloat) | ~4â€“6 GB (edge table + working columns) |
| Numerical output | max, min, mean per row per variable | **Identical** (same NA handling logic) |
| Trained RF model | Untouched | **Untouched** |
END_RESPONSE>>>
