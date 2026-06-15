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
run_id: Anthropic_F02_T07_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is twofold:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups
- **String key construction:** `paste(id, year, sep="_")` is called millions of times, and named-vector lookup via character keys (`idx_lookup[neighbor_keys]`) is hash-table lookup repeated for every single row. With ~6.46M rows and an average of ~4 rook neighbors per cell, this is ~25.8 million string constructions and hash lookups just for the neighbor resolution step.
- **`lapply` over 6.46M elements:** Pure R loop overhead is enormous. Each iteration also allocates small vectors (paste results, index subsets), creating massive GC pressure.

### 2. `compute_neighbor_stats` â€” Another `lapply` over 6.46M rows computing max/min/mean by subsetting a numeric vector
- Each call to `vals[idx]` and the subsequent `max/min/mean` is fast individually, but repeated 6.46M Ã— 5 variables = ~32.3 million times total, the R interpreter overhead dominates.
- `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is itself slow and memory-hungry.

### Memory
With 6.46M rows Ã— 110 columns, the data frame alone is ~5â€“6 GB. The `neighbor_lookup` list (6.46M elements, each a small integer vector) adds another ~1â€“2 GB. Headroom on a 16 GB machine is tight.

---

## Optimization Strategy

| Problem | Solution |
|---|---|
| Per-row string pasting and hash lookup in `build_neighbor_lookup` | Replace with integer arithmetic: encode `(id, year)` as a single integer key, use `data.table` fast joins or direct integer indexing to resolve neighbor row indices in bulk (vectorized). |
| `lapply` over 6.46M rows for neighbor stats | Replace with a **flat edge-list approach**: expand all neighbor relationships into a long `data.table` of `(row_i, neighbor_row_j)`, join the variable values, then compute grouped `max/min/mean` by `row_i` using `data.table` aggregation â€” fully vectorized, no R-level loop. |
| `do.call(rbind, ...)` on millions of small vectors | Eliminated by the grouped `data.table` aggregation which returns a single `data.table` directly. |
| Memory pressure | The flat edge list for 6.46M rows Ã— ~4 neighbors â‰ˆ 25.8M rows Ã— 2 integer columns â‰ˆ 0.4 GB â€” manageable. We avoid duplicating the full data; we join only the single variable column needed. Process one variable at a time and discard intermediate objects. |
| 86+ hours runtime | Expected reduction to **minutes** (the vectorized `data.table` join + grouped aggregation on ~25.8M rows is very fast). |

**Key invariant preserved:** The numerical output (max, min, mean of rook-neighbor values per cell-year) is identical. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build a flat edge list (row_i â†’ neighbor_row_j) â€” VECTORIZED
#
# This replaces build_neighbor_lookup entirely.
# Instead of a 6.46M-element list, we produce a data.table with ~25.8M
# rows of (row_idx, neighbor_row_idx) that can be reused for every variable.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_neighbor_edge_list <- function(cell_dt, id_order, neighbors) {
  # cell_dt must be a data.table with columns 'id' and 'year'
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  # --- Map each cell ID to its position in id_order (1-based ref index) ---
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # --- Build a cell-level edge list: (ref_idx â†’ neighbor_ref_idx) ---
  #     Expand the nb list into two parallel integer vectors.
  n_neighbors <- lengths(neighbors)                 # integer vector, length = #cells
  from_ref    <- rep(seq_along(neighbors), n_neighbors)
  to_ref      <- unlist(neighbors, use.names = FALSE)

  cell_edges <- data.table(from_ref = from_ref, to_ref = to_ref)

  # --- Map ref indices back to actual cell IDs ---
  cell_edges[, from_id := id_order[from_ref]]
  cell_edges[, to_id   := id_order[to_ref]]

  # --- Create a row-index lookup table: (id, year) â†’ row position in cell_dt ---
  cell_dt[, .row_idx := .I]
  row_lookup <- cell_dt[, .(.row_idx, id, year)]
  setkey(row_lookup, id, year)

  # --- Get the unique years present in the data ---
  years <- sort(unique(cell_dt$year))

  # --- Cross-join cell_edges Ã— years, then resolve row indices for both
  #     the focal cell and the neighbor cell.
  #     To avoid a massive cross join in memory, we do two keyed joins. ---

  # Expand: for every (from_id, to_id) pair, replicate across all years.
  # But not every cell is present in every year, so we join rather than cross.

  # Approach: start from cell_dt rows, attach their ref_idx, then join neighbors.
  cell_dt[, ref_idx := id_to_ref[as.character(id)]]

  # Focal rows: (row_idx, ref_idx, year)
  focal <- cell_dt[, .(focal_row = .row_idx, ref_idx, year)]
  setkey(focal, ref_idx)

  # For each focal row, find its neighbor ref indices via cell_edges
  setkey(cell_edges, from_ref)
  # Join: for each focal row, get all neighbor ref indices
  edge_expanded <- cell_edges[focal, on = .(from_ref = ref_idx),
                              .(focal_row, to_ref, year),
                              allow.cartesian = TRUE,
                              nomatch = NULL]

  # Now resolve neighbor rows: need (to_ref â†’ to_id), then join (to_id, year) â†’ neighbor_row
  edge_expanded[, neighbor_id := id_order[to_ref]]

  # Join to get neighbor row index
  setkey(edge_expanded, neighbor_id, year)
  setkey(row_lookup, id, year)

  edge_expanded[row_lookup,
                neighbor_row := i..row_idx,
                on = .(neighbor_id = id, year = year)]

  # Drop rows where the neighbor wasn't found (boundary / missing year)
  edge_list <- edge_expanded[!is.na(neighbor_row),
                             .(focal_row, neighbor_row)]

  # Clean up temporary columns from cell_dt
  cell_dt[, c(".row_idx", "ref_idx") := NULL]

  return(edge_list)
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Compute neighbor stats for one variable â€” VECTORIZED
#
# This replaces compute_neighbor_stats.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_stats_fast <- function(cell_dt, edge_list, var_name) {
  # Extract the variable values for all neighbor rows
  vals <- cell_dt[[var_name]]

  # Build a working table: focal_row + neighbor's value
  work <- copy(edge_list)
  work[, nval := vals[neighbor_row]]

  # Remove NA neighbor values (matches original behavior)
  work <- work[!is.na(nval)]

  # Grouped aggregation â€” fully vectorized via data.table
  stats <- work[, .(nb_max  = max(nval),
                     nb_min  = min(nval),
                     nb_mean = mean(nval)),
                keyby = .(focal_row)]

  # Allocate output columns (NA by default, matching original behavior for
  # cells with no valid neighbors)
  n <- nrow(cell_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  out_max[stats$focal_row]  <- stats$nb_max
  out_min[stats$focal_row]  <- stats$nb_min
  out_mean[stats$focal_row] <- stats$nb_mean

  # Name the output columns consistently with the original pipeline
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")

  cell_dt[, (col_max)  := out_max]
  cell_dt[, (col_min)  := out_min]
  cell_dt[, (col_mean) := out_mean]

  invisible(cell_dt)
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Main driver â€” drop-in replacement for the outer loop
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

run_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {

  # Convert to data.table if needed (by reference â€” no copy)
  if (!is.data.table(cell_data)) {
    setDT(cell_data)
  }

  message("Building vectorized neighbor edge list ...")
  edge_list <- build_neighbor_edge_list(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("  Edge list: %s rows (%.1f MB)",
                  formatC(nrow(edge_list), big.mark = ","),
                  object.size(edge_list) / 1e6))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for '%s' ...", var_name))
    compute_neighbor_stats_fast(cell_data, edge_list, var_name)
  }

  message("Done.")
  return(cell_data)
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Usage (drop-in replacement for the original outer loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data <- run_neighbor_features(cell_data, id_order, rook_neighbors_unique)
#
# Then proceed to prediction with the already-trained Random Forest model
# exactly as before. The new columns have the same names and identical
# numerical values as the originals.
```

---

## Why This Works â€” Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor resolution** | 6.46M R-level iterations with string pasting + hash lookup | Single vectorized `data.table` keyed join (~25.8M rows) |
| **Stat computation** | 6.46M R-level iterations Ã— 5 vars = 32.3M iterations | 5 Ã— one grouped `data.table` aggregation over ~25.8M rows |
| **Memory for lookup** | List of 6.46M small integer vectors (~1â€“2 GB + GC overhead) | One 2-column integer `data.table` (~0.4 GB) |
| **Expected runtime** | 86+ hours | **~5â€“15 minutes** (dominated by the keyed joins and grouped aggregations) |
| **Numerical output** | max, min, mean of non-NA neighbor values | **Identical** â€” same subsetting logic, same aggregation functions |
| **RF model** | Trained model used downstream | **Untouched** â€” only the feature-engineering step is optimized |

The key insight is replacing millions of R-interpreter-level iterations with bulk vectorized operations via `data.table`'s radix-sort-based keyed joins and grouped aggregation, which are implemented in optimized C and handle this scale comfortably within 16 GB RAM.
END_RESPONSE>>>
