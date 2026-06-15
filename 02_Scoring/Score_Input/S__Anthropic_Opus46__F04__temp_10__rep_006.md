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
run_id: Anthropic_F04_T10_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is clearly in `build_neighbor_lookup`, not in `compute_neighbor_stats` or the Random Forest. Here's why:

**`build_neighbor_lookup`** iterates over ~6.46 million rows via `lapply`, and for each row it performs:
1. A named-vector character lookup (`id_to_ref[as.character(...)]`)
2. Construction of `paste(..., sep="_")` key strings for every neighbor
3. Another named-vector character lookup (`idx_lookup[neighbor_keys]`) against a 6.46-million-element named character vector

Named vector lookups in R are O(n) scans or hash lookups on character keys. Doing this ~6.46 million times, each time against a 6.46M-element named vector, is catastrophically slow. The `paste`/character coercion per row compounds this. Estimated complexity: **O(N Ã— k Ã— lookup_cost)** where N â‰ˆ 6.46M, k â‰ˆ average neighbor count (~4 for rook), and lookup_cost is nontrivial for large named vectors.

**`compute_neighbor_stats`** is also row-level `lapply` over 6.46M rows but does only numeric subsetting and simple aggregationsâ€”much cheaper, though still improvable.

## Optimization Strategy

1. **Replace character-key lookups with integer-indexed hash maps** using `data.table` or environment-based hashing.
2. **Vectorize the neighbor lookup construction**: instead of per-row `lapply`, build a flat edge list (source_row â†’ neighbor_row) for all rows at once using `data.table` joins, then use `split()` or group-by operations.
3. **Vectorize `compute_neighbor_stats`**: use `data.table` grouped aggregation on the flat edge list instead of row-wise `lapply`.
4. **Avoid 6.46M-iteration `lapply` entirely.**

## Optimized R Code

```r
library(data.table)

# ===========================================================================
# STEP 1: Build a flat edge-list mapping every row to its neighbor rows
#         using vectorized data.table joins. Replaces build_neighbor_lookup.
# ===========================================================================

build_neighbor_edgelist <- function(data_dt, id_order, neighbors) {
  # data_dt: a data.table with columns id, year (and others)
  #          and a column .row_idx = 1:.N already added
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors)

  # --- 1a. Expand the nb object into a flat edge list of (cell_id -> neighbor_cell_id)

  n_cells <- length(id_order)
  from_ref <- rep(seq_len(n_cells), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  # Remove zero-length / self-referencing entries produced by nb objects
  valid <- to_ref > 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]

  edges_cell <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
  # edges_cell has ~1,373,394 rows (one per directed rook-neighbor pair)

  # --- 1b. Cross-join with years to get (from_id, year) -> (to_id, year) pairs
  years <- sort(unique(data_dt$year))

  # Use CJ and merge to build the full edgelist keyed to row indices
  # First create a lookup: (id, year) -> row_idx
  row_lookup <- data_dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # Expand edges_cell Ã— years  (~1.37M Ã— 28 = ~38.5M rows â€” fits in RAM)
  edges_full <- edges_cell[, .(from_id, to_id, year = rep(list(years), .N)),
                           env = list()][
    , .(year = unlist(year)), by = .(from_id, to_id)]

  # Alternative, more memory-efficient expansion:
  edges_full <- CJ(edge_idx = seq_len(nrow(edges_cell)), year = years)
  edges_full[, `:=`(from_id = edges_cell$from_id[edge_idx],
                     to_id   = edges_cell$to_id[edge_idx])]
  edges_full[, edge_idx := NULL]

  # --- 1c. Map (from_id, year) -> source row index
  edges_full[row_lookup, src_row := i..row_idx, on = .(from_id = id, year)]

  # --- 1d. Map (to_id, year) -> neighbor row index
  edges_full[row_lookup, nbr_row := i..row_idx, on = .(to_id = id, year)]

  # Drop edges where either side has no matching row
  edges_full <- edges_full[!is.na(src_row) & !is.na(nbr_row)]

  # Keep only what we need
  edges_full[, .(src_row, nbr_row)]
}

# ===========================================================================
# STEP 2: Vectorized neighbor stats via data.table grouped aggregation.
#         Replaces compute_neighbor_stats + the per-variable loop.
# ===========================================================================

compute_all_neighbor_features <- function(data_dt, edges_dt, var_names) {
  # edges_dt has columns: src_row, nbr_row
  # For each var_name, compute max, min, mean of neighbor values grouped by src_row

  for (vn in var_names) {
    cat("Computing neighbor features for:", vn, "\n")

    # Pull neighbor values into the edge table
    edges_dt[, nbr_val := data_dt[[vn]][nbr_row]]

    # Grouped aggregation â€” fully vectorized, single pass per variable
    agg <- edges_dt[!is.na(nbr_val),
                    .(nb_max  = max(nbr_val),
                      nb_min  = min(nbr_val),
                      nb_mean = mean(nbr_val)),
                    by = src_row]

    # Initialize new columns with NA
    max_col  <- paste0("max_",  vn)
    min_col  <- paste0("min_",  vn)
    mean_col <- paste0("mean_", vn)

    data_dt[, (max_col)  := NA_real_]
    data_dt[, (min_col)  := NA_real_]
    data_dt[, (mean_col) := NA_real_]

    # Assign results back by row index
    data_dt[agg$src_row, (max_col)  := agg$nb_max]
    data_dt[agg$src_row, (min_col)  := agg$nb_min]
    data_dt[agg$src_row, (mean_col) := agg$nb_mean]

    # Clean up
    edges_dt[, nbr_val := NULL]
  }

  invisible(data_dt)
}

# ===========================================================================
# STEP 3: Main execution â€” drop-in replacement for the original outer loop
# ===========================================================================

# Convert to data.table if not already; add row index
cell_dt <- as.data.table(cell_data)
cell_dt[, .row_idx := .I]

cat("Building neighbor edge list (vectorized)...\n")
edges_dt <- build_neighbor_edgelist(cell_dt, id_order, rook_neighbors_unique)
cat(sprintf("Edge list: %s rows\n", format(nrow(edges_dt), big.mark = ",")))

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
compute_all_neighbor_features(cell_dt, edges_dt, neighbor_source_vars)

# Remove helper column, convert back to data.frame for downstream RF predict()
cell_dt[, .row_idx := NULL]
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is untouched.
# Column names (max_ntl, min_ntl, mean_ntl, ...) match the original schema,
# and the numerical values are identical (same max/min/mean operations on the
# same neighbor sets), preserving the original numerical estimand.
```

## Why This Is Fast

| Aspect | Original | Optimized |
|---|---|---|
| Lookup structure | Named character vector scan Ã—6.46M | `data.table` keyed binary-search join (once) |
| Neighbor resolution | Per-row `paste` + character match | Flat integer edge list built vectorized |
| Stats computation | `lapply` over 6.46M rows, R-level loop | `data.table` grouped `[.data.table` â€” C-level group-by |
| Estimated wall time | **86+ hours** | **~2â€“5 minutes** on same laptop |
| Peak memory | Moderate (but slow) | ~38.5M-row edge table Ã— 2 int cols â‰ˆ **0.6 GB** + aggregation overhead; fits in 16 GB |

The numerical results are identical: for each `(cell, year)` row, the same set of rook-neighbor rows is identified and the same `max`, `min`, `mean` are computed. The trained Random Forest model is never touched.
END_RESPONSE>>>
