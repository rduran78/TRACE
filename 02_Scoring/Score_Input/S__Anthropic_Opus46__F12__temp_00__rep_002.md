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
run_id: Anthropic_F12_T00_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, result)` on a list of 6.46 million small 3-element vectors is not free, it is a single vectorized C-level operation that completes in seconds. There is no "repeated list binding" inside `compute_neighbor_stats()` â€” it uses `lapply` to build a list in one pass, then binds once. This is standard and efficient.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row `paste()` and character lookup in a named vector â€” 6.46 million times.** For each of the ~6.46M rows, the inner `lapply` function calls `as.character()`, indexes into `id_to_ref`, calls `paste()` to construct `neighbor_keys` (one per neighbor), and then performs named-vector lookup into `idx_lookup` (which itself is a named character vector of length ~6.46M). Named vector lookup in R is O(n) hash-table probing per call, and doing this millions of times with multi-element key vectors is catastrophically slow.

2. **The `idx_lookup` named vector has ~6.46M entries.** Each lookup into it requires hashing a character string and probing a large table. With an average of ~4 rook neighbors per cell, that's ~25.8 million individual character-key lookups into a 6.46M-entry named vector, nested inside a 6.46M-iteration `lapply`. The total character operations (paste + hash + match) dominate runtime.

3. **This function is called once, but it takes the vast majority of the 86+ hours.** `compute_neighbor_stats` is called only 5 times and each call is a simple numeric indexing operation (`vals[idx]`) which is O(1) per element â€” fast.

**In summary:** The bottleneck is the O(n Ã— k) character-key construction and named-vector lookup inside `build_neighbor_lookup()`, not the `do.call(rbind, ...)` in `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Eliminate all character key construction and named-vector lookup.** Replace `paste()`-based keys and named-vector indexing with pure integer arithmetic. Since years are contiguous (1992â€“2019), we can map each `(id, year)` pair to a row index using a precomputed integer matrix or a direct offset formula.

2. **Vectorize the neighbor lookup construction** using `data.table` for fast group-indexed operations, or â€” even better â€” use a direct integer-offset scheme: if data is sorted by `(id, year)`, then for a given cell `id` at row position `base_row`, all its year-rows are at known offsets, and neighbor cells' rows can be computed by integer arithmetic alone.

3. **Vectorize `compute_neighbor_stats()`** by unrolling the neighbor list into a long vector, using `grouping` to compute `max/min/mean` in one vectorized pass via `data.table` or `collapse`.

4. **Preserve the trained Random Forest model** â€” we only change feature-engineering code, not the model or the numerical results.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED build_neighbor_lookup
# ==============================================================================
# Strategy: avoid all paste() and named-vector character lookups.
# Use integer arithmetic with a pre-built (id, year) -> row_index hash via
# data.table, then expand neighbor pairs with integer indexing only.

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # data must have columns: id, year
  # id_order: vector of unique cell IDs in the order matching neighbors (nb object)
  # neighbors: list of integer neighbor indices (spdep nb object)

  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Build a keyed lookup: (id, year) -> row index
  # Using integer-keyed data.table join (very fast)
  setkey(dt, id, year)

  # Map each unique cell id to its position in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Get unique years (sorted)
  years <- sort(unique(dt$year))
  n_years <- length(years)

  # --- Build the full neighbor edge list at the cell level ---
  # For each cell position p in id_order, neighbors[[p]] gives neighbor positions
  # Expand to (focal_id, neighbor_id) pairs
  n_cells <- length(id_order)

  focal_pos <- rep(seq_len(n_cells), lengths(neighbors))
  neighbor_pos <- unlist(neighbors, use.names = FALSE)

  # Convert positions back to actual IDs
  focal_ids <- id_order[focal_pos]
  neighbor_ids <- id_order[neighbor_pos]

  # --- Cross with years to get (focal_id, year, neighbor_id) triples ---
  # Instead of crossing everything (expensive in memory), we build the lookup
  # row-by-row using a merge approach.

  # Create edge data.table at cell level
  edges <- data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)

  # For the row-index lookup, we need (id, year) -> row_idx
  # Build this as a keyed data.table for fast joins
  row_map <- dt[, .(id, year, row_idx)]
  setkey(row_map, id, year)

  # Now, for each row in dt, we need:
  #   1. Find which cell position this row's id maps to
  #   2. Get that cell's neighbor IDs
  #   3. Look up (neighbor_id, same year) in row_map
  #
  # We can do this as a large join:

  # Step 1: For each row in dt, attach its neighbor IDs
  dt[, cell_pos := id_to_pos[as.character(id)]]

  # Build focal_row -> neighbor_ids mapping
  # edges is at cell level; we need to expand by year
  # Approach: join dt with edges on focal_id, then join with row_map on (neighbor_id, year)

  cat("Building neighbor edge list with year expansion...\n")

  # Add year info to edges by joining with dt (focal side)
  # dt has (id, year, row_idx); edges has (focal_id, neighbor_id)
  setkey(edges, focal_id)

  # For each (focal_id, year) combination, get all neighbor_ids
  # Then look up (neighbor_id, year) -> neighbor_row_idx
  focal_rows <- dt[, .(focal_row = row_idx, focal_id = id, year)]

  # Join focal_rows with edges to get (focal_row, year, neighbor_id)
  setkey(focal_rows, focal_id)
  setkey(edges, focal_id)
  expanded <- edges[focal_rows, on = "focal_id", allow.cartesian = TRUE,
                    nomatch = NULL]
  # expanded has: focal_id, neighbor_id, focal_row, year

  # Now join with row_map to get neighbor_row_idx
  setnames(row_map, c("id", "year", "row_idx"), c("neighbor_id", "year", "neighbor_row_idx"))
  setkey(row_map, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  expanded <- row_map[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # Keep only matched rows (neighbor exists in that year)
  expanded <- expanded[!is.na(neighbor_row_idx)]

  cat("Assembling lookup list...\n")

  # Build the lookup list: for each focal_row, a vector of neighbor_row_idx
  setkey(expanded, focal_row)
  n_rows <- nrow(dt)

  # Split neighbor_row_idx by focal_row
  lookup_dt <- expanded[, .(neighbors = list(neighbor_row_idx)), by = focal_row]
  setkey(lookup_dt, focal_row)

  # Initialize full lookup (some rows may have no neighbors)
  neighbor_lookup <- vector("list", n_rows)
  neighbor_lookup[lookup_dt$focal_row] <- lookup_dt$neighbors

  # Fill empties with integer(0)
  empty <- which(vapply(neighbor_lookup, is.null, logical(1)))
  if (length(empty) > 0) {
    neighbor_lookup[empty] <- list(integer(0))
  }

  return(neighbor_lookup)
}


# ==============================================================================
# OPTIMIZED compute_neighbor_stats (vectorized, no per-row lapply)
# ==============================================================================
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)

  # Unroll the neighbor lookup into a long (focal_row, neighbor_row) table
  lens <- lengths(neighbor_lookup)
  focal_rows <- rep(seq_len(n), lens)
  neighbor_rows <- unlist(neighbor_lookup, use.names = FALSE)

  # Get neighbor values
  neighbor_vals <- vals[neighbor_rows]

  # Remove NAs
  valid <- !is.na(neighbor_vals)
  focal_rows <- focal_rows[valid]
  neighbor_vals <- neighbor_vals[valid]

  # Compute grouped stats using data.table
  stats_dt <- data.table(focal = focal_rows, val = neighbor_vals)
  agg <- stats_dt[, .(
    max_val  = max(val),
    min_val  = min(val),
    mean_val = mean(val)
  ), by = focal]

  # Map back to full result matrix
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  result[agg$focal, 1] <- agg$max_val
  result[agg$focal, 2] <- agg$min_val
  result[agg$focal, 3] <- agg$mean_val

  return(result)
}


# ==============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# ==============================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  cat(sprintf("Computing neighbor stats for: %s\n", var_name))
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)

  data[[paste0(var_name, "_neighbor_max")]]  <- stats_mat[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats_mat[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats_mat[, 3]

  return(data)
}


# ==============================================================================
# MAIN PIPELINE (drop-in replacement)
# ==============================================================================
# cell_data, id_order, rook_neighbors_unique are assumed to exist in the environment.
# The trained Random Forest model is NOT touched.

cat("Building optimized neighbor lookup...\n")
system.time({
  neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
})

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  system.time({
    cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_lookup)
  })
}

cat("Done. Neighbor features added. RF model unchanged.\n")
```

---

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| **Alleged bottleneck** | `do.call(rbind, result)` and list binding in `compute_neighbor_stats()` | **Wrong.** This is a single bind of a pre-built list â€” seconds at most. |
| **True bottleneck** | â€” | `build_neighbor_lookup()`: 6.46M iterations of `paste()` + named-vector character lookup into a 6.46M-entry table (~25.8M hash probes). This is where 86+ hours are spent. |
| **Optimization** | â€” | Replace all character-key operations with integer-indexed `data.table` equi-joins. Vectorize `compute_neighbor_stats()` by unrolling to a long table and using grouped aggregation. |
| **Expected speedup** | â€” | From 86+ hours to **minutes** (the join-based lookup and vectorized stats should complete in ~2â€“10 minutes on 16 GB RAM). |
| **RF model** | â€” | Untouched. Same numerical output (max, min, mean of neighbor values), same column names. |
END_RESPONSE>>>
