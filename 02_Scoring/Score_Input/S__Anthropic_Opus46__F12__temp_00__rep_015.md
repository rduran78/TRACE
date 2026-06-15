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
run_id: Anthropic_F12_T00_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized concatenation step per variable â€” only 5 calls total. Each inner function returns a fixed-length-3 numeric vector, so there is no repeated list *growth* (no quadratic copy pattern). This is O(n) and takes seconds, not hours.

**The true bottleneck is `build_neighbor_lookup()`.**

Specifically, inside the `lapply` over **6.46 million rows**, every iteration performs:

1. **`as.character(data$id[i])` â†’ named-vector lookup `id_to_ref[...]`**: character conversion and name-based lookup on a 344K-element named vector, repeated 6.46M times.
2. **`paste(neighbor_cell_ids, data$year[i], sep = "_")`**: string concatenation for every neighbor of every row (~1,373,394 directed relationships Ã— 28 years â‰ˆ tens of millions of `paste` calls).
3. **`idx_lookup[neighbor_keys]`**: name-based lookup on a **6.46-million-element** named character vector, repeated for every neighbor key of every row.

Named-vector lookup in R is **O(n)** linear scan (not hashed), so each of the ~6.46M iterations does a linear scan of a 6.46M-element vector for each neighbor key. This is effectively **O(nÂ² Â· k)** where k is the average neighbor count (~4 for rook contiguity). That is the source of the 86+ hour runtime.

`compute_neighbor_stats()`, by contrast, does only integer-indexed subsetting (`vals[idx]`) â€” which is O(1) per element â€” and the `do.call(rbind, ...)` is a single O(n) operation per variable.

## Optimization Strategy

1. **Replace all named-vector lookups with hash-table lookups** using R `environment`-based hashing or, better, `data.table` keyed joins.
2. **Vectorize `build_neighbor_lookup` entirely**: instead of looping row-by-row, expand the neighbor relationships once at the cell level, then join against all years simultaneously using `data.table` keyed merge â€” eliminating the 6.46M-iteration `lapply` and all per-row `paste`/lookup.
3. **Replace `do.call(rbind, ...)` with pre-allocated matrix** in `compute_neighbor_stats` (minor, but clean).

The key insight: rook neighbors are defined at the **cell** level (344K cells), not the cell-year level (6.46M rows). We should expand neighbors Ã— years via a vectorized join, not row-by-row string matching.

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. OPTIMIZED build_neighbor_lookup
#    Strategy: work at the cell level, then expand to cell-year via
#    data.table keyed join. No per-row paste or named-vector lookup.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert to data.table if not already; add original row index
  dt <- as.data.table(data)
  dt[, .row_idx := .I]

  # --- Step A: Build cell-level edge list (vectorized) ---
  # neighbors is an nb object: list of integer vectors (indices into id_order)
  n_cells <- length(id_order)
  # Number of neighbors per cell
  n_neighbors <- vapply(neighbors, length, integer(1))
  # Source cell index (into id_order) repeated for each neighbor
  src_idx <- rep(seq_len(n_cells), n_neighbors)
  # Destination cell index (into id_order)
  dst_idx <- unlist(neighbors, use.names = FALSE)

  # Map indices to actual cell IDs
  edge_dt <- data.table(
    src_id = id_order[src_idx],
    dst_id = id_order[dst_idx]
  )

  # --- Step B: Build a row-index lookup keyed on (id, year) ---
  key_dt <- dt[, .(id, year, .row_idx)]
  setkey(key_dt, id, year)

  # --- Step C: Get unique years ---
  years <- sort(unique(dt$year))

  # --- Step D: Cross-join edges Ã— years, then look up row indices ---
  # For each edge (src_id, dst_id) and each year, we need:
  #   - the row index of (src_id, year)  â†’ this is the "focal" row
  #   - the row index of (dst_id, year)  â†’ this is the neighbor row

  # Expand edges by years (vectorized via CJ-merge)
  edge_year <- edge_dt[, .(src_id, dst_id, year = rep(list(years), .N))]

  # More memory-efficient: use CJ directly
  edge_year <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
  edge_year[, src_id := edge_dt$src_id[edge_idx]]
  edge_year[, dst_id := edge_dt$dst_id[edge_idx]]
  edge_year[, edge_idx := NULL]

  # Look up focal row index
  edge_year[key_dt, focal_row := i..row_idx, on = .(src_id = id, year = year)]
  # Look up neighbor row index
  edge_year[key_dt, nbr_row := i..row_idx, on = .(dst_id = id, year = year)]

  # Drop edges where either focal or neighbor row doesn't exist
  edge_year <- edge_year[!is.na(focal_row) & !is.na(nbr_row)]

  # --- Step E: Assemble into a list indexed by focal row ---
  setkey(edge_year, focal_row)
  n_rows <- nrow(dt)

  # Split neighbor row indices by focal row
  # Use edge_year to build the lookup list
  lookup_dt <- edge_year[, .(nbr_rows = list(nbr_row)), by = focal_row]

  # Initialize full lookup (empty integer(0) for rows with no neighbors)
  neighbor_lookup <- vector("list", n_rows)
  for (i in seq_len(n_rows)) {
    neighbor_lookup[[i]] <- integer(0)
  }
  neighbor_lookup[lookup_dt$focal_row] <- lookup_dt$nbr_rows

  return(neighbor_lookup)
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. OPTIMIZED compute_neighbor_stats
#    Strategy: fully vectorized using data.table grouping.
#    Avoids lapply over 6.46M rows entirely.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  n <- nrow(data)
  vals <- data[[var_name]]

  # Build an edge table: focal_row â†’ neighbor_row
  focal_lens <- vapply(neighbor_lookup, length, integer(1))
  focal_rows <- rep(seq_len(n), focal_lens)
  nbr_rows   <- unlist(neighbor_lookup, use.names = FALSE)

  if (length(nbr_rows) == 0) {
    # No neighbors at all â€” return all NA
    out <- matrix(NA_real_, nrow = n, ncol = 3)
    colnames(out) <- paste0(var_name, c("_max", "_min", "_mean"))
    return(out)
  }

  # Get neighbor values
  nbr_vals <- vals[nbr_rows]

  # Build data.table for grouped aggregation
  edge_dt <- data.table(focal = focal_rows, val = nbr_vals)
  # Remove NAs in neighbor values
  edge_dt <- edge_dt[!is.na(val)]

  # Aggregate
  agg <- edge_dt[, .(
    v_max  = max(val),
    v_min  = min(val),
    v_mean = mean(val)
  ), by = focal]

  # Map back to full row set
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  out[agg$focal, 1] <- agg$v_max
  out[agg$focal, 2] <- agg$v_min
  out[agg$focal, 3] <- agg$v_mean
  colnames(out) <- paste0(var_name, c("_max", "_min", "_mean"))

  return(out)
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. OPTIMIZED compute_and_add_neighbor_features
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  cnames <- colnames(stats_mat)
  for (j in seq_len(ncol(stats_mat))) {
    data[[cnames[j]]] <- stats_mat[, j]
  }
  return(data)
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. FULL PIPELINE (drop-in replacement)
#    Preserves the trained Random Forest model and original estimand.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Build the lookup ONCE (now takes minutes, not days)
neighbor_lookup <- build_neighbor_lookup_fast(
  cell_data, id_order, rook_neighbors_unique
)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_lookup
  )
}

# The trained Random Forest model is untouched.
# The new columns (e.g., ntl_max, ntl_min, ntl_mean, ...) are numerically
# identical to the originals, preserving the original estimand.
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M iterations, each doing named-vector O(n) scans on a 6.46M-element vector | Eliminated |
| `build_neighbor_lookup` | O(nÂ² Â· k) â‰ˆ trillions of character comparisons | O(n Â· k) via `data.table` keyed join; minutes |
| `compute_neighbor_stats` | O(n Â· k) via integer indexing (already fast); `do.call(rbind,...)` is O(n) | O(n Â· k) fully vectorized grouped aggregation; seconds |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes on the same laptop |
| RF model | Preserved âœ“ | Preserved âœ“ |
| Numerical output | Original estimand | Identical values âœ“ |

The colleague's diagnosis targeted a minor O(n) cost. The real culprit was the O(nÂ²Â·k) named-vector lookup pattern inside `build_neighbor_lookup()`, which is eliminated by vectorized `data.table` keyed joins.
END_RESPONSE>>>
