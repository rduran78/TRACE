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
run_id: Anthropic_F12_T07_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized concatenation step and runs in seconds to low minutes at most. The `lapply` inside `compute_neighbor_stats()` is also relatively lean â€” it's just indexing into a pre-extracted numeric vector and computing three summary statistics.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Massive per-row string operations**: For each of the ~6.46 million rows, it calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` to construct string keys, then does **named-vector lookup** (`idx_lookup[neighbor_keys]`) against a character vector of 6.46 million names. Named vector lookup by character key in R is O(n) per probe in the worst case (it uses a linear hash with potential collisions), and this is done for every neighbor of every row. With ~1.37 million directed neighbor relationships spread across 28 years, the total number of key lookups is on the order of **tens of millions**, each probing a 6.46-million-entry named character vector.

2. **Redundant work across years**: The neighbor *structure* is identical across all 28 years â€” the same grid cell has the same rook neighbors every year. Yet `build_neighbor_lookup()` recomputes the string-key-based lookup independently for every cell-year row, doing 28Ã— the work necessary.

3. **`as.character()` and `paste()` allocations**: Each of the 6.46 million iterations allocates new character vectors for `as.character(data$id[i])`, the `paste(...)` call, and the subsetting result. This creates enormous GC (garbage collection) pressure.

In contrast, `compute_neighbor_stats()` merely indexes a numeric vector and computes `max/min/mean` â€” these are trivially fast operations. And `do.call(rbind, result)` on a list of 6.46M three-element vectors is equivalent to `matrix(unlist(result), ncol=3, byrow=TRUE)`, which takes seconds.

**Conclusion**: The bottleneck is `build_neighbor_lookup()` â€” specifically its per-row string construction and named-character-vector lookup over a 6.46M-entry table, repeated redundantly for all 28 years. This is where the 86+ hours are being spent.

---

## Optimization Strategy

1. **Eliminate string keys entirely.** Replace the character-based `idx_lookup` with integer arithmetic. Since the data is a panel (cell Ã— year), we can map `(id, year)` â†’ row index using an integer hash (e.g., via `data.table` or a direct integer-keyed environment/match).

2. **Exploit year-invariant neighbor structure.** Build the neighbor mapping once at the cell level (344,208 cells), then expand to cell-year rows via vectorized integer indexing â€” no per-row `lapply` needed.

3. **Vectorize `compute_neighbor_stats()`.** Replace the per-row `lapply` with a single grouped operation using `data.table` or vectorized indexing with `rowMeans`-style operations.

4. **Replace `do.call(rbind, ...)` with `matrix(unlist(...), ...)`** as a minor secondary improvement.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED build_neighbor_lookup
# ==============================================================================
# Key insight: neighbor structure is IDENTICAL across all 28 years.
# We build a cell-level neighbor list once, then expand to row-level
# using pure integer arithmetic â€” no string keys, no paste, no named vectors.

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  
  dt <- as.data.table(data)
  
  # Step 1: Create integer mappings
  # Map each unique id to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map each unique year to a sequential integer
  years_sorted <- sort(unique(dt$year))
  n_years <- length(years_sorted)
  year_to_idx <- setNames(seq_along(years_sorted), as.character(years_sorted))
  
  # Step 2: Build a fast (id, year) -> row_index lookup using data.table
  # Add row indices to dt
  dt[, row_idx := .I]
  setkey(dt, id, year)
  
  # Step 3: Build cell-level neighbor ID list (done once for 344K cells, not 6.46M rows)
  # For each cell in id_order, get the IDs of its rook neighbors
  n_cells <- length(id_order)
  
  # Precompute: for each cell index in id_order, what are the neighbor cell IDs?
  cell_neighbor_ids <- lapply(seq_len(n_cells), function(ref) {
    nb_indices <- neighbors[[ref]]
    # Remove 0s (spdep uses 0 for no-neighbor sentinel)
    nb_indices <- nb_indices[nb_indices > 0L]
    if (length(nb_indices) == 0L) return(integer(0))
    id_order[nb_indices]
  })
  names(cell_neighbor_ids) <- as.character(id_order)
  
  # Step 4: For each row in data, look up neighbor rows using data.table join
  # Instead of lapply over 6.46M rows, we do this vectorized:
  
  # Build an edge list: (focal_id, focal_year, neighbor_id) 
  # Then join to get neighbor row indices
  
  # First, create a compact representation: for each cell, its neighbors
  # Expand to an edge data.table
  from_ids <- rep(id_order, times = vapply(cell_neighbor_ids, length, integer(1)))
  to_ids   <- unlist(cell_neighbor_ids, use.names = FALSE)
  
  edges <- data.table(focal_id = from_ids, neighbor_id = to_ids)
  
  # Cross with years to get (focal_id, year, neighbor_id)
  # But this could be huge. Instead, we work row-by-row more cleverly:
  # For each row i with (id_i, year_i), neighbors are rows with (neighbor_id, year_i)
  
  # Build the lookup: for each (id, year), what is the row index?
  row_lookup <- dt[, .(id, year, row_idx)]
  setkey(row_lookup, id, year)
  
  # For each row, get its cell's neighbor IDs, then find rows with those IDs and same year
  # We do this via a merge rather than per-row lapply
  
  # Create focal edge table: each row's id and year, crossed with its neighbors
  # focal_row_idx, focal_id, focal_year -> expand by neighbor_id
  
  focal_info <- dt[, .(focal_row = row_idx, focal_id = id, focal_year = year)]
  
  # Map focal_id to its ref index in id_order
  focal_info[, ref_idx := id_to_ref[as.character(focal_id)]]
  
  # Now we need to expand: for each focal row, one record per neighbor
  # This is the key vectorized step
  
  # Get number of neighbors per cell
  n_neighbors_per_cell <- vapply(cell_neighbor_ids, length, integer(1))
  
  # Map each focal row to its cell's neighbor count
  focal_info[, n_nb := n_neighbors_per_cell[ref_idx]]
  
  # Expand focal_info: repeat each row n_nb times
  expanded <- focal_info[rep(seq_len(.N), n_nb)]
  
  # Add the neighbor_id column
  # For each focal row, the neighbor IDs come from cell_neighbor_ids[[ref_idx]]
  # We need to generate the neighbor_id vector in the same order as the expansion
  neighbor_id_vec <- unlist(cell_neighbor_ids[focal_info$ref_idx], use.names = FALSE)
  expanded[, neighbor_id := neighbor_id_vec]
  
  # Now join to find the row index of (neighbor_id, focal_year)
  setnames(expanded, "focal_year", "year")
  expanded[, id := neighbor_id]
  
  # Keyed join
  expanded[row_lookup, neighbor_row := i.row_idx, on = .(id, year)]
  
  # Remove NAs (neighbors that don't exist in data for that year)
  expanded <- expanded[!is.na(neighbor_row)]
  
  # Now split by focal_row to get the neighbor_lookup list
  # Sort by focal_row for efficient splitting
  setkey(expanded, focal_row)
  
  # Pre-allocate result list
  n_rows <- nrow(dt)
  neighbor_lookup <- vector("list", n_rows)
  
  # Split neighbor_row by focal_row
  split_result <- split(expanded$neighbor_row, expanded$focal_row)
  
  # Fill in the lookup (rows with no neighbors remain NULL -> handle in stats)
  for (nm in names(split_result)) {
    neighbor_lookup[[as.integer(nm)]] <- as.integer(split_result[[nm]])
  }
  
  return(neighbor_lookup)
}

# ==============================================================================
# OPTIMIZED compute_neighbor_stats
# ==============================================================================
# Replace lapply + do.call(rbind, ...) with fully vectorized data.table operations

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  
  # Build a flat edge table: (focal_row, neighbor_value)
  focal_rows <- rep(seq_len(n), times = vapply(neighbor_lookup, function(x) {
    if (is.null(x)) 0L else length(x)
  }, integer(1)))
  
  neighbor_rows <- unlist(neighbor_lookup, use.names = FALSE)
  
  if (length(neighbor_rows) == 0) {
    # Edge case: no neighbors at all
    out <- matrix(NA_real_, nrow = n, ncol = 3)
    colnames(out) <- paste0("neighbor_", c("max", "min", "mean"), "_", var_name)
    return(out)
  }
  
  neighbor_vals <- vals[neighbor_rows]
  
  # Use data.table for grouped aggregation
  edge_dt <- data.table(
    focal = focal_rows,
    nval  = neighbor_vals
  )
  
  # Remove NA neighbor values
  edge_dt <- edge_dt[!is.na(nval)]
  
  # Compute grouped stats
  stats_dt <- edge_dt[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal]
  
  # Map back to full row set
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  out[stats_dt$focal, 1] <- stats_dt$nb_max
  out[stats_dt$focal, 2] <- stats_dt$nb_min
  out[stats_dt$focal, 3] <- stats_dt$nb_mean
  
  colnames(out) <- paste0("neighbor_", c("max", "min", "mean"), "_", var_name)
  return(out)
}

# ==============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# ==============================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  for (j in seq_len(ncol(stats_mat))) {
    data[[colnames(stats_mat)[j]]] <- stats_mat[, j]
  }
  return(data)
}

# ==============================================================================
# MAIN PIPELINE (drop-in replacement)
# ==============================================================================
# Build the lookup ONCE (this is the former bottleneck â€” now fast)
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_lookup)
}

# The trained Random Forest model is untouched â€” use it for prediction as before.
# predict(rf_model, cell_data)  # unchanged
```

---

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| **Alleged bottleneck** | `do.call(rbind, result)` and list binding in `compute_neighbor_stats()` | **Wrong.** This step is a minor cost. |
| **True bottleneck** | â€” | `build_neighbor_lookup()`: 6.46M iterations of `paste()` + named-character-vector lookup against a 6.46M-entry index, redundantly recomputing year-invariant structure 28Ã—. |
| **Root cause** | â€” | O(tens of millions) string allocations and hash lookups in a single-threaded `lapply`. |
| **Fix** | â€” | Replace string keys with integer-keyed `data.table` joins; exploit year-invariant neighbor structure via vectorized edge-list expansion and merge. |
| **Expected speedup** | â€” | From ~86+ hours to **minutes** (vectorized joins on ~38M edges). |
| **RF model** | â€” | Preserved unchanged; only the feature-engineering pipeline is optimized. |
| **Numerical equivalence** | â€” | Identical `max`, `min`, `mean` computed on the same neighbor sets â†’ same estimand. |
END_RESPONSE>>>
