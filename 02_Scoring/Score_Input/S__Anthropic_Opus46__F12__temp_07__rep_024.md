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
run_id: Anthropic_F12_T07_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding in `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not ideal, it is a one-time matrix assembly per variable (5 calls total). Each call binds ~6.46M rows Ã— 3 columns â€” this takes seconds, not hours.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** Inside the `lapply`, for every single row `i`, the code calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` and then does a named-vector lookup `idx_lookup[neighbor_keys]`. Named vector lookup in R is hash-based but has significant per-call overhead, and this is executed ~6.46 million times.

2. **`as.character()` conversion on every iteration.** `id_to_ref[as.character(data$id[i])]` converts a single value to character 6.46 million times.

3. **The total number of neighbor key lookups is enormous.** With ~1.37 million directed neighbor relationships spread across 344,208 cells and 28 years, each cell has ~4 neighbors on average (rook contiguity). That means the inner `lapply` performs roughly 6.46M Ã— 4 = ~25.8 million `paste` + hash-lookup operations, all in an interpreted R loop.

4. **The lookup is rebuilt identically for every row-year of the same cell.** A cell's rook neighbors don't change across years. Yet the code recomputes the neighbor mapping for all 28 year-copies of every cell independently. This is a 28Ã— redundancy.

**Summary:** `build_neighbor_lookup()` is O(N Ã— k) with large constant factors in interpreted R, where N = 6.46M and k â‰ˆ 4. This dominates runtime by orders of magnitude. `compute_neighbor_stats()` is comparatively fast because `vals[idx]` is vectorized integer subsetting.

---

## Optimization Strategy

1. **Build the neighbor lookup at the cell level (344K cells), not the cell-year level (6.46M rows).** Since rook neighbors are time-invariant, compute a cell-level neighbor map once, then expand to row-level using vectorized operations.

2. **Replace per-row `paste`/named-vector lookups with integer-indexed operations.** Use `data.table` or base R vectorized merges to map (cell_id, year) â†’ row index, then expand cell-level neighbors to row-level neighbors entirely via vectorized joins.

3. **Replace `do.call(rbind, ...)` with pre-allocated matrix output** in `compute_neighbor_stats()` for a modest secondary speedup.

4. **Preserve the trained Random Forest model** â€” we only change how features are computed, not the features themselves. The numerical output is identical.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# OPTIMIZED build_neighbor_lookup
# =============================================================================
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert data to data.table for fast indexed operations
  dt <- as.data.table(data)
  dt[, row_idx := .I]
  
  # --- Step 1: Build CELL-level neighbor map (344K cells, not 6.46M rows) ---
  # id_to_ref: map cell id -> position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # For each cell, find its neighbor cell ids (time-invariant)
  # neighbors is an nb object: neighbors[[ref_idx]] gives integer indices into id_order
  # We build an edge list: (cell_id, neighbor_cell_id)
  edge_list <- rbindlist(lapply(seq_along(id_order), function(ref_idx) {
    nb_indices <- neighbors[[ref_idx]]
    if (length(nb_indices) == 0L) return(NULL)
    data.table(cell_id = id_order[ref_idx],
               neighbor_cell_id = id_order[nb_indices])
  }))
  
  # --- Step 2: Build row-index lookup keyed on (id, year) ---
  setkey(dt, id, year)
  
  # --- Step 3: Expand to row-level neighbor lookup via vectorized join ---
  # For each row i, its neighbors are all rows with (neighbor_cell_id, same year).
  # Instead of looping 6.46M times, we join:
  #   rows (with id, year, row_idx) -> edge_list (cell_id -> neighbor_cell_id) -> rows again
  
  # Get unique (id, year, row_idx) â€” each row
  row_info <- dt[, .(id, year, row_idx)]
  
  # Join row_info with edge_list on id == cell_id to get neighbor_cell_ids
  setnames(edge_list, c("cell_id", "neighbor_cell_id"))
  setkey(edge_list, cell_id)
  setkey(row_info, id)
  
  # For each row, find its neighbor cells
  expanded <- edge_list[row_info, on = .(cell_id = id),
                        .(row_idx = i.row_idx,
                          year = i.year,
                          neighbor_cell_id = x.neighbor_cell_id),
                        allow.cartesian = TRUE]
  
  # Remove rows where there was no neighbor (cell on boundary with 0 neighbors won't appear)
  expanded <- expanded[!is.na(neighbor_cell_id)]
  
  # Now join to find the row index of each (neighbor_cell_id, year)
  neighbor_row_info <- dt[, .(neighbor_row_idx = row_idx, id, year)]
  setkey(neighbor_row_info, id, year)
  setkey(expanded, neighbor_cell_id, year)
  
  matched <- neighbor_row_info[expanded,
                                on = .(id = neighbor_cell_id, year = year),
                                .(row_idx = i.row_idx,
                                  neighbor_row_idx = x.neighbor_row_idx),
                                nomatch = NA]
  
  # Remove unmatched (neighbor cell-year not in data)
  matched <- matched[!is.na(neighbor_row_idx)]
  
  # --- Step 4: Convert to list indexed by row_idx ---
  setkey(matched, row_idx)
  n_rows <- nrow(data)
  
  # Split neighbor_row_idx by row_idx
  lookup <- vector("list", n_rows)
  
  # Fast split using data.table
  split_result <- split(matched$neighbor_row_idx, matched$row_idx)
  
  # Assign to lookup (split keys are character; convert)
  idx_keys <- as.integer(names(split_result))
  for (j in seq_along(idx_keys)) {
    lookup[[idx_keys[j]]] <- as.integer(split_result[[j]])
  }
  
  # Rows with no neighbors remain NULL; replace with integer(0)
  null_mask <- vapply(lookup, is.null, logical(1))
  lookup[null_mask] <- list(integer(0))
  
  lookup
}


# =============================================================================
# OPTIMIZED compute_neighbor_stats (pre-allocated matrix, no do.call(rbind))
# =============================================================================
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  result_mat <- matrix(NA_real_, nrow = n, ncol = 3)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0L) next
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(idx) == 0L) next
    result_mat[i, 1] <- max(neighbor_vals)
    result_mat[i, 2] <- min(neighbor_vals)
    result_mat[i, 3] <- mean(neighbor_vals)
  }
  
  result_mat
}


# =============================================================================
# ALTERNATIVE: Fully vectorized compute_neighbor_stats using data.table
# (avoids the 6.46M R-level loop entirely)
# =============================================================================
compute_neighbor_stats_vectorized <- function(data, matched_dt, var_name) {
  # matched_dt has columns: row_idx, neighbor_row_idx (from build step)
  vals <- data[[var_name]]
  
  dt <- copy(matched_dt)
  dt[, neighbor_val := vals[neighbor_row_idx]]
  dt <- dt[!is.na(neighbor_val)]
  
  agg <- dt[, .(nb_max  = max(neighbor_val),
                nb_min  = min(neighbor_val),
                nb_mean = mean(neighbor_val)),
            by = row_idx]
  
  n <- nrow(data)
  result_mat <- matrix(NA_real_, nrow = n, ncol = 3)
  result_mat[agg$row_idx, 1] <- agg$nb_max
  result_mat[agg$row_idx, 2] <- agg$nb_min
  result_mat[agg$row_idx, 3] <- agg$nb_mean
  
  result_mat
}


# =============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# =============================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_nb_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_nb_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_nb_mean")]] <- stats[, 3]
  data
}

# Vectorized version (even faster â€” no R loop at all)
compute_and_add_neighbor_features_vec <- function(data, var_name, matched_dt) {
  stats <- compute_neighbor_stats_vectorized(data, matched_dt, var_name)
  data[[paste0(var_name, "_nb_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_nb_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_nb_mean")]] <- stats[, 3]
  data
}


# =============================================================================
# MAIN PIPELINE (drop-in replacement)
# =============================================================================

# Option A: Using the list-based lookup (compatible with original interface)
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_lookup)
}

# Option B: Fully vectorized (fastest â€” keeps matched_dt in data.table form)
# Requires retaining matched_dt from the build step. Modified build below:

build_neighbor_matched_dt <- function(data, id_order, neighbors) {
  dt <- as.data.table(data)
  dt[, row_idx := .I]
  
  edge_list <- rbindlist(lapply(seq_along(id_order), function(ref_idx) {
    nb_indices <- neighbors[[ref_idx]]
    if (length(nb_indices) == 0L) return(NULL)
    data.table(cell_id = id_order[ref_idx],
               neighbor_cell_id = id_order[nb_indices])
  }))
  
  row_info <- dt[, .(id, year, row_idx)]
  
  expanded <- edge_list[row_info, on = .(cell_id = id),
                        .(row_idx = i.row_idx,
                          year = i.year,
                          neighbor_cell_id = x.neighbor_cell_id),
                        allow.cartesian = TRUE]
  expanded <- expanded[!is.na(neighbor_cell_id)]
  
  neighbor_row_info <- dt[, .(id, year, neighbor_row_idx = row_idx)]
  
  matched <- neighbor_row_info[expanded,
                                on = .(id = neighbor_cell_id, year = year),
                                .(row_idx = i.row_idx,
                                  neighbor_row_idx = x.neighbor_row_idx),
                                nomatch = NA]
  matched <- matched[!is.na(neighbor_row_idx)]
  setkey(matched, row_idx)
  matched
}

# Usage for Option B:
matched_dt <- build_neighbor_matched_dt(cell_data, id_order, rook_neighbors_unique)

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_vec(cell_data, var_name, matched_dt)
}

# The trained Random Forest model is untouched.
# The computed features (nb_max, nb_min, nb_mean) are numerically identical.
```

---

## Expected Performance Improvement

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M interpreted R iterations with `paste` + hash lookup | One vectorized `data.table` join (~seconds to low minutes) |
| `compute_neighbor_stats` (per variable) | 6.46M R-loop iterations + `do.call(rbind)` | **Option A:** Pre-allocated matrix (modest gain) / **Option B:** Single `data.table` grouped aggregation (~seconds) |
| **Total (5 variables)** | **~86+ hours** | **~5â€“15 minutes** |

The dominant speedup comes from eliminating the 6.46M-iteration interpreted R loop in `build_neighbor_lookup` and replacing it with vectorized `data.table` joins that exploit the time-invariance of the neighbor structure. The secondary speedup from fixing `compute_neighbor_stats` is real but comparatively minor â€” confirming that the colleague's diagnosis was aimed at the wrong function.
END_RESPONSE>>>
