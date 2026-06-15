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
run_id: Anthropic_F12_T00_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, result)` on ~6.46 million small 3-element vectors is not free, it is a single vectorized bind operation per variable (only 5 calls total). The actual cost there is modest â€” binding 6.46M rows Ã— 3 columns takes seconds, not hours.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** Inside the `lapply`, for every single row `i`, the code:
   - Calls `as.character(data$id[i])` â€” 6.46M times.
   - Looks up `id_to_ref[as.character(...)]` â€” named character vector lookup, O(n) hash probe Ã— 6.46M.
   - Retrieves `id_order[neighbors[[ref_idx]]]` â€” fine, but done per-row.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” creates ~4 strings per row (avg ~4 rook neighbors) Ã— 6.46M rows â‰ˆ 25.8M string allocations.
   - Looks up `idx_lookup[neighbor_keys]` â€” probing a **named vector of 6.46 million entries** using string keys, ~4 times per row = ~25.8M hash lookups into a 6.46M-entry hash table.

2. **This is done once but takes the vast majority of the 86+ hours.** The `lapply` over 6.46M rows with string operations and named-vector lookups is catastrophically slow in R. Named vector lookup in R is not O(1) â€” it uses linear hashing with poor cache behavior at this scale.

3. **The lookup is redundant across variables.** The `neighbor_lookup` is built once and reused for all 5 variables â€” that's correct. But the construction itself is the wall-clock killer.

4. **`compute_neighbor_stats` is comparatively cheap.** It's just integer indexing into a numeric vector (`vals[idx]`) and computing `max/min/mean` â€” all fast vectorized operations. The `do.call(rbind, ...)` on a list of 6.46M length-3 vectors takes ~10-30 seconds, not hours.

## Optimization Strategy

1. **Eliminate all string operations.** Replace `paste(id, year)` key construction and named-vector string lookups with pure integer arithmetic. Create a direct integer mapping from `(id_index, year_index)` â†’ row number using a matrix or a computed offset, since years are contiguous (1992â€“2019, 28 years).

2. **Vectorize the neighbor lookup construction.** Instead of an `lapply` over 6.46M rows, expand the neighbor list (which is per-cell, ~344K entries) into a flat edge list, then broadcast across all 28 years using vectorized integer operations.

3. **Replace `do.call(rbind, lapply(...))` with pre-allocated matrix and direct column computation** using vectorized grouped operations (via `data.table` or manual sparse-vector indexing).

4. **Preserve the trained Random Forest model** â€” we only change feature engineering, not the model. The numerical outputs (max, min, mean of neighbor values) remain identical.

## Working R Code

```r
library(data.table)

# =============================================================================
# OPTIMIZED build_neighbor_lookup
# =============================================================================
# Key insight: since years are contiguous 1992-2019 (28 years) and every cell
# appears in every year, we can compute a direct integer mapping:
#   row_number = (cell_index - 1) * n_years + year_index
# This eliminates ALL string operations.

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert to data.table for fast operations if not already
  dt <- as.data.table(data)
  
  n_cells <- length(id_order)
  years <- sort(unique(dt$year))
  n_years <- length(years)
  
  # Create integer mappings
  id_to_cellidx <- setNames(seq_along(id_order), as.character(id_order))
  year_to_yearidx <- setNames(seq_along(years), as.character(years))
  
  # Ensure data is sorted by (id, year) so that row = (cellidx-1)*n_years + yearidx
  # First, compute cellidx and yearidx for each row
  dt[, cellidx := id_to_cellidx[as.character(id)]]
  dt[, yearidx := year_to_yearidx[as.character(year)]]
  
  # Sort by cellidx, then yearidx â€” this gives us a predictable row layout
  setorder(dt, cellidx, yearidx)
  
  # Now row i in dt corresponds to cellidx = ((i-1) %/% n_years) + 1,
  #                                 yearidx = ((i-1) %% n_years) + 1
  # Verify this assumption holds (all cells have all years):
  stopifnot(nrow(dt) == n_cells * n_years)
  
  # Build flat neighbor edge list at the CELL level (not cell-year level)
  # neighbors[[c]] gives the neighbor indices for cell c in id_order
  # We expand this to cell-year level using integer arithmetic
  
  # For each cell c with neighbor cells n1, n2, ..., nk:
  #   For each year y (yearidx 1..28):
  #     source row = (c - 1) * n_years + y
  #     neighbor rows = (n1 - 1) * n_years + y, (n2 - 1) * n_years + y, ...
  
  # Build the lookup as a list of length nrow(dt)
  # But do it vectorized: first build cell-level, then replicate across years
  
  # Cell-level neighbor list (already have this: neighbors)
  # Convert to cellidx-based if not already
  # neighbors is an nb object: neighbors[[i]] gives indices into id_order
  # So neighbors[[cellidx]] gives neighbor cell indices â€” already in cellidx space
  
  # Pre-compute neighbor count per cell
  n_neighbors <- lengths(neighbors)  # length n_cells
  
  # Flatten the cell-level neighbor list
  # For cell c, neighbors are neighbors[[c]]
  flat_source_cell <- rep(seq_len(n_cells), times = n_neighbors)
  flat_target_cell <- unlist(neighbors)
  
  # Now expand across years: each (source_cell, target_cell) pair generates
  # n_years entries, one per year
  n_edges_cell <- length(flat_source_cell)  # ~1,373,394 / 2 directed? 
  # Actually ~1,373,394 directed edges total
  
  # For each year, compute source_row and target_row
  # source_row = (source_cell - 1) * n_years + yearidx
  # target_row = (target_cell - 1) * n_years + yearidx
  
  # Expand: repeat each edge n_years times, cycle through years
  flat_source_cell_exp <- rep(flat_source_cell, each = n_years)
  flat_target_cell_exp <- rep(flat_target_cell, each = n_years)
  year_idx_exp <- rep(seq_len(n_years), times = n_edges_cell)
  
  flat_source_row <- (flat_source_cell_exp - 1L) * n_years + year_idx_exp
  flat_target_row <- (flat_target_cell_exp - 1L) * n_years + year_idx_exp
  
  # Now build the lookup list: for each source_row, collect all target_rows
  # Use split() which is vectorized and fast
  neighbor_lookup <- split(flat_target_row, flat_source_row)
  
  # split() returns a named list with names as character(source_row)
  # We need a list of length nrow(dt), indexed 1..nrow(dt)
  # Rows with no neighbors won't appear in the split result
  
  full_lookup <- vector("list", nrow(dt))
  idx_present <- as.integer(names(neighbor_lookup))
  full_lookup[idx_present] <- neighbor_lookup
  
  # Return both the reordered data and the lookup
  # IMPORTANT: dt is now reordered by (cellidx, yearidx). We must return
  # the reordered data so downstream code uses consistent row indices.
  
  # Remove helper columns
  dt[, c("cellidx", "yearidx") := NULL]
  
  list(data = as.data.frame(dt), neighbor_lookup = full_lookup)
}


# =============================================================================
# OPTIMIZED compute_neighbor_stats â€” fully vectorized, no lapply
# =============================================================================
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- nrow(data)
  
  # Flatten the lookup
  source_rows <- rep(seq_len(n), times = lengths(neighbor_lookup))
  target_rows <- unlist(neighbor_lookup)
  
  if (length(target_rows) == 0) {
    return(cbind(rep(NA_real_, n), rep(NA_real_, n), rep(NA_real_, n)))
  }
  
  neighbor_vals <- vals[target_rows]
  
  # Remove NAs: filter out entries where neighbor_vals is NA
  valid <- !is.na(neighbor_vals)
  source_rows_v <- source_rows[valid]
  neighbor_vals_v <- neighbor_vals[valid]
  
  # Use data.table for fast grouped aggregation
  edge_dt <- data.table(src = source_rows_v, val = neighbor_vals_v)
  
  agg <- edge_dt[, .(
    nmax = max(val),
    nmin = min(val),
    nmean = mean(val)
  ), by = src]
  
  # Initialize result matrix with NAs
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  result[agg$src, 1] <- agg$nmax
  result[agg$src, 2] <- agg$nmin
  result[agg$src, 3] <- agg$nmean
  
  result
}


# =============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# =============================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]
  data
}


# =============================================================================
# MAIN PIPELINE â€” drop-in replacement
# =============================================================================
# Step 1: Build optimized neighbor lookup (replaces the old build_neighbor_lookup)
message("Building optimized neighbor lookup...")
system.time({
  lookup_result <- build_neighbor_lookup_fast(
    cell_data, id_order, rook_neighbors_unique
  )
})

# IMPORTANT: use the reordered data from here on
cell_data <- lookup_result$data
neighbor_lookup <- lookup_result$neighbor_lookup

# Step 2: Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor statistics...")
system.time({
  for (var_name in neighbor_source_vars) {
    message("  Processing: ", var_name)
    cell_data <- compute_and_add_neighbor_features_fast(
      cell_data, var_name, neighbor_lookup
    )
  }
})

message("Done. Neighbor features added to cell_data.")

# The trained Random Forest model is untouched.
# The numerical estimand (max, min, mean of neighbor values) is preserved exactly.
# Predicted runtime: ~2-10 minutes instead of 86+ hours.
```

## Summary of Changes and Expected Speedup

| Component | Original | Optimized | Speedup Factor |
|---|---|---|---|
| `build_neighbor_lookup` | `lapply` over 6.46M rows with `paste()` + named-vector string lookup | Vectorized integer arithmetic + `split()` on flat edge arrays | ~1000â€“5000Ã— |
| `compute_neighbor_stats` | `lapply` over 6.46M rows + `do.call(rbind, ...)` | Flat vectorized indexing + `data.table` grouped aggregation | ~50â€“200Ã— |
| **Total estimated time** | **86+ hours** | **~2â€“10 minutes** | **~500â€“2500Ã—** |

**Key principle:** The original code performed **~6.46 million iterations** of string construction and hash-table probing in pure R. The optimized code replaces this with **vectorized integer arithmetic** (cell-to-row mapping via `(cellidx - 1) * n_years + yearidx`), a single `split()` call, and `data.table` grouped aggregation â€” all of which operate on contiguous integer/numeric vectors and leverage R's internal C-level loops.
END_RESPONSE>>>
