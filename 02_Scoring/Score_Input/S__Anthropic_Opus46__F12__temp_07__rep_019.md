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
run_id: Anthropic_F12_T07_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized bind operation on pre-allocated 3-element vectors â€” it completes in seconds for this data size.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **6.46 million iterations of `paste()` + character hash lookups per row.** For each of the ~6.46M rows, the function calls `as.character()`, indexes into `id_to_ref`, calls `paste()` to build neighbor keys, and then does named-vector lookups into `idx_lookup`. Named vector lookup in R is O(n) per query in the worst case (hashing helps, but with ~6.46M names the hash table is enormous and collision-prone).

2. **Redundant recomputation across years.** The neighbor *topology* is purely spatial â€” cell A's neighbors are the same cells regardless of year. Yet `build_neighbor_lookup` recomputes neighbor keys for every cell-year row. With 344,208 cells Ã— 28 years, the spatial neighbor resolution is repeated 28 times unnecessarily.

3. **`paste()` string construction at scale.** Building ~6.46M keys and then for each row building multiple neighbor keys (average ~4 rook neighbors Ã— 6.46M â‰ˆ 25.8M `paste` calls inside the `lapply`) is extremely expensive.

4. **`idx_lookup` is a named character vector of length 6.46M.** Every `idx_lookup[neighbor_keys]` call does a name-based search into this massive vector. This is the single most expensive operation in the entire pipeline, dwarfing the `do.call(rbind, ...)`.

**Estimated cost breakdown:**
- `build_neighbor_lookup`: ~80+ hours (millions of named-vector lookups into a 6.46M-entry vector, plus string operations)
- `compute_neighbor_stats` (Ã—5 variables): ~minutes total (pure numeric indexing + simple arithmetic)
- `do.call(rbind, result)`: ~seconds per call

## Optimization Strategy

1. **Separate spatial topology from temporal indexing.** Build the neighbor lookup by exploiting the panel structure: compute spatial neighbors once (344K cells), then map to row indices using integer arithmetic, not string hashing.

2. **Replace named-vector lookups with integer-indexed structures.** Use `match()` once to create an integer mapping, then use direct integer indexing throughout.

3. **Vectorize `compute_neighbor_stats` using `data.table` or pre-allocated matrices** instead of `lapply` + `do.call(rbind, ...)` â€” a secondary optimization.

4. **Pre-compute a row-index matrix** keyed by (cell_integer_index, year_integer_index) so that finding "neighbor cell X in year Y" is a single matrix lookup O(1), not a hash-table search.

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build neighbor lookup efficiently
# ============================================================

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert to data.table for speed; keep original row order
  dt <- as.data.table(data)
  dt[, orig_row := .I]
  
  # --- Spatial mapping (done once for 344,208 cells, not 6.46M rows) ---
  # Map each id to a contiguous integer index matching id_order
  id_order_vec <- as.integer(id_order)
  n_cells <- length(id_order_vec)
  
  # id_to_pos: given a cell id, what is its position in id_order?
  # Use match for a one-time O(n) operation
  unique_ids_in_data <- sort(unique(dt$id))
  id_to_pos <- match(id_order_vec, id_order_vec)  # identity, but we need the reverse
  # Actually: we need  cell_id -> position in id_order
  # id_order[pos] == cell_id
  # So: given cell_id, pos = match(cell_id, id_order_vec)
  
  # --- Temporal mapping ---
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_idx <- match(years, years)  # 1..n_years
  names(year_to_idx) <- as.character(years)
  
  # --- Build a row-index matrix: row_matrix[cell_pos, year_idx] = row in data ---
  # This allows O(1) lookup of any (cell, year) -> row index
  
  # Map each row's cell id to its position in id_order
  dt[, cell_pos := match(id, id_order_vec)]
  dt[, year_idx := match(year, years)]
  
  # Pre-allocate matrix with NA
  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_matrix[cbind(dt$cell_pos, dt$year_idx)] <- dt$orig_row
  
  # --- Build neighbor lookup: for each row, find row indices of neighbors ---
  # neighbors[[cell_pos]] gives neighbor positions in id_order
  
  # Pre-expand: for each row, get neighbor row indices via matrix lookup
  # Vectorized approach using data.table
  
  # Build edge list of spatial neighbors (done once, ~1.37M edges for 344K cells)
  edge_list <- rbindlist(lapply(seq_len(n_cells), function(pos) {
    nb <- neighbors[[pos]]
    if (length(nb) == 0) return(NULL)
    data.table(cell_pos = pos, neighbor_pos = as.integer(nb))
  }))
  
  # For each row in data, we need: all rows whose cell is a neighbor AND same year
  # Strategy: join (cell_pos, year_idx) with edge_list to get (neighbor_pos, year_idx),
  # then look up row_matrix[neighbor_pos, year_idx]
  
  # Build a data.table of (orig_row, cell_pos, year_idx)
  row_info <- dt[, .(orig_row, cell_pos, year_idx)]
  
  # Join with edge_list on cell_pos
  # Result: for each orig_row, all (neighbor_pos, year_idx) pairs
  row_edges <- merge(row_info, edge_list, by = "cell_pos", allow.cartesian = TRUE)
  
  # Look up neighbor row indices from the matrix
  row_edges[, neighbor_row := row_matrix[cbind(neighbor_pos, year_idx)]]
  
  # Remove NAs (neighbor cell-year combinations not in data)
  row_edges <- row_edges[!is.na(neighbor_row)]
  
  # Build the lookup as a list indexed by orig_row
  # Sort by orig_row for efficient split
  setkey(row_edges, orig_row)
  
  n_rows <- nrow(dt)
  
  # Split neighbor_row by orig_row
  lookup_list <- vector("list", n_rows)
  
  # Use split (fast on keyed data.table)
  split_result <- split(row_edges$neighbor_row, row_edges$orig_row)
  
  # Fill in the list (some rows may have no neighbors)
  filled_indices <- as.integer(names(split_result))
  for (j in seq_along(filled_indices)) {
    lookup_list[[filled_indices[j]]] <- split_result[[j]]
  }
  
  # Rows with no neighbors remain NULL â€” convert to integer(0)
  for (j in seq_len(n_rows)) {
    if (is.null(lookup_list[[j]])) lookup_list[[j]] <- integer(0)
  }
  
  return(lookup_list)
}

# ============================================================
# STEP 2: Compute neighbor stats efficiently (vectorized)
# ============================================================

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  
  # Pre-allocate output matrix
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(out) <- paste0("neighbor_", c("max_", "min_", "mean_"), var_name)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0L) next
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    out[i, 1L] <- max(nv)
    out[i, 2L] <- min(nv)
    out[i, 3L] <- mean(nv)
  }
  
  return(out)
}

# ============================================================
# STEP 3: Wrapper to add features to data
# ============================================================

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  col_names <- colnames(stats)
  for (j in seq_along(col_names)) {
    data[[col_names[j]]] <- stats[, j]
  }
  return(data)
}

# ============================================================
# STEP 4: Full pipeline (drop-in replacement)
# ============================================================

# Build lookup ONCE (fast: minutes instead of 80+ hours)
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_lookup)
}

# The trained Random Forest model is untouched â€” use it for prediction as before:
# predictions <- predict(rf_model, cell_data)
```

## Expected Performance Improvement

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~80+ hours (string hashing, 6.46M named-vector lookups) | **~2â€“5 minutes** (integer matrix lookup, vectorized merge) |
| `compute_neighbor_stats` (Ã—5) | ~minutes | **~minutes** (comparable, slightly faster with pre-allocation) |
| `do.call(rbind, ...)` | ~seconds | **Eliminated** (direct matrix pre-allocation) |
| **Total** | **~86+ hours** | **~10â€“20 minutes** |

The key insight: the original code treated a **panel data problem** as a **flat row problem**, doing expensive string-based lookups 6.46 million times. By factoring the problem into spatial topology (344K cells, computed once) Ã— temporal index (28 years, mapped via integer matrix), we reduce the core operation from O(6.46M Ã— string-hash) to O(1.37M spatial edges) + O(1) matrix lookups, yielding roughly a **300â€“500Ã— speedup**. The trained Random Forest model and all numerical outputs are preserved exactly.
END_RESPONSE>>>
