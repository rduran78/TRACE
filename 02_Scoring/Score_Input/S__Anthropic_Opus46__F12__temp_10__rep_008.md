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
run_id: Anthropic_F12_T10_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and "repeated list binding" inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. Here's why:

**`compute_neighbor_stats()` is actually efficient.** It uses `lapply` to build a list of fixed-length vectors (length 3), then calls `do.call(rbind, ...)` exactly once. There is no repeated list binding â€” it's a single final assembly step. For ~6.46M rows, `do.call(rbind, list_of_vectors)` on uniform-length numeric vectors is fast (seconds, not hours). This is a well-known R idiom and not a meaningful bottleneck at this scale.

**The true bottleneck is `build_neighbor_lookup()`.** This function executes an `lapply` over **every one of the ~6.46 million rows**, and inside the loop body it performs:

1. **`as.character(data$id[i])` and a named-vector lookup `id_to_ref[...]`** â€” repeated 6.46M times, each involving a hash/name lookup.
2. **`paste(neighbor_cell_ids, data$year[i], sep = "_")`** â€” creates character key vectors 6.46M times, each with ~4 neighbor keys (avg neighbors per cell â‰ˆ 4 for rook contiguity on a grid).
3. **`idx_lookup[neighbor_keys]`** â€” named-vector character lookup against a named vector of length 6.46M, repeated 6.46M times.

The `idx_lookup` named vector has **6.46 million names**. R's named-vector lookup is O(n) linear scan or uses an internal hash, but repeatedly querying it with character string construction inside a 6.46M-iteration loop is devastating. The `paste()` calls alone generate ~25.8 million temporary strings. This single function likely accounts for **>95% of the 86+ hour runtime**.

Additionally, this lookup is **year-invariant in structure**: every cell has the same neighbors across all 28 years. Yet the function redundantly recomputes the neighbor index set for every cell-year row, inflating work by a factor of 28.

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup` entirely** â€” eliminate the row-level `lapply` loop.
2. **Exploit the year-invariance**: compute neighbor relationships once per cell, then replicate across years using integer arithmetic.
3. **Replace named-vector character lookups** with direct integer indexing using `data.table` or `match()`-based precomputation.
4. **Vectorize `compute_neighbor_stats`** using a grouped operation on a pre-built edge table (or a flat vector gather + group-by-row aggregation).
5. **Preserve the trained Random Forest model** â€” only the feature-engineering pipeline changes, producing numerically identical output columns.

## Working R Code

```r
library(data.table)

# ===========================================================================
# OPTIMIZED PIPELINE â€” Preserves original numerical results exactly.
# ===========================================================================

#' Step 1: Build a flat edge table (row_index -> neighbor_row_index)
#'
#' Instead of building a per-row list in a 6.46M-iteration R loop with
#' character paste/lookup, we:
#'   (a) Build a cell-level edge list from the nb object (344K cells).
#'   (b) Cross with years via integer arithmetic to get row-level edges.
#'
#' Assumptions (from the original code's logic):
#'   - `data` is ordered consistently (we use data.table for safety).
#'   - Each row is uniquely identified by (id, year).
#'   - `id_order` gives the cell IDs in the order matching `neighbors` (nb object).
#'   - `neighbors[[k]]` gives integer indices into `id_order` for the
#'     neighbors of cell `id_order[k]`.

build_edge_table <- function(data, id_order, neighbors) {
  dt <- as.data.table(data)
  
  # --- (a) Cell-level edge list from nb object ---
  # neighbors[[k]] is an integer vector of indices into id_order
  # We need: from_cell_id -> to_cell_id
  n_cells <- length(id_order)
  from_ref <- rep(seq_len(n_cells), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)
  
  # Remove the spdep "no-neighbor" sentinel (0L) if present
  valid <- to_ref > 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]
  
  from_cell_id <- id_order[from_ref]
  to_cell_id   <- id_order[to_ref]
  
  cell_edges <- data.table(from_id = from_cell_id, to_id = to_cell_id)
  
  # --- (b) Map (id, year) -> row index in data ---
  dt[, row_idx := .I]
  
  # Unique years
  years <- sort(unique(dt$year))
  
  # Create lookup: keyed by (id, year) -> row_idx
  setkey(dt, id, year)
  
  # --- (c) Expand cell_edges Ã— years to row-level edges ---
  # For each year, join from_id and to_id to get from_row and to_row
  year_dt <- data.table(year = years)
  
  # Cross join edges with years
  row_edges <- cell_edges[, .(from_id, to_id)]
  row_edges <- row_edges[rep(seq_len(.N), length(years))]
  row_edges[, year := rep(years, each = nrow(cell_edges))]
  
  # Join to get from_row_idx
  row_edges[dt, on = .(from_id = id, year = year), from_row := i.row_idx]
  # Join to get to_row_idx (neighbor's row in same year)
  row_edges[dt, on = .(to_id = id, year = year), to_row := i.row_idx]
  
  # Drop edges where either side has no matching row (boundary / missing data)
  row_edges <- row_edges[!is.na(from_row) & !is.na(to_row)]
  
  # Return only the integer columns we need
  row_edges[, .(from_row, to_row)]
}


#' Step 2: Compute neighbor stats using the edge table (fully vectorized)
#'
#' For each row i in `data`, we need max, min, mean of var[neighbors of i].
#' This is a grouped aggregation on the edge table.

compute_neighbor_stats_fast <- function(data, edge_table, var_name) {
  dt <- as.data.table(data)
  n <- nrow(dt)
  
  vals <- dt[[var_name]]
  
  # Attach neighbor values to edge table
  edges <- copy(edge_table)
  edges[, neighbor_val := vals[to_row]]
  
  # Drop NA neighbor values (mirrors original: neighbor_vals[!is.na(...)])
  edges <- edges[!is.na(neighbor_val)]
  
  # Grouped aggregation
  agg <- edges[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = from_row]
  
  # Initialize result columns with NA (mirrors original default)
  max_col  <- rep(NA_real_, n)
  min_col  <- rep(NA_real_, n)
  mean_col <- rep(NA_real_, n)
  
  max_col[agg$from_row]  <- agg$nb_max
  min_col[agg$from_row]  <- agg$nb_min
  mean_col[agg$from_row] <- agg$nb_mean
  
  # Build column names consistent with original pipeline
  # (adjust naming convention to match whatever compute_and_add_neighbor_features used)
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  dt[, (col_max)  := max_col]
  dt[, (col_min)  := min_col]
  dt[, (col_mean) := mean_col]
  
  as.data.frame(dt)
}


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

# Build edge table ONCE (replaces build_neighbor_lookup)
# Expected: ~38.5M edges (1,373,394 directed edges Ã— 28 years)
# Runs in seconds, not hours.
message("Building edge table...")
edge_table <- build_edge_table(cell_data, id_order, rook_neighbors_unique)
message("Edge table built: ", nrow(edge_table), " row-level edges.")

# Compute neighbor features for each variable (replaces the for-loop)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var_name)
  cell_data <- compute_neighbor_stats_fast(cell_data, edge_table, var_name)
}

message("Done. Neighbor features added. RF model unchanged.")

# The trained Random Forest model is untouched â€” only the feature columns in
# cell_data have been populated, producing numerically identical values to
# the original pipeline.
```

## Performance Comparison

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-loop iterations with `paste()` + named-vector lookup against a 6.46M-name vector | One vectorized `data.table` join; runs in **~30â€“60 seconds** |
| `compute_neighbor_stats` (Ã—5 vars) | `lapply` over 6.46M rows (acceptable but not ideal) | `data.table` grouped aggregation on ~38.5M-row edge table; runs in **~10â€“20 seconds per variable** |
| `do.call(rbind, ...)` | Binding 6.46M 3-element vectors â€” actually **fast** (~2â€“5 sec), not a bottleneck | Eliminated (results written directly) |
| **Total estimated time** | **86+ hours** | **~2â€“5 minutes** |

## Summary

The colleague misidentified the bottleneck. `do.call(rbind, ...)` on a list of uniform-length numeric vectors is a near-trivial operation. The catastrophic bottleneck is `build_neighbor_lookup()`, which performs **6.46 million iterations** of character string construction (`paste`) and named-vector character-key lookups against a **6.46-million-entry named vector**, redundantly recomputing year-invariant neighbor structures 28 times. The fix is to vectorize the entire operation using `data.table` joins: build a cell-level edge list once from the `nb` object, cross it with years, join to row indices, and then aggregate neighbor statistics via grouped `data.table` operations. This reduces runtime from 86+ hours to minutes while producing numerically identical features, leaving the trained Random Forest model completely untouched.
END_RESPONSE>>>
