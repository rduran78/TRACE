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
run_id: Anthropic_F12_T03_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not ideal, it is a one-shot operation per variable (5 times total) and runs in seconds to low minutes at most. The `lapply` inside `compute_neighbor_stats()` does simple numeric indexing and three summary statistics â€” this is lightweight.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **`paste()` key construction and named-vector lookup over 6.46M rows:** `idx_lookup` is a named integer vector with ~6.46 million entries. For each of the 6.46 million rows, the function does `id_to_ref[as.character(...)]`, constructs `paste(neighbor_cell_ids, data$year[i], sep="_")`, and then performs `idx_lookup[neighbor_keys]` â€” a **named character lookup into a 6.46M-element named vector**. Named vector lookup in R is **O(n)** linear scan per query (R's named vectors use linear search, not hashing), making the total complexity approximately **O(n Ã— k Ã— N)** where n = 6.46M rows, k â‰ˆ average neighbors (~4 for rook), and N = 6.46M (the size of `idx_lookup`). This is catastrophically slow.

2. **Redundant recomputation across years:** Every cell has the same neighbors in every year. Yet the function recomputes neighbor relationships for all 6.46M cell-year rows independently, rather than computing once per cell (344,208 cells) and reusing across 28 years.

3. **`as.character()` coercion** is called 6.46 million times inside the `lapply`.

In summary: `build_neighbor_lookup()` performs ~6.46 million named-character lookups into a 6.46M-length named vector, each involving string construction and linear search. This is the operation that drives the 86+ hour runtime, not the `rbind` or the stats computation.

## Optimization Strategy

1. **Replace named-vector lookups with environment/hash-based lookups** (or better, pure integer indexing).
2. **Exploit the panel structure:** Neighbors are a spatial property â€” they don't change across years. Compute neighbor indices once per cell (344,208 cells), then expand to cell-years using integer arithmetic.
3. **Vectorize `compute_neighbor_stats()`** using `data.table` grouped operations or matrix indexing, eliminating the per-row `lapply` entirely.
4. **Preserve the trained Random Forest model** â€” we only change feature engineering, producing numerically identical columns.

## Working R Code

```r
library(data.table)

# ===========================================================================
# STEP 1: Build neighbor lookup ONCE per cell (not per cell-year)
#         Uses environment-based hashing for O(1) lookups.
# ===========================================================================

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # data must be a data.table (or will be converted)
  dt <- as.data.table(data)
  
  # --- Part A: Map each (id, year) to its row index using a hash (environment) ---
  # Build a hash: key = "id_year" -> value = row index
  id_year_hash <- new.env(hash = TRUE, parent = emptyenv(), size = nrow(dt))
  ids   <- dt$id
  years <- dt$year
  for (i in seq_len(nrow(dt))) {
    key <- paste0(ids[i], "_", years[i])
    id_year_hash[[key]] <- i
  }
  # Note: the above loop is O(n) with O(1) per insert into a hashed environment.
  # For 6.46M rows this takes ~30-60 seconds, vs. hours for the original.
  
  # --- Part B: Build cell-level neighbor mapping (only 344K cells) ---
  # id_order[j] gives the cell id at position j in the nb object
  # neighbors[[j]] gives the neighbor positions for cell at position j
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # For each cell, store its neighbor cell IDs (not row indices yet)
  # This is done once for 344,208 cells.
  cell_neighbor_ids <- lapply(seq_along(id_order), function(j) {
    nb_positions <- neighbors[[j]]
    if (length(nb_positions) == 0L || (length(nb_positions) == 1L && nb_positions[1] == 0L)) {
      return(integer(0))
    }
    id_order[nb_positions]
  })
  names(cell_neighbor_ids) <- as.character(id_order)
  
  # --- Part C: For each row, resolve neighbor row indices via hash ---
  unique_years <- sort(unique(years))
  
  # Pre-group rows by cell id for efficiency
  dt[, row_idx := .I]
  cell_year_map <- dt[, .(row_idx = row_idx, year = year), by = id]
  
  # Allocate result list
  neighbor_lookup <- vector("list", nrow(dt))
  
  # Process cell by cell (344K cells), expand across years
  unique_cell_ids <- unique(ids)
  
  # For speed, iterate by cell and resolve all its years at once
  cell_rows <- dt[, .(rows = list(row_idx), years = list(year)), by = id]
  
  for (ci in seq_len(nrow(cell_rows))) {
    cid       <- cell_rows$id[ci]
    row_idxs  <- cell_rows$rows[[ci]]   # row indices for this cell across years
    yr_vals   <- cell_rows$years[[ci]]   # corresponding years
    nb_cids   <- cell_neighbor_ids[[as.character(cid)]]  # neighbor cell IDs
    
    if (length(nb_cids) == 0L) {
      for (ri in seq_along(row_idxs)) {
        neighbor_lookup[[row_idxs[ri]]] <- integer(0)
      }
      next
    }
    
    # For each year this cell appears in, find neighbor rows
    for (ri in seq_along(row_idxs)) {
      yr <- yr_vals[ri]
      nb_rows <- integer(length(nb_cids))
      valid   <- logical(length(nb_cids))
      for (ni in seq_along(nb_cids)) {
        key <- paste0(nb_cids[ni], "_", yr)
        val <- id_year_hash[[key]]
        if (!is.null(val)) {
          nb_rows[ni] <- val
          valid[ni]   <- TRUE
        }
      }
      neighbor_lookup[[row_idxs[ri]]] <- nb_rows[valid]
    }
  }
  
  dt[, row_idx := NULL]
  neighbor_lookup
}

# ===========================================================================
# STEP 2: Vectorized compute_neighbor_stats using data.table
#         Avoids per-row lapply; uses fast group-by operations.
# ===========================================================================

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  n <- length(neighbor_lookup)
  vals <- data[[var_name]]
  
  # Build an edge list: parent_row -> neighbor_row
  # Then do grouped aggregation
  parent_lengths <- vapply(neighbor_lookup, length, integer(1))
  total_edges    <- sum(parent_lengths)
  
  parent_idx <- rep.int(seq_len(n), parent_lengths)
  child_idx  <- unlist(neighbor_lookup, use.names = FALSE)
  
  if (length(child_idx) == 0L) {
    # No neighbors at all â€” return NA matrix
    out <- matrix(NA_real_, nrow = n, ncol = 3)
    colnames(out) <- paste0("neighbor_", c("max", "min", "mean"), "_", var_name)
    return(out)
  }
  
  # Extract neighbor values
  neighbor_vals <- vals[child_idx]
  
  # Build data.table for grouped aggregation
  edge_dt <- data.table(
    parent = parent_idx,
    nval   = neighbor_vals
  )
  
  # Remove NAs before aggregation
  edge_dt <- edge_dt[!is.na(nval)]
  
  # Grouped aggregation â€” extremely fast in data.table
  agg <- edge_dt[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = parent]
  
  # Map back to full row set
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  out[agg$parent, 1] <- agg$nb_max
  out[agg$parent, 2] <- agg$nb_min
  out[agg$parent, 3] <- agg$nb_mean
  colnames(out) <- paste0("neighbor_", c("max", "min", "mean"), "_", var_name)
  out
}

# ===========================================================================
# STEP 3: Wrapper that adds features to cell_data (preserves original columns)
# ===========================================================================

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  for (j in seq_len(ncol(stats_mat))) {
    data[[colnames(stats_mat)[j]]] <- stats_mat[, j]
  }
  data
}

# ===========================================================================
# STEP 4: Full optimized pipeline (drop-in replacement for outer loop)
# ===========================================================================

# Convert to data.table for speed (if not already)
cell_data <- as.data.table(cell_data)

# Build neighbor lookup â€” now O(n) with hashing instead of O(n^2) with named vectors
# Expected time: ~5-15 minutes instead of 86+ hours
neighbor_lookup <- build_neighbor_lookup_fast(
  cell_data, id_order, rook_neighbors_unique
)

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor features for:", var_name, "\n")
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_lookup
  )
}

# The trained Random Forest model is untouched.
# The numerical output (max, min, mean of neighbor values) is identical
# to the original implementation â€” only the computational method changed.
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup structure** | Named vector (O(n) linear scan per query) | `environment` hash (O(1) per query) |
| **Lookup granularity** | Per cell-year (6.46M iterations) | Per cell (344K), expanded across years |
| **Stats computation** | `lapply` + `do.call(rbind,...)` over 6.46M rows | Vectorized edge-list + `data.table` grouped aggregation |
| **Estimated runtime** | 86+ hours | ~10â€“30 minutes |
| **Numerical output** | max/min/mean per row | Identical max/min/mean per row |
| **RF model** | Preserved | Preserved (no retraining) |

The true bottleneck was **`build_neighbor_lookup()`** â€” specifically the O(n) named-vector character lookups repeated 6.46 million times, not the `do.call(rbind, ...)` in `compute_neighbor_stats()`.
END_RESPONSE>>>
