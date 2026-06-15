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
run_id: Anthropic_F12_T00_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not ideal, it is a one-shot operation per variable (5 times total) and each invocation merely stacks pre-allocated 3-element vectors. This is a minor cost.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row `paste()` and character-keyed lookup for 6.46 million rows inside `lapply`**: For every single row, the function does `as.character(data$id[i])`, a named-vector lookup `id_to_ref[...]`, then constructs `paste(neighbor_cell_ids, data$year[i], sep = "_")` strings, and performs *another* named-vector lookup `idx_lookup[neighbor_keys]`. Named vector lookup in R is **O(n)** linear scan per query (R's named vectors use linear search, not hashing), and `idx_lookup` has 6.46 million entries. This means each of the 6.46 million rows performs multiple linear scans over a 6.46-million-element named vector.

2. **Quadratic-class complexity**: With ~6.46M rows and ~4 neighbors per cell on average, the lookup into `idx_lookup` (a named character vector) is approximately O(6.46M Ã— 4 Ã— 6.46M) character comparisons in the worst case. Even with partial optimizations in R internals, this is catastrophically slow and is the source of the 86+ hour runtime.

3. `compute_neighbor_stats()` by contrast is a simple numeric indexing operation (`vals[idx]`) â€” integer-indexed subsetting is O(1) per element. The `do.call(rbind, ...)` on the result list is O(n) and takes seconds, not hours.

**Conclusion**: The bottleneck is the construction of `neighbor_lookup` via repeated character-key lookups in a named vector of length 6.46M, not the `rbind` or list operations in `compute_neighbor_stats()`.

## Optimization Strategy

1. **Replace the named-vector lookup with an environment (hash map)** or, better yet, eliminate character-key lookups entirely by using `data.table` keyed joins or direct integer arithmetic.

2. **Exploit the panel structure**: Since every cell appears once per year in a regular panel (344,208 cells Ã— 28 years), we can compute a direct integer mapping from `(id, year)` â†’ row index using arithmetic rather than string matching. If the data is sorted by `(id, year)`, the row index is deterministic: `(cell_position - 1) * 28 + (year - 1991)`. Even if not sorted, we can build a hash map via `data.table` or an environment once.

3. **Vectorize `compute_neighbor_stats()`** using matrix operations instead of per-row `lapply`.

4. **Preserve the trained Random Forest model** â€” we only change feature engineering, producing numerically identical columns.

## Working R Code

```r
library(data.table)

# ==============================================================
# OPTIMIZED build_neighbor_lookup using data.table hash joins
# ==============================================================
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert to data.table for keyed joins
  dt <- as.data.table(data)
  dt[, row_idx := .I]
  
  # Build (id, year) -> row_idx hash via data.table keyed lookup
  setkey(dt, id, year)
  
  # Build id -> reference index mapping (position in id_order)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Pre-extract columns as vectors for speed
  data_id   <- dt$id
  data_year <- dt$year
  n_rows    <- nrow(dt)
  
  # ---- Step 1: Build an edge list of (source_row, neighbor_id, year) ----
  # For each row i, we need neighbor cell IDs and the same year.
  # Instead of looping per row, we vectorize over all rows.
  
  # Map each row's id to its ref_idx in the neighbor list
  ref_indices <- id_to_ref[as.character(data_id)]  # length n_rows
  
  # For each row, get the neighbor cell IDs
  # neighbors[[ref_idx]] gives indices into id_order
  # We build this as a flat edge list using vectorized operations
  
  # Number of neighbors per row
  n_neighbors <- vapply(ref_indices, function(ri) {
    if (is.na(ri)) 0L else length(neighbors[[ri]])
  }, integer(1))
  
  total_edges <- sum(n_neighbors)
  
  # Pre-allocate flat vectors
  source_rows    <- integer(total_edges)
  neighbor_ids   <- integer(total_edges)
  neighbor_years <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n_rows)) {
    ri <- ref_indices[i]
    if (!is.na(ri)) {
      nb <- neighbors[[ri]]
      nn <- length(nb)
      if (nn > 0L) {
        idx_range <- pos:(pos + nn - 1L)
        source_rows[idx_range]    <- i
        neighbor_ids[idx_range]   <- id_order[nb]
        neighbor_years[idx_range] <- data_year[i]
        pos <- pos + nn
      }
    }
  }
  
  # ---- Step 2: Join to find row indices of neighbors ----
  edges_dt <- data.table(
    source_row    = source_rows,
    neighbor_id   = neighbor_ids,
    neighbor_year = neighbor_years
  )
  
  # Keyed join: find the row_idx for each (neighbor_id, neighbor_year)
  setkey(edges_dt, neighbor_id, neighbor_year)
  setkey(dt, id, year)
  
  edges_dt[dt, neighbor_row := i.row_idx,
           on = .(neighbor_id = id, neighbor_year = year)]
  
  # Remove edges where neighbor row was not found
  edges_dt <- edges_dt[!is.na(neighbor_row)]
  
  # ---- Step 3: Split into per-row lists ----
  # Order by source_row for efficient splitting
  setkey(edges_dt, source_row)
  
  # Use split to create the lookup list
  lookup_list <- vector("list", n_rows)
  
  # Initialize all entries as empty integer vectors
  for (i in seq_len(n_rows)) {
    lookup_list[[i]] <- integer(0)
  }
  
  # Fill in from edges
  split_result <- split(edges_dt$neighbor_row, edges_dt$source_row)
  split_names  <- as.integer(names(split_result))
  for (j in seq_along(split_names)) {
    lookup_list[[split_names[j]]] <- as.integer(split_result[[j]])
  }
  
  return(lookup_list)
}

# ==============================================================
# OPTIMIZED compute_neighbor_stats using vectorized operations
# ==============================================================
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals    <- data[[var_name]]
  n_rows  <- length(neighbor_lookup)
  
  # Build flat vectors for vectorized computation
  # Map each neighbor index to its source row
  lens <- lengths(neighbor_lookup)
  
  source_row    <- rep.int(seq_len(n_rows), lens)
  neighbor_row  <- unlist(neighbor_lookup, use.names = FALSE)
  neighbor_vals <- vals[neighbor_row]
  
  # Remove NAs in neighbor values
  valid <- !is.na(neighbor_vals)
  source_row    <- source_row[valid]
  neighbor_vals <- neighbor_vals[valid]
  
  # Compute grouped statistics using data.table
  stats_dt <- data.table(src = source_row, val = neighbor_vals)
  agg <- stats_dt[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = src]
  
  # Initialize result matrix with NAs
  result <- matrix(NA_real_, nrow = n_rows, ncol = 3)
  colnames(result) <- c("max", "min", "mean")
  
  result[agg$src, 1] <- agg$nb_max
  result[agg$src, 2] <- agg$nb_min
  result[agg$src, 3] <- agg$nb_mean
  
  return(result)
}

# ==============================================================
# OPTIMIZED compute_and_add_neighbor_features
# ==============================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  
  data[[paste0(var_name, "_nb_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_nb_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_nb_mean")]] <- stats[, 3]
  
  return(data)
}

# ==============================================================
# MAIN PIPELINE (drop-in replacement)
# ==============================================================

# Build the lookup ONCE â€” now O(n log n) via data.table keyed joins
# instead of O(n^2) via named-vector linear scans
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
# The numerical outputs (nb_max, nb_min, nb_mean) are identical
# to the original implementation â€” only the computation path changed.
```

## Further Optimization: Eliminate the Per-Row Loop in Lookup Construction

The `for` loop in `build_neighbor_lookup_fast` that constructs the flat edge list can itself be slow for 6.46M rows. Here is a fully vectorized version:

```r
build_neighbor_lookup_fast_v2 <- function(data, id_order, neighbors) {
  dt <- as.data.table(data)
  dt[, row_idx := .I]
  
  id_to_ref <- match(dt$id, id_order)  # integer vector, no names
  
  # Vectorized edge-list construction
  # For each unique cell, get its neighbors once, then expand across years
  unique_cells <- data.table(
    cell_id = id_order,
    ref_idx = seq_along(id_order)
  )
  
  # Build cell-level neighbor edges (cell_id -> neighbor_cell_id)
  cell_nb_list <- lapply(seq_along(neighbors), function(ri) {
    nb <- neighbors[[ri]]
    if (length(nb) == 0L) return(data.table(cell_id = integer(0), nb_id = integer(0)))
    data.table(cell_id = id_order[ri], nb_id = id_order[nb])
  })
  cell_edges <- rbindlist(cell_nb_list)
  
  # Cross-join with years: each cell-neighbor pair exists for every year
  years <- sort(unique(dt$year))
  cell_edges_expanded <- cell_edges[, .(year = years), by = .(cell_id, nb_id)]
  
  # Join to get source row index
  setkey(dt, id, year)
  cell_edges_expanded[dt, source_row := i.row_idx,
                      on = .(cell_id = id, year = year)]
  
  # Join to get neighbor row index
  cell_edges_expanded[dt, neighbor_row := i.row_idx,
                      on = .(nb_id = id, year = year)]
  
  # Drop unmatched
  cell_edges_expanded <- cell_edges_expanded[
    !is.na(source_row) & !is.na(neighbor_row)
  ]
  
  # Split into per-row lookup
  setkey(cell_edges_expanded, source_row)
  n_rows <- nrow(dt)
  
  lookup_list <- vector("list", n_rows)
  for (i in seq_len(n_rows)) lookup_list[[i]] <- integer(0)
  
  split_result <- split(cell_edges_expanded$neighbor_row,
                        cell_edges_expanded$source_row)
  split_idx    <- as.integer(names(split_result))
  lookup_list[split_idx] <- lapply(split_result, as.integer)
  
  return(lookup_list)
}
```

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| **Alleged bottleneck** | `do.call(rbind, ...)` and list binding in `compute_neighbor_stats()` | **Wrong** â€” this is O(n) and takes seconds |
| **True bottleneck** | â€” | `build_neighbor_lookup()`: named-vector lookups (`idx_lookup[neighbor_keys]`) over a 6.46M-element character-named vector are O(n) per query Ã— 6.46M queries â‰ˆ O(nÂ²) |
| **Fix** | â€” | Replace character-named vector lookups with `data.table` keyed joins (O(n log n)) and vectorize the edge-list construction |
| **Expected speedup** | â€” | From ~86+ hours to **minutes** (estimated 5â€“15 min on the same laptop) |
| **RF model** | â€” | Preserved, untouched |
| **Numerical output** | â€” | Identical `nb_max`, `nb_min`, `nb_mean` values |
END_RESPONSE>>>
