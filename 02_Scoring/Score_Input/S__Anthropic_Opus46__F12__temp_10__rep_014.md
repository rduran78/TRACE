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
run_id: Anthropic_F12_T10_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working Code

## Diagnosis

The colleague is **partially right but misses the deeper bottleneck**. Let me audit both functions:

### `compute_neighbor_stats()` â€” Colleague's Claim

`do.call(rbind, result)` on a list of ~6.46 million 3-element vectors is indeed not free â€” it builds a ~6.46M Ã— 3 matrix from a list. However, this is a **one-shot operation per variable** (5 times total). With modern R, `do.call(rbind, ...)` on a list of equal-length numeric vectors is reasonably efficient. The inner `lapply` does simple subsetting, `max`, `min`, `mean` â€” these are vectorized C-level operations on small vectors (typically ~4 rook neighbors). **This is not the main bottleneck.**

### `build_neighbor_lookup()` â€” The True Bottleneck

This function runs `lapply` over **every row** (~6.46 million rows), and for each row it:

1. Calls `as.character(data$id[i])` â€” character conversion per row.
2. Performs **named vector lookup** `id_to_ref[...]` â€” hash-based, but done 6.46M times in an R-level loop.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” string construction per row for ~4 neighbors = ~25.8 million `paste` calls.
4. Performs **named vector lookup** `idx_lookup[neighbor_keys]` â€” on a named vector of length 6.46M, done 6.46M times, each time looking up ~4 keys.

The core problem: **~6.46 million R-level iterations, each doing string concatenation and named-vector lookups against a 6.46M-entry vector.** Named vector lookup in R is O(n) or uses hashing but with significant per-call overhead. Doing this ~25.8 million times (6.46M rows Ã— ~4 neighbors) inside an interpreted R loop is catastrophically slow.

**This is the dominant bottleneck** â€” it likely accounts for 80%+ of the 86-hour runtime. The neighbor structure is **year-invariant** (same spatial grid across all 28 years), yet the code redundantly recomputes neighbor row indices for every cell-year combination. The same spatial cell appears 28 times (once per year), and each time the code re-derives the same spatial neighbors and re-looks up their row indices for that year.

### Secondary Issue

`compute_neighbor_stats` could also be optimized, but the gain there is modest compared to fixing `build_neighbor_lookup`.

---

## Optimization Strategy

1. **Exploit year-invariance**: The rook neighbors are purely spatial â€” cell A's neighbors don't change across years. Instead of iterating over 6.46M cell-year rows, iterate over 344,208 cells once to build a spatial neighbor map, then replicate across years using vectorized row-index arithmetic.

2. **Replace named-vector lookups with integer-indexed lookups**: Use `match()` once to build integer index mappings, then use direct integer subsetting (O(1)) instead of named lookups.

3. **Vectorize `compute_neighbor_stats`**: Replace `lapply` + `do.call(rbind, ...)` with pre-allocated matrix output and, where possible, use vectorized group operations.

4. **Preserve the trained Random Forest model**: We only change the feature-engineering pipeline, producing numerically identical columns. The model object is untouched.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED build_neighbor_lookup
# =============================================================================
# Key insight: neighbor relationships are spatial (year-invariant).
# Instead of 6.46M R-loop iterations with string-based lookups,
# we do 344,208 cell iterations once, then compute row indices
# via vectorized arithmetic.

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # --- Step 1: Establish consistent ordering assumptions ---
  # data must be sorted by (id, year) or (year, id). We determine the layout.
  n_cells <- length(id_order)
  unique_years <- sort(unique(data$year))
  n_years <- length(unique_years)
  
  # Map cell IDs to spatial index 1..n_cells
  id_to_spatial <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map years to year index 1..n_years
  year_to_idx <- setNames(seq_along(unique_years), as.character(unique_years))
  
  # Compute spatial index and year index for every row (vectorized)
  spatial_idx <- id_to_spatial[as.character(data$id)]
  year_idx    <- year_to_idx[as.character(data$year)]
  
  # --- Step 2: Build a row-index matrix: row_matrix[cell, year] = row in data ---
  # This replaces ALL string-based idx_lookup operations.
  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_matrix[cbind(spatial_idx, year_idx)] <- seq_len(nrow(data))
  
  # --- Step 3: For each cell, get its neighbor spatial indices (once) ---
  # neighbors is an nb object: neighbors[[i]] gives spatial indices of 
  # neighbors of id_order[i].
  
  # --- Step 4: Build the full lookup list ---
  # For row r with spatial_idx s and year_idx t:
  #   neighbor spatial indices = neighbors[[s]]
  #   neighbor row indices = row_matrix[neighbors[[s]], t]
  
  # Pre-allocate output list
  n_rows <- nrow(data)
  neighbor_lookup <- vector("list", n_rows)
  
  for (t in seq_len(n_years)) {
    # Which rows in data correspond to this year?
    rows_this_year <- which(year_idx == t)
    # Extract the year's column from row_matrix (all cells' row indices for year t)
    year_col <- row_matrix[, t]
    
    for (r in rows_this_year) {
      s <- spatial_idx[r]
      nb_spatial <- neighbors[[s]]
      if (length(nb_spatial) == 0L) {
        neighbor_lookup[[r]] <- integer(0)
      } else {
        nb_rows <- year_col[nb_spatial]
        neighbor_lookup[[r]] <- nb_rows[!is.na(nb_rows)]
      }
    }
  }
  
  neighbor_lookup
}

# =============================================================================
# EVEN FASTER: Fully vectorized build using data.table
# =============================================================================
# This avoids the nested R loop entirely.

build_neighbor_lookup_vectorized <- function(data, id_order, neighbors) {
  require(data.table)
  
  n_cells <- length(id_order)
  unique_years <- sort(unique(data$year))
  n_years <- length(unique_years)
  
  # Spatial index for each cell ID
  id_to_spatial <- setNames(seq_along(id_order), as.character(id_order))
  year_to_idx   <- setNames(seq_along(unique_years), as.character(unique_years))
  
  spatial_idx <- as.integer(id_to_spatial[as.character(data$id)])
  year_idx    <- as.integer(year_to_idx[as.character(data$year)])
  
  # Row-index matrix: cell (row) Ã— year (col) â†’ row number in data
  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_matrix[cbind(spatial_idx, year_idx)] <- seq_len(nrow(data))
  
  n_rows <- nrow(data)
  
  # --- Flatten the neighbor list into a lookup table ---
  # For each spatial cell s, neighbors[[s]] gives neighbor spatial indices.
  # Pre-expand: for every data row, get all neighbor rows at once.
  
  # Build flat edge list: (source_spatial, neighbor_spatial)
  nb_lengths <- lengths(neighbors)
  source_spatial <- rep(seq_len(n_cells), nb_lengths)
  target_spatial <- unlist(neighbors, use.names = FALSE)
  # Now source_spatial[k] is neighbor of target_spatial[k] (or vice versa,
  # depending on nb convention â€” spdep::nb is symmetric for rook).
  
  # For each row in data, we need to cross this edge list with the row's year.
  # Strategy: for each year t, all cells present in that year need their
  # neighbors' row indices from year t.
  
  neighbor_lookup <- vector("list", n_rows)
  # Initialize all as empty integer
  for (i in seq_len(n_rows)) neighbor_lookup[[i]] <- integer(0)
  
  for (t in seq_len(n_years)) {
    year_col <- row_matrix[, t]  # length n_cells; year_col[s] = row in data
    
    # Which source cells exist in year t?
    present <- which(!is.na(year_col))
    present_set <- logical(n_cells)
    present_set[present] <- TRUE
    
    # Filter edges where source is present in this year
    edge_mask <- present_set[source_spatial]
    src_s <- source_spatial[edge_mask]
    tgt_s <- target_spatial[edge_mask]
    
    # Map source spatial â†’ data row, target spatial â†’ data row
    src_rows <- year_col[src_s]
    tgt_rows <- year_col[tgt_s]
    
    # Remove edges where target doesn't exist in this year
    valid <- !is.na(tgt_rows)
    src_rows <- src_rows[valid]
    tgt_rows <- tgt_rows[valid]
    
    # Group target rows by source row using split
    if (length(src_rows) > 0) {
      grouped <- split(tgt_rows, src_rows)
      source_row_ids <- as.integer(names(grouped))
      for (j in seq_along(grouped)) {
        neighbor_lookup[[source_row_ids[j]]] <- as.integer(grouped[[j]])
      }
    }
  }
  
  neighbor_lookup
}

# =============================================================================
# OPTIMIZED compute_neighbor_stats (matrix pre-allocation, no do.call(rbind))
# =============================================================================

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  
  result_mat <- matrix(NA_real_, nrow = n, ncol = 3)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0L) next
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    result_mat[i, 1] <- max(nv)
    result_mat[i, 2] <- min(nv)
    result_mat[i, 3] <- mean(nv)
  }
  
  result_mat
}

# =============================================================================
# FULLY VECTORIZED compute_neighbor_stats using data.table
# =============================================================================
# Eliminates the R-level loop over 6.46M rows entirely.

compute_neighbor_stats_vectorized <- function(data, neighbor_lookup, var_name) {
  require(data.table)
  
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  
  # Flatten neighbor_lookup into (source_row, neighbor_row) pairs
  src_lengths <- lengths(neighbor_lookup)
  source_row <- rep(seq_len(n), src_lengths)
  neighbor_row <- unlist(neighbor_lookup, use.names = FALSE)
  
  if (length(neighbor_row) == 0) {
    return(matrix(NA_real_, nrow = n, ncol = 3))
  }
  
  # Get neighbor values
  neighbor_vals <- vals[neighbor_row]
  
  # Build data.table for grouped aggregation
  dt <- data.table(src = source_row, val = neighbor_vals)
  dt <- dt[!is.na(val)]
  
  # Vectorized grouped aggregation â€” this is the key speedup
  agg <- dt[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = src]
  
  # Map back to full result matrix
  result_mat <- matrix(NA_real_, nrow = n, ncol = 3)
  result_mat[agg$src, 1] <- agg$nb_max
  result_mat[agg$src, 2] <- agg$nb_min
  result_mat[agg$src, 3] <- agg$nb_mean
  
  result_mat
}

# =============================================================================
# OPTIMIZED wrapper (drop-in replacement for compute_and_add_neighbor_features)
# =============================================================================

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_vectorized(data, neighbor_lookup, var_name)
  
  data[[paste0("nb_max_", var_name)]]  <- stats[, 1]
  data[[paste0("nb_min_", var_name)]]  <- stats[, 2]
  data[[paste0("nb_mean_", var_name)]] <- stats[, 3]
  
  data
}

# =============================================================================
# OPTIMIZED OUTER PIPELINE
# =============================================================================

# Build the neighbor lookup ONCE (this was already the case, but now much faster)
neighbor_lookup <- build_neighbor_lookup_vectorized(
  cell_data, id_order, rook_neighbors_unique
)

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_lookup
  )
}

# The trained Random Forest model is unchanged â€” use it directly for prediction:
# predictions <- predict(rf_model, cell_data)
```

---

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| `do.call(rbind, result)` | "Main bottleneck" | Moderate cost (~5 calls), but **not dominant** |
| Inner `lapply` in `compute_neighbor_stats` | "Repeated list binding" | No list binding occurs; it's simple subsetting. Minor issue. |
| **`build_neighbor_lookup`** | Not mentioned | **TRUE DOMINANT BOTTLENECK**: 6.46M R-loop iterations, each doing `paste()` string construction and named-vector hash lookups against a 6.46M-entry vector. ~25.8M string concatenations + hash lookups. |

**Verdict: Reject the colleague's diagnosis.** The real bottleneck is `build_neighbor_lookup()`, which redundantly recomputes year-invariant spatial neighbor relationships for every cell-year row via expensive string-key lookups in an R-level loop.

### Expected Speedup

| Operation | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-loop iters Ã— string ops | 28 year-iterations Ã— vectorized integer matrix indexing |
| `compute_neighbor_stats` (Ã—5) | `lapply` + `do.call(rbind, ...)` | Fully vectorized `data.table` grouped aggregation |
| **Estimated total time** | **86+ hours** | **~10â€“30 minutes** |

The optimization preserves numerical identity of all output columns and requires no changes to the trained Random Forest model.
END_RESPONSE>>>
