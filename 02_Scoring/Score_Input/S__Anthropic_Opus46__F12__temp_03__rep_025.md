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
run_id: Anthropic_F12_T03_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single matrix assembly step per variable â€” only 5 calls total. Each inner function returns a fixed-length-3 numeric vector, so there is no "repeated list binding" inside `compute_neighbor_stats()`; it's a straightforward `lapply` returning pre-allocated vectors. That step is O(n) and relatively fast.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Character key construction and named-vector lookup over 6.46 million rows.** `paste(data$id, data$year, sep = "_")` creates 6.46M character keys. Then, for *each* of the 6.46M rows, it does:
   - `as.character(data$id[i])` â€” character conversion per row.
   - `id_to_ref[as.character(...)]` â€” named vector lookup (hash-based, but called 6.46M times individually inside `lapply`).
   - `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” creates ~4 character keys per row (rook neighbors), totaling ~26M `paste` calls.
   - `idx_lookup[neighbor_keys]` â€” named vector lookup against a 6.46M-element named vector, called 6.46M times with ~4 keys each = ~26M hash lookups.

2. **Total cost:** ~6.46 million R-level function calls via `lapply`, each doing multiple character allocations, paste operations, and hash lookups. This is the dominant O(n Ã— k) bottleneck with enormous constant factors from R's character handling overhead. On a laptop, this alone accounts for the vast majority of the 86+ hour runtime.

3. By contrast, `compute_neighbor_stats()` simply indexes into a numeric vector with pre-computed integer indices â€” extremely fast. And `do.call(rbind, result)` on 6.46M length-3 vectors takes seconds, not hours.

## Optimization Strategy

1. **Replace character-key hashing with integer arithmetic.** Map `(id, year)` pairs to integer indices using a direct integer lookup table instead of character paste + named vector lookup. Since years span 1992â€“2019 (28 years), we can encode `(id, year)` as a single integer and use `match()` or a direct-index table.

2. **Vectorize `build_neighbor_lookup` entirely** â€” eliminate the per-row `lapply` by expanding all neighbor relationships at once using vectorized operations, then splitting by row.

3. **Replace `do.call(rbind, ...)` with direct matrix pre-allocation** in `compute_neighbor_stats` (a minor but easy win).

4. **Preserve the trained Random Forest model** â€” we only change feature engineering, producing numerically identical columns.

## Working R Code

```r
# =============================================================================
# OPTIMIZED build_neighbor_lookup â€” fully vectorized, no character keys
# =============================================================================
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  n <- nrow(data)
  num_ids <- length(id_order)
  
  # Step 1: Create integer mapping from id -> reference index (position in id_order)
  # Use match() once, vectorized over all rows
  id_ref <- match(data$id, id_order)  # length n, integer vector
  
  # Step 2: Create a fast (id, year) -> row mapping using an integer-keyed approach.
  # Map years to 1..28
  years <- sort(unique(data$year))
  year_offset <- match(data$year, years)  # length n, integer vector
  
  num_years <- length(years)
  
  # Encode (id_position_in_id_order, year_offset) as a single integer key
  # key = (id_index_in_id_order - 1) * num_years + year_offset
  # This gives a unique integer in 1..(num_ids * num_years) for each (id, year)
  
  row_keys <- (match(data$id, id_order) - 1L) * num_years + year_offset
  
  # Build reverse lookup: key -> row index
  # Pre-allocate a vector of size num_ids * num_years (344208 * 28 â‰ˆ 9.6M â€” fits easily)
  max_key <- num_ids * num_years
  key_to_row <- integer(max_key)
  key_to_row[row_keys] <- seq_len(n)
  
  # Step 3: Expand all neighbor relationships vectorized
  # For each row i, we need neighbors[[id_ref[i]]] mapped to the same year.
  # 
  # Strategy: build an edge list (row_index, neighbor_id_ref) for all rows,
  # then compute neighbor keys and look up row indices.
  
  # Precompute neighbor lengths
  nb_lengths <- lengths(neighbors)  # length = num_ids
  row_nb_lengths <- nb_lengths[id_ref]  # length = n, neighbors per row
  
  # Total edges â‰ˆ sum of row_nb_lengths (â‰ˆ 6.46M * ~4 = ~26M)
  total_edges <- sum(row_nb_lengths)
  
  # Row indices repeated by their neighbor count
  row_rep <- rep.int(seq_len(n), row_nb_lengths)
  
  # Year offsets repeated
  year_rep <- year_offset[row_rep]
  
  # Neighbor id references: unlist neighbors in id_ref order, then index
  # neighbors is an nb object indexed by position in id_order.
  # For row i, the neighbor refs are neighbors[[id_ref[i]]].
  # We need to unlist neighbors[id_ref] in order.
  
  nb_expanded <- neighbors[id_ref]  # list of length n, reordered
  neighbor_refs <- unlist(nb_expanded, use.names = FALSE)  # integer vector, length = total_edges
  
  # Compute keys for all neighbor (id_ref, year) pairs
  neighbor_keys <- (neighbor_refs - 1L) * num_years + year_rep
  
  # Look up row indices
  neighbor_rows <- key_to_row[neighbor_keys]
  
  # Remove invalid (0 means no matching row in data)
  valid <- neighbor_rows > 0L
  row_rep_valid <- row_rep[valid]
  neighbor_rows_valid <- neighbor_rows[valid]
  
  # Step 4: Split neighbor row indices by source row
  # Use split with factor to preserve all levels (rows with no valid neighbors get integer(0))
  lookup <- split(neighbor_rows_valid, factor(row_rep_valid, levels = seq_len(n)))
  
  # Ensure each element is integer
  lookup <- lapply(lookup, as.integer)
  
  lookup
}

# =============================================================================
# OPTIMIZED compute_neighbor_stats â€” pre-allocated matrix, no do.call(rbind)
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
    if (length(neighbor_vals) == 0L) next
    result_mat[i, 1L] <- max(neighbor_vals)
    result_mat[i, 2L] <- min(neighbor_vals)
    result_mat[i, 3L] <- mean(neighbor_vals)
  }
  
  result_mat
}

# =============================================================================
# ALTERNATIVE: Fully vectorized compute_neighbor_stats using group operations
# (avoids the R-level for loop entirely for even more speed)
# =============================================================================
compute_neighbor_stats_vectorized <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  
  # Build flat vectors: source row and neighbor value
  lens <- lengths(neighbor_lookup)
  src <- rep.int(seq_len(n), lens)
  nb_idx <- unlist(neighbor_lookup, use.names = FALSE)
  nb_vals <- vals[nb_idx]
  
  # Remove NAs
  valid <- !is.na(nb_vals)
  src <- src[valid]
  nb_vals <- nb_vals[valid]
  
  # Use data.table for fast grouped aggregation
  if (requireNamespace("data.table", quietly = TRUE)) {
    dt <- data.table::data.table(src = src, val = nb_vals)
    agg <- dt[, .(mx = max(val), mn = min(val), mn2 = mean(val)), by = src]
    
    result_mat <- matrix(NA_real_, nrow = n, ncol = 3)
    result_mat[agg$src, 1L] <- agg$mx
    result_mat[agg$src, 2L] <- agg$mn
    result_mat[agg$src, 3L] <- agg$mn2
  } else {
    # Fallback: tapply
    f <- factor(src, levels = seq_len(n))
    result_mat <- matrix(NA_real_, nrow = n, ncol = 3)
    result_mat[, 1L] <- as.numeric(tapply(nb_vals, f, max))
    result_mat[, 2L] <- as.numeric(tapply(nb_vals, f, min))
    result_mat[, 3L] <- as.numeric(tapply(nb_vals, f, mean))
  }
  
  result_mat
}

# =============================================================================
# Wrapper: compute_and_add_neighbor_features (preserves original column names)
# =============================================================================
compute_and_add_neighbor_features <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_vectorized(data, neighbor_lookup, var_name)
  
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1L]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2L]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3L]
  
  data
}

# =============================================================================
# MAIN PIPELINE â€” drop-in replacement
# =============================================================================

# Build the optimized lookup (runs once; minutes instead of tens of hours)
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for all 5 source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
}

# The trained Random Forest model is untouched â€” use it directly for prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Complexity Improvements

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | O(n) R-level calls, each doing `paste` + hash lookup against 6.46M-element named vector. **~26M character hash lookups.** | **Fully vectorized** integer arithmetic + single `split()`. Zero per-row R function calls. |
| Key encoding | Character paste `"id_year"` â†’ named vector | Integer key `(id_idx-1)*28 + year_idx` â†’ direct integer indexing into pre-allocated vector |
| `compute_neighbor_stats` | `lapply` + `do.call(rbind, ...)` | Vectorized `unlist` â†’ `data.table` grouped aggregation, or pre-allocated matrix loop |
| **Estimated runtime** | **86+ hours** | **~5â€“15 minutes** on the same laptop |

**Key insight:** The bottleneck was never the `rbind` or the stats computation â€” it was the 6.46 million individual R-level iterations in `build_neighbor_lookup`, each performing expensive character string construction and hash-table lookups against a massive named vector. Replacing character keys with integer arithmetic and vectorizing the entire expansion eliminates this bottleneck by orders of magnitude.
END_RESPONSE>>>
