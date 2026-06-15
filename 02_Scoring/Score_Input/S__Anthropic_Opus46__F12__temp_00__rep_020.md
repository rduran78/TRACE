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
run_id: Anthropic_F12_T00_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single matrix assembly step per variable (5 calls total). The actual cost there is modest â€” binding 6.46M pre-allocated 3-element vectors into a matrix is an O(n) memory copy.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here is the evidence from the code:

1. **Massive character key construction and named-vector lookup (O(n) hash lookups inside an O(n) `lapply`):** For each of the ~6.46 million rows, the function:
   - Calls `as.character(data$id[i])` â€” 6.46M character coercions.
   - Looks up `id_to_ref[as.character(...)]` â€” 6.46M named-vector hash lookups against a 344,208-element named vector.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” constructs on average ~4 string keys per row (rook neighbors), so ~25.8M `paste` operations.
   - Looks up `idx_lookup[neighbor_keys]` â€” ~25.8M named-vector hash lookups against a **6.46-million-element** named character vector.

   Named vector lookup in R uses internal hashing, but building and probing a 6.46M-entry hash table 25.8M times, all inside a sequential `lapply` over 6.46M iterations with per-element R function call overhead, is the dominant cost. This single function likely accounts for **>90% of the 86+ hour runtime**.

2. **The `paste`-based string key strategy is fundamentally expensive.** String construction, hashing, and comparison are far slower than integer arithmetic.

3. **`compute_neighbor_stats()` is comparatively cheap:** It does only integer indexing into a numeric vector (`vals[idx]`), then `max`, `min`, `mean` on ~4 values. The `do.call(rbind, result)` at the end is a single operation. This function is called only 5 times total.

**Conclusion:** The bottleneck is `build_neighbor_lookup()` â€” specifically the per-row string key construction and repeated hash-table probing against a 6.46M-entry named character vector. The colleague's diagnosis is wrong.

---

## Optimization Strategy

1. **Eliminate all string/paste operations.** Replace the `paste(id, year)` key scheme with direct integer arithmetic: encode each (id, year) pair as a single integer `(id_index - 1) * n_years + year_index`. Use a pre-built integer matrix for O(1) lookup.

2. **Vectorize `build_neighbor_lookup()`** by pre-expanding the neighbor relationships across all years at once using vectorized operations, avoiding the 6.46M-iteration `lapply` entirely.

3. **Replace `do.call(rbind, lapply(...))` in `compute_neighbor_stats()`** with a fully vectorized approach using the pre-expanded neighbor index pairs and `rowMeans`/group operations, or at minimum use `vapply` for pre-allocated output.

4. **Preserve the trained Random Forest model** â€” we only change feature engineering, producing numerically identical columns.

5. **Preserve the original numerical estimand** â€” all computed neighbor max/min/mean values will be identical.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” replaces build_neighbor_lookup + compute_neighbor_stats
# Produces numerically identical results to the original code.
# =============================================================================

library(data.table)

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
 # -------------------------------------------------------------------
 # Instead of building string keys and doing millions of hash lookups,
 # we use integer encoding:  row = (id_index - 1) * n_years + year_index
 # -------------------------------------------------------------------

 # Step 1: Map cell IDs to contiguous 1-based integer indices
 n_cells <- length(id_order)
 id_to_idx <- integer(max(id_order))
 id_to_idx[id_order] <- seq_len(n_cells)
 # If IDs are not guaranteed to be small positive integers, use:
 # id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))
 # and adjust below. But for spatial grid IDs this is typically fine.

 # Step 2: Map years to contiguous 1-based integer indices
 years_unique <- sort(unique(data$year))
 n_years <- length(years_unique)
 year_to_idx <- integer(max(years_unique))
 year_to_idx[years_unique] <- seq_len(n_years)

 # Step 3: Build a row-position matrix: row_pos[cell_idx, year_idx] = row in data
 #   This replaces the 6.46M-entry named character vector idx_lookup entirely.
 data_id_idx  <- id_to_idx[data$id]
 data_yr_idx  <- year_to_idx[data$year]
 row_pos <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
 row_pos[cbind(data_id_idx, data_yr_idx)] <- seq_len(nrow(data))

 # Step 4: Build the neighbor lookup as a list of integer vectors
 #   For each row i in data, find which rows correspond to its
 #   rook neighbors in the same year.
 #
 #   Vectorized approach: expand all (row, neighbor_row) pairs at once.

 # 4a: For every row, get its cell's neighbor cell-indices
 #     neighbors[[cell_idx]] gives neighbor cell indices (already indices into id_order)
 #     We need to map data$id indices through id_to_idx, then use neighbors list.

 # Pre-expand: for each data row, get the list of neighbor cell indices
 # Then for each neighbor cell index, look up row_pos[neighbor_cell_idx, year_idx]

 # To avoid a 6.46M lapply, we use a vectorized expansion:

 # Number of neighbors per cell
 n_neighbors <- lengths(neighbors)  # length = n_cells

 # For each data row, the number of neighbors = n_neighbors[data_id_idx[i]]
 row_n_neighbors <- n_neighbors[data_id_idx]

 # Total directed neighbor-year pairs
 total_pairs <- sum(as.numeric(row_n_neighbors))
 cat("Total neighbor-year pairs to resolve:", total_pairs, "\n")

 # Expand: for each data row i, repeat i  n_neighbors[cell(i)] times
 # and pair with each neighbor cell index
 # Use rep() for vectorized expansion

 # Source row indices, repeated
 src_rows <- rep.int(seq_len(nrow(data)), times = row_n_neighbors)

 # Neighbor cell indices (into id_order), expanded
 # neighbors[[cell_idx]] already returns indices into id_order
 # We need neighbors[[data_id_idx[i]]] for each row i
 # Expand all neighbor lists for the cells that appear in data
 neighbor_cell_indices <- unlist(neighbors[data_id_idx], use.names = FALSE)

 # Year index for each pair = year index of the source row
 pair_yr_idx <- data_yr_idx[src_rows]

 # Look up the target row in data for each (neighbor_cell, year) pair
 target_rows <- row_pos[cbind(neighbor_cell_indices, pair_yr_idx)]

 # Remove NA targets (neighbor cell has no data for that year)
 valid <- !is.na(target_rows)
 src_rows_valid    <- src_rows[valid]
 target_rows_valid <- target_rows[valid]

 # Free memory
 rm(src_rows, neighbor_cell_indices, pair_yr_idx, target_rows, valid)
 gc()

 # Build the lookup list using split()
 # split is highly optimized in R for integer grouping
 lookup <- vector("list", nrow(data))
 # Initialize all to integer(0)
 lookup[] <- list(integer(0))

 split_result <- split(target_rows_valid,
                       factor(src_rows_valid, levels = seq_len(nrow(data))))
 # Assign (split returns named list with all levels)
 for (nm in names(split_result)) {
   lookup[[as.integer(nm)]] <- as.integer(split_result[[nm]])
 }

 # More memory-efficient alternative using data.table:
 # dt_pairs <- data.table(src = src_rows_valid, tgt = target_rows_valid)
 # lookup <- dt_pairs[, .(tgt = list(tgt)), by = src]
 # ... but the split approach above is clear and fast enough.

 rm(split_result, src_rows_valid, target_rows_valid)
 gc()

 lookup
}


compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
 # ---------------------------------------------------------------
 # Vectorized computation of neighbor max, min, mean.
 # Uses the pre-built expanded pairs for fully vectorized indexing.
 # Produces numerically identical results to the original.
 # ---------------------------------------------------------------
 vals <- data[[var_name]]
 n <- nrow(data)

 # Use vapply for pre-allocated output (avoids do.call(rbind,...) overhead)
 result <- vapply(neighbor_lookup, function(idx) {
   if (length(idx) == 0L) return(c(NA_real_, NA_real_, NA_real_))
   nv <- vals[idx]
   nv <- nv[!is.na(nv)]
   if (length(nv) == 0L) return(c(NA_real_, NA_real_, NA_real_))
   c(max(nv), min(nv), mean(nv))
 }, numeric(3))

 # vapply returns 3 x n matrix; transpose to n x 3
 t(result)
}


# Even faster: fully vectorized stats using data.table grouping
# (eliminates the 6.46M lapply entirely for the stats computation)
compute_neighbor_stats_vectorized <- function(data, neighbor_lookup, var_name) {
 vals <- data[[var_name]]
 n <- nrow(data)

 # Expand to (source_row, neighbor_value) pairs
 src <- rep.int(seq_len(n), times = lengths(neighbor_lookup))
 tgt <- unlist(neighbor_lookup, use.names = FALSE)

 if (length(tgt) == 0L) {
   return(matrix(NA_real_, nrow = n, ncol = 3))
 }

 neighbor_vals <- vals[tgt]

 dt <- data.table(src = src, nv = neighbor_vals)
 # Remove NAs in neighbor values
 dt <- dt[!is.na(nv)]

 # Compute grouped stats
 stats <- dt[, .(nmax = max(nv), nmin = min(nv), nmean = mean(nv)), by = src]

 # Map back to full n-row output
 out <- matrix(NA_real_, nrow = n, ncol = 3)
 out[stats$src, 1L] <- stats$nmax
 out[stats$src, 2L] <- stats$nmin
 out[stats$src, 3L] <- stats$nmean

 out
}


# =============================================================================
# Optimized wrapper (drop-in replacement for compute_and_add_neighbor_features)
# =============================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
 stats_mat <- compute_neighbor_stats_vectorized(data, neighbor_lookup, var_name)

 data[[paste0(var_name, "_neighbor_max")]]  <- stats_mat[, 1L]
 data[[paste0(var_name, "_neighbor_min")]]  <- stats_mat[, 2L]
 data[[paste0(var_name, "_neighbor_mean")]] <- stats_mat[, 3L]

 data
}


# =============================================================================
# MAIN EXECUTION â€” drop-in replacement for the outer loop
# =============================================================================

# Build the optimized lookup (runs once; minutes instead of tens of hours)
cat("Building optimized neighbor lookup...\n")
neighbor_lookup <- build_neighbor_lookup_fast(
 cell_data, id_order, rook_neighbors_unique
)
cat("Neighbor lookup complete.\n")

# Compute neighbor features for all 5 source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
 cat("Computing neighbor stats for:", var_name, "\n")
 cell_data <- compute_and_add_neighbor_features_fast(
   cell_data, var_name, neighbor_lookup
 )
}

cat("All neighbor features computed.\n")

# The trained Random Forest model is untouched.
# The computed neighbor max/min/mean values are numerically identical
# to the original implementation.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup()`: 6.46M iterations Ã— string paste Ã— hash probe against 6.46M-entry named vector | Eliminated |
| **Key construction** | `paste(id, year)` strings, ~32M string ops | Pure integer arithmetic, zero strings |
| **Lookup structure** | 6.46M-entry named character vector | Integer matrix `row_pos[cell, year]` â€” O(1) direct indexing |
| **Neighbor expansion** | Sequential `lapply` over 6.46M rows | Fully vectorized `rep.int` + `unlist` + matrix indexing |
| **Stats computation** | `lapply` + `do.call(rbind,...)` | `data.table` grouped aggregation (vectorized C code) |
| **Estimated runtime** | 86+ hours | **~5â€“15 minutes** on the same laptop |
| **Numerical output** | Identical | Identical |
| **RF model** | Preserved (untouched) | Preserved (untouched) |

The colleague was looking at the wrong function. The real cost was millions of string constructions and hash-table probes in `build_neighbor_lookup()`, not the matrix binding in `compute_neighbor_stats()`.
END_RESPONSE>>>
