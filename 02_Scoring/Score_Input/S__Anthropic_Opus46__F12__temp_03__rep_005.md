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
run_id: Anthropic_F12_T03_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it's a single matrix assembly step per variable â€” only 5 calls total. Each inner function returns a fixed-length numeric vector of length 3, so there is no "repeated list binding" happening inside `compute_neighbor_stats()`; it's a straightforward `lapply` returning pre-allocated 3-element vectors.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** Inside the `lapply`, for every single row `i`, the code calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` and then does `idx_lookup[neighbor_keys]` â€” a named character vector lookup. Named vector lookup in R is **O(n)** per query on average for long vectors (it uses linear hashing that degrades), and `idx_lookup` has ~6.46 million entries. This is executed ~6.46 million times, each time for multiple neighbors.

2. **Repeated `as.character()` conversion** of `data$id[i]` and the `id_to_ref` lookup, 6.46 million times.

3. **Total string operations**: With ~6.46M rows and an average of ~4 rook neighbors per cell, the inner `lapply` performs roughly **25.8 million `paste` operations** and **25.8 million named-vector lookups** against a 6.46M-length character-keyed vector. This is the dominant cost â€” likely accounting for the vast majority of the 86+ hour runtime.

4. `compute_neighbor_stats()` by contrast does only cheap numeric indexing (`vals[idx]`) and simple arithmetic â€” it is fast.

**Conclusion:** The bottleneck is the O(N Ã— k) string-key construction and character-based lookup in `build_neighbor_lookup()`, not the `do.call(rbind, ...)` in `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Eliminate all string operations in `build_neighbor_lookup()`**: Replace the `paste`-based character key with integer arithmetic. Encode each `(id, year)` pair as a single integer: `id_index * N_YEARS + year_index`. Use integer-keyed lookup via direct vector indexing (O(1) per access) instead of named character vector lookup.

2. **Vectorize `build_neighbor_lookup()`**: Instead of an `lapply` over 6.46M rows, use `data.table` to expand neighbor relationships and join, or use vectorized integer indexing.

3. **Vectorize `compute_neighbor_stats()`**: Replace `lapply` + `do.call(rbind, ...)` with grouped vectorized operations using `data.table`, or at minimum use `vapply` (which pre-allocates the output matrix).

4. **Preserve the trained Random Forest model**: No changes to model or features â€” only the computation of the same neighbor lookup and the same summary statistics (max, min, mean) are optimized.

---

## Working R Code

```r
library(data.table)

# ===========================================================================
# OPTIMIZED build_neighbor_lookup â€” integer-key approach, fully vectorized
# ===========================================================================
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # data must have columns: id, year
  # id_order: vector of unique spatial IDs in the order matching `neighbors`
  # neighbors: spdep nb object (list of integer index vectors into id_order)

  n_ids   <- length(id_order)
  years   <- sort(unique(data$year))
  n_years <- length(years)

  # Integer mappings (1-based)
  id_to_ref   <- setNames(seq_along(id_order), as.character(id_order))
  year_to_idx <- setNames(seq_along(years), as.character(years))

  # Encode every (id, year) pair as a unique integer key
  # key = (ref_idx - 1) * n_years + year_idx
  # This gives a dense integer space of size n_ids * n_years
  data_ref_idx  <- id_to_ref[as.character(data$id)]
  data_year_idx <- year_to_idx[as.character(data$year)]
  data_key      <- (data_ref_idx - 1L) * n_years + data_year_idx

  # Build reverse map: key -> row index in data
  # (dense vector, NA where no data exists)
  max_key <- n_ids * n_years
  key_to_row <- rep(NA_integer_, max_key)
  key_to_row[data_key] <- seq_len(nrow(data))

  # --- Expand neighbor pairs (vectorized) ---
  # For each spatial cell ref_idx, get its neighbor ref_idxs
  # Then cross with all years

  # Build edge list: from_ref -> to_ref (directed, one entry per neighbor pair)
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors)

  # Remove zero-neighbor entries (spdep uses 0L for no-neighbor cells)
  valid <- to_ref != 0L

  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]

  n_edges <- length(from_ref)

  # For every row in data, we need to know its ref_idx and year_idx
  # Then for each row, its neighbors are: all to_ref where from_ref == row's ref_idx,

  # crossed with the row's year_idx.
  #
  # Strategy: build a data.table of (row_i, neighbor_row_j) directly.

  # Step 1: For each ref_idx, which rows in data belong to it?
  # ref_idx -> year_idx -> row mapping is already in key_to_row

  # Step 2: For each (from_ref, to_ref) edge and each year, compute:
  #   row_i = key_to_row[(from_ref - 1) * n_years + year_idx]
  #   row_j = key_to_row[(to_ref   - 1) * n_years + year_idx]

  # Expand edges Ã— years
  year_idxs <- seq_len(n_years)

  # Use rep to cross edges with years
  edge_from <- rep(from_ref, each = n_years)
  edge_to   <- rep(to_ref,   each = n_years)
  edge_year <- rep(year_idxs, times = n_edges)

  key_from <- (edge_from - 1L) * n_years + edge_year
  key_to   <- (edge_to   - 1L) * n_years + edge_year

  row_i <- key_to_row[key_from]
  row_j <- key_to_row[key_to]

  # Remove pairs where either row doesn't exist in data
  valid2 <- !is.na(row_i) & !is.na(row_j)
  row_i  <- row_i[valid2]
  row_j  <- row_j[valid2]

  # Build the lookup as a list indexed by row_i
  # Use data.table for fast split
  dt_edges <- data.table(row_i = row_i, row_j = row_j, key = "row_i")

  n_rows <- nrow(data)
  lookup <- vector("list", n_rows)

  # Split row_j by row_i
  split_result <- dt_edges[, .(nbrs = list(row_j)), by = row_i]
  lookup[split_result$row_i] <- split_result$nbrs

  # Fill remaining with empty integer vectors
  empty_idx <- which(lengths(lookup) == 0L)
  if (length(empty_idx) > 0) {
    lookup[empty_idx] <- list(integer(0))
  }

  lookup
}

# ===========================================================================
# OPTIMIZED compute_neighbor_stats â€” fully vectorized with data.table
# ===========================================================================
compute_neighbor_stats_fast <- function(data, neighbor_edge_dt, var_name) {
  # neighbor_edge_dt: data.table with columns row_i, row_j (precomputed)
  # Returns a matrix with columns: max, min, mean â€” one row per data row

  n_rows <- nrow(data)
  vals   <- data[[var_name]]

  # Attach neighbor values
  dt <- copy(neighbor_edge_dt)
  dt[, val := vals[row_j]]

  # Remove NA neighbor values
  dt <- dt[!is.na(val)]

  # Compute grouped stats
  stats <- dt[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = row_i]

  # Allocate output matrix (NA by default)
  out <- matrix(NA_real_, nrow = n_rows, ncol = 3)
  colnames(out) <- c("max", "min", "mean")

  out[stats$row_i, 1] <- stats$nb_max
  out[stats$row_i, 2] <- stats$nb_min
  out[stats$row_i, 3] <- stats$nb_mean

  out
}

# ===========================================================================
# OPTIMIZED compute_and_add_neighbor_features (drop-in replacement)
# ===========================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_edge_dt) {
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_edge_dt, var_name)

  suffix <- var_name
  data[[paste0("nb_max_",  suffix)]] <- stats_mat[, 1]
  data[[paste0("nb_min_",  suffix)]] <- stats_mat[, 2]
  data[[paste0("nb_mean_", suffix)]] <- stats_mat[, 3]

  data
}

# ===========================================================================
# OPTIMIZED build_neighbor_edge_dt â€” returns a data.table of (row_i, row_j)
# This is the key structure reused across all 5 variables.
# ===========================================================================
build_neighbor_edge_dt <- function(data, id_order, neighbors) {
  n_ids   <- length(id_order)
  years   <- sort(unique(data$year))
  n_years <- length(years)

  id_to_ref   <- setNames(seq_along(id_order), as.character(id_order))
  year_to_idx <- setNames(seq_along(years), as.character(years))

  data_ref_idx  <- id_to_ref[as.character(data$id)]
  data_year_idx <- year_to_idx[as.character(data$year)]
  data_key      <- (data_ref_idx - 1L) * n_years + data_year_idx

  max_key    <- n_ids * n_years
  key_to_row <- rep(NA_integer_, max_key)
  key_to_row[data_key] <- seq_len(nrow(data))

  # Expand neighbor edge list

  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors)
  valid    <- to_ref != 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]
  n_edges  <- length(from_ref)

  # Cross with years â€” use chunking to limit peak memory
  # Each edge Ã— 28 years â‰ˆ 1.37M edges Ã— 28 = ~38.5M rows (manageable)
  year_idxs <- seq_len(n_years)

  edge_from <- rep(from_ref, each = n_years)
  edge_to   <- rep(to_ref,   each = n_years)
  edge_year <- rep(year_idxs, times = n_edges)

  key_from <- (edge_from - 1L) * n_years + edge_year
  key_to   <- (edge_to   - 1L) * n_years + edge_year

  row_i <- key_to_row[key_from]
  row_j <- key_to_row[key_to]

  valid2 <- !is.na(row_i) & !is.na(row_j)

  dt <- data.table(
    row_i = row_i[valid2],
    row_j = row_j[valid2]
  )
  setkey(dt, row_i)
  dt
}

# ===========================================================================
# MAIN PIPELINE (drop-in replacement for the outer loop)
# ===========================================================================

# Build the edge table ONCE (replaces build_neighbor_lookup)
neighbor_edge_dt <- build_neighbor_edge_dt(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for all 5 variables (reuses the same edge table)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_edge_dt
  )
}

# The trained Random Forest model is untouched.
# The numerical outputs (nb_max_*, nb_min_*, nb_mean_*) are identical to the original.
```

---

## Summary of Changes and Expected Speedup

| Aspect | Original | Optimized |
|---|---|---|
| **Key encoding** | `paste()` string keys, 6.46M-entry named character vector lookup | Integer arithmetic keys, O(1) direct vector indexing |
| **Lookup construction** | `lapply` over 6.46M rows, each doing paste + named lookup | Fully vectorized: expand edge list Ã— years, direct integer index, one `data.table` group-by |
| **Neighbor stats** | `lapply` over 6.46M rows + `do.call(rbind, ...)` | Single `data.table` grouped aggregation on pre-built edge table |
| **Edge table reuse** | Lookup rebuilt implicitly per variable (if called repeatedly) | Built **once**, reused for all 5 variables |
| **Estimated runtime** | 86+ hours | **Minutes** (vectorized integer ops + data.table grouped aggregation on ~38.5M edge rows) |
| **RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical results** | Original max/min/mean | Identical âœ“ |

The dominant cost was always the **6.46 million iterations of string construction and character-keyed lookup** in `build_neighbor_lookup()`, not the `do.call(rbind, ...)` in `compute_neighbor_stats()`. Replacing strings with integer arithmetic and replacing row-wise `lapply` with vectorized `data.table` operations eliminates the bottleneck.
END_RESPONSE>>>
