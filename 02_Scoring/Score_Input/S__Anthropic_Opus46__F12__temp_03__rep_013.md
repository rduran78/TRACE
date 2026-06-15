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
run_id: Anthropic_F12_T03_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single matrix assembly step per variable (5 calls total). Each call binds ~6.46M rows Ã— 3 columns â€” this completes in seconds.

**The true bottleneck is `build_neighbor_lookup()`.**

Here is the evidence from the code:

1. **Repeated `paste()` and character-keyed named-vector lookups over 6.46 million rows.** `idx_lookup` is a named integer vector with ~6.46M entries keyed by `paste(id, year, sep="_")`. Named vector lookup in R is O(n) per probe (linear scan or hash with overhead). Inside the `lapply`, for each of the 6.46M rows, the function:
   - Calls `as.character(data$id[i])` â€” 6.46M coercions.
   - Looks up `id_to_ref[...]` â€” 6.46M named-vector lookups against a 344K-length vector.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep="_")` â€” creates ~4 strings per row (rook neighbors), so ~25.8M `paste` calls total.
   - Probes `idx_lookup[neighbor_keys]` â€” ~25.8M named-vector lookups against a 6.46M-length named vector.

2. **This is called once, but it dominates everything.** The average rook cell has ~4 neighbors. That means ~25.8 million hash lookups into a 6.46M-entry named character vector. R's named vector lookup is backed by a hash table that is rebuilt on first access and has significant per-call overhead when called millions of times from an `lapply`. This single function likely accounts for 80%+ of the 86-hour runtime.

3. **`compute_neighbor_stats()` is comparatively cheap.** Once `neighbor_lookup` exists, each call just does integer indexing into a numeric vector (`vals[idx]`) and computes `max`, `min`, `mean` â€” all fast vectorized operations. The `do.call(rbind, result)` on a list of 6.46M length-3 vectors takes a few seconds at most.

**Conclusion:** The deep bottleneck is the O(n Ã— k) character-key construction and lookup in `build_neighbor_lookup()`, not the `do.call(rbind, ...)` in `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Eliminate all character-key lookups in `build_neighbor_lookup()`.** Replace the `paste`/named-vector approach with direct integer arithmetic. If we sort or index the data by `(id, year)`, we can compute row positions arithmetically rather than via hash lookups. Specifically, if we create a mapping from `id â†’ integer index` and `year â†’ integer index`, then the row for `(id_i, year_j)` can be found via a precomputed integer matrix or a simple formula (if data is sorted).

2. **Vectorize the neighbor lookup construction.** Instead of an `lapply` over 6.46M rows, expand all neighbor relationships at once using vectorized operations, then split by row.

3. **Keep `compute_neighbor_stats()` largely as-is**, but replace `do.call(rbind, result)` with a pre-allocated matrix for marginal improvement.

4. **Preserve the trained Random Forest model** â€” we change only the feature-engineering pipeline, not the model or the numerical values produced.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE
# =============================================================================
# Key insight: replace all character-paste + named-vector lookups with
# integer arithmetic. This reduces build_neighbor_lookup from hours to seconds.
# =============================================================================

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # ---- Step 1: Build integer mappings ----
  n_ids  <- length(id_order)
  
  # Map each id to a contiguous integer 1..n_ids
  # Use match() or a fast environment-based lookup
  id_int <- match(data$id, id_order)  
  # This is vectorized over all 6.46M rows â€” fast.
  
  # Map each year to a contiguous integer 1..n_years
  years_sorted <- sort(unique(data$year))
  n_years      <- length(years_sorted)
  year_int     <- match(data$year, years_sorted)
  
  # ---- Step 2: Build a row-index matrix: row_matrix[id_idx, year_idx] = row in data ----
  # This replaces the named character vector idx_lookup entirely.
  # Pre-allocate with NA
  row_matrix <- matrix(NA_integer_, nrow = n_ids, ncol = n_years)
  row_matrix[cbind(id_int, year_int)] <- seq_len(nrow(data))
  
  # ---- Step 3: Expand all neighbor pairs (vectorized) ----
  # For each row i, we need: neighbors of data$id[i] in the same year.
  # 
  # Instead of lapply over 6.46M rows, we:
  #   (a) Expand the neighbor list into a flat edge table (cell_idx -> neighbor_cell_idx)
  #   (b) Cross with years using vectorized indexing into row_matrix
  
  n_rows <- nrow(data)
  
  # Build flat neighbor edge list: from_id_idx -> to_id_idx
  # neighbors is an nb object: neighbors[[j]] gives integer indices into id_order
  # that are neighbors of id_order[j].
  from_id_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_id_idx   <- unlist(neighbors, use.names = FALSE)
  # This gives us ~1.37M directed pairs
  
  n_edges <- length(from_id_idx)
  
  # ---- Step 4: For each row in data, find its neighbors ----
  # A row is identified by (id_int[i], year_int[i]).
  # Its neighbors are: for each to_id in neighbors[[id_int[i]]], 
  #   the row at row_matrix[to_id, year_int[i]].
  #
  # Strategy: group data rows by id_int, then for each id, 
  # expand across its neighbors and all its years.
  
  # For each id_idx j (1..n_ids), find which rows in data belong to it
  # and which neighbor id_idxs it has.
  
  # rows_by_id[[j]] = vector of row indices in data where id_int == j
  rows_by_id <- split(seq_len(n_rows), id_int)
  # Ensure indexed by integer (names are character, but we'll use direct indexing)
  rows_by_id_vec <- vector("list", n_ids)
  for (nm in names(rows_by_id)) {
    rows_by_id_vec[[as.integer(nm)]] <- rows_by_id[[nm]]
  }
  
  # years_by_id[[j]] = the year_int values for those rows (same order)
  # We'll compute on the fly.
  
  # Pre-allocate the result as a list of integer vectors
  neighbor_lookup <- vector("list", n_rows)
  
  # ---- Step 5: Iterate over id_idx (344K iterations, not 6.46M) ----
  # For each cell, get its neighbor cell indices, then for each year that
  # cell appears in, look up the neighbor rows via row_matrix.
  
  for (j in seq_len(n_ids)) {
    my_rows <- rows_by_id_vec[[j]]
    if (is.null(my_rows) || length(my_rows) == 0L) next
    
    nb_ids <- neighbors[[j]]  # neighbor id indices (into id_order)
    if (length(nb_ids) == 0L) {
      for (r in my_rows) {
        neighbor_lookup[[r]] <- integer(0)
      }
      next
    }
    
    my_years <- year_int[my_rows]  # which year slots these rows occupy
    
    # For each year this cell appears in, gather neighbor rows
    # row_matrix[nb_ids, yr] gives the row indices of all neighbors in year yr
    # This is a submatrix extraction: row_matrix[nb_ids, my_years]
    # Result: matrix of dim (length(nb_ids), length(my_years))
    nb_row_mat <- row_matrix[nb_ids, my_years, drop = FALSE]
    # Each column corresponds to one of my_rows; each column gives neighbor row indices
    
    for (k in seq_along(my_rows)) {
      col_vals <- nb_row_mat[, k]
      neighbor_lookup[[my_rows[k]]] <- col_vals[!is.na(col_vals)]
    }
  }
  
  neighbor_lookup
}


compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals   <- data[[var_name]]
  n      <- length(neighbor_lookup)
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0L) next
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    result[i, 1] <- max(nv)
    result[i, 2] <- min(nv)
    result[i, 3] <- mean(nv)
  }
  result
}


compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]
  data
}


# =============================================================================
# MAIN EXECUTION (drop-in replacement)
# =============================================================================

# Build the lookup â€” now O(n_ids * avg_neighbors * avg_years) with integer ops
# instead of O(n_rows * avg_neighbors) character hash lookups.
# ~344K cells Ã— ~4 neighbors Ã— ~28 years â‰ˆ 38.5M integer lookups (fast).
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
# The numerical values produced are identical to the original pipeline.
# Estimated runtime: minutes instead of 86+ hours.
```

---

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| **Alleged bottleneck** | `do.call(rbind, ...)` and list binding in `compute_neighbor_stats()` | **Wrong.** This is a minor cost (~seconds per variable). |
| **True bottleneck** | â€” | `build_neighbor_lookup()`: 6.46M iterations each doing `paste()` + named-vector hash probes into a 6.46M-entry vector. |
| **Root cause** | â€” | Character key construction (`paste`) and R's named-vector lookup overhead at scale: ~25.8M hash probes into a 6.46M-key table, all inside a scalar `lapply`. |
| **Fix** | â€” | Replace character keys with integer arithmetic via `match()` + a pre-built `row_matrix[id_idx, year_idx]`. Loop over 344K cells (not 6.46M rows). Use matrix subindexing for neighbor row resolution. |
| **Expected speedup** | â€” | From ~86+ hours to **minutes** (roughly 1,000â€“5,000Ã—). |
| **Model preservation** | â€” | âœ… Trained RF untouched. Numerical output identical. |
END_RESPONSE>>>
