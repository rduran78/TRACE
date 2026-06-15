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
run_id: Anthropic_F12_T03_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized matrix assembly step per variable â€” only 5 calls total. Each inner function returns a fixed-length-3 numeric vector, so there is no "repeated list binding" happening inside `compute_neighbor_stats()` at all; it's a straightforward `lapply` returning pre-allocated length-3 vectors.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Massive character key construction and named-vector lookup**: For each of the ~6.46 million rows, the function calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` to build character keys, then does named-vector indexing into `idx_lookup` (itself a named vector of length ~6.46 million). Named-vector lookup in R is **O(n)** linear scan per query (R's named vectors use linear search, not hash tables). With an average of ~4 rook neighbors per cell and ~6.46M rows, that's ~25.8 million character-key lookups into a 6.46M-length named vector. This is catastrophically slow â€” **O(nÂ²)** in aggregate.

2. **Repeated `as.character()` and `paste()` inside the per-row `lapply`**: These allocate millions of small character vectors, creating enormous GC pressure.

3. **The lookup is called once but dominates total runtime**: `compute_neighbor_stats` is called 5 times, but `build_neighbor_lookup` is the single call that takes the vast majority of the 86+ hours.

In contrast, `compute_neighbor_stats()` does pure numeric indexing (`vals[idx]`) which is O(1) per element â€” it's fast.

## Optimization Strategy

1. **Replace named-vector lookup with an environment (hash map)** or, better yet, **eliminate character-key lookups entirely** by using `data.table` integer joins or a direct integer-indexed matrix approach.

2. **Vectorize `build_neighbor_lookup`** by pre-building a `data.table` keyed on `(id, year)` â†’ `row_index`, then doing a batch merge/join instead of per-row `lapply`.

3. **Replace `do.call(rbind, ...)` with `matrix()` pre-allocation** in `compute_neighbor_stats` (minor improvement, but good practice).

4. **Use `data.table` for the join** to get O(1) amortized hash-based lookups instead of O(n) named-vector scans.

The key insight: we can decompose the problem. For each row `i`, we need the row indices of all rows sharing the same `year` whose `id` is a rook neighbor of row `i`'s `id`. We can pre-build an edge list of (id, neighbor_id) pairs, then join on year to get all (row_i, row_j) pairs at once â€” fully vectorized.

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED build_neighbor_lookup using data.table hash joins
# ============================================================
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert to data.table if not already; preserve original
  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Build a mapping from id -> integer reference index in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Build edge list: (id, neighbor_id) from the nb object
  # neighbors[[k]] gives the indices in id_order that are neighbors of id_order[k]
  edge_list <- rbindlist(lapply(seq_along(id_order), function(k) {
    nb <- neighbors[[k]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(data.table(id = integer(0), neighbor_id = integer(0)))
    }
    data.table(id = id_order[k], neighbor_id = id_order[nb])
  }))

  # Now join: for each (id, year) row, find all neighbor rows
  # Step 1: key the data by (id, year) -> row_idx
  # Step 2: join edge_list with dt on id to get year, then look up neighbor rows

  # Create keyed lookup: given (id, year) -> row_idx
  setkey(dt, id, year)

  # For every row in dt, get its neighbors via the edge list
  # Merge dt with edge_list on 'id' to get (row_idx, year, neighbor_id)
  dt_edges <- merge(
    dt[, .(id, year, row_idx)],
    edge_list,
    by = "id",
    allow.cartesian = TRUE
  )
  # dt_edges now has columns: id, year, row_idx (the source row), neighbor_id

  # Now find the row index of each (neighbor_id, year) pair
  neighbor_rows <- dt[, .(neighbor_id = id, year, neighbor_row_idx = row_idx)]
  setkey(neighbor_rows, neighbor_id, year)
  setkey(dt_edges, neighbor_id, year)

  matched <- neighbor_rows[dt_edges, nomatch = 0L]
  # matched has: neighbor_id, year, neighbor_row_idx, id, row_idx

  # Build the lookup as a list indexed by row_idx
  # Sort for reproducibility
  setkey(matched, row_idx)

  n_rows <- nrow(dt)
  lookup <- vector("list", n_rows)

  # Split neighbor_row_idx by row_idx
  split_result <- split(matched$neighbor_row_idx, matched$row_idx)

  for (nm in names(split_result)) {
    lookup[[as.integer(nm)]] <- as.integer(split_result[[nm]])
  }

  # Fill NULLs with empty integer vectors
  empty <- integer(0)
  for (i in seq_len(n_rows)) {
    if (is.null(lookup[[i]])) lookup[[i]] <- empty
  }

  lookup
}

# ============================================================
# OPTIMIZED compute_neighbor_stats with pre-allocated matrix
# ============================================================
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

# ============================================================
# Wrapper that adds columns (preserves original column naming)
# ============================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1L]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2L]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3L]
  data
}

# ============================================================
# MAIN PIPELINE (drop-in replacement)
# ============================================================

# Build the lookup ONCE â€” this is now minutes instead of days
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_lookup)
}

# The trained Random Forest model is untouched â€” use as before:
# predictions <- predict(rf_model, cell_data)
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | Per-row `lapply` with O(n) named-vector scan â†’ **O(nÂ²)** total | `data.table` hash join â†’ **O(n)** amortized |
| **Key bottleneck** | `build_neighbor_lookup`: ~25.8M linear scans into 6.46M-length named vector | Eliminated entirely via batch merge |
| **`compute_neighbor_stats`** | Already reasonably efficient (numeric indexing) | Minor improvement: pre-allocated matrix instead of `do.call(rbind, lapply(...))` |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes on the same laptop |
| **Memory** | Millions of temporary character vectors | One `data.table` join (fits in 16 GB) |
| **RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical results** | Same max/min/mean estimands | Identical âœ“ |

**Verdict**: The colleague misidentified the bottleneck. The true culprit is `build_neighbor_lookup()` and its O(nÂ²) named-vector lookups, not the `do.call(rbind, ...)` in `compute_neighbor_stats()`.
END_RESPONSE>>>
