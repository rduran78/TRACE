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
run_id: Anthropic_F12_T07_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working Code

## Diagnosis

The colleague is **partially correct but misses the deeper bottleneck**. Let me audit both functions:

### `compute_neighbor_stats()` â€” Colleague's Claim

`do.call(rbind, result)` on a list of ~6.46 million 3-element vectors is indeed not optimal â€” it builds a large list and then binds. However, this operation on numeric vectors of fixed length (3) is actually reasonably fast. The `lapply` loop itself is simple indexing and basic summary stats. For 5 variables, this runs 5 Ã— 6.46M iterations, but each iteration is trivial (subset a numeric vector, compute max/min/mean). This is **not the primary bottleneck**.

### `build_neighbor_lookup()` â€” The True Deep Bottleneck

**This is where the real cost hides**, and the colleague missed it entirely. Examine what happens inside the `lapply` over all ~6.46 million rows:

1. **`as.character(data$id[i])` and `id_to_ref[as.character(...)]`** â€” called 6.46M times, each doing a named-vector character lookup.
2. **`paste(neighbor_cell_ids, data$year[i], sep = "_")`** â€” creates character key strings for every neighbor of every row. With ~1.37M directed neighbor relationships spread across 344K cells (average ~4 neighbors per cell) and 28 years, this is ~6.46M Ã— ~4 = ~25.8 million `paste` calls inside the loop.
3. **`idx_lookup[neighbor_keys]`** â€” named character vector lookup, 25.8 million times, against a named vector of length 6.46 million.

**Named vector lookup in R is O(n) linear scan** (not hashed). So each of the ~25.8M lookups scans a 6.46M-length character vector. This is catastrophically slow â€” effectively **O(25.8M Ã— 6.46M) character comparisons**. This is the reason the pipeline takes 86+ hours.

Additionally, `build_neighbor_lookup` is called **once** and its output is reused for all 5 variables, so the bottleneck is concentrated in a single massive computation, not spread across the loop.

### Verdict

**Reject the colleague's diagnosis.** The main bottleneck is `build_neighbor_lookup()`, specifically the repeated named-character-vector lookups (`idx_lookup[neighbor_keys]`) inside a 6.46M-iteration `lapply`. Named vector lookup in R uses linear search, making this an O(NÂ²)-scale operation on millions of elements. The `do.call(rbind, ...)` in `compute_neighbor_stats` is a secondary, much smaller concern.

---

## Optimization Strategy

1. **Replace named vector lookups with environment-based hash maps** (R environments use hashing, giving O(1) amortized lookup instead of O(N) linear scan). This transforms `build_neighbor_lookup` from O(N Ã— M) to O(N) where N = rows, M = average neighbors.

2. **Vectorize `build_neighbor_lookup`** â€” instead of looping per-row, exploit the fact that neighbor relationships are static across years. Pre-build a cell-level neighbor map, then expand across all 28 years using vectorized operations.

3. **Replace `do.call(rbind, result)` in `compute_neighbor_stats`** with pre-allocated matrix output and direct assignment â€” eliminates the secondary bottleneck the colleague identified.

4. **Preserve the trained Random Forest model** â€” we only change the feature-engineering pipeline, not the model.

5. **Preserve the original numerical estimand** â€” the optimized code computes identical max, min, mean neighbor statistics.

---

## Working Optimized R Code

```r
# =============================================================================
# OPTIMIZED build_neighbor_lookup
# =============================================================================
# Key changes:
#   - Use environment (hash map) instead of named vector for idx_lookup
#   - Vectorize the year-expansion rather than looping per row
#   - Pre-compute cell-level neighbor structure, then expand across years
# =============================================================================

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {

  n_rows <- nrow(data)

  # --- Step 1: Build a hash-map from (id, year) -> row index ----------------
  # Environments in R use hashing: O(1) average lookup vs O(N) for named vectors
  idx_env <- new.env(hash = TRUE, parent = emptyenv(), size = n_rows * 1.2)
  keys <- paste(data$id, data$year, sep = "_")
  for (j in seq_len(n_rows)) {
    idx_env[[ keys[j] ]] <- j
  }

  # --- Step 2: Build cell-level neighbor map (cell index -> neighbor cell ids)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Unique cell IDs in the data (preserving order of id_order)
  unique_ids <- unique(data$id)

  # For each unique cell, find its neighbor cell IDs (done once, not per row)
  cell_neighbor_ids <- lapply(unique_ids, function(cell_id) {
    ref_idx <- id_to_ref[as.character(cell_id)]
    nb_indices <- neighbors[[ref_idx]]
    id_order[nb_indices]
  })
  names(cell_neighbor_ids) <- as.character(unique_ids)

  # --- Step 3: For each row, resolve neighbor row indices using the hash map -
  # Group rows by cell ID to avoid redundant neighbor-id lookups
  cell_id_char <- as.character(data$id)
  years <- data$year

  # Pre-allocate output list
  neighbor_lookup <- vector("list", n_rows)

  for (i in seq_len(n_rows)) {
    nb_cell_ids <- cell_neighbor_ids[[ cell_id_char[i] ]]
    if (length(nb_cell_ids) == 0L) {
      neighbor_lookup[[i]] <- integer(0)
      next
    }
    nb_keys <- paste(nb_cell_ids, years[i], sep = "_")
    matched <- integer(length(nb_keys))
    count <- 0L
    for (k in seq_along(nb_keys)) {
      val <- idx_env[[ nb_keys[k] ]]
      if (!is.null(val)) {
        count <- count + 1L
        matched[count] <- val
      }
    }
    neighbor_lookup[[i]] <- matched[seq_len(count)]
  }

  neighbor_lookup
}


# =============================================================================
# OPTIMIZED compute_neighbor_stats
# =============================================================================
# Key changes:
#   - Pre-allocate output matrix instead of do.call(rbind, list(...))
#   - Direct assignment into matrix rows
#   - Identical numerical results: max, min, mean of non-NA neighbor values
# =============================================================================

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)

  # Pre-allocate output matrix (avoids do.call(rbind, ...) entirely)
  out <- matrix(NA_real_, nrow = n, ncol = 3)

  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0L) next
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0L) next
    out[i, 1L] <- max(neighbor_vals)
    out[i, 2L] <- min(neighbor_vals)
    out[i, 3L] <- mean(neighbor_vals)
  }

  out
}


# =============================================================================
# OPTIMIZED compute_and_add_neighbor_features (wrapper)
# =============================================================================
# Preserves the original column-naming convention and numerical estimand.
# =============================================================================

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)

  data[[paste0(var_name, "_neighbor_max")]]  <- stats_mat[, 1L]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats_mat[, 2L]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats_mat[, 3L]

  data
}


# =============================================================================
# OPTIMIZED OUTER PIPELINE
# =============================================================================
# - The trained Random Forest model is untouched.
# - The numerical estimand (neighbor max/min/mean) is preserved exactly.
# - Only the feature-engineering step is optimized.
# =============================================================================

message("Building optimized neighbor lookup (hash-based)...")
t0 <- Sys.time()
neighbor_lookup <- build_neighbor_lookup_fast(
  cell_data, id_order, rook_neighbors_unique
)
message("Neighbor lookup built in ", round(difftime(Sys.time(), t0, units = "mins"), 1), " min")

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var_name)
  t1 <- Sys.time()
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_lookup
  )
  message("  Done in ", round(difftime(Sys.time(), t1, units = "secs"), 1), " sec")
}

message("Feature engineering complete. Trained RF model is unchanged.")
```

---

## Expected Performance Improvement

| Component | Original | Optimized | Speedup Factor |
|---|---|---|---|
| `build_neighbor_lookup` (idx_lookup) | O(N) per lookup Ã— ~25.8M lookups = O(NÃ—M) â‰ˆ hours | O(1) per lookup Ã— ~25.8M = O(M) â‰ˆ minutes | **~1000Ã—+** |
| `compute_neighbor_stats` (rbind) | `do.call(rbind, 6.46M-list)` | Pre-allocated matrix, direct write | **~5-10Ã—** |
| **Total pipeline** | **~86+ hours** | **~15-45 minutes** (estimated) | **~100-300Ã—** |

The dominant gain comes from replacing the O(N) named-vector scan in `build_neighbor_lookup` with O(1) environment-based hash lookups. The `compute_neighbor_stats` optimization is real but secondary â€” confirming that the colleague's diagnosis was aimed at the wrong function.
END_RESPONSE>>>
