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
run_id: Anthropic_F12_T00_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single matrix assembly step per variable (5 calls total). Each call binds ~6.46M rows Ã— 3 columns â€” this completes in seconds.

**The true bottleneck is `build_neighbor_lookup()`.**

Here is the evidence from the code:

1. **Massive character key construction and named-vector lookup (O(n) repeated hashing):** `build_neighbor_lookup` creates `idx_lookup`, a named integer vector with ~6.46 million entries keyed by `paste(id, year, sep="_")`. It then iterates over every one of those 6.46M rows, and for each row:
   - Converts `data$id[i]` to character and looks it up in `id_to_ref` (named vector lookup).
   - Extracts neighbor cell IDs, pastes them with the current year to form `neighbor_keys`.
   - Performs named lookup into the 6.46M-element `idx_lookup` vector **for each neighbor key**.

2. **Scale of the problem:** With ~6.46M rows and an average of ~4 rook neighbors per cell, the inner `lapply` performs ~25.8 million `paste()` + named-vector lookups against a 6.46M-length named vector. Named vector lookup in R is hash-based but carries significant per-call overhead when done millions of times inside an `lapply`. The `paste` calls alone generate tens of millions of temporary strings.

3. **This function is called once, but it dominates runtime.** The neighbor lookup is reused across the 5 variables, so `compute_neighbor_stats` runs 5 times on a prebuilt lookup â€” those are fast vectorized index operations. The one-time cost of `build_neighbor_lookup` dwarfs everything else.

4. **`compute_neighbor_stats` is actually efficient in structure:** `vals[idx]` is integer-index subsetting (fast), and the summary stats are computed on small vectors (~4 elements). The `do.call(rbind, result)` on a list of 6.46M length-3 vectors takes a few seconds at most. This is not the bottleneck.

**Conclusion:** The bottleneck is the row-by-row `paste`-and-lookup pattern in `build_neighbor_lookup()`, which performs tens of millions of string operations and hash lookups in an interpreted R loop.

---

## Optimization Strategy

1. **Replace the row-level `lapply` in `build_neighbor_lookup` with a vectorized merge/join approach.** Instead of iterating over every row and constructing string keys, we:
   - Expand the neighbor relationships into a flat edge table (cell_id â†’ neighbor_id) once.
   - Join this with the data's (id, year) â†’ row_index mapping using `data.table` equi-joins, which are orders of magnitude faster than named-vector lookups in a loop.
   - Split the result into a list indexed by source row.

2. **Replace `do.call(rbind, result)` in `compute_neighbor_stats` with a direct matrix construction** (minor improvement, but clean).

3. **Preserve the trained Random Forest model** â€” we change only the feature-engineering pipeline, not the model or the numerical values produced.

4. **Preserve the original numerical estimand** â€” the optimized code computes identical `max`, `min`, `mean` neighbor statistics.

---

## Working R Code

```r
library(data.table)

# =============================================================================
# OPTIMIZED build_neighbor_lookup
# =============================================================================
# Replaces the row-by-row paste-and-lookup with a vectorized data.table join.
#
# Inputs are identical to the original:
#   data       â€” data.frame/data.table with columns $id and $year (and others)
#   id_order   â€” vector of cell IDs in the order matching the nb object
#   neighbors  â€” spdep nb object (list of integer index vectors into id_order)
#
# Output is identical: a list of length nrow(data), where each element is an
# integer vector of row indices into `data` for that row's spatial neighbors
# in the same year.
# =============================================================================

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {

  n_cells <- length(id_order)

  # --- Step 1: Build a flat edge list from the nb object ---
  # Each entry neighbors[[i]] is an integer vector of indices into id_order.
  # We expand this into a two-column data.table: (cell_id, neighbor_id).

  # Precompute lengths to allocate once
  n_neighbors <- vapply(neighbors, length, integer(1))
  total_edges <- sum(n_neighbors)

  src_idx <- rep.int(seq_len(n_cells), n_neighbors)
  dst_idx <- unlist(neighbors, use.names = FALSE)

  # Remove the spdep "no neighbors" sentinel (0)
  valid <- dst_idx != 0L
  src_idx <- src_idx[valid]
  dst_idx <- dst_idx[valid]

  edge_dt <- data.table(
    cell_id     = id_order[src_idx],
    neighbor_id = id_order[dst_idx]
  )

  # --- Step 2: Build a row-index table from data ---
  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Table keyed by (id, year) for the SOURCE rows
  source_dt <- dt[, .(id, year, src_row = row_idx)]

  # Table keyed by (id, year) for the NEIGHBOR rows
  neighbor_dt <- dt[, .(id, year, nbr_row = row_idx)]

  # --- Step 3: Join edges with source rows, then with neighbor rows ---
  # For every (source row) we find its neighbor cell IDs, then find the
  # row indices of those neighbor cells in the same year.

  # Join source rows to edge list on cell_id
  # Result: for each source row, all its neighbor cell IDs
  setkey(source_dt, id)
  setkey(edge_dt, cell_id)

  expanded <- edge_dt[source_dt,
    .(neighbor_id, year, src_row),
    on = .(cell_id = id),
    allow.cartesian = TRUE,
    nomatch = 0L
  ]

  # Now join to find the row index of each neighbor in the same year

  setkey(expanded, neighbor_id, year)
  setkey(neighbor_dt, id, year)

  matched <- neighbor_dt[expanded,
    .(src_row, nbr_row),
    on = .(id = neighbor_id, year = year),
    nomatch = 0L
  ]

  # --- Step 4: Split into a list indexed by source row ---
  # We need a list of length nrow(data). Rows with no neighbors get integer(0).

  n_rows <- nrow(data)

  # Order by src_row for efficient splitting
  setkey(matched, src_row)

  lookup_list <- vector("list", n_rows)
  # Fill all with integer(0) default
  for (i in seq_len(n_rows)) lookup_list[[i]] <- integer(0)

  # Use split (vectorized) â€” much faster than row-by-row
  split_result <- split(matched$nbr_row, matched$src_row)
  idx_names <- as.integer(names(split_result))
  for (j in seq_along(idx_names)) {
    lookup_list[[idx_names[j]]] <- as.integer(split_result[[j]])
  }

  lookup_list
}


# =============================================================================
# OPTIMIZED compute_neighbor_stats
# =============================================================================
# Replaces do.call(rbind, ...) with direct matrix allocation, and uses
# vectorized grouped operations via data.table for the heavy lifting.
#
# Produces IDENTICAL numerical output: an nrow(data) Ã— 3 matrix with columns
# max, min, mean of the neighbor values â€” preserving the original estimand.
# =============================================================================

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {

  vals <- data[[var_name]]
  n <- length(neighbor_lookup)

  # Flatten the lookup into a long-form data.table for vectorized computation
  lens <- vapply(neighbor_lookup, length, integer(1))
  total <- sum(lens)

  if (total == 0L) {
    out <- matrix(NA_real_, nrow = n, ncol = 3)
    colnames(out) <- c("max", "min", "mean")
    return(out)
  }

  src_row <- rep.int(seq_len(n), lens)
  nbr_row <- unlist(neighbor_lookup, use.names = FALSE)

  long_dt <- data.table(
    src  = src_row,
    val  = vals[nbr_row]
  )

  # Remove NAs in neighbor values (matches original behavior)
  long_dt <- long_dt[!is.na(val)]

  # Compute grouped stats
  stats_dt <- long_dt[, .(
    vmax  = max(val),
    vmin  = min(val),
    vmean = mean(val)
  ), by = src]

  # Allocate output matrix (NA by default, matching original for rows with no

  # valid neighbors)
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(out) <- c("max", "min", "mean")

  out[stats_dt$src, 1L] <- stats_dt$vmax
  out[stats_dt$src, 2L] <- stats_dt$vmin
  out[stats_dt$src, 3L] <- stats_dt$vmean

  out
}


# =============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# =============================================================================
# Drop-in replacement that calls the fast versions above and adds the same
# columns to cell_data that the original pipeline produced.
# =============================================================================

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats_mat[, 1L]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats_mat[, 2L]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats_mat[, 3L]
  data
}


# =============================================================================
# OPTIMIZED OUTER LOOP
# =============================================================================
# Drop-in replacement for the original outer loop.
# The trained Random Forest model is untouched.
# =============================================================================

# Build the lookup ONCE (this is now minutes instead of tens of hours)
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

# cell_data now has the same 15 new columns (5 vars Ã— 3 stats) with
# numerically identical values. Feed directly to the existing trained
# Random Forest model for prediction â€” no retraining required.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: ~6.46M row-level `paste` + named-vector hash lookups (~25.8M string ops) | Eliminated |
| **Lookup construction** | `lapply` over 6.46M rows, each doing `paste` + `idx_lookup[keys]` | Single `data.table` equi-join on ~27.4M edge-year pairs |
| **Stats computation** | `lapply` + `do.call(rbind, ...)` (minor cost) | Vectorized `data.table` grouped aggregation |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes on the same laptop |
| **Numerical output** | max, min, mean per neighbor set | Identical values |
| **Trained RF model** | â€” | Preserved, untouched |

The colleague misidentified `do.call(rbind, ...)` as the bottleneck. The real culprit is the millions of `paste()` calls and named-vector lookups inside the `lapply` of `build_neighbor_lookup()`. Replacing these with vectorized `data.table` joins eliminates the bottleneck entirely.
END_RESPONSE>>>
