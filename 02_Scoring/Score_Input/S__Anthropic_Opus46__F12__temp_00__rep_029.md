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
run_id: Anthropic_F12_T00_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not ideal, it is a one-shot operation per variable (5 times total) and each invocation merely stacks pre-allocated 3-element vectors. This is a minor cost.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string operations at scale**: For each of the ~6.46 million rows, the function calls `as.character(data$id[i])`, performs a named-vector lookup (`id_to_ref[...]`), then constructs `paste(neighbor_cell_ids, data$year[i], sep = "_")` for every neighbor, and finally does another named-vector lookup (`idx_lookup[neighbor_keys]`). Named vector lookup in R is **O(n)** hash-table probing per call, and `paste()` allocates new strings each time.

2. **~6.46 million `lapply` iterations with string key construction**: Each row has on average ~4 rook neighbors (1,373,394 directed relationships / 344,208 cells â‰ˆ 4). So the function constructs and looks up roughly **25.8 million paste-generated string keys** via named-vector indexing. This is astronomically expensive in R's single-threaded interpreted loop.

3. **The lookup is called once but dominates wall time**: `compute_neighbor_stats()` is called 5 times (once per variable) but uses only fast integer indexing (`vals[idx]`). `build_neighbor_lookup()` is called once but does all the expensive string work. On a dataset of this size, this single call likely accounts for the vast majority of the 86+ hour runtime.

4. **`do.call(rbind, result)` is a secondary concern**: Binding 6.46M three-element numeric vectors is not free, but it's dwarfed by the string-construction bottleneck. Replacing it with a pre-allocated matrix is a minor optimization.

## Optimization Strategy

1. **Eliminate all string key construction and named-vector lookups.** Replace the string-keyed approach with pure integer arithmetic. Since the data has a regular panel structure (344,208 cells Ã— 28 years), we can compute row indices directly: if the data is sorted by `(id, year)`, then the row for cell `c` in year `y` is `(cell_index - 1) * n_years + (year - min_year + 1)`. This turns every lookup into O(1) integer arithmetic.

2. **Vectorize `compute_neighbor_stats()`** by pre-allocating a matrix instead of using `do.call(rbind, ...)`.

3. **Preserve the trained Random Forest model** â€” we only change the feature-engineering pipeline, not the model.

4. **Preserve the original numerical estimand** â€” the computed neighbor max/min/mean values are identical, just computed faster.

## Working R Code

```r
# ==============================================================================
# OPTIMIZED PIPELINE
# ==============================================================================

# ----------------------------------------------------------
# Step 0: Ensure data is sorted by (id, year) â€” required for
#         direct integer index arithmetic.
# ----------------------------------------------------------
cell_data <- cell_data[order(cell_data$id, cell_data$year), ]

# ----------------------------------------------------------
# Step 1: Build neighbor lookup using pure integer arithmetic.
#         No string keys, no paste(), no named-vector lookups.
# ----------------------------------------------------------
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  n_cells <- length(id_order)
  years   <- sort(unique(data$year))
  n_years <- length(years)
  min_year <- min(years)

  # Map each spatial id to its 1-based cell index in the sorted data.
  # id_order gives the spatial ids in the order matching the nb object.
  # We need: for each unique id in the data, what is its positional index

  # in the sorted-by-id data?
  unique_ids_sorted <- unique(data$id)  # already sorted because data is sorted by (id, year)
  # cell_pos: maps from id_order index (nb index) -> position among unique sorted ids
  id_to_sorted_pos <- match(id_order, unique_ids_sorted)

  # For each cell (in nb ordering), get the sorted-data positions of its neighbors
  # Then expand across all years.

  # Pre-compute neighbor sorted positions for each cell in nb ordering
  # neighbors[[k]] gives nb-indices of neighbors of cell k
  # We need sorted-data positions of those neighbors.

  # Total number of (row, neighbor_row) pairs to estimate memory:
  # ~6.46M rows * ~4 neighbors = ~25.8M pairs. Stored as a list of integer

  # vectors: ~200 MB, fits in 16 GB.

  cat("Building fast neighbor lookup for", nrow(data), "rows...\n")

  # For each nb-index cell, get the sorted-position indices of its neighbors
  neighbor_sorted_pos <- lapply(seq_along(neighbors), function(k) {
    nb_indices <- neighbors[[k]]
    if (length(nb_indices) == 0L) return(integer(0))
    pos <- id_to_sorted_pos[nb_indices]
    pos[!is.na(pos)]
  })

  # Now build the row-level lookup.
  # Row for cell at sorted position `p` (1-based) in year `y` is:
  #   (p - 1) * n_years + (y - min_year + 1)
  # Neighbor rows for that row: for each neighbor sorted position `np`:
  #   (np - 1) * n_years + (y - min_year + 1)
  # i.e., same year offset, different cell block.

  # We can vectorize this across all years for each cell.
  # Strategy: iterate over cells (344K), and for each cell expand across years.

  lookup <- vector("list", nrow(data))

  year_offsets <- seq_len(n_years)  # 1 to 28

  for (p in seq_along(unique_ids_sorted)) {
    nb_pos <- neighbor_sorted_pos[which(id_order == unique_ids_sorted[p])]
    if (length(nb_pos) == 0L) {
      # No match in id_order â€” shouldn't happen, but handle gracefully
      for (t in year_offsets) {
        row_i <- (p - 1L) * n_years + t
        lookup[[row_i]] <- integer(0)
      }
      next
    }
    nb_pos <- nb_pos[[1]]  # integer vector of neighbor sorted positions
    if (length(nb_pos) == 0L) {
      for (t in year_offsets) {
        row_i <- (p - 1L) * n_years + t
        lookup[[row_i]] <- integer(0)
      }
      next
    }
    # Base row indices for neighbors (year offset = 0)
    nb_base <- (nb_pos - 1L) * n_years
    for (t in year_offsets) {
      row_i <- (p - 1L) * n_years + t
      lookup[[row_i]] <- nb_base + t
    }
  }

  lookup
}

# Even faster: fully vectorized version avoiding the inner year loop
build_neighbor_lookup_vectorized <- function(data, id_order, neighbors) {
  n_cells_nb <- length(id_order)
  years       <- sort(unique(data$year))
  n_years     <- length(years)

  unique_ids_sorted <- unique(data$id)  # sorted because data is sorted by (id, year)
  n_cells_data      <- length(unique_ids_sorted)

  # Map id_order (nb ordering) -> sorted-data cell position
  id_to_sorted_pos <- match(id_order, unique_ids_sorted)

  # Map sorted-data cell position -> nb index
  sorted_pos_to_nb <- integer(n_cells_data)
  valid <- !is.na(id_to_sorted_pos)
  sorted_pos_to_nb[id_to_sorted_pos[valid]] <- which(valid)

  cat("Building vectorized neighbor lookup for", nrow(data), "rows...\n")

  # For each data cell (sorted position p), find its nb index, then its neighbors
  # Pre-compute: for each sorted position p, the neighbor sorted positions
  nb_sorted <- vector("list", n_cells_data)
  for (p in seq_len(n_cells_data)) {
    nb_idx <- sorted_pos_to_nb[p]
    if (nb_idx == 0L) {
      nb_sorted[[p]] <- integer(0)
      next
    }
    nb_indices <- neighbors[[nb_idx]]
    if (length(nb_indices) == 0L) {
      nb_sorted[[p]] <- integer(0)
      next
    }
    pos <- id_to_sorted_pos[nb_indices]
    nb_sorted[[p]] <- pos[!is.na(pos)]
  }

  # Now expand to row-level lookup.
  # Row index for cell p, year-offset t: (p-1)*n_years + t
  # Neighbor row indices: (nb_p - 1)*n_years + t  for each nb_p in nb_sorted[[p]]
  lookup <- vector("list", n_cells_data * n_years)

  for (p in seq_len(n_cells_data)) {
    nb_pos <- nb_sorted[[p]]
    if (length(nb_pos) == 0L) {
      for (t in seq_len(n_years)) {
        lookup[[(p - 1L) * n_years + t]] <- integer(0)
      }
    } else {
      nb_base <- (nb_pos - 1L) * n_years
      for (t in seq_len(n_years)) {
        lookup[[(p - 1L) * n_years + t]] <- nb_base + t
      }
    }
  }

  lookup
}

# ----------------------------------------------------------
# Step 2: Optimized compute_neighbor_stats with pre-allocated matrix
# ----------------------------------------------------------
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n    <- length(neighbor_lookup)
  out  <- matrix(NA_real_, nrow = n, ncol = 3)

  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0L) next
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    out[i, 1] <- max(nv)
    out[i, 2] <- min(nv)
    out[i, 3] <- mean(nv)
  }

  out
}

# ----------------------------------------------------------
# Step 3: Compute and add neighbor features (preserving column names)
# ----------------------------------------------------------
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]
  data
}

# ----------------------------------------------------------
# Step 4: Run the optimized pipeline
# ----------------------------------------------------------
# Build lookup ONCE (the former main bottleneck, now fast)
neighbor_lookup <- build_neighbor_lookup_vectorized(
  cell_data, id_order, rook_neighbors_unique
)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor features for:", var_name, "\n")
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_lookup
  )
}

# ----------------------------------------------------------
# The trained Random Forest model is untouched.
# The numerical estimand is preserved: max, min, mean of
# neighbor values are computed identically.
# ----------------------------------------------------------
```

## Summary

| Aspect | Colleague's Claim | Actual Finding |
|---|---|---|
| **Alleged bottleneck** | `do.call(rbind, ...)` and list binding in `compute_neighbor_stats()` | **Wrong.** This is a minor cost (5 calls, each stacking 6.46M Ã— 3 vectors). |
| **True bottleneck** | â€” | `build_neighbor_lookup()`: 6.46 million iterations of `paste()` + named-vector string lookups, constructing ~25.8 million string keys. |
| **Root cause** | â€” | O(n) string hashing per row, single-threaded, on 6.46M rows. |
| **Fix** | â€” | Replace all string key construction with O(1) integer arithmetic exploiting the regular panel structure: row = `(cell_pos - 1) Ã— n_years + year_offset`. |
| **Expected speedup** | â€” | From ~86+ hours to minutes. The cell-level loop (344K iterations) with a trivial inner year loop (28 iterations of integer addition) replaces 6.46M iterations of string construction and hash-table lookup. |
| **Model preservation** | â€” | âœ… Random Forest untouched. |
| **Numerical equivalence** | â€” | âœ… Same max/min/mean computed from identical neighbor sets. |
END_RESPONSE>>>
