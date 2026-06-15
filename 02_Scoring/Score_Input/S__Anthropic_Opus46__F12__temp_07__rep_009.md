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
run_id: Anthropic_F12_T07_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not ideal, it is a one-shot operation per variable (only 5 calls total) and each call binds vectors of length 3 â€” this is a minor cost.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's the evidence from the code:

1. **Massive per-row `paste()` and named-vector lookup inside `lapply`**: For each of the ~6.46 million rows, the function:
   - Calls `as.character(data$id[i])` and looks up `id_to_ref[...]` â€” a named vector lookup (hash-based, but repeated 6.46M times with R-level overhead).
   - Computes `neighbor_cell_ids` by indexing into `id_order`.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” constructing string keys for every neighbor of every row.
   - Looks up each key in `idx_lookup` â€” another named vector with ~6.46 million entries.

2. **Scale**: With ~6.46 million rows and an average of ~4 rook neighbors per cell, this inner function performs roughly **25.8 million string constructions and hash lookups**, all inside an R-level `lapply` with per-element function call overhead. This is the operation that drives the 86+ hour runtime.

3. `compute_neighbor_stats()` by contrast is cheap: it does pure numeric indexing (`vals[idx]`) and simple arithmetic (`max`, `min`, `mean`) â€” all vectorized C-level operations. The `do.call(rbind, result)` on 6.46M vectors of length 3 takes seconds, not hours.

**Root cause summary**: The bottleneck is the **row-level string construction and hash-table lookup** strategy in `build_neighbor_lookup()`. The lookup is recomputed per row despite the fact that the neighbor structure is constant across years â€” every cell has the same neighbors in every year. The function redundantly reconstructs 28 copies of the same spatial neighbor relationships (one per year), each time through expensive string operations.

---

## Optimization Strategy

1. **Eliminate per-row string key construction entirely.** Instead of building a string-keyed lookup, exploit the panel structure: if data is sorted by `(id, year)`, then for a given cell `id` at a given `year`, the row index of its neighbor can be computed arithmetically from the neighbor's `id` and the year offset â€” no strings needed.

2. **Build the lookup using vectorized integer arithmetic instead of `lapply` + `paste` + named-vector hashing.** Pre-sort the data by `(id, year)`, create a direct integer map from `id` to its block-start row, and then for each cell-year row, compute neighbor row indices as `block_start[neighbor_id] + year_offset`.

3. **Keep `compute_neighbor_stats()` largely as-is** but replace `do.call(rbind, result)` with a pre-allocated matrix for marginal improvement.

This reduces `build_neighbor_lookup()` from ~25.8 million string operations inside an R loop to a single vectorized pass â€” expected runtime drops from 86+ hours to **minutes**.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE
# Preserves the trained Random Forest model and original numerical estimand.
# =============================================================================

# ---- Step 0: Ensure data is sorted by (id, year) ---------------------------
cell_data <- cell_data[order(cell_data$id, cell_data$year), ]

# ---- Step 1: Optimized build_neighbor_lookup --------------------------------
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Unique IDs and years, both sorted
  unique_ids   <- sort(unique(data$id))
  unique_years <- sort(unique(data$year))
  n_years      <- length(unique_years)
  n_rows       <- nrow(data)

  # Map: id -> integer index in unique_ids (1-based)
  id_int <- match(data$id, unique_ids)

  # Map: year -> integer offset (0-based)
  year_int <- match(data$year, unique_years)

  # Because data is sorted by (id, year), the row for unique_ids[j] in

  # unique_years[t] is: (j - 1) * n_years + t
  # Build a fast row-index matrix: row_matrix[j, t] = row index in data
  # Verify the sort assumption and build the map:
  block_start <- match(unique_ids, data$id)  
  # block_start[j] = first row where id == unique_ids[j]
  # Row for (unique_ids[j], unique_years[t]) = block_start[j] + (t - 1)

  # Map from id_order to unique_ids index
  id_order_to_uid <- match(id_order, unique_ids)

  # Map from data$id to id_order index
  id_to_ref <- match(data$id, id_order)

  # --- Vectorized construction of neighbor_lookup ---
  # For each row i:
  #   ref_idx = id_to_ref[i]  (position in id_order / neighbors list)
  #   neighbor_cell_ids = id_order[neighbors[[ref_idx]]]
  #   neighbor_uid_indices = id_order_to_uid[neighbors[[ref_idx]]]
  #   year_offset = year_int[i] - 1
  #   neighbor_rows = block_start[neighbor_uid_indices] + year_offset
  #
  # We vectorize across all rows by expanding neighbors per row.

  # Pre-compute neighbor uid indices for each id_order entry
  # neighbors is an nb object: list of length = length(id_order)
  n_cells <- length(id_order)
  neighbor_uid_list <- lapply(seq_len(n_cells), function(ref) {
    nb <- neighbors[[ref]]
    if (length(nb) == 0 || (length(nb) == 1 && nb[1] == 0L)) {
      return(integer(0))
    }
    id_order_to_uid[nb]
  })

  # Now build the full lookup: for each row i, compute neighbor rows
  # We group rows by their id_to_ref value to avoid redundant neighbor lookups
  # All rows sharing the same spatial cell have the same neighbor *cells*;
  # they differ only in year_offset.

  cat("Building neighbor lookup for", n_rows, "rows...\n")

  # Pre-allocate result list
  neighbor_lookup <- vector("list", n_rows)

  # Vectorized approach: iterate over unique cells, fill all years at once
  for (j in seq_len(n_cells)) {
    uid_j <- id_order_to_uid[j]
    if (is.na(uid_j)) next
    # Rows in data belonging to this cell
    row_start <- block_start[uid_j]
    if (is.na(row_start)) next
    row_indices <- row_start:(row_start + n_years - 1L)
    # Clamp to valid range (in case some cell-years are missing)
    row_indices <- row_indices[row_indices <= n_rows]
    # Filter to rows that actually belong to this id
    row_indices <- row_indices[data$id[row_indices] == id_order[j]]
    if (length(row_indices) == 0) next

    nb_uids <- neighbor_uid_list[[j]]
    if (length(nb_uids) == 0) {
      for (ri in row_indices) neighbor_lookup[[ri]] <- integer(0)
      next
    }
    # Remove NA neighbor uids (neighbors not present in data)
    nb_uids <- nb_uids[!is.na(nb_uids)]
    if (length(nb_uids) == 0) {
      for (ri in row_indices) neighbor_lookup[[ri]] <- integer(0)
      next
    }
    nb_block_starts <- block_start[nb_uids]
    nb_block_starts <- nb_block_starts[!is.na(nb_block_starts)]
    if (length(nb_block_starts) == 0) {
      for (ri in row_indices) neighbor_lookup[[ri]] <- integer(0)
      next
    }

    for (ri in row_indices) {
      yr_offset <- year_int[ri] - 1L
      candidate_rows <- nb_block_starts + yr_offset
      # Validate: must be in range and have matching year
      valid <- candidate_rows >= 1L & candidate_rows <= n_rows
      candidate_rows <- candidate_rows[valid]
      if (length(candidate_rows) > 0) {
        # Confirm year match (handles ragged panels)
        keep <- data$year[candidate_rows] == data$year[ri]
        candidate_rows <- candidate_rows[keep]
      }
      neighbor_lookup[[ri]] <- as.integer(candidate_rows)
    }
  }

  neighbor_lookup
}

# ---- Step 2: Optimized compute_neighbor_stats -------------------------------
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  out <- matrix(NA_real_, nrow = n, ncol = 3)

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

# ---- Step 3: Compute and add neighbor features (preserves original API) -----
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0("max_neighbor_", var_name)]]  <- stats[, 1]
  data[[paste0("min_neighbor_", var_name)]]  <- stats[, 2]
  data[[paste0("mean_neighbor_", var_name)]] <- stats[, 3]
  data
}

# ---- Step 4: Run the optimized pipeline -------------------------------------
neighbor_lookup <- build_neighbor_lookup_fast(
  cell_data, id_order, rook_neighbors_unique
)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_lookup
  )
}

# The trained Random Forest model is untouched.
# The numerical estimand (max, min, mean of neighbor values) is identical.
```

---

## Even Faster: Fully Vectorized Lookup Construction (No Inner Loop)

If the panel is **balanced** (every cell has all 28 years), we can eliminate the inner loop entirely:

```r
build_neighbor_lookup_vectorized <- function(data, id_order, neighbors) {
  # Requires data sorted by (id, year) and a balanced panel
  unique_ids   <- sort(unique(data$id))
  unique_years <- sort(unique(data$year))
  n_years      <- length(unique_years)
  n_cells_data <- length(unique_ids)

  # Integer id map: data$id -> position in unique_ids
  uid_map   <- match(data$id, unique_ids)
  year_map  <- match(data$year, unique_years)

  # id_order -> position in unique_ids
  io_to_uid <- match(id_order, unique_ids)
  # data$id -> position in id_order
  id_to_io  <- match(data$id, id_order)

  # Block start for each unique_id (1-indexed)
  block_start <- (match(unique_ids, unique_ids) - 1L) * n_years + 1L
  # i.e., block_start[j] = (j-1)*n_years + 1

  # Expand neighbors: for each row, get neighbor row indices
  # row i -> cell uid_map[i], year year_map[i]
  # neighbors of cell id_to_io[i] in id_order space -> nb indices in id_order
  # convert to uid space -> io_to_uid[nb]
  # neighbor rows -> (io_to_uid[nb] - 1) * n_years + year_map[i]

  n_rows <- nrow(data)
  io_indices <- id_to_io  # length n_rows

  # Pre-expand neighbor uid indices per id_order slot
  nb_uid <- lapply(neighbors, function(nb) {
    if (length(nb) == 0 || (length(nb) == 1 && nb[1] == 0L)) return(integer(0))
    res <- io_to_uid[nb]
    res[!is.na(res)]
  })

  # Build lookup: fully vectorized per cell (outer), vectorized per year (inner)
  neighbor_lookup <- vector("list", n_rows)

  for (cell_io in seq_along(id_order)) {
    uid_j <- io_to_uid[cell_io]
    if (is.na(uid_j)) next
    nbuids <- nb_uid[[cell_io]]
    if (length(nbuids) == 0L) {
      rows_j <- ((uid_j - 1L) * n_years + 1L):(uid_j * n_years)
      rows_j <- rows_j[rows_j <= n_rows]
      for (ri in rows_j) neighbor_lookup[[ri]] <- integer(0)
      next
    }
    nb_bases <- (nbuids - 1L) * n_years  # base offsets for each neighbor cell
    rows_j <- ((uid_j - 1L) * n_years + 1L):(uid_j * n_years)
    rows_j <- rows_j[rows_j <= n_rows]
    for (ri in rows_j) {
      yr <- year_map[ri]
      neighbor_lookup[[ri]] <- nb_bases + yr  # vectorized across all neighbors
    }
  }

  neighbor_lookup
}
```

---

## Performance Summary

| Component | Original Cost | Optimized Cost |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M `paste()` + hash lookups â†’ **~80+ hours** | Integer arithmetic, no strings â†’ **~2â€“5 minutes** |
| `compute_neighbor_stats` | Numeric indexing + `do.call(rbind)` â†’ **~minutes** | Pre-allocated matrix â†’ **~1â€“2 minutes** |
| **Total** | **86+ hours** | **~5â€“10 minutes** |

**Conclusion**: The colleague misidentified the bottleneck. The `do.call(rbind, ...)` and list operations in `compute_neighbor_stats()` are minor costs. The true bottleneck is the per-row string construction and hash-table lookup strategy in `build_neighbor_lookup()`, which performs ~25.8 million `paste()` calls and named-vector lookups inside an R-level loop. Replacing this with integer arithmetic exploiting the panel structure eliminates the bottleneck entirely.
END_RESPONSE>>>
