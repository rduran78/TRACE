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
run_id: Anthropic_F12_T10_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague is **partially right but misses the deeper bottleneck**. Let me audit both functions:

### `compute_neighbor_stats()` â€” Minor bottleneck
- `do.call(rbind, result)` on ~6.46M rows is indeed slow (binding millions of 3-element vectors), but this runs only **5 times** (once per variable). With modern R, binding ~6.46M small vectors takes perhaps 30â€“60 seconds per call â€” annoying but not 86 hours.
- There is no "repeated list binding" inside `compute_neighbor_stats()` â€” it uses `lapply` which pre-allocates a list. The colleague's description is factually wrong about internal list binding.

### `build_neighbor_lookup()` â€” **THE TRUE BOTTLENECK**

This is where the 86+ hours come from. Examine what happens inside the `lapply` over **6.46 million rows**:

1. **`as.character(data$id[i])`** â€” called 6.46M times (minor).
2. **`id_order[neighbors[[ref_idx]]]`** â€” repeated neighbor vector subsetting (moderate).
3. **`paste(neighbor_cell_ids, data$year[i], sep = "_")`** â€” 6.46M calls to `paste()` creating character key vectors (significant).
4. **`idx_lookup[neighbor_keys]`** â€” **6.46M named-vector lookups by character key** (SEVERE).

The named vector `idx_lookup` has ~6.46M entries. Named vector lookup in R is **O(n)** per query (linear scan, not hashed). Performing ~6.46M lookups, each scanning a ~6.46M-length named vector, yields **O(nÂ²) â‰ˆ 4.17 Ã— 10Â¹Â³ operations**. This is the dominant bottleneck â€” not `do.call(rbind, ...)`.

Additionally, the `build_neighbor_lookup` creates a list of ~6.46M integer vectors, each holding neighbor row indices. With ~4 rook neighbors per cell on average, that's ~25.8M integers stored across 6.46M list elements â€” significant memory pressure on a 16 GB machine, especially when the full data frame with 110 columns is also in memory.

## Optimization Strategy

1. **Replace named-vector lookup with an `environment`-based hash map (or `data.table` join)** â€” converts O(n) per lookup to O(1), reducing `build_neighbor_lookup` from O(nÂ²) to O(n).
2. **Vectorize the neighbor lookup construction** â€” instead of row-by-row `lapply` over 6.46M rows, exploit the fact that many rows share the same `id` (each id has 28 year-rows). Build a **cell-level** neighbor map once (344K entries), then expand to row-level using vectorized joins.
3. **Vectorize `compute_neighbor_stats`** â€” replace `lapply` + `do.call(rbind, ...)` with direct matrix indexing and `vapply` or grouped column operations.
4. **Preserve the trained Random Forest model** â€” no changes to model or features, only to how neighbor features are computed (same numerical results).

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” preserves exact numerical output
# =============================================================================

library(data.table)

# ---- Step 1: Optimized neighbor lookup builder (O(n) instead of O(nÂ²)) ------

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert data to data.table for fast operations (non-destructive)
  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Build cell-level neighbor map: for each cell id, which cell ids are neighbors?
  # id_order is the vector of cell ids; neighbors is the nb object (list of integer indices)
  # neighbors[[k]] gives the indices into id_order for neighbors of id_order[k]

  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Get unique cell ids present in data
  unique_ids <- unique(dt$id)

  # Pre-build: for each unique cell id, the neighbor cell ids
  cell_neighbor_ids <- lapply(as.character(unique_ids), function(cid) {
    ref_idx <- id_to_ref[cid]
    if (is.na(ref_idx)) return(integer(0))
    nb_indices <- neighbors[[ref_idx]]
    # nb objects use 0 to indicate no neighbors
    nb_indices <- nb_indices[nb_indices > 0]
    id_order[nb_indices]
  })
  names(cell_neighbor_ids) <- as.character(unique_ids)

  # Build a fast row lookup: (id, year) -> row_idx using data.table keyed join
  setkey(dt, id, year)

  # Now, for each row i, we need the row indices of (neighbor_id, same year) combos.
  # Strategy: build an edge table (focal_row, neighbor_id, year), then join to dt
  # to get neighbor_row indices. Finally, split by focal_row.

  # Step A: for each row, get its id and year, then its neighbor cell ids
  # Use dt to expand: each row -> its neighbor cell ids
  cat("Building edge table...\n")

  # Map each id to an integer group for fast lapply
  id_levels <- as.character(unique_ids)

  # Expand: for each (id, year, row_idx), create (row_idx, neighbor_id, year) rows
  # Do this per unique id to keep memory bounded

  # Pre-compute: group rows by id
  id_groups <- split(seq_len(nrow(dt)), dt$id)

  # Build edge list in chunks
  edge_list <- vector("list", length(unique_ids))

  for (k in seq_along(unique_ids)) {
    cid <- as.character(unique_ids[k])
    nb_ids <- cell_neighbor_ids[[cid]]
    if (length(nb_ids) == 0) next

    focal_rows <- id_groups[[cid]]
    focal_years <- dt$year[focal_rows]

    # Create all (focal_row, neighbor_id, year) combinations
    edge_list[[k]] <- data.table(
      focal_row   = rep(focal_rows, each = length(nb_ids)),
      neighbor_id = rep(nb_ids, times = length(focal_rows)),
      year        = rep(focal_years, each = length(nb_ids))
    )
  }

  cat("Combining edge table...\n")
  edges <- rbindlist(edge_list)
  rm(edge_list)
  gc()

  # Step B: Join edges to dt to get neighbor row indices
  cat("Joining to get neighbor row indices...\n")
  setkey(edges, neighbor_id, year)

  # Create a join target: (id, year) -> row_idx
  row_lookup <- dt[, .(neighbor_id = id, year, neighbor_row = row_idx)]
  setkey(row_lookup, neighbor_id, year)

  edges <- row_lookup[edges, nomatch = 0L]
  # Now edges has columns: neighbor_id, year, neighbor_row, focal_row

  # Step C: Split neighbor_row by focal_row
  cat("Splitting into per-row neighbor lists...\n")
  n_rows <- nrow(dt)

  # Pre-allocate result as list of integer(0)
  neighbor_lookup <- vector("list", n_rows)
  # Fill with empty integer vectors
  for (i in seq_len(n_rows)) neighbor_lookup[[i]] <- integer(0)

  # Use split (fast in data.table)
  split_result <- split(edges$neighbor_row, edges$focal_row)
  split_names  <- as.integer(names(split_result))

  for (j in seq_along(split_names)) {
    neighbor_lookup[[split_names[j]]] <- as.integer(split_result[[j]])
  }

  rm(edges, split_result, row_lookup, dt)
  gc()

  cat("Neighbor lookup built.\n")
  return(neighbor_lookup)
}

# ---- Step 2: Optimized neighbor stats computation ---------------------------

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)

  # Use vapply for pre-allocated matrix output (avoids do.call(rbind, ...))
  result <- vapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, numeric(3))

  # vapply returns 3 x n matrix; transpose to n x 3
  t(result)
}

# ---- Step 3: Feature addition wrapper (preserves original column names) -----

compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)

  # Assign columns â€” must match original naming convention
  data[[paste0(var_name, "_neighbor_max")]]  <- stats_mat[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats_mat[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats_mat[, 3]

  return(data)
}

# ---- Step 4: Run the optimized pipeline -------------------------------------

cat("Starting optimized neighbor feature pipeline...\n")
t0 <- Sys.time()

# Build the lookup ONCE (this is the former bottleneck)
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

t1 <- Sys.time()
cat(sprintf("Pipeline completed in %.1f minutes.\n", difftime(t1, t0, units = "mins")))

# The trained Random Forest model is untouched â€” use it as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Changes and Expected Speedup

| Component | Original Complexity | Optimized Complexity | Speedup Factor |
|---|---|---|---|
| `build_neighbor_lookup` (idx_lookup named-vector scan) | O(nÂ²) â‰ˆ 4.2Ã—10Â¹Â³ ops | O(n log n) via data.table keyed join | **~1,000,000Ã—** |
| `compute_neighbor_stats` (`do.call(rbind,...)`) | O(n) with high overhead | O(n) via `vapply` + `t()` (pre-allocated) | **~3â€“5Ã—** |
| **Total estimated runtime** | **86+ hours** | **~10â€“30 minutes** | **~200â€“500Ã—** |

**Key insight**: The colleague focused on a cosmetic inefficiency (`do.call(rbind, ...)` runs 5 times, ~5 minutes total) while the real bottleneck was the **O(nÂ²) named-vector lookup** in `build_neighbor_lookup()`, which executes 6.46 million linear scans through a 6.46-million-element named vector. Replacing this with `data.table` keyed joins reduces the entire pipeline from days to minutes while preserving the exact same numerical output and leaving the trained Random Forest model untouched.
END_RESPONSE>>>
