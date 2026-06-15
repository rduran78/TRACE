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
run_id: Anthropic_F12_T03_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of ~6.46 million small 3-element vectors is not free, it is a single vectorized C-level operation that completes in seconds. There is no "repeated list binding" inside `compute_neighbor_stats()` â€” it uses `lapply` to build a list in one pass, then binds once. This is standard and efficient R.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **`paste()` + named-vector lookup for every row:** `build_neighbor_lookup` iterates over **~6.46 million rows**. For each row `i`, it:
   - Calls `as.character(data$id[i])` â€” scalar character conversion.
   - Indexes into `id_to_ref` by name â€” a named-vector lookup (linear hash probe per call).
   - Constructs `neighbor_keys` via `paste()` for every neighbor of every row.
   - Performs named-vector lookup into `idx_lookup` (a named vector of length ~6.46 million) for each neighbor key.

2. **Scale of the problem:** With ~6.46M rows and an average of ~4 rook neighbors per cell, this inner function executes ~25.8 million `paste()` calls and ~25.8 million named-vector lookups into a 6.46M-entry named vector. Named vector lookup in R is O(n) in the worst case per probe (it's a linear scan of the names, not a hash table). Even if R internally hashes, the overhead of 25.8M individual character-key lookups into a 6.46M-entry structure is enormous.

3. **The `lapply` over 6.46M elements** with non-trivial per-element work (character coercion, paste, named lookup) is the dominant cost â€” likely accounting for 80%+ of the 86-hour runtime.

`compute_neighbor_stats()` by contrast does only integer indexing (`vals[idx]`) and simple arithmetic â€” this is fast.

## Optimization Strategy

1. **Replace named-vector lookups with `data.table` hash joins or environment-based hashing.** Use `data.table` for O(1) keyed lookups.
2. **Vectorize `build_neighbor_lookup` entirely:** Instead of building a per-row list of neighbor indices (6.46M list elements), pre-build the entire neighbor-row mapping as a flat table using vectorized joins, then split once.
3. **Vectorize `compute_neighbor_stats`:** Use `data.table` grouped aggregation on the flat neighbor table instead of `lapply` over 6.46M elements.
4. **Preserve the trained Random Forest model** â€” we only change feature engineering, producing identical numerical columns.

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED build_neighbor_lookup (vectorized via data.table)
# ============================================================
# Instead of returning a list of length nrow(data), we return
# a data.table mapping each row index to its neighbor row indices.
# This avoids 6.46M iterations with paste + named-vector lookups.

build_neighbor_map_dt <- function(data, id_order, neighbors) {
  # data must have columns: id, year
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  dt <- as.data.table(data)
  dt[, row_idx := .I]

  # Step 1: Build a mapping from cell id -> position in id_order
  id_to_ref <- data.table(
    cell_id = id_order,
    ref_idx = seq_along(id_order)
  )

  # Step 2: Build a flat edge list from the nb object:
  #   (ref_idx_from, ref_idx_to)
  # This is done once for the ~344K cells, not per row.
  edge_list <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb <- neighbors[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(data.table(from_ref = integer(0), to_ref = integer(0)))
    }
    data.table(from_ref = i, to_ref = as.integer(nb))
  }))

  # Map ref indices back to cell IDs
  edge_list[, from_id := id_order[from_ref]]
  edge_list[, to_id   := id_order[to_ref]]

  # Step 3: Build a keyed lookup: (id, year) -> row_idx
  setkey(dt, id, year)

  # Step 4: For each row in dt, find its ref_idx, then its neighbor cell IDs,
  #         then the row indices of those neighbors in the same year.
  #         We do this via vectorized joins.

  # Attach ref_idx to each row
  dt_with_ref <- merge(dt[, .(row_idx, id, year)], id_to_ref,
                        by.x = "id", by.y = "cell_id", all.x = TRUE)

  # Join with edge_list to get neighbor cell IDs for each row
  # dt_with_ref has: row_idx, id, year, ref_idx
  # edge_list has: from_ref, to_ref, from_id, to_id
  neighbor_expand <- merge(
    dt_with_ref[, .(row_idx, year, ref_idx)],
    edge_list[, .(from_ref, to_id)],
    by.x = "ref_idx", by.y = "from_ref",
    all.x = FALSE,       # inner join: rows with no neighbors are dropped
    allow.cartesian = TRUE
  )
  # neighbor_expand now has: ref_idx, row_idx (of the focal cell-year),
  #                          year, to_id (neighbor cell id)

  # Step 5: Look up the row_idx of each neighbor in the same year
  # Build a keyed table for lookup
  row_lookup <- dt[, .(neighbor_row_idx = row_idx, id, year)]
  setkey(row_lookup, id, year)

  setnames(neighbor_expand, "to_id", "neighbor_id")
  setkey(neighbor_expand, neighbor_id, year)

  # Keyed join: find the row index of (neighbor_id, year)
  neighbor_map <- row_lookup[neighbor_expand,
                             .(row_idx = i.row_idx,
                               neighbor_row_idx = x.neighbor_row_idx),
                             on = .(id = neighbor_id, year = year),
                             nomatch = NA]

  # Drop NAs (neighbor cell-year combinations that don't exist in data)
  neighbor_map <- neighbor_map[!is.na(neighbor_row_idx)]

  return(neighbor_map)
  # Columns: row_idx (focal row), neighbor_row_idx (neighbor row)
}


# ============================================================
# OPTIMIZED compute_neighbor_stats (vectorized via data.table)
# ============================================================
# Instead of lapply over 6.46M rows, we do grouped aggregation.

compute_neighbor_stats_dt <- function(data_dt, neighbor_map, var_name, nrow_data) {
  # data_dt: data.table with at least column [[var_name]] and row order preserved
  # neighbor_map: data.table with (row_idx, neighbor_row_idx)
  # var_name: character, the variable to aggregate
  # nrow_data: total number of rows in the original data

  # Extract neighbor values via integer indexing (vectorized)
  nm <- copy(neighbor_map)
  nm[, val := data_dt[[var_name]][neighbor_row_idx]]

  # Drop NAs in val
  nm_valid <- nm[!is.na(val)]

  # Grouped aggregation
  stats <- nm_valid[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = row_idx]

  # Build full result (NA for rows with no valid neighbors)
  result <- data.table(
    row_idx = seq_len(nrow_data),
    nb_max  = NA_real_,
    nb_min  = NA_real_,
    nb_mean = NA_real_
  )
  result[stats, on = "row_idx",
         `:=`(nb_max = i.nb_max, nb_min = i.nb_min, nb_mean = i.nb_mean)]

  return(result)
}


# ============================================================
# OPTIMIZED compute_and_add_neighbor_features
# ============================================================
compute_and_add_neighbor_features_dt <- function(data_dt, var_name,
                                                  neighbor_map, nrow_data) {
  stats <- compute_neighbor_stats_dt(data_dt, neighbor_map, var_name, nrow_data)

  # Name columns to match original pipeline output
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)

  data_dt[, (max_col)  := stats$nb_max]
  data_dt[, (min_col)  := stats$nb_min]
  data_dt[, (mean_col) := stats$nb_mean]

  return(data_dt)
}


# ============================================================
# MAIN PIPELINE (drop-in replacement for the outer loop)
# ============================================================

# Convert to data.table (in-place if possible)
cell_data_dt <- as.data.table(cell_data)

# Build the vectorized neighbor map ONCE (~344K cells Ã— ~4 neighbors Ã— 28 years)
# This replaces the 6.46M-iteration lapply in build_neighbor_lookup
cat("Building neighbor map (vectorized)...\n")
neighbor_map <- build_neighbor_map_dt(cell_data_dt, id_order, rook_neighbors_unique)
cat("Neighbor map built:", nrow(neighbor_map), "directed cell-year-neighbor links\n")

nrow_data <- nrow(cell_data_dt)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  cell_data_dt <- compute_and_add_neighbor_features_dt(
    cell_data_dt, var_name, neighbor_map, nrow_data
  )
}

# Convert back to data.frame if downstream code (e.g., the trained RF) expects it
cell_data <- as.data.frame(cell_data_dt)

# The trained Random Forest model is untouched.
# The numerical output columns (nb_max_*, nb_min_*, nb_mean_*) are identical
# to the original pipeline's output â€” same max/min/mean over the same
# integer-indexed neighbor values.
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M `lapply` iterations with `paste()` + named-vector probes | Single vectorized `data.table` merge (~38.4M rows built in seconds) |
| **Stat computation** | 6.46M `lapply` iterations with per-element `max/min/mean` | `data.table` grouped `[, .(max, min, mean), by=row_idx]` â€” columnar, cache-friendly |
| **`do.call(rbind, ...)`** | Binds 6.46M 3-element vectors (colleague's suspect â€” actually fast) | Eliminated entirely; results written directly to columns |
| **Expected runtime** | ~86+ hours | **Minutes** (dominated by the single `merge` in `build_neighbor_map_dt`) |
| **Memory** | ~6.46M-element list of integer vectors | ~38.4M-row flat `data.table` (~600 MB) â€” fits in 16 GB |
| **Trained RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical output** | Identical âœ“ | Identical âœ“ |

The true bottleneck was the **6.46 million scalar iterations in `build_neighbor_lookup()`**, each performing character coercion, `paste()`, and named-vector lookups into a 6.46M-entry vector â€” not the `do.call(rbind, ...)` in `compute_neighbor_stats()`.
END_RESPONSE>>>
