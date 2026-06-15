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
run_id: Anthropic_F12_T10_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, result)` on ~6.46 million small 3-element vectors is not free, it is a single vectorizable matrix assembly step and takes on the order of seconds, not hours.

**The true bottleneck is `build_neighbor_lookup()`.**

Here is the evidence from the code:

1. **Massive per-row character key construction and named-vector lookup**: `build_neighbor_lookup()` runs an `lapply` over **every row** (~6.46 million rows). For each row, it:
   - Converts `data$id[i]` to character and performs a named-vector lookup (`id_to_ref[as.character(data$id[i])]`) â€” this is O(n) named-vector search repeated 6.46M times.
   - Retrieves the neighbor cell IDs and calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` to build string keys â€” millions of string allocations.
   - Performs **another** named-vector lookup `idx_lookup[neighbor_keys]` where `idx_lookup` is a named vector of length **6.46 million** â€” named-vector lookup on a vector this size is extremely slow (R uses linear hashing with string comparison, and doing this 6.46M Ã— ~4 neighbors â‰ˆ 25.8 billion character comparisons).

2. **The result is invariant across variables but recomputed only once** â€” that's fine, but the single computation itself is the 86+ hour bottleneck. The `paste`-based string key lookups against a 6.46M-entry named vector are catastrophically slow. Named vector indexing in R with string keys is not O(1) hash-table lookup; performance degrades severely at this scale.

3. `compute_neighbor_stats()` by contrast is simple: it indexes a numeric vector by integer positions (fast) and computes `max/min/mean` on small neighbor sets. The `do.call(rbind, ...)` on the result list is a single operation. Even if suboptimal, it accounts for seconds, not hours.

**Root cause summary**: The pipeline spends virtually all its time in `build_neighbor_lookup()` doing repeated string construction (`paste`) and named-character-vector indexing (`idx_lookup[neighbor_keys]`) at a scale of ~6.46M Ã— ~4 = ~25M lookups against a 6.46M-length named vector.

---

## Optimization Strategy

1. **Replace string-key lookups with integer arithmetic.** Instead of building `"id_year"` string keys and looking them up in a named vector, use the structure of the panel: every cell appears in every year (balanced panel, 28 years). Compute row positions arithmetically: `row = (cell_position - 1) * n_years + year_offset`. This is O(1) per lookup with no string allocation.

2. **Vectorize `build_neighbor_lookup()`** â€” eliminate the per-row `lapply` entirely. Pre-expand the neighbor relationships into a full edge list (cell_i â†’ cell_j for each year), compute target row indices with integer math, and store the result as a grouped integer list.

3. **Vectorize `compute_neighbor_stats()`** â€” replace `lapply` + `do.call(rbind, ...)` with column-wise grouped aggregation using the edge list and `data.table` or vectorized split/vapply.

4. **Preserve the trained Random Forest model** â€” we only change feature engineering speed; the output columns are numerically identical (same max, min, mean of the same neighbor values).

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Ensure data is a data.table, sorted by (id, year)
# ============================================================
cell_dt <- as.data.table(cell_data)

# Sort by id then year â€” critical for the arithmetic indexing trick
setorder(cell_dt, id, year)

# The unique IDs and years, in sorted order
unique_ids   <- sort(unique(cell_dt$id))
unique_years <- sort(unique(cell_dt$year))
n_ids   <- length(unique_ids)   # 344,208
n_years <- length(unique_years) # 28

# Verify balanced panel
stopifnot(nrow(cell_dt) == n_ids * n_years)

# ============================================================
# STEP 1: Build integer mappings (replaces all paste/named-vector lookups)
# ============================================================

# Map each unique cell id to its 1-based position in the sorted id vector
id_to_pos <- setNames(seq_along(unique_ids), as.character(unique_ids))

# Map each year to its 1-based offset
year_to_offset <- setNames(seq_along(unique_years), as.character(unique_years))

# With data sorted by (id, year), the row index of cell at position p
# in year with offset y is:  (p - 1) * n_years + y
# This replaces the entire idx_lookup named vector and paste() calls.

# ============================================================
# STEP 2: Build the neighbor edge list ONCE using integer arithmetic
#          (replaces build_neighbor_lookup entirely)
# ============================================================

# rook_neighbors_unique is an nb object: a list of length n_ids,
# where element [[p]] gives the positions (in id_order) of neighbors of
# the p-th cell in id_order.
# id_order is the vector of cell IDs in the order matching the nb object.

# Map id_order positions to our sorted-id positions
id_order_to_pos <- id_to_pos[as.character(id_order)]

build_neighbor_edge_list <- function(rook_neighbors_unique,
                                     id_order_to_pos,
                                     n_years) {
  # For each cell position p in id_order, get its neighbors' positions
  # and create edges (source_pos, target_pos)
  n_cells <- length(rook_neighbors_unique)

  # Pre-compute lengths for pre-allocation
  lens <- lengths(rook_neighbors_unique)
  total_edges_per_year <- sum(lens)  # ~1.37M directed edges

  # Pre-allocate edge list for ONE year
  source_pos <- integer(total_edges_per_year)
  target_pos <- integer(total_edges_per_year)

  offset <- 0L
  for (p in seq_len(n_cells)) {
    nb <- rook_neighbors_unique[[p]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) next
    n_nb <- length(nb)
    idx_range <- (offset + 1L):(offset + n_nb)
    source_pos[idx_range] <- id_order_to_pos[p]
    target_pos[idx_range] <- id_order_to_pos[nb]
    offset <- offset + n_nb
  }

  # Trim if any cells had 0 neighbors
  if (offset < total_edges_per_year) {
    source_pos <- source_pos[1:offset]
    target_pos <- target_pos[1:offset]
  }

  # Now expand across all years using integer row arithmetic
  # Row of cell at position p in year-offset y = (p-1)*n_years + y
  source_rows <- integer(offset * n_years)
  target_rows <- integer(offset * n_years)

  for (y in seq_len(n_years)) {
    rng <- ((y - 1L) * offset + 1L):(y * offset)
    source_rows[rng] <- (source_pos - 1L) * n_years + y
    target_rows[rng] <- (target_pos - 1L) * n_years + y
  }

  data.table(source_row = source_rows, target_row = target_rows)
}

cat("Building neighbor edge list...\n")
system.time({
  edge_dt <- build_neighbor_edge_list(rook_neighbors_unique,
                                      id_order_to_pos,
                                      n_years)
})
# Expected: ~38.4M rows (1.37M edges Ã— 28 years), built in seconds

# ============================================================
# STEP 3: Vectorized compute_neighbor_stats using data.table grouping
#          (replaces compute_neighbor_stats and the for loop)
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics for", length(neighbor_source_vars), "variables...\n")

for (var_name in neighbor_source_vars) {
  cat("  Processing:", var_name, "\n")

  # Extract the variable values for all target (neighbor) rows
  edge_dt[, val := cell_dt[[var_name]][target_row]]

  # Remove NAs from neighbor values before aggregation
  valid_edges <- edge_dt[!is.na(val)]

  # Grouped aggregation: max, min, mean by source_row
  agg <- valid_edges[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = source_row]

  # Initialize result columns with NA
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  # Fill in computed values
  cell_dt[agg$source_row, (max_col)  := agg$nb_max]
  cell_dt[agg$source_row, (min_col)  := agg$nb_min]
  cell_dt[agg$source_row, (mean_col) := agg$nb_mean]
}

# Clean up temporary column
edge_dt[, val := NULL]

# ============================================================
# STEP 4: Convert back to data.frame if the RF model expects one
# ============================================================
cell_data <- as.data.frame(cell_dt)

cat("Done. Neighbor features added to cell_data.\n")
```

---

## Performance Comparison

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` (string paste + named-vector lookup over 6.46M rows) | **~85+ hours** (dominant bottleneck) | **Eliminated** â€” replaced by integer arithmetic edge list built in ~5â€“15 seconds |
| `compute_neighbor_stats` (lapply + do.call rbind, 5 vars) | ~minutes | **~1â€“3 minutes total** via `data.table` grouped aggregation on ~38.4M-row edge list |
| **Total estimated runtime** | **86+ hours** | **< 5 minutes** |

## Why This Preserves Correctness

- The neighbor relationships are identical: same rook neighbors, same year matching, same directed edges.
- The aggregation functions (`max`, `min`, `mean`) are computed on exactly the same sets of non-NA neighbor values.
- The output columns are numerically identical to the original pipeline's output.
- The trained Random Forest model is not modified or retrained â€” it simply receives the same feature columns it was trained on.
END_RESPONSE>>>
