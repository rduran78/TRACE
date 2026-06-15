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
run_id: Anthropic_F08_T10_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` treats the neighbor relationship as if it varies per **cell-year row** (6.46 million rows), when in fact the spatial neighbor topology is **static across all 28 years**. This means:

1. **`build_neighbor_lookup` creates 6.46 million entries** â€” one per row â€” even though there are only 344,208 unique spatial cells. Each entry redundantly recomputes the same neighbor cell IDs, merely re-keying them to a different year. This produces a list of ~6.46M integer vectors, consuming enormous memory and time.

2. **`compute_neighbor_stats` then iterates over those 6.46M entries**, indexing into the data for each one. The repeated random-access indexing and per-row `lapply` with R-level loops is extremely slow.

3. The combination of rebuilding identical neighbor structures 28 times over, plus 5 variables Ã— 6.46M rows of R-level looping, yields the estimated 86+ hour runtime.

**Key insight:** The neighbor graph is a property of the **cell**, not the **cell-year**. The variables that change by year are just column values. Therefore:

- Build the neighbor topology **once** over 344,208 cells (static).
- For each year, extract the variable vector for that year's slice and compute neighbor stats using the **cell-level** lookup â€” 28 Ã— 344,208 = 6.46M operations, but with a much smaller, cache-friendly lookup structure and vectorized operations.

## Optimization Strategy

1. **Separate static topology from dynamic data.** Build a cell-level neighbor index list of length 344,208 (not 6.46M). This is just `rook_neighbors_unique` re-indexed to a dense integer mapping â€” essentially already available.

2. **Vectorize the neighbor-stat computation per year.** For each year, subset the data to that year's rows (344,208 rows, aligned by cell order). For each variable, use the cell-level neighbor list plus vectorized C-backed operations (`vapply` over 344K cells, not 6.46M) to compute max, min, mean of neighbor values.

3. **Use matrix indexing and `vapply` for speed.** Pre-extract the variable as a numeric vector aligned to cell order, then compute stats via the small neighbor list.

4. **Optionally leverage `data.table` for fast split-apply.** Use `data.table` keyed joins and ordering to guarantee cell-order alignment within each year, making the vectorized approach safe and fast.

5. **Preserve the numerical estimand exactly.** The same `max`, `min`, `mean` of the same neighbor values are computed â€” only the iteration structure changes.

Estimated speedup: The lookup shrinks from 6.46M to 344K entries (â‰ˆ19Ã—). The per-year inner loop is vectorized and runs 28 times. Total wall-clock time should drop from 86+ hours to **minutes**.

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build a STATIC cell-level neighbor lookup (done ONCE)
# ==============================================================================
# rook_neighbors_unique is an nb object: a list of length 344,208
# where each element is an integer vector of neighbor indices (1-based)
# referencing positions in id_order.
#
# id_order is a vector of 344,208 cell IDs in the canonical order
# matching rook_neighbors_unique.
#
# We simply keep rook_neighbors_unique as-is â€” it IS the static cell-level
# neighbor lookup. We just need a mapping from cell ID to its position.

build_cell_neighbor_lookup <- function(id_order, neighbors_nb) {
  # neighbors_nb: spdep nb object (list of integer vectors of neighbor positions)
  # id_order: vector of cell IDs aligned with neighbors_nb
  #
  # Returns a list:
  #   $id_order       - the canonical cell ID vector
  #   $id_to_pos      - named integer vector mapping cell ID -> position in id_order

  #   $neighbor_pos   - list of integer vectors; neighbor_pos[[i]] gives positions
  #                     of neighbors of cell i in id_order
  
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # spdep nb objects store 0L for cells with no neighbors; clean that up
  neighbor_pos <- lapply(neighbors_nb, function(nb) {
    nb <- nb[nb != 0L]
    as.integer(nb)
  })
  
  list(
    id_order     = id_order,
    id_to_pos    = id_to_pos,
    neighbor_pos = neighbor_pos
  )
}

# ==============================================================================
# STEP 2: Compute neighbor stats for ONE variable across ALL years, vectorized
# ==============================================================================
compute_neighbor_features_fast <- function(dt, var_name, cell_lookup) {
  # dt: data.table with columns: id, year, <var_name>
  # cell_lookup: output of build_cell_neighbor_lookup
  # 
  # Adds three new columns to dt (by reference):
  #   <var_name>_neighbor_max
  #   <var_name>_neighbor_min
  #   <var_name>_neighbor_mean
  
  id_order     <- cell_lookup$id_order
  id_to_pos    <- cell_lookup$id_to_pos
  neighbor_pos <- cell_lookup$neighbor_pos
  n_cells      <- length(id_order)
  
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  # Pre-allocate output columns with NA
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]
  
  # Ensure data.table is keyed for fast subsetting
  if (!identical(key(dt), c("year", "id"))) {
    setkey(dt, year, id)
  }
  
  years <- sort(unique(dt$year))
  
  for (yr in years) {
    # Extract this year's slice, ordered by id (due to key)
    yr_rows <- dt[.(yr)]  # keyed lookup: all rows for this year
    
    # Build a values vector aligned to id_order (position-indexed)
    # yr_rows is sorted by id because of the key, but the ids may not
    # match id_order's ordering exactly. Use explicit alignment.
    vals_by_id <- rep(NA_real_, n_cells)
    pos_in_order <- id_to_pos[as.character(yr_rows$id)]
    valid <- !is.na(pos_in_order)
    vals_by_id[pos_in_order[valid]] <- yr_rows[[var_name]][valid]
    
    # Compute neighbor stats for each cell using the static neighbor list
    # This is the inner loop: 344,208 iterations (not 6.46M)
    stats <- vapply(seq_len(n_cells), function(i) {
      nb_idx <- neighbor_pos[[i]]
      if (length(nb_idx) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      nb_vals <- vals_by_id[nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      c(max(nb_vals), min(nb_vals), mean(nb_vals))
    }, numeric(3))
    # stats is a 3 x n_cells matrix; rows = max, min, mean
    
    # Write results back into dt for the rows matching this year
    # Identify the row indices in dt for this year
    # Because dt is keyed on (year, id), we can find them:
    row_indices <- dt[.(yr), which = TRUE]
    
    # Align: row_indices correspond to yr_rows, which has yr_rows$id
    # Map each yr_rows$id to its position in id_order to get the stat
    cell_positions <- pos_in_order[valid]
    
    # For rows that have a valid cell position, assign the stats
    set(dt, i = row_indices[valid], j = col_max,  value = stats[1, cell_positions])
    set(dt, i = row_indices[valid], j = col_min,  value = stats[2, cell_positions])
    set(dt, i = row_indices[valid], j = col_mean, value = stats[3, cell_positions])
  }
  
  invisible(dt)
}

# ==============================================================================
# STEP 3: Main pipeline â€” drop-in replacement for the original outer loop
# ==============================================================================

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Build the static cell-level neighbor lookup ONCE
cell_lookup <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)

# Neighbor source variables (unchanged)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute neighbor features for each variable
for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  compute_neighbor_features_fast(cell_data, var_name, cell_lookup)
}

# cell_data now has 15 new columns (3 per variable Ã— 5 variables):
#   ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean,
#   ec_neighbor_max, ec_neighbor_min, ec_neighbor_mean,
#   pop_density_neighbor_max, pop_density_neighbor_min, pop_density_neighbor_mean,
#   def_neighbor_max, def_neighbor_min, def_neighbor_mean,
#   usd_est_n2_neighbor_max, usd_est_n2_neighbor_min, usd_est_n2_neighbor_mean

# ==============================================================================
# STEP 4: Predict with the pre-trained Random Forest (UNCHANGED)
# ==============================================================================
# The trained RF model is used as-is. No retraining.
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Changes

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup size | 6.46M entries (one per cell-year) | 344K entries (one per cell) |
| Lookup build time | Hashes 6.46M keys, resolves 6.46MÃ—~4 neighbor keys | Reuses existing `nb` object directly |
| Inner loop iterations | 5 vars Ã— 6.46M = 32.3M R-level calls | 5 vars Ã— 28 years Ã— 344K = 48.2M `vapply` calls (but `vapply` is C-dispatched and the per-call work is trivial) |
| Memory for lookup | ~6.46M-element list of integer vectors | ~344K-element list (already in memory as `nb` object) |
| Year handling | Embedded in row keys; year-agnostic flat iteration | Explicit year loop; vectorized extraction per year-slice |
| Numerical result | max/min/mean of non-NA neighbor values | **Identical** â€” same values, same neighbors, same aggregation functions |
| RF model | Untouched | Untouched |
| Estimated runtime | 86+ hours | ~10â€“30 minutes |
END_RESPONSE>>>
