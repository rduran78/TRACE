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
run_id: Anthropic_F08_T10_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is slow (~86+ hours) for one fundamental reason: **`build_neighbor_lookup` rebuilds the neighbor relationship for every cell-year row (~6.46 million rows), even though the spatial neighbor topology is identical across all 28 years.** Specifically:

1. **Redundant work in `build_neighbor_lookup`:** For each of the ~6.46M rows, it looks up the cell's neighbors (which are purely spatial and year-invariant), then searches for those neighbors' rows in the current year. This means the same spatial neighbor lookup is performed 28 times per cell. The `paste`/`match` key construction and lookup is O(n) string work repeated millions of times.

2. **Redundant work in `compute_neighbor_stats`:** The `neighbor_lookup` is a list of ~6.46M elements. For each of the 5 variables, the function iterates over all 6.46M entries. This means ~32.3M R-level list iterations with per-element anonymous function calls â€” extremely slow in base R.

3. **Memory pressure:** Storing a 6.46M-element list of integer vectors (the neighbor lookup) is memory-intensive and cache-unfriendly.

**Key insight from the prompt:** *"The neighbor relationship among cells does not change across years, while variables attached to cells do change by year."* This means we should:
- Build the neighbor topology **once** at the cell level (344K cells), not the cell-year level (6.46M rows).
- For each year, use that static topology to gather variable values and compute stats.

---

## Optimization Strategy

### 1. Separate static structure from dynamic data

Build a **cell-level** neighbor lookup once: for each cell index `i` (1 to 344,208), store the integer vector of neighbor cell indices. This is derived directly from `rook_neighbors_unique` (the `nb` object) and is trivially available â€” `rook_neighbors_unique[[i]]` already gives the neighbor indices for cell `i`.

### 2. Process year-by-year with vectorized operations

For each year:
- Subset (or index into) the data to get that year's variable values as a simple vector aligned to cell order.
- Use the static cell-level neighbor lookup to compute max, min, mean via vectorized `vapply` over 344K cells (not 6.46M rows).

### 3. Use matrix indexing and `vapply` instead of `lapply` + `do.call(rbind, ...)`

`vapply` with a fixed return length avoids the overhead of `do.call(rbind, lapply(...))`.

### 4. Complexity reduction

| Aspect | Before | After |
|---|---|---|
| Neighbor lookup size | 6.46M entries | 344K entries (built once) |
| Lookup construction | String-key hash over 6.46M rows | Direct integer index from `nb` object |
| Stats computation per variable | 6.46M iterations | 28 Ã— 344K = 9.64M iterations (same total, but no string ops, simpler indexing) |
| Total estimated time | 86+ hours | **~5â€“15 minutes** |

The speedup comes from eliminating millions of string-paste and hash-lookup operations and working with pure integer-indexed vectors.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Exploits: static neighbor topology + year-varying variable values
# =============================================================================

#' Build a cell-level neighbor lookup (done ONCE, independent of year).
#'
#' @param id_order   Integer vector of cell IDs in the order matching the nb object.
#' @param nb_object  A spdep::nb object (list of integer vectors of neighbor indices).
#' @return A list of length n_cells, where each element is an integer vector
#'         of neighbor cell-position indices (1-based, into id_order).
build_static_neighbor_lookup <- function(id_order, nb_object) {
  n <- length(id_order)
  stopifnot(length(nb_object) == n)
  
  # spdep::nb objects store neighbors as integer indices into the original

  # spatial object, with 0L meaning "no neighbors". We clean that up.
  lapply(seq_len(n), function(i) {
    nbrs <- nb_object[[i]]
    # spdep uses 0L to represent "no neighbours" in a single-element vector
    nbrs <- nbrs[nbrs != 0L]
    as.integer(nbrs)
  })
}

#' Compute neighbor max, min, mean for one variable across ALL years,
#' using the static neighbor lookup.
#'
#' @param cell_data           data.frame/data.table with columns: id, year, and <var_name>.
#' @param var_name            Character: name of the source variable.
#' @param id_order            Integer vector of cell IDs in canonical order.
#' @param years               Integer vector of all years (sorted).
#' @param static_nb_lookup    List from build_static_neighbor_lookup().
#' @return A data.frame with 3 columns: <var>_neighbor_max, <var>_neighbor_min,
#'         <var>_neighbor_mean, with nrow == nrow(cell_data).
compute_neighbor_stats_optimized <- function(cell_data,
                                              var_name,
                                              id_order,
                                              years,
                                              static_nb_lookup) {
  
  n_cells <- length(id_order)
  n_years <- length(years)
  n_rows  <- nrow(cell_data)
  
  # --- Pre-allocate output vectors ---
  out_max  <- rep(NA_real_, n_rows)
  out_min  <- rep(NA_real_, n_rows)
  out_mean <- rep(NA_real_, n_rows)
  
  # --- Build a mapping from cell ID to canonical position index ---
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # --- Process each year independently ---
  for (yr in years) {
    
    # Row mask for this year
    yr_mask <- cell_data$year == yr
    yr_indices <- which(yr_mask)  # row positions in cell_data for this year
    
    if (length(yr_indices) == 0L) next
    
    # Get cell IDs and variable values for this year
    yr_ids  <- cell_data$id[yr_indices]
    yr_vals <- cell_data[[var_name]][yr_indices]
    
    # Map each cell ID in this year's subset to its canonical position
    yr_positions <- id_to_pos[as.character(yr_ids)]
    
    # Build a full-length value vector indexed by canonical cell position
    # (so that static_nb_lookup indices work directly)
    vals_by_pos <- rep(NA_real_, n_cells)
    valid <- !is.na(yr_positions)
    vals_by_pos[yr_positions[valid]] <- yr_vals[valid]
    
    # Now compute neighbor stats for each cell present this year
    # using vectorized vapply over this year's rows only
    stats <- vapply(seq_along(yr_indices), function(j) {
      if (!valid[j]) return(c(NA_real_, NA_real_, NA_real_))
      pos <- yr_positions[j]
      nbr_idx <- static_nb_lookup[[pos]]
      if (length(nbr_idx) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      nbr_vals <- vals_by_pos[nbr_idx]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
    }, numeric(3))
    # stats is 3 x length(yr_indices)
    
    out_max[yr_indices]  <- stats[1L, ]
    out_min[yr_indices]  <- stats[2L, ]
    out_mean[yr_indices] <- stats[3L, ]
  }
  
  result <- data.frame(out_max, out_min, out_mean)
  colnames(result) <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  result
}


# =============================================================================
# MAIN PIPELINE (drop-in replacement for the outer loop)
# =============================================================================

# --- Step 1: Build the static neighbor lookup ONCE ---
static_nb_lookup <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

# --- Step 2: Identify the years in the data ---
years <- sort(unique(cell_data$year))

# --- Step 3: Define the neighbor source variables ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# --- Step 4: Compute and attach neighbor features for each variable ---
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  stats_df <- compute_neighbor_stats_optimized(
    cell_data        = cell_data,
    var_name         = var_name,
    id_order         = id_order,
    years            = years,
    static_nb_lookup = static_nb_lookup
  )
  # Attach the 3 new columns to cell_data
  cell_data[[paste0(var_name, "_neighbor_max")]]  <- stats_df[[1]]
  cell_data[[paste0(var_name, "_neighbor_min")]]  <- stats_df[[2]]
  cell_data[[paste0(var_name, "_neighbor_mean")]] <- stats_df[[3]]
}

# --- Step 5: Predict with the pre-trained Random Forest (UNCHANGED) ---
# The RF model is already trained; we only use predict().
# cell_data now has all the same columns with the same names and
# numerically identical neighbor feature values.
# predictions <- predict(rf_model, newdata = cell_data[, predictor_vars])
```

---

## Why This Preserves the Original Numerical Estimand

The refactored code computes **exactly the same quantities**: for each cell-year row, it gathers the variable values of the cell's rook neighbors *in the same year*, then computes `max`, `min`, and `mean` over non-NA values. The column names are identical. The only difference is *how* the neighbor indices are resolved â€” via a static cell-position lookup instead of year-specific string key hashing â€” but the resulting index sets and therefore the resulting statistics are identical.

The pre-trained Random Forest is **not retrained** â€” it is used as-is in the prediction step with the same feature columns.

---

## Optional Further Speedup: data.table Version

If even more speed is desired (pushing from ~5â€“15 minutes to ~1â€“3 minutes), here is a `data.table` variant that eliminates the inner `vapply` loop using batch vectorization:

```r
library(data.table)

compute_neighbor_stats_dt <- function(cell_data_dt,
                                       var_name,
                                       id_order,
                                       static_nb_lookup) {
  
  n_cells <- length(id_order)
  id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))
  
  # Add canonical cell position
  cell_data_dt[, cell_pos := id_to_pos[as.character(id)]]
  
  # Build an edge table from the static neighbor lookup (done once per call)
  # Each row: (focal_pos, neighbor_pos)
  edges <- rbindlist(lapply(seq_len(n_cells), function(i) {
    nbrs <- static_nb_lookup[[i]]
    if (length(nbrs) == 0L) return(NULL)
    data.table(focal_pos = i, nbr_pos = nbrs)
  }))
  
  # For each year, join edges with variable values to compute stats
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  cell_data_dt[, (max_col)  := NA_real_]
  cell_data_dt[, (min_col)  := NA_real_]
  cell_data_dt[, (mean_col) := NA_real_]
  
  for (yr in sort(unique(cell_data_dt$year))) {
    # Extract this year's values by cell position
    yr_dt <- cell_data_dt[year == yr, .(cell_pos, val = get(var_name))]
    setkey(yr_dt, cell_pos)
    
    # Join: for each edge, get the neighbor's value
    edge_vals <- merge(edges, yr_dt, by.x = "nbr_pos", by.y = "cell_pos",
                       all.x = FALSE, allow.cartesian = FALSE)
    
    # Aggregate by focal_pos
    agg <- edge_vals[!is.na(val),
                     .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)),
                     by = focal_pos]
    
    # Write back into cell_data_dt
    yr_rows <- cell_data_dt[year == yr, which = TRUE]
    yr_cell_pos <- cell_data_dt$cell_pos[yr_rows]
    
    # Match aggregated results to the year's rows
    m <- match(yr_cell_pos, agg$focal_pos)
    cell_data_dt[yr_rows, (max_col)  := agg$nb_max[m]]
    cell_data_dt[yr_rows, (min_col)  := agg$nb_min[m]]
    cell_data_dt[yr_rows, (mean_col) := agg$nb_mean[m]]
  }
  
  cell_data_dt[, cell_pos := NULL]
  invisible(cell_data_dt)
}

# Usage:
cell_data_dt <- as.data.table(cell_data)
static_nb_lookup <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  compute_neighbor_stats_dt(cell_data_dt, var_name, id_order, static_nb_lookup)
}

cell_data <- as.data.frame(cell_data_dt)
```

---

## Summary

| Item | Original | Optimized |
|---|---|---|
| Neighbor lookup granularity | 6.46M cell-year rows | 344K cells (static, built once) |
| Key mechanism | String paste + hash table | Direct integer indexing |
| Stats loop iterations per variable | 6.46M | 28 Ã— 344K (same count, but no string ops) |
| Estimated runtime | 86+ hours | ~5â€“15 min (base R) / ~1â€“3 min (data.table) |
| RF model | Unchanged | Unchanged |
| Numerical results | â€” | Identical |
END_RESPONSE>>>
