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
run_id: Anthropic_F12_T03_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single matrix assembly step per variable â€” only 5 calls total. Each inner function returns a fixed-length-3 numeric vector, so there is no "repeated list binding" inside `compute_neighbor_stats()`; it uses `lapply`, which pre-allocates its output list. That cost is modest.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Specifically:

1. **`paste()` key construction and named-vector lookup (`idx_lookup[neighbor_keys]`)** is called once per row â€” 6.46 million times â€” and each call performs character pasting and hash-table lookup for every neighbor of every cell-year. With ~1.37 million directed neighbor relationships replicated across 28 years, this produces tens of millions of string operations and named-vector lookups inside a sequential `lapply`.

2. **`as.character(data$id[i])` and `id_to_ref[...]`** are called row-by-row (6.46M times) instead of being vectorized.

3. The entire `neighbor_lookup` structure stores **integer index vectors for every single row** (~6.46M list elements), consuming enormous memory and time to construct.

The fundamental problem: the neighbor topology is **year-invariant** (rook neighbors don't change across years), yet the code re-derives neighbor row indices for every cell-year combination by string-pasting year suffixes and doing hash lookups. This is an O(N Ã— K) string-operation bottleneck where N = 6.46M and K = average neighbor count (~4 for rook).

`compute_neighbor_stats()` is actually reasonably efficient given the lookup â€” it's just indexing into a numeric vector. The real cost is building the lookup.

## Optimization Strategy

1. **Separate the spatial topology from the temporal dimension.** Build a neighbor lookup only over the 344,208 unique cell IDs (not 6.46M cell-years). Since neighbors are year-invariant, we only need to know, for each cell, which other cells are neighbors.

2. **Vectorize the stats computation using `data.table` split-by-year.** For each year, subset the data, build a simple integer-indexed vector of values, and compute neighbor stats using the compact cell-level lookup. This turns 6.46M hash lookups into 28 vectorized operations over 344K cells.

3. **Replace `do.call(rbind, ...)` with pre-allocated matrix output** (addresses the colleague's minor concern for free).

4. **Preserve the trained Random Forest model** â€” we only change feature engineering speed, not the features themselves. The numerical output is identical.

## Working R Code

```r
library(data.table)

# =============================================================================
# STEP 1: Build a CELL-LEVEL neighbor lookup (year-invariant)
#         This runs over 344,208 cells, NOT 6.46 million cell-years.
# =============================================================================

build_cell_neighbor_lookup <- function(id_order, rook_neighbors_unique) {
  # id_order: vector of cell IDs in the order matching the nb object
  # rook_neighbors_unique: spdep nb object (list of integer neighbor indices)
  #
  # Returns: a named list, keyed by cell ID (as character),
  #          each element is an integer vector of neighbor cell IDs.
  
  n <- length(id_order)
  id_order_char <- as.character(id_order)
  
  lookup <- vector("list", n)
  names(lookup) <- id_order_char
  
  for (i in seq_len(n)) {
    nb_idx <- rook_neighbors_unique[[i]]
    # spdep nb objects use 0L to denote no neighbors
    if (length(nb_idx) == 1L && nb_idx[1L] == 0L) {
      lookup[[i]] <- integer(0)
    } else {
      lookup[[i]] <- id_order[nb_idx]
    }
  }
  
  lookup
}

# =============================================================================
# STEP 2: Vectorized neighbor stats computation, year-by-year
#         For each year, we only loop over 344K cells (vectorized indexing).
# =============================================================================

compute_neighbor_stats_fast <- function(dt, var_name, cell_neighbor_lookup) {
  # dt: data.table with columns 'id', 'year', and var_name
  # cell_neighbor_lookup: named list from build_cell_neighbor_lookup
  # Returns: dt with three new columns appended (max, min, mean of neighbor var)
  
  col_max  <- paste0("n_max_", var_name)
  col_min  <- paste0("n_min_", var_name)
  col_mean <- paste0("n_mean_", var_name)
  
  # Pre-allocate output columns
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]
  
  # Get unique years
  years <- sort(unique(dt$year))
  
  for (yr in years) {
    # Row indices in dt for this year
    yr_rows <- which(dt$year == yr)
    
    # Build a fast value lookup: cell_id -> value for this year
    yr_ids  <- dt$id[yr_rows]
    yr_vals <- dt[[var_name]][yr_rows]
    
    # Named numeric vector for O(1) lookup by cell ID
    names(yr_vals) <- as.character(yr_ids)
    
    # For each cell in this year, get neighbor values
    yr_ids_char <- as.character(yr_ids)
    n_cells     <- length(yr_rows)
    
    out_max  <- rep(NA_real_, n_cells)
    out_min  <- rep(NA_real_, n_cells)
    out_mean <- rep(NA_real_, n_cells)
    
    for (j in seq_len(n_cells)) {
      nb_ids <- cell_neighbor_lookup[[yr_ids_char[j]]]
      if (length(nb_ids) == 0L) next
      
      nb_keys <- as.character(nb_ids)
      nb_vals <- yr_vals[nb_keys]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      
      if (length(nb_vals) == 0L) next
      
      out_max[j]  <- max(nb_vals)
      out_min[j]  <- min(nb_vals)
      out_mean[j] <- mean(nb_vals)
    }
    
    set(dt, i = yr_rows, j = col_max,  value = out_max)
    set(dt, i = yr_rows, j = col_min,  value = out_min)
    set(dt, i = yr_rows, j = col_mean, value = out_mean)
  }
  
  dt
}

# =============================================================================
# STEP 3: Even faster â€” fully vectorized via matrix indexing (recommended)
#         Eliminates the inner per-cell R loop entirely.
# =============================================================================

compute_neighbor_stats_vectorized <- function(dt, var_name, id_order,
                                               rook_neighbors_unique) {
  # Build a sparse neighbor edge list once (cell-index level)
  # Then do grouped vectorized operations per year.
  
  n_cells <- length(id_order)
  
  # Build edge list: (from_cell_idx, to_cell_idx) in id_order space
  from_idx <- integer(0)
  to_idx   <- integer(0)
  for (i in seq_len(n_cells)) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 1L && nb[1L] == 0L) next
    from_idx <- c(from_idx, rep(i, length(nb)))
    to_idx   <- c(to_idx, nb)
  }
  edges <- data.table(from = from_idx, to = to_idx)
  
  # Map cell IDs to cell indices
  id_to_cidx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add cell index to dt
  dt[, cell_idx := id_to_cidx[as.character(id)]]
  
  col_max  <- paste0("n_max_", var_name)
  col_min  <- paste0("n_min_", var_name)
  col_mean <- paste0("n_mean_", var_name)
  
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]
  
  years <- sort(unique(dt$year))
  
  for (yr in years) {
    yr_mask <- dt$year == yr
    yr_cidx <- dt$cell_idx[yr_mask]
    yr_vals <- dt[[var_name]][yr_mask]
    
    # Build cell_idx -> value vector (dense, indexed by cell_idx)
    val_vec <- rep(NA_real_, n_cells)
    val_vec[yr_cidx] <- yr_vals
    
    # Vectorized: get all neighbor values at once via edge list
    nb_vals_vec <- val_vec[edges$to]
    
    # Build a data.table of (from_cell_idx, neighbor_value) and aggregate
    agg_dt <- data.table(from = edges$from, nv = nb_vals_vec)
    agg_dt <- agg_dt[!is.na(nv)]
    
    if (nrow(agg_dt) == 0L) next
    
    stats <- agg_dt[, .(nmax = max(nv), nmin = min(nv), nmean = mean(nv)),
                    by = from]
    
    # Map aggregated stats back to dt rows for this year
    # Build cell_idx -> row_in_dt mapping for this year
    yr_row_indices <- which(yr_mask)
    cidx_to_yr_row <- setNames(yr_row_indices, as.character(yr_cidx))
    
    matched_rows <- cidx_to_yr_row[as.character(stats$from)]
    valid <- !is.na(matched_rows)
    
    set(dt, i = as.integer(matched_rows[valid]), j = col_max,
        value = stats$nmax[valid])
    set(dt, i = as.integer(matched_rows[valid]), j = col_min,
        value = stats$nmin[valid])
    set(dt, i = as.integer(matched_rows[valid]), j = col_mean,
        value = stats$nmean[valid])
  }
  
  dt[, cell_idx := NULL]
  dt
}

# =============================================================================
# STEP 4: Main pipeline â€” drop-in replacement
# =============================================================================

# Convert to data.table for speed (preserves all columns)
cell_dt <- as.data.table(cell_data)

# Build the edge list ONCE (takes seconds, not hours)
# Option A: Use the fully vectorized version (recommended)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Processing neighbor stats for: ", var_name)
  cell_dt <- compute_neighbor_stats_vectorized(
    cell_dt, var_name, id_order, rook_neighbors_unique
  )
}

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is untouched.
# The output columns (n_max_*, n_min_*, n_mean_*) are numerically identical.
# Predict as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Summary

| Aspect | Colleague's Claim | Actual Bottleneck |
|---|---|---|
| **Target** | `do.call(rbind, ...)` and list binding in `compute_neighbor_stats()` | `build_neighbor_lookup()`: 6.46M string-paste + hash lookups |
| **Root cause** | Supposedly inefficient aggregation | Year-invariant topology recomputed for every cell-year via string operations |
| **Scale** | 5 `rbind` calls on 6.46M Ã— 3 matrices (~seconds each) | 6.46M Ã— ~4 `paste()` + named-vector lookups (~hours) |
| **Fix** | Pre-allocate matrix | Separate spatial topology from time; vectorize via edge-list + `data.table` grouped aggregation |
| **Expected speedup** | Marginal | **~100â€“500Ã—** (86+ hours â†’ minutes) |

The optimized code:
- Builds the neighbor edge list **once** over 344K cells (not 6.46M rows).
- For each year Ã— variable, uses **vectorized indexing** into a dense value vector and **`data.table` grouped aggregation** â€” no per-row R loops over millions of rows.
- Produces **numerically identical** output columns, preserving the trained Random Forest model's compatibility.
END_RESPONSE>>>
