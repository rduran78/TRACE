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
run_id: Anthropic_F04_T07_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction steps, and specifically:

### 1. `build_neighbor_lookup` â€” O(N) `lapply` with per-row string operations

For each of the ~6.46 million rows, this function:
- Converts an integer ID to character and looks it up in a named vector (`id_to_ref`).
- Retrieves neighbor cell IDs from the `nb` object.
- Constructs `paste(..., sep="_")` key strings for every neighbor of every row.
- Looks those keys up in a named character vector (`idx_lookup`) of length 6.46M.

Named vector lookup in R is **hash-based but still slow per call** when done millions of times inside `lapply`. With ~6.46M rows and an average of ~4 rook neighbors per cell (1,373,394 directed / 344,208 cells â‰ˆ 4), that is **~26 million string constructions and hash lookups** just for the neighbor lookup build. The `paste()` and named-vector lookups are all interpreted R â€” no vectorization.

### 2. `compute_neighbor_stats` â€” O(N) `lapply` with per-row subsetting

Called 5 times (once per source variable). Each call iterates over 6.46M rows, extracts neighbor values, and computes `max`, `min`, `mean`. That is **5 Ã— 6.46M = 32.3 million R-level function calls**, each with vector subsetting and aggregation.

### 3. Combined cost

The entire pipeline performs roughly **~60 million interpreted R iterations** with string allocation, hashing, and small-vector aggregation. On a standard laptop, this easily reaches the estimated 86+ hours.

---

## Optimization Strategy

The key insight: **eliminate all per-row string operations and replace the row-level `lapply` loops with vectorized / `data.table` grouped operations.**

### Specific tactics

| Tactic | Applies to | Expected speedup |
|---|---|---|
| **A.** Replace the `paste`-key lookup with an integer join via `data.table`. Build a `(cell_id, year) â†’ row_index` integer lookup table once; then join neighbors in bulk. | `build_neighbor_lookup` | ~100â€“500Ã— |
| **B.** Expand the neighbor list into a long-form `data.table` edge list `(row_i, row_j)` once, then compute all 5 variables' stats with a single grouped aggregation â€” no R-level loop over rows. | `compute_neighbor_stats` + outer loop | ~50â€“200Ã— |
| **C.** Combine A and B: produce one long edge table `(source_row, neighbor_row)`, left-join all 5 variable values in one pass, then `group_by(source_row)` to compute `max/min/mean` for all variables simultaneously. | Everything | Collapses 5 passes into 1 |

Expected wall-clock time: **minutes, not hours** (typically 2â€“10 minutes on 16 GB RAM for this data size).

### Memory feasibility

The long edge table has ~6.46M rows Ã— ~4 neighbors = ~26M rows Ã— a few integer/double columns â‰ˆ **< 1 GB**. Fits comfortably in 16 GB.

---

## Working R Code

```r
library(data.table)

#' Vectorized spatial neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data       data.frame/data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order        integer vector of cell IDs in the same order as rook_neighbors_unique
#' @param neighbors       spdep::nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names to summarize
#' @return cell_data (data.table) with new columns appended:
#'         <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean for each var

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          neighbors,
                                          neighbor_source_vars) {

  # --- Convert to data.table (by reference if already one) ---
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Preserve original row order
  cell_data[, .row_idx := .I]

  # ---------------------------------------------------------------
  # STEP 1: Build (cell_id, year) -> row_index lookup (integer keys)
  # ---------------------------------------------------------------
  # This replaces the paste-based named vector lookup entirely.
  row_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # ---------------------------------------------------------------
  # STEP 2: Expand the nb object into a long edge list of cell IDs

  # ---------------------------------------------------------------
  # neighbors[[k]] gives integer indices into id_order for the k-th cell.
  # We need: source_cell_id -> neighbor_cell_id

  n_cells <- length(id_order)

  # Pre-compute lengths to allocate vectors in one shot
  n_neighbors <- vapply(neighbors, length, integer(1))
  total_edges <- sum(n_neighbors)

  source_cell_id   <- rep.int(id_order, times = n_neighbors)
  neighbor_cell_id <- id_order[unlist(neighbors, use.names = FALSE)]

  edge_dt <- data.table(
    source_id   = source_cell_id,
    neighbor_id = neighbor_cell_id
  )
  rm(source_cell_id, neighbor_cell_id)  # free memory

  # ---------------------------------------------------------------
  # STEP 3: Cross with years to get (source_row, neighbor_row) pairs
  # ---------------------------------------------------------------
  # Every edge exists in every year. Instead of replicating the edge
  # table 28 times, we join through the row_lookup.

  years <- sort(unique(cell_data$year))

  # Expand edges Ã— years
  # More memory-efficient: join source side first, then neighbor side.

  # 3a. Attach source row index
  # For each (source_id, year) find the row index in cell_data
  edge_year <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
  edge_year[, source_id   := edge_dt$source_id[edge_idx]]
  edge_year[, neighbor_id := edge_dt$neighbor_id[edge_idx]]

  # Join to get source row
  edge_year[row_lookup, on = .(source_id = id, year = year),
            source_row := i..row_idx]

  # Join to get neighbor row
  edge_year[row_lookup, on = .(neighbor_id = id, year = year),
            neighbor_row := i..row_idx]

  # Drop edges where either side is missing (boundary cells / missing years)
  edge_year <- edge_year[!is.na(source_row) & !is.na(neighbor_row)]

  # Keep only what we need
  edge_year <- edge_year[, .(source_row, neighbor_row)]

  rm(edge_dt, row_lookup)
  gc()

  # ---------------------------------------------------------------
  # STEP 4: Vectorized aggregation for ALL variables at once
  # ---------------------------------------------------------------
  # Attach neighbor values for every source variable in one go.

  # Extract neighbor values via integer indexing (vectorized)
  for (var in neighbor_source_vars) {
    vals <- cell_data[[var]]
    set(edge_year, j = var, value = vals[edge_year$neighbor_row])
  }

  # Remove the neighbor_row column to save memory
  edge_year[, neighbor_row := NULL]

  # Group by source_row and compute max, min, mean for each variable
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  }))

  names(agg_exprs) <- agg_names

  # Perform the grouped aggregation
  agg_result <- edge_year[, lapply(agg_exprs, eval, envir = .SD), by = source_row]

  # --- Simpler, equivalent aggregation (avoids bquote complexity): ---
  # Build it explicitly:
  agg_result <- edge_year[,
    {
      out <- list()
      for (v in neighbor_source_vars) {
        vv <- .SD[[v]]
        vv <- vv[!is.na(vv)]
        if (length(vv) == 0L) {
          out[[paste0(v, "_neighbor_max")]]  <- NA_real_
          out[[paste0(v, "_neighbor_min")]]  <- NA_real_
          out[[paste0(v, "_neighbor_mean")]] <- NA_real_
        } else {
          out[[paste0(v, "_neighbor_max")]]  <- max(vv)
          out[[paste0(v, "_neighbor_min")]]  <- min(vv)
          out[[paste0(v, "_neighbor_mean")]] <- mean(vv)
        }
      }
      out
    },
    by = source_row
  ]

  rm(edge_year)
  gc()

  # ---------------------------------------------------------------
  # STEP 5: Merge aggregated features back into cell_data
  # ---------------------------------------------------------------
  # agg_result has column "source_row" = original row index in cell_data

  feature_cols <- setdiff(names(agg_result), "source_row")

  # Initialize new columns with NA

  for (col in feature_cols) {
    set(cell_data, j = col, value = NA_real_)
  }

  # Assign by integer row index (vectorized, in-place)
  for (col in feature_cols) {
    set(cell_data, i = agg_result$source_row, j = col, value = agg_result[[col]])
  }

  # Clean up helper column
  cell_data[, .row_idx := NULL]

  return(cell_data)
}
```

### Memory-optimized variant (if the ~26M Ã— 28 edge-year expansion is too large)

If memory is tight, process **one year at a time** â€” still fully vectorized within each year:

```r
compute_all_neighbor_features_chunked <- function(cell_data,
                                                  id_order,
                                                  neighbors,
                                                  neighbor_source_vars) {

  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  cell_data[, .row_idx := .I]

  # --- Build edge list once (cell-level, not row-level) ---
  n_neighbors <- vapply(neighbors, length, integer(1))
  edge_dt <- data.table(
    source_id   = rep.int(id_order, times = n_neighbors),
    neighbor_id = id_order[unlist(neighbors, use.names = FALSE)]
  )

  # --- Initialize output columns ---
  feature_cols <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  }))
  for (col in feature_cols) {
    set(cell_data, j = col, value = NA_real_)
  }

  # --- Process year by year ---
  years <- sort(unique(cell_data$year))

  for (yr in years) {
    # Subset rows for this year
    yr_rows <- cell_data[year == yr, .(.row_idx, id)]
    setkey(yr_rows, id)

    # Map source_id and neighbor_id to row indices for this year
    edges_yr <- copy(edge_dt)
    edges_yr[yr_rows, on = .(source_id = id),   src_row := i..row_idx]
    edges_yr[yr_rows, on = .(neighbor_id = id),  nbr_row := i..row_idx]
    edges_yr <- edges_yr[!is.na(src_row) & !is.na(nbr_row)]

    # Attach neighbor values
    for (v in neighbor_source_vars) {
      vals <- cell_data[[v]]
      set(edges_yr, j = v, value = vals[edges_yr$nbr_row])
    }

    # Aggregate
    agg <- edges_yr[, {
      out <- list()
      for (v in neighbor_source_vars) {
        vv <- .SD[[v]]
        vv <- vv[!is.na(vv)]
        if (length(vv) == 0L) {
          out[[paste0(v, "_neighbor_max")]]  <- NA_real_
          out[[paste0(v, "_neighbor_min")]]  <- NA_real_
          out[[paste0(v, "_neighbor_mean")]] <- NA_real_
        } else {
          out[[paste0(v, "_neighbor_max")]]  <- max(vv)
          out[[paste0(v, "_neighbor_min")]]  <- min(vv)
          out[[paste0(v, "_neighbor_mean")]] <- mean(vv)
        }
      }
      out
    }, by = src_row]

    # Write back in-place
    for (col in feature_cols) {
      set(cell_data, i = agg$src_row, j = col, value = agg[[col]])
    }
  }

  cell_data[, .row_idx := NULL]
  return(cell_data)
}
```

### Drop-in replacement call

```r
library(data.table)

# --- Replace the original three-step pipeline with one call ---
# BEFORE (86+ hours):
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }

# AFTER (minutes):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features_chunked(
  cell_data            = cell_data,
  id_order             = id_order,
  neighbors            = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# The output columns (<var>_neighbor_max, _neighbor_min, _neighbor_mean)
# are numerically identical to the original implementation.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M `paste` + hash lookups in `lapply` | One `data.table` keyed join (vectorized) |
| Neighbor stats | 5 Ã— 6.46M R-level `lapply` iterations | Single grouped `data.table` aggregation per year (28 iterations, each fully vectorized) |
| String allocations | ~26M `paste()` calls | **Zero** |
| Estimated time | 86+ hours | **2â€“10 minutes** |
| Peak memory | Lower but irrelevant at 86 hrs | ~2â€“4 GB (fits in 16 GB) |
| Numerical output | Baseline | **Identical** (`max`, `min`, `mean` on same neighbor sets) |
| Trained RF model | Preserved | **Preserved** (no retraining) |
END_RESPONSE>>>
