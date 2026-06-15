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
run_id: Anthropic_F06_T07_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) effective cost
The function iterates over **every row** (~6.46 million cell-year rows) and, for each row, performs:
- Character conversion and named-vector lookup (`id_to_ref`)
- String pasting to build neighbor keys (`paste(...)`)
- Named-vector lookup into `idx_lookup` (a named character vector of length ~6.46M)

Named vector lookups in R use linear hashing that degrades at scale. With ~6.46M rows, each doing multiple `paste` + named-vector lookups, this alone can take tens of hours.

**Key insight:** The neighbor topology is **time-invariant** â€” cell *i*'s rook neighbors are the same in every year. Yet the code rebuilds string keys and re-resolves neighbor indices for every cell-year combination, repeating the same spatial lookup 28 times per cell.

### Bottleneck 2: `compute_neighbor_stats` â€” repeated `lapply` over 6.46M rows
This is called 5 times (once per source variable). Each call iterates over 6.46M entries, subsetting, removing NAs, and computing `max/min/mean`. The `lapply` + `do.call(rbind, ...)` pattern is slow for millions of small vectors.

### Why raster focal/kernel operations don't directly apply
The comment in the prompt asks whether raster focal operations offer a useful analogy. They do conceptually (a rook neighborhood is a 3Ã—3 cross kernel), but the data is stored as an **irregular panel** (not all cells may be present in all years, cells have arbitrary IDs, there are NAs to handle). Converting to a raster stack for 28 years Ã— 5 variables is possible but risks altering the numerical results if the grid has gaps or irregular boundaries. The strategy below preserves exact numerical equivalence by using the same neighbor structure, but computes it **vastly** more efficiently.

---

## Optimization Strategy

| Strategy | Speedup Source |
|---|---|
| **1. Separate spatial and temporal dimensions** | Build neighbor index only over 344K cells (not 6.46M cell-years). Reuse across all 28 years. Eliminates 28Ã— redundancy. |
| **2. Replace named-vector lookups with integer-indexed `data.table` joins** | `data.table` keyed joins are O(n log n) vs. O(nÂ²) for large named vectors. |
| **3. Vectorized matrix operations for stats** | Instead of `lapply` over 6.46M rows, build a sparse neighbor matrix and use matrix multiplication / row operations for mean, and vectorized grouped `max`/`min`. |
| **4. Process all 5 variables in one pass** | Avoid re-traversing the neighbor structure 5 times. |

**Expected speedup:** From 86+ hours â†’ **~2â€“10 minutes** on the same hardware.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Preserves exact numerical results and the trained Random Forest model.
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # -------------------------------------------------------------------
  # STEP 0: Convert to data.table for performance (non-destructive)
  # -------------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Ensure 'id' and 'year' columns exist
  stopifnot(all(c("id", "year") %in% names(dt)))

  # -------------------------------------------------------------------
  # STEP 1: Build a SPATIAL-ONLY neighbor edge list (time-invariant)

  #   rook_neighbors_unique is an nb object: a list of length = # cells,
  #   where each element contains integer indices into id_order of neighbors.
  #   We convert this to an edge list of (focal_id, neighbor_id).
  # -------------------------------------------------------------------
  n_cells <- length(id_order)
  stopifnot(length(rook_neighbors_unique) == n_cells)

  # Build edge list: focal_cell_id -> neighbor_cell_id
  focal_idx <- rep(seq_len(n_cells),
                   times = lengths(rook_neighbors_unique))
  neighbor_idx <- unlist(rook_neighbors_unique)

  # Remove the 0-neighbor sentinel that spdep uses (integer(0) becomes empty,
  # but some nb objects use 0L to indicate no neighbors)
  valid <- neighbor_idx > 0L
  focal_idx <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]

  edges <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )

  # -------------------------------------------------------------------
  # STEP 2: Join edges with panel data to get neighbor values
  #
  #   For each (focal_id, year), we need the variable values of all
  #   its rook neighbors in that same year. We accomplish this with a
  #   keyed join: edges Ã— dt on (neighbor_id == id, year == year).
  # -------------------------------------------------------------------

  # Key the main data for fast joining
  # We need to join on neighbor_id = id AND year = year
  # Create a slim table with just id, year, and the source variables
  keep_cols <- c("id", "year", neighbor_source_vars)
  dt_slim <- dt[, ..keep_cols]
  setnames(dt_slim, "id", "neighbor_id")
  setkey(dt_slim, neighbor_id, year)

  # Add year to edges by cross-joining edges with unique years
  # WRONG approach: that would be huge. Instead, replicate edges per year

  # BETTER: join edges to the focal data to get (focal_id, year, neighbor_id),
  # then join to dt_slim to get neighbor values.

  # Actually, the most memory-efficient approach:
  # For each year, do the join. But 28 iterations is fine.

  # Alternatively (and faster): build the full join table at once.
  # edges has ~1.37M rows. Adding year: 1.37M * 28 = ~38.4M rows.
  # Each row needs the neighbor variable values. With 5 numeric vars,
  # that's ~38.4M * 5 * 8 bytes â‰ˆ 1.5 GB â€” fits in 16 GB RAM.

  years <- sort(unique(dt$year))

  # Expand edges across all years
  edges_expanded <- CJ(edge_idx = seq_len(nrow(edges)), year = years)
  edges_expanded[, focal_id    := edges$focal_id[edge_idx]]
  edges_expanded[, neighbor_id := edges$neighbor_id[edge_idx]]
  edges_expanded[, edge_idx := NULL]

  # Join to get neighbor variable values
  setkey(edges_expanded, neighbor_id, year)
  edges_expanded <- dt_slim[edges_expanded, on = .(neighbor_id, year)]

  # Now edges_expanded has columns:
  #   neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, focal_id

  # -------------------------------------------------------------------
  # STEP 3: Compute grouped max, min, mean per (focal_id, year)
  # -------------------------------------------------------------------
  setkey(edges_expanded, focal_id, year)

  # Compute stats for all variables at once using data.table aggregation
  stat_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))

  stat_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the aggregation call dynamically
  # data.table's .SDcols approach is cleaner here:
  stats_dt <- edges_expanded[,
    {
      result <- vector("list", length(neighbor_source_vars) * 3L)
      k <- 1L
      for (v in neighbor_source_vars) {
        vals <- get(v)
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0L) {
          result[[k]]     <- NA_real_
          result[[k + 1]] <- NA_real_
          result[[k + 2]] <- NA_real_
        } else {
          result[[k]]     <- max(vals)
          result[[k + 1]] <- min(vals)
          result[[k + 2]] <- mean(vals)
        }
        k <- k + 3L
      }
      names(result) <- stat_names
      result
    },
    by = .(focal_id, year)
  ]

  # -------------------------------------------------------------------
  # STEP 4: Handle Inf/-Inf from max/min of empty groups
  #   (already handled above with the length check, but belt-and-suspenders)
  # -------------------------------------------------------------------
  for (col in stat_names) {
    vals <- stats_dt[[col]]
    vals[is.infinite(vals)] <- NA_real_
    set(stats_dt, j = col, value = vals)
  }

  # -------------------------------------------------------------------
  # STEP 5: Merge stats back into the original data
  # -------------------------------------------------------------------
  setnames(stats_dt, "focal_id", "id")
  setkey(stats_dt, id, year)
  setkey(dt, id, year)

  dt <- stats_dt[dt, on = .(id, year)]

  # -------------------------------------------------------------------
  # STEP 6: Return as data.frame (or data.table, depending on downstream)
  # -------------------------------------------------------------------
  as.data.frame(dt)
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is NOT retrained.
# Use it exactly as before for prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Even Leaner Alternative (Lower Peak Memory)

If the ~38M-row expanded edge table strains RAM, process year-by-year:

```r
compute_all_neighbor_features_lowmem <- function(cell_data,
                                                  id_order,
                                                  rook_neighbors_unique,
                                                  neighbor_source_vars) {
  library(data.table)
  dt <- as.data.table(cell_data)

  # Build spatial edge list (once)
  n_cells <- length(id_order)
  focal_idx <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  neighbor_idx <- unlist(rook_neighbors_unique)
  valid <- neighbor_idx > 0L
  edges <- data.table(
    focal_id    = id_order[focal_idx[valid]],
    neighbor_id = id_order[neighbor_idx[valid]]
  )

  keep_cols <- c("id", "year", neighbor_source_vars)
  dt_slim <- dt[, ..keep_cols]

  stat_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  years <- sort(unique(dt$year))
  results_list <- vector("list", length(years))

  for (yi in seq_along(years)) {
    yr <- years[yi]
    dt_yr <- dt_slim[year == yr]
    setnames(dt_yr, "id", "neighbor_id")

    # Join edges with this year's data
    merged <- edges[dt_yr, on = "neighbor_id", nomatch = 0L, allow.cartesian = TRUE]

    stats_yr <- merged[,
      {
        result <- vector("list", length(neighbor_source_vars) * 3L)
        k <- 1L
        for (v in neighbor_source_vars) {
          vals <- get(v)
          vals <- vals[!is.na(vals)]
          if (length(vals) == 0L) {
            result[[k]] <- NA_real_; result[[k+1]] <- NA_real_; result[[k+2]] <- NA_real_
          } else {
            result[[k]] <- max(vals); result[[k+1]] <- min(vals); result[[k+2]] <- mean(vals)
          }
          k <- k + 3L
        }
        names(result) <- stat_names
        result
      },
      by = .(focal_id)
    ]
    stats_yr[, year := yr]
    results_list[[yi]] <- stats_yr
  }

  all_stats <- rbindlist(results_list)

  # Fix infinities
  for (col in stat_names) {
    v <- all_stats[[col]]
    v[is.infinite(v)] <- NA_real_
    set(all_stats, j = col, value = v)
  }

  setnames(all_stats, "focal_id", "id")
  setkey(all_stats, id, year)
  setkey(dt, id, year)
  dt <- all_stats[dt, on = .(id, year)]

  as.data.frame(dt)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | 6.46M string-key lookups in named vector | Built once for 344K cells; expanded via `data.table` join |
| Stats computation | `lapply` over 6.46M rows Ã— 5 variables | `data.table` grouped aggregation (vectorized C internals) |
| Total passes over data | 5 (one per variable) | 1 (all variables in single grouped operation) |
| Estimated runtime | 86+ hours | **~2â€“10 minutes** |
| Numerical results | Baseline | **Identical** (same max/min/mean over same neighbor sets) |
| Random Forest model | Unchanged | **Unchanged â€” no retraining** |
| Peak RAM | Moderate (but slow) | ~2â€“4 GB (fits in 16 GB) |
END_RESPONSE>>>
