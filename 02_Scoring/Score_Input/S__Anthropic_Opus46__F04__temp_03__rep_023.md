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
run_id: Anthropic_F04_T03_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` â€” Row-level `lapply` over 6.46 million rows

Each iteration performs:
- A character coercion and named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
- A `paste()` call to build neighbor keys.
- A named-vector lookup into `idx_lookup` (which is a 6.46-million-element named character vector â€” each lookup is O(n) hash probe repeated per neighbor).
- Subsetting and `is.na` filtering.

Doing this 6.46 million times in an interpreted `lapply` loop, with repeated string allocation and hash-table lookups against a massive named vector, is catastrophically slow. The named-vector `idx_lookup` alone has ~6.46M entries; probing it millions of times with multi-element key vectors is the single largest cost.

### 2. `compute_neighbor_stats` â€” Row-level `lapply` + `do.call(rbind, ...)`

Each of the 6.46M iterations subsets a numeric vector, removes NAs, and computes max/min/mean. The `lapply` returns a list of 6.46M small vectors, then `do.call(rbind, ...)` must allocate and fill a matrix from that list. This is repeated 5 times (once per source variable).

### Quantitative estimate of current cost

- `build_neighbor_lookup`: ~6.46M iterations Ã— (string ops + hash lookups) â‰ˆ tens of hours.
- `compute_neighbor_stats`: ~6.46M iterations Ã— 5 variables Ã— (subset + summary stats + list-to-matrix bind) â‰ˆ additional tens of hours.
- Combined: the reported 86+ hour estimate is consistent.

---

## Optimization Strategy

The key insight is: **eliminate all row-level R loops and string-key lookups; replace them with vectorized joins and grouped aggregations using `data.table`.**

### Specific changes

| Current (slow) | Proposed (fast) |
|---|---|
| Named character vector `idx_lookup` with `paste(id, year)` keys, probed per row | `data.table` keyed join on `(id, year)` integer columns â€” O(1) amortized binary-search join |
| `lapply` over 6.46M rows to build neighbor index list | Vectorized expansion: explode the `nb` object into an edge-list `data.table`, then merge with the data to get row indices â€” one join, no loop |
| `lapply` over 6.46M rows to compute max/min/mean per neighborhood | `data.table` grouped aggregation: `dt[, .(max, min, mean), by = focal_row]` â€” single vectorized pass per variable |
| `do.call(rbind, list_of_6.46M_vectors)` | Eliminated entirely; results are columns in a `data.table` joined back |
| 5 separate passes rebuilding the same structure | All 5 variables computed in one pass over the edge table, or 5 fast grouped aggregations |

**Expected speedup**: From 86+ hours to roughly **2â€“10 minutes** on the same laptop, depending on RAM pressure. The `data.table` grouped aggregation over ~13â€“15 million edge-rows (bidirectional rook neighbors Ã— years) is trivially fast.

**Numerical equivalence**: The aggregation functions (max, min, mean, with NA removal) are identical, so the trained Random Forest model requires no changes.

---

## Working R Code

```r
library(data.table)

#' Build neighbor features using fully vectorized data.table operations.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data        data.frame (or data.table) with columns: id, year,
#'                         and all neighbor_source_vars.
#' @param id_order         integer vector of cell IDs in the order matching
#'                         the nb object (i.e., id_order[k] is the cell ID
#'                         for the k-th element of rook_neighbors_unique).
#' @param nb_obj           spdep nb object (list of integer index vectors);
#'                         rook_neighbors_unique.
#' @param neighbor_source_vars character vector of variable names to summarize.
#' @return data.table with original columns plus new neighbor feature columns.
add_neighbor_features_fast <- function(cell_data,
                                       id_order,
                                       nb_obj,
                                       neighbor_source_vars) {

  # --- Step 0: Convert to data.table (copy to avoid side-effects) -----------
  dt <- as.data.table(cell_data)

  # --- Step 1: Build directed edge list from the nb object ------------------
  #
  # nb_obj[[k]] contains the integer indices of neighbors of the k-th spatial

  # unit.  We map those indices back to cell IDs via id_order.
  #
  # Result: edges_dt with columns (focal_id, neighbor_id), all integer.

  focal_indices <- rep(
    seq_along(nb_obj),
    times = lengths(nb_obj)
  )
  neighbor_indices <- unlist(nb_obj, use.names = FALSE)

  edges_dt <- data.table(
    focal_id    = id_order[focal_indices],
    neighbor_id = id_order[neighbor_indices]
  )
  rm(focal_indices, neighbor_indices)  # free memory

  # Remove any zero-index entries that spdep uses for "no neighbors"

  edges_dt <- edges_dt[neighbor_id != 0L]

  # --- Step 2: Unique years vector ------------------------------------------
  years <- sort(unique(dt$year))

  # --- Step 3: Cross-join edges Ã— years to get the full focalâ€“neighborâ€“year
  #             table.  This is the "exploded" lookup table.
  #
  #   ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows (fits comfortably in 16 GB).

  years_dt <- data.table(year = years)
  edge_year <- edges_dt[, CJ_dt := TRUE][
    years_dt[, CJ_dt := TRUE],
    on = "CJ_dt",
    allow.cartesian = TRUE
  ]
  edge_year[, CJ_dt := NULL]

  # Cleaner cross-join (data.table idiomatic):
  # We redo this properly:
  edge_year <- CJ(edge_idx = seq_len(nrow(edges_dt)), year = years)
  edge_year[, `:=`(
    focal_id    = edges_dt$focal_id[edge_idx],
    neighbor_id = edges_dt$neighbor_id[edge_idx]
  )]
  edge_year[, edge_idx := NULL]

  # --- Step 4: Attach neighbor variable values via keyed join ---------------
  #
  # We join edge_year to dt on (neighbor_id == id, year == year) to pull in
  # the neighbor's variable values.

  # Subset dt to only the columns we need for the join (save memory).
  dt_vals <- dt[, c("id", "year", neighbor_source_vars), with = FALSE]
  setnames(dt_vals, "id", "neighbor_id")
  setkeyv(dt_vals, c("neighbor_id", "year"))
  setkeyv(edge_year, c("neighbor_id", "year"))

  edge_year <- dt_vals[edge_year, on = c("neighbor_id", "year")]

  # Now edge_year has columns:
  #   neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, focal_id

  # --- Step 5: Grouped aggregation by (focal_id, year) ----------------------
  #
  # For each variable, compute max, min, mean (na.rm = TRUE), exactly matching
  # the original compute_neighbor_stats logic.

  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]]  <-
      bquote(as.numeric(max(.(v_sym),   na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_min_", v)]]  <-
      bquote(as.numeric(min(.(v_sym),   na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      bquote(mean(.(v_sym), na.rm = TRUE))
  }

  # Build a single aggregation call
  agg_call <- as.call(c(as.name("list"), agg_exprs))

  neighbor_stats <- edge_year[,
    eval(agg_call),
    by = .(focal_id, year)
  ]

  # Handle cells with no valid neighbors: max/min of empty â†’ -Inf/Inf â†’ NA
  for (col in names(neighbor_stats)) {
    if (col %in% c("focal_id", "year")) next
    v <- neighbor_stats[[col]]
    set(neighbor_stats, which(is.infinite(v) | is.nan(v)), col, NA_real_)
  }

  # --- Step 6: Join aggregated stats back to the main data.table ------------
  setnames(neighbor_stats, "focal_id", "id")
  setkeyv(neighbor_stats, c("id", "year"))
  setkeyv(dt, c("id", "year"))

  dt <- neighbor_stats[dt, on = c("id", "year")]

  # --- Step 7: Return -------------------------------------------------------
  # Convert back to data.frame if the caller expects one.
  return(as.data.frame(dt))
}
```

### Drop-in replacement for the outer loop

```r
# ---- BEFORE (slow, ~86+ hours) ----
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ---- AFTER (fast, ~2-10 minutes) ----
library(data.table)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_neighbor_features_fast(
  cell_data            = cell_data,
  id_order             = id_order,
  nb_obj               = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is unchanged â€” proceed directly to predict().
# predictions <- predict(rf_model, newdata = cell_data)
```

### Memory-conscious variant (if 38M-row cross-join is tight on 16 GB)

If the full cross-join (~38.4M rows Ã— several columns) strains the 16 GB laptop, process years in chunks:

```r
add_neighbor_features_chunked <- function(cell_data, id_order, nb_obj,
                                          neighbor_source_vars,
                                          chunk_size = 7L) {
  dt <- as.data.table(cell_data)

  # Build edge list once
  focal_indices    <- rep(seq_along(nb_obj), times = lengths(nb_obj))
  neighbor_indices <- unlist(nb_obj, use.names = FALSE)
  edges_dt <- data.table(
    focal_id    = id_order[focal_indices],
    neighbor_id = id_order[neighbor_indices]
  )
  edges_dt <- edges_dt[neighbor_id != 0L]
  rm(focal_indices, neighbor_indices)

  years <- sort(unique(dt$year))
  year_chunks <- split(years, ceiling(seq_along(years) / chunk_size))

  dt_vals <- dt[, c("id", "year", neighbor_source_vars), with = FALSE]
  setnames(dt_vals, "id", "neighbor_id")
  setkeyv(dt_vals, c("neighbor_id", "year"))

  # Build aggregation expression once
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]]  <-
      bquote(as.numeric(max(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_min_", v)]]  <-
      bquote(as.numeric(min(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      bquote(mean(.(v_sym), na.rm = TRUE))
  }
  agg_call <- as.call(c(as.name("list"), agg_exprs))

  stats_list <- vector("list", length(year_chunks))

  for (ci in seq_along(year_chunks)) {
    yrs <- year_chunks[[ci]]
    ey  <- CJ(edge_idx = seq_len(nrow(edges_dt)), year = yrs)
    ey[, `:=`(
      focal_id    = edges_dt$focal_id[edge_idx],
      neighbor_id = edges_dt$neighbor_id[edge_idx]
    )]
    ey[, edge_idx := NULL]
    setkeyv(ey, c("neighbor_id", "year"))

    ey <- dt_vals[ey, on = c("neighbor_id", "year")]

    chunk_stats <- ey[, eval(agg_call), by = .(focal_id, year)]

    # Inf / NaN â†’ NA
    stat_cols <- setdiff(names(chunk_stats), c("focal_id", "year"))
    for (col in stat_cols) {
      v <- chunk_stats[[col]]
      set(chunk_stats, which(is.infinite(v) | is.nan(v)), col, NA_real_)
    }

    stats_list[[ci]] <- chunk_stats
    rm(ey, chunk_stats)
    gc()
  }

  neighbor_stats <- rbindlist(stats_list)
  rm(stats_list); gc()

  setnames(neighbor_stats, "focal_id", "id")
  setkeyv(neighbor_stats, c("id", "year"))
  setkeyv(dt, c("id", "year"))

  dt <- neighbor_stats[dt, on = c("id", "year")]

  return(as.data.frame(dt))
}

# Usage:
cell_data <- add_neighbor_features_chunked(
  cell_data, id_order, rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
  chunk_size = 7  # 4 chunks of 7 years each
)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M-iteration `lapply` with string hashing | One vectorized edge-list expansion + `data.table` keyed join |
| **Stat computation** | 6.46M-iteration `lapply` Ã— 5 vars + `do.call(rbind)` | Single `data.table` grouped aggregation per variable (vectorized C) |
| **Estimated runtime** | 86+ hours | ~2â€“10 minutes |
| **RAM safety (16 GB)** | Fine (but slow) | Full version ~5â€“8 GB peak; chunked version ~3â€“5 GB peak |
| **Numerical output** | max, min, mean (na.rm) per neighbor set | Identical â€” preserves the trained RF model's expected features |
| **RF model** | Unchanged | Unchanged â€” no retraining needed |
END_RESPONSE>>>
