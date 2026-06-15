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
run_id: Anthropic_F04_T03_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and specifically:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string operations

For every one of the ~6.46 million rows, the function:
- Converts an integer ID to character and looks it up in a named vector (`id_to_ref`).
- Retrieves neighbor cell IDs from the `nb` object.
- Calls `paste()` to construct `cell_year` key strings for every neighbor of every row.
- Performs named-vector lookup (`idx_lookup[neighbor_keys]`) â€” which in R is an **O(k)** hash lookup per key, but the overhead of constructing millions of character key vectors and performing named-vector subsetting in an interpreted `lapply` loop is enormous.

With ~6.46M rows and an average of ~4 rook neighbors per cell, this loop performs roughly **25.8 million** string paste + hash-lookup operations inside an R-level loop. The result is a list of 6.46M integer vectors â€” itself a large, fragmented memory structure.

### 2. `compute_neighbor_stats` â€” O(n) `lapply` over 6.46 million rows (Ã—5 variables)

For each of the 5 source variables, another `lapply` iterates over all 6.46M rows, subsets a numeric vector by the index list, removes NAs, and computes `max`, `min`, `mean`. That is **32.3 million** R-level function calls total. The final `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is also very slow.

### Summary of root causes

| Cause | Impact |
|---|---|
| Row-level `lapply` over 6.46M rows (interpreted R loop) | Dominant wall-clock cost |
| Per-row `paste()` + named-vector character lookup in `build_neighbor_lookup` | Millions of transient string allocations |
| `do.call(rbind, ...)` on a 6.46M-element list | Slow list-to-matrix coercion |
| Repeated per-variable `lapply` (Ã—5) in `compute_neighbor_stats` | Multiplies the loop cost |

---

## Optimization Strategy

**Replace all row-level R loops with vectorized `data.table` grouped joins and aggregations.**

The key insight: the neighbor lookup and the neighbor statistics can both be expressed as **equi-joins** followed by **grouped aggregations** â€” operations that `data.table` executes in optimized C.

### Steps

1. **Build an edge table** (once): Expand the `nb` object into a two-column `data.table` of `(cell_id, neighbor_cell_id)` â€” ~1.37M rows.
2. **Join with the panel**: Inner-join the edge table with the panel data on `(neighbor_cell_id, year)` to get, for every `(cell_id, year)`, the variable values of all its neighbors. This is a single keyed join â€” no string pasting, no row-level loop.
3. **Grouped aggregation**: Group by `(cell_id, year)` and compute `max`, `min`, `mean` for all 5 variables simultaneously in one pass.
4. **Left-join back** to the original data to attach the 15 new columns.

This eliminates every `lapply`, every `paste`, every named-vector lookup, and every `do.call(rbind, ...)`.

**Expected speedup**: From ~86+ hours to **minutes** (typically 2â€“10 minutes on a 16 GB laptop).

**Numerical equivalence**: The aggregation functions (`max`, `min`, `mean` after dropping NAs) are identical, so the trained Random Forest model's predictions are unchanged.

---

## Working R Code

```r
library(data.table)

#' Optimized spatial neighbor feature construction.
#'
#' @param cell_data       data.frame (or data.table) with columns: id, year,
#'                        and all columns named in neighbor_source_vars.
#' @param id_order        integer vector â€” the cell IDs in the order that
#'                        corresponds to the index positions in the nb object.
#' @param rook_neighbors_unique  spdep nb object (list of integer index vectors).
#' @param neighbor_source_vars   character vector of variable names to aggregate.
#'
#' @return data.table equal to the input with 3 new columns per source variable
#'         appended: <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean.
build_neighbor_features_fast <- function(cell_data,
                                         id_order,
                                         rook_neighbors_unique,
                                         neighbor_source_vars) {

  # ---- 0. Convert to data.table (by reference if already one) ---------------
  dt <- as.data.table(cell_data)

  # ---- 1. Build the directed edge list from the nb object -------------------
  #
  # rook_neighbors_unique[[i]] contains the *index positions* (into id_order)

  # of the neighbors of the cell whose ID is id_order[i].
  # An nb entry of integer(0) (or the sentinel 0L used by spdep) means no

  # neighbors.

  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    # spdep uses 0L as a sentinel for "no neighbors"
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(cell_id     = id_order[i],
               neighbor_id = id_order[nb_idx])
  }))
  # edges is ~1.37 M rows â€” small and fast to build.

  # ---- 2. Key the panel for fast join ---------------------------------------
  #
  # We need to look up neighbor variable values by (neighbor_id, year).
  # Select only the columns we need to keep the join memory-lean.

  cols_needed <- unique(c("id", "year", neighbor_source_vars))
  neighbor_dt <- dt[, ..cols_needed]
  setnames(neighbor_dt, "id", "neighbor_id")
  setkey(neighbor_dt, neighbor_id, year)

  # ---- 3. Expand edges Ã— years via keyed join ------------------------------
  #
  # For every (cell_id -> neighbor_id) edge, pull in the neighbor's variable
  # values for the matching year.  We achieve this by first attaching the
  # focal cell's year to the edge table, then joining.

  # Get the unique (id, year) pairs from the panel.
  focal_keys <- unique(dt[, .(cell_id = id, year)])

  # Merge focal_keys with edges to get (cell_id, year, neighbor_id).
  # This is an equi-join on cell_id.
  setkey(edges, cell_id)
  setkey(focal_keys, cell_id)
  expanded <- edges[focal_keys, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: cell_id, neighbor_id, year
  # Rows â‰ˆ 6.46M Ã— avg_neighbors â‰ˆ 25.8M  (fits in RAM at ~0.6 GB for 3 int cols)

  # ---- 4. Join to get neighbor variable values ------------------------------
  setkey(expanded, neighbor_id, year)
  expanded <- neighbor_dt[expanded, on = .(neighbor_id, year), nomatch = NA]
  # Now expanded has: neighbor_id, year, <source_vars>, cell_id

  # ---- 5. Grouped aggregation -----------------------------------------------
  #
  # For each (cell_id, year), compute max / min / mean of each source variable
  # across all neighbors, dropping NAs as the original code does.

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

  # Build the j-expression programmatically.
  # Using a simpler, robust approach:
  agg_fun <- function(sub_dt) {
    out <- list()
    for (v in neighbor_source_vars) {
      vals <- sub_dt[[v]]
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[paste0(v, "_neighbor_max")]]  <- NA_real_
        out[[paste0(v, "_neighbor_min")]]  <- NA_real_
        out[[paste0(v, "_neighbor_mean")]] <- NA_real_
      } else {
        out[[paste0(v, "_neighbor_max")]]  <- max(vals)
        out[[paste0(v, "_neighbor_min")]]  <- min(vals)
        out[[paste0(v, "_neighbor_mean")]] <- mean(vals)
      }
    }
    as.list(out)
  }

  # More efficient: use data.table's native grouped aggregation via .SDcols
  stats <- expanded[,
    {
      out <- vector("list", length(neighbor_source_vars) * 3L)
      k <- 0L
      for (v in neighbor_source_vars) {
        vals <- get(v)
        vals <- vals[!is.na(vals)]
        n <- length(vals)
        k <- k + 1L; out[[k]] <- if (n == 0L) NA_real_ else max(vals)
        k <- k + 1L; out[[k]] <- if (n == 0L) NA_real_ else min(vals)
        k <- k + 1L; out[[k]] <- if (n == 0L) NA_real_ else mean(vals)
      }
      names(out) <- agg_names
      out
    },
    by = .(cell_id, year)
  ]

  # ---- 6. Handle cells with no neighbors (they won't appear in stats) -------
  #
  # These cells should get NA for all neighbor features, which is the default
  # after a left join.

  # ---- 7. Join aggregated stats back to the original panel ------------------
  setkey(stats, cell_id, year)
  setkey(dt, id, year)
  dt <- stats[dt, on = .(cell_id = id, year = year)]

  # Restore the original column name for the cell identifier.
  setnames(dt, "cell_id", "id")

  # ---- 8. Replace -Inf / Inf from max/min of empty sets (safety) ------------
  for (nm in agg_names) {
    set(dt, which(is.infinite(dt[[nm]])), nm, NA_real_)
  }

  return(dt[])
}


# ============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Proceed directly to predict():
# preds <- predict(rf_model, newdata = cell_data)
```

---

## Why This Works and What Changed

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M-iteration `lapply` with `paste` + named-vector hash | Single `data.table` keyed join (~1.37M edges Ã— 28 years) |
| Neighbor stats | 5 Ã— 6.46M-iteration `lapply` with per-row subsetting | One grouped `data.table` aggregation over ~25.8M rows |
| Result assembly | `do.call(rbind, <6.46M-element list>)` | `data.table` join (column bind by reference) |
| String allocations | ~25.8M transient `paste` results | Zero |
| Estimated time | 86+ hours | ~2â€“10 minutes |
| Peak RAM | Fragmented 6.46M-element list of integer vectors | ~2â€“3 GB contiguous `data.table` columns |
| Numerical result | max / min / mean (NA-removed) | Identical max / min / mean (NA-removed) |
| Trained RF model | Unchanged | Unchanged |

The fundamental transformation is: **loop over rows â†’ join + group-by**, which moves all iteration from interpreted R into `data.table`'s C backend.
END_RESPONSE>>>
