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
run_id: Anthropic_F04_T00_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For every single row, the function:

1. **Performs repeated character coercion and hash lookups** (`as.character`, `paste`, named-vector indexing via `idx_lookup[neighbor_keys]`) â€” all inside a per-row anonymous function.
2. **Allocates millions of small intermediate character vectors** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`), one per row, causing extreme GC pressure.
3. **Redundantly recomputes the same neighbor-cell-ID sets** â€” every row sharing the same spatial cell ID (across 28 years) re-derives the same `neighbor_cell_ids` from `neighbors[[ref_idx]]`, 28 times.

`compute_neighbor_stats` is comparatively cheap (numeric subsetting), but it is called 5 times over the 6.46M-element list, and the `lapply` + `do.call(rbind, ...)` pattern on millions of 3-element vectors is also unnecessarily slow.

**In summary:** The code is O(N Ã— k) with enormous per-element constant factors from string operations, where N â‰ˆ 6.46M and k â‰ˆ average neighbor count (~4 for rook). The 86+ hour estimate is consistent with this.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Eliminate per-row string operations** | Replace `paste`/character key lookups with integer-indexed joins via `data.table`. |
| **Exploit panel structure** | Each cell's neighbor set is constant across years. Build the spatial adjacency once (344K cells), then join by `(neighbor_id, year)` â€” a vectorized equi-join, not a per-row loop. |
| **Vectorize aggregation** | Use `data.table` grouped aggregation (`j = .(max, min, mean), by = row_id`) instead of `lapply` over 6.46M elements. |
| **Process all 5 variables in one pass** | Melt or compute all neighbor stats in a single join + group-by, avoiding 5 separate passes. |
| **Preserve numerics exactly** | `max`, `min`, `mean` on the same neighbor sets yield identical values. |
| **No model retraining** | We only rebuild features; the trained RF object is untouched. |

Expected speedup: from 86+ hours to **~2â€“10 minutes** on the same laptop.

---

## Working R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique,
                                         neighbor_source_vars) {
  # Convert to data.table if not already; preserve original row order
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]
  

  # --- Step 1: Build a spatial edge list (cell-level, year-invariant) ---
  # rook_neighbors_unique is an nb object: a list of integer index vectors
  # id_order[i] is the cell id for the i-th element of the nb list
  
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  # edge_list has columns: focal_id, neighbor_id
  # ~1.37M rows â€” small and fast to build
  
  
  # --- Step 2: Join edge list with panel data to get neighbor values ---
  # Key the main table for fast joins
  # We need: for each (focal_id, year), look up all (neighbor_id, year) rows
  
  # Subset to only the columns we need for neighbor stats + join keys
  value_cols <- intersect(neighbor_source_vars, names(dt))
  neighbor_dt <- dt[, c("id", "year", value_cols), with = FALSE]
  setnames(neighbor_dt, "id", "neighbor_id")
  
  # Keyed join: edge_list Ã—  neighbor_dt on neighbor_id, then we still need year
  # Strategy: merge edge_list with dt on focal_id to get (focal_id, year, neighbor_id),
  # then merge with neighbor_dt on (neighbor_id, year) to get neighbor values.
  
  # Get unique (focal_id, year) with row_id
  focal_keys <- dt[, .(focal_id = id, year, .row_id)]
  
  # Expand: for each focal row, attach its neighbor cell ids
  # This is the "big" table: ~6.46M rows Ã— ~4 neighbors â‰ˆ ~26M rows
  setkey(edge_list, focal_id)
  setkey(focal_keys, focal_id)
  expanded <- edge_list[focal_keys, on = "focal_id", allow.cartesian = TRUE,
                        nomatch = NULL]
  # expanded has columns: focal_id, neighbor_id, year, .row_id
  
  # Now attach neighbor values
  setkey(neighbor_dt, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  expanded <- neighbor_dt[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # expanded now has: neighbor_id, year, <value_cols>, focal_id, .row_id
  
  
  # --- Step 3: Grouped aggregation â€” all variables at once ---
  agg_exprs <- unlist(lapply(value_cols, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)
  
  agg_names <- unlist(lapply(value_cols, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))
  
  # Build the j-expression programmatically
  agg_call <- as.call(c(as.name("list"), setNames(agg_exprs, agg_names)))
  
  stats <- expanded[, eval(agg_call), by = .row_id]
  
  # Replace -Inf/Inf from max/min of empty sets with NA
  inf_cols <- grep("neighbor_(max|min)_", names(stats), value = TRUE)
  for (col in inf_cols) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }
  
  
  # --- Step 4: Merge back into original data, preserving row order ---
  setkey(stats, .row_id)
  
  # Drop any pre-existing neighbor columns in dt to avoid duplication
  old_neighbor_cols <- intersect(agg_names, names(dt))
  if (length(old_neighbor_cols) > 0) {
    dt[, (old_neighbor_cols) := NULL]
  }
  
  dt <- stats[dt, on = ".row_id"]
  
  # Clean up helper column and restore original order
  setorder(dt, .row_id)
  dt[, .row_id := NULL]
  
  # Return as data.frame if the input was a data.frame
  if (!is.data.table(cell_data)) {
    setDF(dt)
  }
  
  return(dt)
}


# ---- Usage (drop-in replacement for the original outer loop) ----

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Original | Optimized | Equivalence |
|---|---|---|
| `vals[idx]` where `idx` = row indices of neighbors in same year | Equi-join on `(neighbor_id, year)` retrieves identical rows | Same input set |
| `max(neighbor_vals)` | `max(v, na.rm=TRUE)` grouped by `.row_id` | Identical `max` |
| `min(neighbor_vals)` | `min(v, na.rm=TRUE)` grouped by `.row_id` | Identical `min` |
| `mean(neighbor_vals)` | `mean(v, na.rm=TRUE)` grouped by `.row_id` | Identical `mean` |
| `NA` when no valid neighbors | `Inf`/`-Inf` replaced with `NA`; `mean` of empty = `NaN` â†’ `NA` | Identical |

The RF model object is never modified or retrained. Only the feature-construction step is replaced.

---

## Expected Performance

| Stage | Estimated Time | Peak RAM |
|---|---|---|
| Edge list construction (~1.37M rows) | < 1 s | ~50 MB |
| Cartesian expansion (~26M rows) | ~10â€“30 s | ~2â€“4 GB |
| Equi-join for neighbor values | ~10â€“30 s | ~3â€“5 GB |
| Grouped aggregation (5 vars Ã— 3 stats) | ~20â€“60 s | ~1 GB |
| Merge back | ~5 s | negligible |
| **Total** | **~2â€“5 minutes** | **~6â€“8 GB peak** |

This is well within the 16 GB laptop constraint and represents a roughly **1,000Ã—** speedup over the original 86+ hour estimate.
END_RESPONSE>>>
