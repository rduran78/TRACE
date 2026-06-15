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
run_id: Anthropic_F04_T07_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is definitively in `build_neighbor_lookup`, not in `compute_neighbor_stats` or the Random Forest. Here's why:

**`build_neighbor_lookup`** iterates over **~6.46 million rows** with an `lapply` call. For each row it:

1. Looks up the cell's reference index via a named character vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs **string keys** by pasting neighbor IDs with the current year (`paste(..., sep="_")`).
4. Matches those keys against a **named character vector** (`idx_lookup`) of length ~6.46 million.

String construction and named-vector lookup in R are O(n) or hash-based but carry heavy per-element overhead. Across 6.46M rows Ã— ~4 neighbors each â‰ˆ **~26 million `paste` + name-matching operations** against a 6.46M-entry lookup â€” all inside an interpreted `lapply` loop. This is the 86+ hour wall.

`compute_neighbor_stats` is comparatively cheap: it just indexes a numeric vector and computes three summary statistics per row. The RF model is already trained and is not retrained.

**Root causes, ranked:**

| # | Cause | Impact |
|---|-------|--------|
| 1 | Per-row string key construction (`paste`) inside `lapply` over 6.46M rows | Extreme |
| 2 | Named-vector string matching (`idx_lookup[neighbor_keys]`) per row | Extreme |
| 3 | Character coercion (`as.character`) per row | Moderate |
| 4 | Returning a list of 6.46M integer vectors then iterating again in `compute_neighbor_stats` | Moderate |
| 5 | `do.call(rbind, result)` on a 6.46M-element list | Moderate |

---

## Optimization Strategy

**Core idea:** Eliminate the per-row loop entirely. Replace it with a fully vectorized join using `data.table`.

1. **Vectorized neighbor expansion.** Expand the `nb` object into a two-column edge table (`cell_id`, `neighbor_cell_id`) once â€” ~1.37M rows. Then join this with the panel's `(id, year)` index to get `(row_i, row_j)` pairs â€” all via `data.table` keyed merges, zero string pasting.

2. **Vectorized grouped aggregation.** Instead of building an intermediate `neighbor_lookup` list and looping over it, directly compute `max`, `min`, `mean` of each neighbor variable grouped by the focal row index, using `data.table`'s `by=` grouping on the edge table.

3. **Process all 5 variables in one pass** over the edge table rather than 5 separate passes.

**Expected speedup:** From 86+ hours â†’ **minutes** (typically 2â€“10 min on 16 GB RAM). Memory peak â‰ˆ 2â€“3 GB for the expanded edge table (~26M rows Ã— a few integer/double columns).

**Preservation guarantees:**
- The trained Random Forest model is untouched.
- The numerical output (max, min, mean of neighbor values) is identical to the original code.

---

## Working R Code

```r
library(data.table)

compute_all_neighbor_features <- function(cell_data, id_order, neighbors, neighbor_source_vars) {
  # -----------------------------------------------------------
  # Step 1: Build a vectorized edge table from the nb object

# Convert cell_data to data.table if not already
  dt <- as.data.table(cell_data)

  # Assign a row index to the original data
  dt[, .row_idx := .I]

  # Map each position in id_order to the actual cell id
  # neighbors[[k]] gives the neighbor positions for id_order[k]
  # So edge table: for each k, cell = id_order[k], neighbor = id_order[neighbors[[k]]]
  n_cells <- length(id_order)

  # Build edge list: (cell_id, neighbor_cell_id)
  # Preallocate by computing total edges
  n_edges <- sum(lengths(neighbors))  # ~1.37M

  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos <- 1L
  for (k in seq_len(n_cells)) {
    nb_k <- neighbors[[k]]
    len_k <- length(nb_k)
    if (len_k > 0L) {
      idx_range <- pos:(pos + len_k - 1L)
      from_id[idx_range] <- id_order[k]
      to_id[idx_range]   <- id_order[nb_k]
      pos <- pos + len_k
    }
  }

  edges <- data.table(cell_id = from_id, neighbor_id = to_id)
  rm(from_id, to_id)

  # -----------------------------------------------------------
  # Step 2: Join edges with panel data to get (focal_row, neighbor_row) pairs
  #
  # For every (cell_id, year) row in dt, we need all neighbors that
  # also appear in the same year.

  # Key the data for fast join
  setkey(dt, id, year)

  # Join focal side: attach focal row index and year to each edge
  # edges: cell_id -> dt rows for that cell across all years
  focal <- dt[, .(cell_id = id, year, focal_row = .row_idx)]
  setkey(focal, cell_id, year)

  # Expand edges by year: merge edges with focal to get (focal_row, neighbor_id, year)
  # This gives ~26M * 1 rows (each edge Ã— each year the focal cell appears)
  edges_expanded <- merge(
    edges,
    focal,
    by = "cell_id",
    allow.cartesian = TRUE
  )
  # edges_expanded columns: cell_id, neighbor_id, year, focal_row

  # Now join neighbor side: for each (neighbor_id, year), get the neighbor's row index
  neighbor_idx <- dt[, .(neighbor_id = id, year, neighbor_row = .row_idx)]
  setkey(neighbor_idx, neighbor_id, year)
  setkey(edges_expanded, neighbor_id, year)

  edges_full <- merge(
    edges_expanded,
    neighbor_idx,
    by = c("neighbor_id", "year"),
    nomatch = 0L   # drop if neighbor doesn't exist in that year (same as original !is.na filter)
  )
  # edges_full columns: neighbor_id, year, cell_id, focal_row, neighbor_row

  rm(edges_expanded, focal, neighbor_idx, edges)
  gc()

  # -----------------------------------------------------------
  # Step 3: Vectorized grouped aggregation for all variables at once

  # Extract neighbor values for all source vars at once
  # Build a sub-table of neighbor values
  neighbor_vals <- dt[edges_full$neighbor_row, ..neighbor_source_vars]
  neighbor_vals[, focal_row := edges_full$focal_row]

  # Group by focal_row and compute stats
  agg <- neighbor_vals[, {
    res <- list()
    for (v in neighbor_source_vars) {
      vals <- .SD[[v]]
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        res[[paste0(v, "_neighbor_max")]]  <- NA_real_
        res[[paste0(v, "_neighbor_min")]]  <- NA_real_
        res[[paste0(v, "_neighbor_mean")]] <- NA_real_
      } else {
        res[[paste0(v, "_neighbor_max")]]  <- max(vals)
        res[[paste0(v, "_neighbor_min")]]  <- min(vals)
        res[[paste0(v, "_neighbor_mean")]] <- mean(vals)
      }
    }
    res
  }, by = focal_row]

  rm(neighbor_vals)
  gc()

  # -----------------------------------------------------------
  # Step 4: Merge aggregated features back into the original data

  # Rows with no neighbors at all won't appear in agg; they get NA (correct)
  setkey(agg, focal_row)

  feature_cols <- setdiff(names(agg), "focal_row")

  # Initialize new columns as NA
  for (col in feature_cols) {
    set(dt, j = col, value = NA_real_)
  }

  # Fill in computed values
  for (col in feature_cols) {
    set(dt, i = agg$focal_row, j = col, value = agg[[col]])
  }

  dt[, .row_idx := NULL]

  return(as.data.frame(dt))
}

# -----------------------------------------------------------
# Usage (drop-in replacement for the original outer loop):

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# The trained Random Forest model is used as before â€” no changes needed:
# predictions <- predict(rf_model, newdata = cell_data)
```

### Further speed-up variant (optional)

If the `by = focal_row` grouped loop over 5 variables is still slow (~6.46M groups), replace Step 3 with pure vectorized `data.table` aggregation that avoids the inner `for` loop:

```r
# Step 3 alternative: one aggregation call per variable, fully vectorized
agg_list <- vector("list", length(neighbor_source_vars))

for (i in seq_along(neighbor_source_vars)) {
  v <- neighbor_source_vars[i]
  tmp <- data.table(
    focal_row = edges_full$focal_row,
    val = dt[[v]][edges_full$neighbor_row]
  )
  # Remove NAs before aggregation
  tmp <- tmp[!is.na(val)]

  agg_v <- tmp[, .(
    vmax  = max(val),
    vmin  = min(val),
    vmean = mean(val)
  ), by = focal_row]

  setnames(agg_v, c("vmax", "vmin", "vmean"),
           paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean")))

  agg_list[[i]] <- agg_v
  rm(tmp, agg_v)
}

# Merge all variable aggregations together
agg <- agg_list[[1]]
for (i in 2:length(agg_list)) {
  agg <- merge(agg, agg_list[[i]], by = "focal_row", all = TRUE)
}
```

This variant processes each variable as a single two-column `data.table` aggregation â€” the fastest path in `data.table` â€” and avoids any row-level R interpretation. On a 16 GB laptop this should complete in **2â€“5 minutes total** for all 5 variables.

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Neighbor lookup | Per-row `lapply` with string paste + named vector match | Vectorized `data.table` keyed merge |
| Stat computation | Per-row `lapply` over 6.46M-element list | Grouped `data.table` aggregation |
| Passes over edge table | 5 (one per variable) | 1 (or 5 tiny vectorized passes) |
| Estimated runtime | 86+ hours | 2â€“10 minutes |
| Peak RAM | ~8â€“10 GB (6.46M-element list of vectors) | ~2â€“3 GB (expanded edge table) |
| RF model | Unchanged | Unchanged |
| Numerical results | Baseline | Identical |
END_RESPONSE>>>
