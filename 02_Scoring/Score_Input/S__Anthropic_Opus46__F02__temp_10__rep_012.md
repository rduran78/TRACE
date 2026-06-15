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
run_id: Anthropic_F02_T10_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The two critical bottlenecks are:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting and named-vector lookups

* `paste(neighbor_cell_ids, data$year[i], sep = "_")` and indexing into `idx_lookup` (a named character vector) is called **once per row**. Named character vector lookup is O(n) internally per call because R hashes lazily and the vector has 6.46M names.
* The entire structure (`neighbor_lookup`) is a **list of 6.46 million integer vectors**, consuming enormous memory (each list element has overhead ~128 bytes + data). Estimated memory: â‰¥ 2â€“4 GB just for list overhead, plus the integer vectors themselves.

### 2. `compute_neighbor_stats` â€” also O(n) `lapply` over 6.46 million rows, called 5 times

* Each call iterates all 6.46M rows, subsetting `vals[idx]` and computing `max/min/mean`. This is pure R with no vectorization.
* `do.call(rbind, result)` on 6.46M small vectors is notoriously slow.

### Combined effect
The nested string operations, per-row R-level iteration over millions of rows, and list-of-vectors memory layout explain the 86+ hour estimate.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Eliminate string-key lookups** | Use `data.table` keyed joins (binary search, O(log n)) instead of named-vector lookup on pasted strings. |
| **Vectorize neighbor expansion** | Expand the neighbor list into a single `data.table` of `(id, neighbor_id)` pairs, then join to get `(row_index, neighbor_row_index)` in one vectorized pass â€” no per-row `lapply`. |
| **Vectorize aggregation** | Group-by aggregation (`data.table`'s `[, .(max, min, mean), by=row_idx]`) replaces 6.46M R-level `lapply` iterations. |
| **Process all 5 variables in one pass** | Compute stats for all neighbor source variables simultaneously inside a single grouped aggregation, avoiding 5 separate scans. |
| **Avoid giant intermediate lists** | The neighbor lookup becomes a two-column `data.table` (~22M rows for directed pairs Ã— 28 years) instead of a 6.46M-element list. |

**Expected speedup**: from 86+ hours to roughly **10â€“30 minutes** on the same laptop, with peak RAM well within 16 GB.

**Preservation guarantees**: The code only adds new columns (neighbor feature columns) to the existing data. The trained Random Forest model object is never touched. The numerical values produced (max, min, mean of neighbors) are identical to the original implementation.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {
  # ---------------------------------------------------------------
  # Step 1: Build a vectorized edge table from the nb object

  # ---------------------------------------------------------------
  # rook_neighbors_unique is a list (spdep nb object) indexed by
  # positional reference into id_order.
  # Expand to a data.table of (id, neighbor_id) pairs.

  n_cells <- length(id_order)
  from_ref <- rep(seq_len(n_cells),
                  times = lengths(rook_neighbors_unique))
  to_ref   <- unlist(rook_neighbors_unique, use.names = FALSE)

  edges <- data.table(
    id          = id_order[from_ref],
    neighbor_id = id_order[to_ref]
  )
  rm(from_ref, to_ref)

  # ---------------------------------------------------------------
  # Step 2: Convert cell_data to data.table and assign row indices

  # ---------------------------------------------------------------
  was_df <- is.data.frame(cell_data) && !is.data.table(cell_data)
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # ---------------------------------------------------------------
  # Step 3: Join edges with data to map each row to its neighbor rows
  #
  # For every (id, year) row, find the matching (neighbor_id, year) rows.
  # This replaces build_neighbor_lookup entirely.
  # ---------------------------------------------------------------

  # Subset columns needed for the join + aggregation
  keep_cols <- c("id", "year", "row_idx", neighbor_source_vars)
  dt_sub <- dt[, ..keep_cols]

  # Key for the "focal" side: get the year for each row
  # Join edges with focal rows to get (id, year, neighbor_id, row_idx of focal)
  focal <- dt_sub[, .(id, year, focal_row_idx = row_idx)]
  setkey(focal, id)
  setkey(edges, id)

  # Merge: for each focal (id, year) attach all neighbor_ids
  # Result: (id, year, focal_row_idx, neighbor_id)
  edge_year <- edges[focal, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  rm(focal)

  # Now join to get the neighbor's variable values in that same year
  # Prepare neighbor data keyed on (neighbor_id aliased as id, year)
  neighbor_dt <- copy(dt_sub)
  setnames(neighbor_dt, "id", "neighbor_id")
  setnames(neighbor_dt, "row_idx", "nb_row_idx")
  setkey(neighbor_dt, neighbor_id, year)
  setkey(edge_year, neighbor_id, year)

  # This is the big join: each focal row_idx paired with its neighbor's values

  joined <- neighbor_dt[edge_year,
                        on = .(neighbor_id, year),
                        allow.cartesian = TRUE,
                        nomatch = 0L]
  rm(edge_year, neighbor_dt, dt_sub)

  # ---------------------------------------------------------------
  # Step 4: Vectorized grouped aggregation for all variables at once
  # ---------------------------------------------------------------
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("nb_max_", v)]]  <-
      bquote(as.numeric(max(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("nb_min_", v)]]  <-
      bquote(as.numeric(min(.(v_sym), na.rm = TRUE)))
    agg_exprs[[paste0("nb_mean_", v)]] <-
      bquote(mean(.(v_sym), na.rm = TRUE))
  }

  # Evaluate all aggregations in one grouped pass
  stats <- joined[, lapply(agg_exprs, eval),
                  by = .(focal_row_idx)]
  rm(joined)

  # Replace -Inf / Inf from max/min of all-NA groups with NA
  for (col_name in names(stats)[-1L]) {
    set(stats, which(is.infinite(stats[[col_name]])), col_name, NA_real_)
  }

  # ---------------------------------------------------------------
  # Step 5: Merge aggregated stats back to the main table
  # ---------------------------------------------------------------
  setkey(stats, focal_row_idx)
  setkey(dt, row_idx)

  # Initialize new columns with NA (handles rows with zero neighbors)
  new_cols <- setdiff(names(stats), "focal_row_idx")
  for (col_name in new_cols) {
    set(dt, j = col_name, value = NA_real_)
  }

  # Update join: only rows present in stats get values
  dt[stats, on = .(row_idx = focal_row_idx),
     (new_cols) := mget(paste0("i.", new_cols))]

  # Clean up helper column
  dt[, row_idx := NULL]

  # ---------------------------------------------------------------
  # Step 6: Return in original format
  # ---------------------------------------------------------------
  if (was_df) {
    return(as.data.frame(dt))
  }
  return(dt[])
}

# ===================================================================
# Usage â€” drop-in replacement for the original outer loop
# ===================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged and can be used directly
# for prediction on the updated cell_data with the new neighbor columns.
```

---

## Memory Management Notes

If RAM is still tight during the large `cartesian` join (the `joined` table can reach ~380 M rows Ã— few columns â‰ˆ 6â€“8 GB), process in **year-chunks**:

```r
# Optional chunked variant â€” replaces Step 3 + 4 above
years <- sort(unique(dt$year))
stats_list <- vector("list", length(years))

for (yi in seq_along(years)) {
  y <- years[yi]
  dt_y <- dt_sub[year == y]
  setkey(dt_y, id)

  focal_y <- dt_y[, .(id, year, focal_row_idx = row_idx)]
  ey <- edges[focal_y, on = "id", allow.cartesian = TRUE, nomatch = 0L]

  nb_y <- copy(dt_y)
  setnames(nb_y, c("id", "row_idx"), c("neighbor_id", "nb_row_idx"))
  setkey(nb_y, neighbor_id)
  setkey(ey, neighbor_id)

  j_y <- nb_y[ey, on = "neighbor_id", allow.cartesian = TRUE, nomatch = 0L]

  stats_list[[yi]] <- j_y[, lapply(agg_exprs, eval), by = .(focal_row_idx)]
  rm(dt_y, focal_y, ey, nb_y, j_y)
}

stats <- rbindlist(stats_list)
rm(stats_list)
```

This keeps peak memory for the join at roughly **1/28th** of the full version (~250 MB per year-chunk), comfortably within 16 GB.

---

### Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M-element R list via `lapply` + string paste | Single vectorized `data.table` keyed join |
| Aggregation | 5 Ã— `lapply` over 6.46M rows | One grouped `data.table` aggregation |
| Estimated time | 86+ hours | ~10â€“30 minutes |
| Peak RAM | Uncontrolled (list overhead) | ~6â€“8 GB (full) or ~2â€“3 GB (chunked) |
| Trained RF model | Preserved | Preserved |
| Numerical output | Identical (max, min, mean) | Identical |
END_RESPONSE>>>
