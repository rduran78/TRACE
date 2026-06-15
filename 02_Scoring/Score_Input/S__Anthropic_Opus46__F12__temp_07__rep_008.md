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
run_id: Anthropic_F12_T07_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The colleague's claim is **partially correct but misses the deeper bottleneck**. Let me audit both functions:

### `compute_neighbor_stats()` â€” Colleague's Claimed Bottleneck
- `do.call(rbind, result)`: Binding ~6.46M small 3-element vectors into a matrix is not cheap, but `do.call(rbind, ...)` on a list of uniform-length numeric vectors is actually reasonably efficient in R. This is a **minor** bottleneck.
- There is no "repeated list binding" inside `compute_neighbor_stats()` â€” it uses `lapply` and returns fixed-length vectors. The colleague's description of the inner workings is factually wrong. This function runs 5 times (once per variable), each time iterating over 6.46M entries with simple numeric subsetting. It's not trivial, but it's **not the dominant cost**.

### `build_neighbor_lookup()` â€” The True Deep Bottleneck
This function is called once but contains the **critical bottleneck**:

1. **`paste()` to build string keys and `idx_lookup` named vector**: It creates a named vector of length ~6.46M using `paste(data$id, data$year, sep = "_")`. Named vector lookup in R uses linear hashing that degrades at scale.

2. **Per-row `lapply` over 6.46M rows**: Inside the lambda:
   - `as.character(data$id[i])` â€” called 6.46M times.
   - `id_to_ref[as.character(...)]` â€” named vector lookup, 6.46M times.
   - `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” string construction for every neighbor of every row, totaling ~1.37M neighbor pairs Ã— 28 years â‰ˆ **38.4M string constructions** (and likely more, since each row's neighbors are looked up individually).
   - `idx_lookup[neighbor_keys]` â€” named vector lookup of ~38.4M keys against a 6.46M-element named vector.

The total work in `build_neighbor_lookup` is **O(N Ã— avg_neighbors)** string concatenations and named-vector lookups, where N = 6.46M and avg_neighbors â‰ˆ 4 (rook neighbors). That's ~25.8M string-match lookups against a 6.46M-entry named vector â€” **this is the 86-hour bottleneck**.

**String-keyed lookup in a named vector is O(n) per miss in the worst case in R's internal hashing, and even at O(1) amortized, the constant factor of `paste` + character matching at this scale is enormous.**

### Verdict: **Reject the colleague's diagnosis.** The dominant bottleneck is `build_neighbor_lookup()`, specifically the row-by-row `lapply` with repeated string construction and named-vector lookups over 6.46M rows.

---

## Optimization Strategy

1. **Eliminate all string-key lookups.** Replace `paste`-based keys with integer arithmetic: encode `(id, year)` as a single integer or use `data.table` keyed joins.

2. **Vectorize `build_neighbor_lookup` entirely.** Instead of iterating row-by-row, expand the neighbor relationships into a full edge table (cell_i, cell_j), join on year to get (row_i, row_j) pairs, then compute grouped statistics using `data.table`.

3. **Vectorize `compute_neighbor_stats`.** Instead of `lapply` over 6.46M elements, use `data.table` grouped aggregation on the edge table.

4. **Preserve the trained Random Forest model** â€” we only change feature engineering, producing identical numerical columns.

### Complexity Reduction
- Old: ~25.8M string constructions + named-vector lookups in an R-level loop = **hours**.
- New: One `data.table` merge + grouped aggregation = **minutes**.

---

## Working R Code

```r
library(data.table)

#' Optimized pipeline: replaces build_neighbor_lookup + compute_neighbor_stats
#' Produces identical numerical output (max, min, mean of neighbor values).
#'
#' @param cell_data        data.frame with columns: id, year, and all var columns
#' @param id_order         integer vector of cell IDs in the order matching the nb object
#' @param rook_neighbors   spdep::nb object (list of integer index vectors)
#' @param neighbor_source_vars character vector of variable names
#' @return cell_data with new neighbor feature columns appended

compute_all_neighbor_features_optimized <- function(cell_data,
                                                     id_order,
                                                     rook_neighbors,
                                                     neighbor_source_vars) {

  # ---- Step 1: Build directed edge list (in terms of cell IDs) ----
  # rook_neighbors[[i]] contains indices into id_order for the neighbors of id_order[i]
  from_idx <- rep(seq_along(rook_neighbors), lengths(rook_neighbors))
  to_idx   <- unlist(rook_neighbors)

  # Convert from nb indices to actual cell IDs
  edge_dt <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )

  # ---- Step 2: Convert cell_data to data.table and key it ----
  dt <- as.data.table(cell_data)

  # Create a row index to preserve original order
  dt[, .row_idx := .I]

  # We need to know which years exist
  years <- sort(unique(dt$year))

  # ---- Step 3: Cross edges with years to get (from_id, year) -> (to_id, year) ----
  # For each edge (from_id, to_id), the neighbor relationship holds across ALL years.
  # So we expand: for each year, (from_id, year) has neighbor row (to_id, year).

  # Build lookup: (id, year) -> row_idx
  # Use integer key for speed
  setkey(dt, id, year)

  # Expand edges Ã— years using a merge approach:
  # "from" rows: merge edge_dt with dt on from_id = id to get (from_row, to_id, year)
  # Then merge on (to_id, year) to get to_row.

  # Step 3a: Get from-side row indices
  # For each edge, we need every year that from_id appears in.
  from_rows <- dt[, .(from_id = id, year, from_row = .row_idx)]
  setkey(from_rows, from_id)
  setkey(edge_dt, from_id)

  # Merge: for each edge, get all (from_row, year, to_id) combinations
  edge_year <- edge_dt[from_rows, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # Columns: from_id, to_id, year, from_row

  # Step 3b: Get to-side row indices
  to_rows <- dt[, .(to_id = id, year, to_row = .row_idx)]
  setkey(edge_year, to_id, year)
  setkey(to_rows, to_id, year)

  edge_full <- edge_year[to_rows, on = c("to_id", "year"), nomatch = 0L]
  # Columns: from_id, to_id, year, from_row, to_row

  # Free intermediate objects
  rm(from_rows, to_rows, edge_year)
  gc()

  # ---- Step 4: For each variable, compute grouped stats ----
  for (var_name in neighbor_source_vars) {

    # Extract neighbor values via to_row indexing (vectorized)
    edge_full[, neighbor_val := dt[[var_name]][to_row]]

    # Compute stats grouped by from_row, excluding NAs
    stats <- edge_full[!is.na(neighbor_val),
                       .(nmax  = max(neighbor_val),
                         nmin  = min(neighbor_val),
                         nmean = mean(neighbor_val)),
                       by = from_row]

    # Initialize new columns with NA
    max_col  <- paste0("n_max_", var_name)
    min_col  <- paste0("n_min_", var_name)
    mean_col <- paste0("n_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign computed values back
    dt[stats$from_row, (max_col)  := stats$nmax]
    dt[stats$from_row, (min_col)  := stats$nmin]
    dt[stats$from_row, (mean_col) := stats$nmean]

    # Clean up
    edge_full[, neighbor_val := NULL]
    rm(stats)
  }

  # ---- Step 5: Restore original order and return as data.frame ----
  setorder(dt, .row_idx)
  dt[, .row_idx := NULL]

  as.data.frame(dt)
}

# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features_optimized(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Predictions proceed as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Works and Performance Estimate

| Aspect | Original | Optimized |
|---|---|---|
| Lookup strategy | 6.46M `paste` + named-vector lookups in R loop | `data.table` keyed binary-search join |
| Neighbor expansion | Per-row `lapply` with string matching | Single vectorized merge (~38.4M rows) |
| Stat computation | `lapply` over 6.46M elements + `do.call(rbind)` | `data.table` grouped `by` aggregation (C-level) |
| Estimated time | **86+ hours** | **~5â€“15 minutes** on same hardware |
| Memory | Moderate (many small allocations) | ~2â€“4 GB peak for edge table (fits in 16 GB) |
| Numerical output | max, min, mean per row | **Identical** max, min, mean per row |
| RF model | Unchanged | **Unchanged** |

### Memory Note
The expanded edge table has ~38.4M rows Ã— 5 columns of integers â‰ˆ ~1.5 GB. With the `neighbor_val` double column added temporarily, peak usage is ~2â€“3 GB for the edge table plus the original data (~6.46M Ã— 110 columns). This fits comfortably within 16 GB RAM.

**Summary:** The real bottleneck is `build_neighbor_lookup()` â€” specifically, 6.46 million iterations of string-key construction and named-vector lookups in an R-level loop. The fix is to replace all string-keyed row-level iteration with vectorized `data.table` joins and grouped aggregation, reducing runtime from 86+ hours to minutes while producing numerically identical features.
END_RESPONSE>>>
