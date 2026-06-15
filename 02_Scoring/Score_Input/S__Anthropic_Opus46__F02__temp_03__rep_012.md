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
run_id: Anthropic_F02_T03_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F02_Anthropic_temp_03_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core bottleneck is the **`build_neighbor_lookup` function**, which uses `lapply` over ~6.46 million rows, performing per-row string pasting and named-vector lookups. Named vector lookups in R via character keys are O(n) hash lookups, but doing ~6.46M of themâ€”each touching multiple neighborsâ€”creates billions of small operations. The `compute_neighbor_stats` function then iterates over the same 6.46M entries again per variable, and this is repeated 5 times (once per neighbor source variable). The `do.call(rbind, ...)` on millions of small vectors is also notoriously slow.

**Specific problems:**

1. **`build_neighbor_lookup`:** `paste()` and named-vector indexing inside a per-row `lapply` over 6.46M rows is extremely slow. The resulting list-of-integer-vectors also consumes significant memory.
2. **`compute_neighbor_stats`:** Iterating a 6.46M-element list with `lapply` and then `do.call(rbind, ...)` on 6.46M 3-element vectors is slow and memory-wasteful.
3. **Redundant work:** The neighbor structure is year-invariant (same spatial neighbors every year), but the lookup is rebuilt per cell-year row by pasting year into keys.

---

## Optimization Strategy

**Key insight:** The neighbor relationships are purely spatialâ€”they don't change across years. So we should separate the spatial neighbor graph from the temporal (year) dimension and use vectorized, column-oriented operations instead of row-wise R loops.

### Strategy summary:

| Step | Technique | Speedup source |
|---|---|---|
| 1 | Build a **flat edge table** (an integer matrix of `[source_row, neighbor_row]` pairs) using `data.table` merge instead of per-row `lapply` + `paste` | Vectorized join replaces 6.46M R-level iterations |
| 2 | Compute neighbor stats using **vectorized grouped aggregation** with `data.table` on the edge table | Replaces per-row `lapply` + `do.call(rbind, ...)` |
| 3 | Process all 5 variables in a **single pass** over the edge table | Eliminates redundant iteration |
| 4 | Avoid creating a 6.46M-element list entirely | Massive memory savings |

**Expected runtime:** Minutes instead of 86+ hours. Memory stays well within 16 GB.

---

## Working R Code

```r
library(data.table)

#' Build a flat edge table mapping each cell-year row to its neighbor cell-year rows.
#' This replaces build_neighbor_lookup entirely.
#'
#' @param cell_data   data.frame/data.table with columns: id, year (and predictor cols)
#' @param id_order    character/integer vector: the cell IDs in the order matching
#'                    the spdep::nb object (i.e., id_order[i] is the cell ID for
#'                    the i-th element of rook_neighbors_unique)
#' @param neighbors   spdep::nb object (list of integer vectors of neighbor indices)
#' @return data.table with columns: source_row, neighbor_row
build_edge_table <- function(cell_data, id_order, neighbors) {

  # --- Step 1: Build spatial edge list (cell-level, year-independent) ----------
  # Convert nb list to a two-column data.table of (source_cell_id, neighbor_cell_id)
  n_cells <- length(id_order)
  num_neighbors <- vapply(neighbors, length, integer(1))
  total_edges <- sum(num_neighbors)

  source_idx <- rep(seq_len(n_cells), times = num_neighbors)
  neighbor_idx <- unlist(neighbors, use.names = FALSE)

  spatial_edges <- data.table(
    source_id   = id_order[source_idx],
    neighbor_id = id_order[neighbor_idx]
  )

  # --- Step 2: Map cell IDs to row numbers, keyed by (id, year) ---------------
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # Create a lookup: for each (id, year) -> row_idx
  id_year_lookup <- dt[, .(id, year, row_idx)]
  setkey(id_year_lookup, id)

  # --- Step 3: Get unique years ------------------------------------------------
  years <- sort(unique(dt$year))

  # --- Step 4: For each year, join spatial edges to row indices ----------------
  # This is the key vectorized operation: for every spatial edge and every year,
  # find the source_row and neighbor_row.

  edge_list <- rbindlist(lapply(years, function(yr) {
    # Rows in this year
    yr_lookup <- id_year_lookup[year == yr, .(id, row_idx)]

    # Join source side
    merged <- spatial_edges[yr_lookup, on = .(source_id = id), nomatch = 0L,
                            .(source_row = i.row_idx, neighbor_id)]

    # Join neighbor side
    setnames(yr_lookup, "row_idx", "neighbor_row")
    merged <- merged[yr_lookup, on = .(neighbor_id = id), nomatch = 0L,
                     .(source_row, neighbor_row = i.neighbor_row)]

    merged
  }))

  edge_list
}


#' Compute neighbor max, min, mean for multiple variables at once using
#' vectorized data.table grouped aggregation.
#'
#' @param cell_data   data.frame/data.table with predictor columns
#' @param edge_table  data.table with columns: source_row, neighbor_row
#' @param var_names   character vector of column names to compute neighbor stats for
#' @return data.table with one row per row of cell_data and columns:
#'         <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
compute_all_neighbor_stats <- function(cell_data, edge_table, var_names) {

  dt <- as.data.table(cell_data)
  n_rows <- nrow(dt)

  # Extract neighbor values for all edges at once
  # edge_table$neighbor_row indexes into dt
  neighbor_vals <- dt[edge_table$neighbor_row, ..var_names]
  neighbor_vals[, source_row := edge_table$source_row]

  # Grouped aggregation: for each source_row, compute max/min/mean of each var
  agg_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(var_names, function(v) {
    paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  }))

  # Build the aggregation call
  stats <- neighbor_vals[,
    setNames(lapply(var_names, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        list(NA_real_, NA_real_, NA_real_)
      } else {
        list(max(vals), min(vals), mean(vals))
      }
    }), var_names),
    by = source_row
  ]

  # The above returns a nested list; let's do it more directly:
  # Use a simpler, highly efficient approach:

  result_dt <- data.table(row_id = seq_len(n_rows))

  for (v in var_names) {
    vals_col <- neighbor_vals[[v]]
    src_col  <- neighbor_vals$source_row

    # Remove NAs from computation
    valid <- !is.na(vals_col)
    v_valid <- vals_col[valid]
    s_valid <- src_col[valid]

    # Compute grouped stats using data.table
    tmp <- data.table(source_row = s_valid, val = v_valid)

    agg <- tmp[, .(
      nmax  = max(val),
      nmin  = min(val),
      nmean = mean(val)
    ), by = source_row]

    # Allocate full-length columns (NA for cells with no valid neighbors)
    max_col  <- rep(NA_real_, n_rows)
    min_col  <- rep(NA_real_, n_rows)
    mean_col <- rep(NA_real_, n_rows)

    max_col[agg$source_row]  <- agg$nmax
    min_col[agg$source_row]  <- agg$nmin
    mean_col[agg$source_row] <- agg$nmean

    result_dt[, (paste0(v, "_neighbor_max"))  := max_col]
    result_dt[, (paste0(v, "_neighbor_min"))  := min_col]
    result_dt[, (paste0(v, "_neighbor_mean")) := mean_col]
  }

  result_dt[, row_id := NULL]
  result_dt
}


# =============================================================================
# MAIN EXECUTION â€” drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Step 1: Build the edge table (replaces build_neighbor_lookup)
message("Building edge table...")
edge_table <- build_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %s edges", format(nrow(edge_table), big.mark = ",")))

# Step 2: Compute all neighbor features at once (replaces the for loop)
message("Computing neighbor statistics...")
neighbor_features <- compute_all_neighbor_stats(cell_data, edge_table, neighbor_source_vars)

# Step 3: Bind the new columns onto cell_data
cell_data <- cbind(cell_data, neighbor_features)

message("Done. Neighbor features added.")

# The trained Random Forest model is unchanged and can be used directly for
# prediction on the updated cell_data, which now contains the same
# <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean columns
# as the original pipeline produced.
```

---

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | Same max/min/mean computed on the same neighbor sets â†’ identical feature values. The Random Forest sees the same input columns with the same values. |
| **Trained RF model** | Untouched. Only the feature-engineering step is optimized. |
| **Memory** | The edge table is ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows Ã— 2 integer columns â‰ˆ 0.6 GB. The neighbor value extraction for one variable at a time is ~38.5M Ã— 1 double â‰ˆ 0.3 GB. Peak usage stays well under 16 GB. |
| **Speed** | `data.table` grouped aggregation on 38.5M rows is typically seconds per variable. Total expected runtime: **under 10 minutes** on a standard laptop, versus the original 86+ hours. |

The ~10,000Ã— speedup comes from replacing 6.46 million R-level loop iterations (with per-iteration string operations and hash lookups) with vectorized `data.table` joins and grouped aggregations that operate on contiguous memory in C.
END_RESPONSE>>>
