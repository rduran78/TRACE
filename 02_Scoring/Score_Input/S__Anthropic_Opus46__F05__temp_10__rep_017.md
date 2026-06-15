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
run_id: Anthropic_F05_T10_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern, not merely a local inefficiency**. Here's the full picture:

### Local Inefficiency
- `paste(data$id, data$year, sep = "_")` constructs ~6.46M string keys to build `idx_lookup`, then inside the `lapply` over ~6.46M rows, each iteration calls `paste()` again and does a named-vector lookup (hash lookup on character keys). That's ~6.46M hash probes Ã— average neighbor count per iteration.

### Broader Algorithmic Problem
The real issue is that `build_neighbor_lookup` solves a problem that doesn't require string hashing at all. The data is a **balanced panel** (344,208 cells Ã— 28 years). Within any given year, the neighbor structure is identicalâ€”it's the same spatial grid. So:

1. **The neighbor graph is time-invariant.** You don't need to re-discover neighbors per cell-year row; you only need to know which rows in the data correspond to cell `j` in the same year as cell `i`.
2. **In a balanced panel sorted by (id, year) or (year, id), the row offset between a cell and its same-year neighbor is deterministic.** No string keys are needed at all.
3. **`compute_neighbor_stats` loops row-by-row in R over 6.46M rows.** Even with the lookup pre-built, computing max/min/mean in an R `lapply` over millions of rows is extremely slow.

### Root Cause Summary

| Layer | Problem |
|-------|---------|
| Key construction | 6.46M `paste()` calls + hash table build â€” unnecessary |
| Neighbor lookup | `lapply` over 6.46M rows doing hash probes â€” unnecessary |
| Stats computation | Row-wise R loop over 6.46M rows â€” should be vectorized |
| Outer loop | Rebuilds nothing per variable, but the stats loop alone Ã—5 vars is brutal |

## Optimization Strategy

1. **Eliminate all string keys.** Use integer arithmetic on a balanced panel. If data is sorted by `(id, year)`, then cell index `c` (1-based among the 344,208 cells) in year index `t` (1-based among 28 years) lives at row `(c - 1) * 28 + t` (if sorted id-major) or `(t - 1) * 344208 + c` (if sorted year-major). A neighbor cell `c'` in the same year is at a known offset.

2. **Vectorize neighbor stats with `data.table` or matrix operations.** Expand the neighbor list into an edge table `(row_i, row_j)`, join variable values, then group-by `row_i` to compute max/min/mean in one vectorized pass.

3. **Process all 5 variables in a single grouped aggregation** rather than 5 separate loops.

This reduces 86+ hours to minutes.

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature construction for a balanced spatial panel.
#'
#' Preserves the original numerical estimand: for each cell-year row and each
#' neighbor source variable, compute max, min, and mean of same-year rook
#' neighbors' values (NA if no valid neighbors).
#'
#' @param cell_data data.frame or data.table with columns: id, year, and all
#'   columns named in neighbor_source_vars.
#' @param id_order integer vector of cell IDs in the order matching
#'   rook_neighbors_unique (i.e., id_order[k] is the cell ID for the k-th
#'   element of the nb object).
#' @param rook_neighbors_unique an nb object (list of integer vectors); the
#'   k-th element lists the neighbor indices (into id_order) for cell
#'   id_order[k].
#' @param neighbor_source_vars character vector of variable names to
#'   compute neighbor stats for.
#' @return data.table equal to cell_data with new columns appended:
#'   <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean for each var.
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # â”€â”€ Step 1: Ensure data is keyed by (id, year) and build integer indices â”€â”€

  # Map cell id -> spatial index (position in id_order / nb object)
  n_cells <- length(id_order)
  id_to_sidx <- setNames(seq_len(n_cells), as.character(id_order))

  # Add spatial index to data
  dt[, sidx := id_to_sidx[as.character(id)]]

  # We need a fast way to go from (sidx, year) -> row number in dt.
  # Create a row-number column, then key by (sidx, year).
  dt[, rownum := .I]
  setkey(dt, sidx, year)

  # â”€â”€ Step 2: Build directed edge list in terms of spatial indices â”€â”€
  # edges: from spatial index i to spatial index j (all directed pairs)

  from_sidx <- rep(
    seq_len(n_cells),
    times = lengths(rook_neighbors_unique)
  )
  to_sidx <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove 0-neighbor entries (nb objects use integer(0) for islands)
  valid <- !is.na(to_sidx) & to_sidx > 0L
  from_sidx <- from_sidx[valid]
  to_sidx   <- to_sidx[valid]

  edges_spatial <- data.table(from_sidx = from_sidx, to_sidx = to_sidx)
  n_edges_spatial <- nrow(edges_spatial)

  cat(sprintf(
    "Spatial edge list: %s directed neighbor pairs\n",
    format(n_edges_spatial, big.mark = ",")
  ))

  # â”€â”€ Step 3: Expand edges across all years â”€â”€
  # For each year, every spatial edge becomes a row-level edge.
  # Instead of a massive cross-join, we look up row numbers.

  years <- sort(unique(dt$year))
  n_years <- length(years)

  # Build a matrix: rows = spatial index, cols = year index -> rownum
  # This is the core insight: balanced panel means we can use a matrix lookup.
  # Create lookup: sidx_year_to_row[sidx, year_idx] = rownum
  # For memory: 344208 * 28 = ~9.6M entries, fine as integer vector.

  year_to_yidx <- setNames(seq_along(years), as.character(years))
  dt[, yidx := year_to_yidx[as.character(year)]]

  # Allocate matrix
  sidx_year_to_row <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  sidx_year_to_row[cbind(dt$sidx, dt$yidx)] <- dt$rownum

  # Now expand: for each year, map spatial edges to row-level edges
  cat("Expanding edge list across years...\n")

  # Vectorized approach: replicate edges for each year
  edge_from_rows <- integer(n_edges_spatial * n_years)
  edge_to_rows   <- integer(n_edges_spatial * n_years)

  for (yi in seq_len(n_years)) {
    offset <- (yi - 1L) * n_edges_spatial
    idx_range <- (offset + 1L):(offset + n_edges_spatial)
    edge_from_rows[idx_range] <- sidx_year_to_row[cbind(edges_spatial$from_sidx, yi)]
    edge_to_rows[idx_range]   <- sidx_year_to_row[cbind(edges_spatial$to_sidx, yi)]
  }

  # Remove any NA (missing cell-year combos, e.g., unbalanced fringes)
  valid_edges <- !is.na(edge_from_rows) & !is.na(edge_to_rows)
  edge_from_rows <- edge_from_rows[valid_edges]
  edge_to_rows   <- edge_to_rows[valid_edges]

  cat(sprintf(
    "Full edge list: %s directed row-level edges\n",
    format(length(edge_from_rows), big.mark = ",")
  ))

  # â”€â”€ Step 4: Compute neighbor stats for all variables in one pass â”€â”€

  edges_dt <- data.table(from_row = edge_from_rows, to_row = edge_to_rows)

  # Restore original row order for dt
  setkey(dt, rownum)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))

    # Attach neighbor values
    edges_dt[, neighbor_val := dt[[var_name]][to_row]]

    # Compute grouped stats, dropping NAs
    stats <- edges_dt[!is.na(neighbor_val),
      .(
        nmax  = max(neighbor_val),
        nmin  = min(neighbor_val),
        nmean = mean(neighbor_val)
      ),
      by = from_row
    ]

    # Initialize result columns with NA
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Fill in computed values
    dt[stats$from_row, (max_col)  := stats$nmax]
    dt[stats$from_row, (min_col)  := stats$nmin]
    dt[stats$from_row, (mean_col) := stats$nmean]
  }

  # â”€â”€ Step 5: Clean up helper columns and restore original order â”€â”€
  dt[, c("sidx", "rownum", "yidx") := NULL]

  # Restore original row order (by the original data's ordering)
  # Since rownum was .I before any rekey, and we set key to rownum above,
  # dt is already in original order.

  cat("Done.\n")
  return(dt)
}

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Usage â€” drop-in replacement for the original outer loop
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names match the originals (<var>_neighbor_max, etc.)
# so predict(rf_model, cell_data) works as before.
```

## Performance Analysis

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Key construction** | 6.46M `paste()` + hash build, repeated inside `lapply` | Zero string ops; integer matrix lookup |
| **Neighbor resolution** | 6.46M R-level iterations, each doing `paste()` + hash probe | One vectorized matrix-index expansion across 28 years |
| **Stats computation** | `lapply` over 6.46M rows Ã— 5 vars = 32.3M R function calls | 5 `data.table` grouped aggregations (C-level) |
| **Memory** | Named character vector of 6.46M keys (~500 MB) | Integer matrix 344K Ã— 28 (~39 MB) + edge table (~220 MB) |
| **Estimated time** | 86+ hours | **~2â€“5 minutes** on a 16 GB laptop |

### Why the estimand is preserved

The optimized code computes, for each cell-year row, `max`, `min`, and `mean` of the same variable values at same-year rook neighbors, with `NA` propagation when no valid neighbors existâ€”identical to the original `compute_neighbor_stats`. The edge list is constructed from the same `rook_neighbors_unique` nb object and the same `id_order` mapping, so the neighbor sets are identical. The Random Forest model is never modified; only its input features are constructed more efficiently.
END_RESPONSE>>>
