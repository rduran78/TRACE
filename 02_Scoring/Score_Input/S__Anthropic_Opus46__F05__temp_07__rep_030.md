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
run_id: Anthropic_F05_T07_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full chain of waste:

### 1. **O(N) string-key construction inside an O(N) loop â†’ O(NÂ²) string work**
`build_neighbor_lookup` constructs `idx_lookup` once (good), but then inside the `lapply` over all ~6.46M rows, it calls `paste()` to build `neighbor_keys` for every row. Each row has ~4 rook neighbors on average (interior cells), so this creates ~25.8M small string vectors across 6.46M iterations, each followed by a named-vector lookup. The `lapply` loop itself is inherently serial in R and carries per-iteration overhead.

### 2. **The neighbor topology is year-invariant, but the code re-discovers it per cell-year**
Rook neighbors are a spatial relationship: cell `A` neighbors cell `B` in **every** year. The current code rebuilds the mapping from `(cell, year)` â†’ `(neighbor_cell, year)` for every row. Since there are 28 years, every spatial neighbor pair is resolved 28 times instead of once.

### 3. **`compute_neighbor_stats` is called 5 times, each iterating over 6.46M rows**
Each call to `compute_neighbor_stats` loops over the full `neighbor_lookup` list (6.46M entries). The list-of-integer-vectors structure also has high memory overhead (~6.46M R integer vectors).

### 4. **Summary of redundancy**
| Source of waste | Magnitude |
|---|---|
| String `paste` + named lookup per row | ~6.46M Ã— ~4 neighbors = ~25.8M paste ops |
| Year-invariant topology resolved per year | 28Ã— redundant |
| Per-variable R-level loop over 6.46M rows | 5Ã— full scan with R `lapply` overhead |
| List-of-vectors memory overhead | ~6.46M small vectors |

**Estimated speedup from the reformulation below: ~200â€“500Ã—**, bringing runtime from 86+ hours to roughly 10â€“25 minutes.

---

## Optimization Strategy

1. **Separate space from time.** Build the neighbor index once at the cell level (344K cells), not the cell-year level (6.46M rows).
2. **Vectorize with a flat edge table + `data.table` grouped aggregation.** Expand the `nb` object into an edge list `(focal_cell_row, neighbor_cell_row)`, join to years via a merge (or row-arithmetic since the panel is balanced), then compute `max/min/mean` per group in one vectorized pass per variable.
3. **Exploit balanced panel structure.** If the data is sorted by `(id, year)` (or we sort it once), the row for `(cell_i, year_t)` is at a deterministic offset, eliminating all hash lookups.
4. **Compute all 5 variables in a single pass** over the edge table using `data.table`.

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and
#'                        all variables in neighbor_source_vars.
#' @param id_order        integer vector of cell IDs in the order matching
#'                        rook_neighbors_unique (i.e., the region.id order from spdep).
#' @param nb              spdep nb object (rook_neighbors_unique).
#' @param neighbor_source_vars character vector of variable names.
#' @return cell_data with new columns appended (same row order as input).
add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      nb,
                                      neighbor_source_vars) {

  # --- 0. Convert to data.table, preserve original row order ----------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
    was_df <- TRUE
  } else {
    was_df <- FALSE
  }
  cell_data[, .row_orig := .I]

  # --- 1. Sort by (id, year) to make row arithmetic possible ----------------
  #     After sorting, cell i (0-indexed in id_order) and year t (0-indexed

  #     from min_year) lives at row:  i * n_years + t + 1
  years     <- sort(unique(cell_data$year))
  n_years   <- length(years)
  n_cells   <- length(id_order)
  stopifnot(nrow(cell_data) == n_cells * n_years)

  # Create a map from cell id -> 0-based spatial index
  id_to_sidx <- setNames(seq_along(id_order) - 1L, as.character(id_order))

  # Add spatial index and year index, then sort
  cell_data[, sidx := id_to_sidx[as.character(id)]]
  year_min <- min(years)
  cell_data[, tidx := as.integer(year - year_min)]
  setorder(cell_data, sidx, tidx)
  # Now row number = sidx * n_years + tidx + 1  (1-based)

  # --- 2. Build spatial edge list from nb object (cell-level, no years) -----
  #     nb[[k]] contains integer indices of neighbors of the k-th cell.
  #     We expand this to a two-column integer matrix: (focal_sidx, neighbor_sidx)
  #     using 0-based spatial indices.
  n_edges <- sum(lengths(nb))
  focal_sidx    <- integer(n_edges)
  neighbor_sidx <- integer(n_edges)
  pos <- 1L
  for (k in seq_along(nb)) {
    nbrs <- nb[[k]]
    if (length(nbrs) == 0L || (length(nbrs) == 1L && nbrs[1L] == 0L)) next
    len <- length(nbrs)
    focal_sidx[pos:(pos + len - 1L)]    <- k - 1L        # 0-based
    neighbor_sidx[pos:(pos + len - 1L)] <- nbrs - 1L      # 0-based (spdep is 1-based)
    pos <- pos + len
  }
  # Trim if any nb entries were empty
  if (pos - 1L < n_edges) {
    focal_sidx    <- focal_sidx[1:(pos - 1L)]
    neighbor_sidx <- neighbor_sidx[1:(pos - 1L)]
  }

  # --- 3. Expand edge list across all years (vectorized) --------------------
  #     For each year index t in 0..(n_years-1), the row (1-based) of

  #     spatial index s is:  s * n_years + t + 1
  #
  #     We replicate the spatial edge list n_years times.
  n_spatial_edges <- length(focal_sidx)
  tidx_rep <- rep(0:(n_years - 1L), each = n_spatial_edges)
  focal_rows    <- rep(focal_sidx,    times = n_years) * n_years + tidx_rep + 1L
  neighbor_rows <- rep(neighbor_sidx, times = n_years) * n_years + tidx_rep + 1L

  edges <- data.table(
    focal_row    = focal_rows,
    neighbor_row = neighbor_rows
  )
  # Free temporaries
  rm(focal_sidx, neighbor_sidx, tidx_rep, focal_rows, neighbor_rows)
  gc()

  # --- 4. Look up neighbor values and aggregate per focal row ---------------
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor stats for: ", var_name)

    # Pull the variable as a plain vector (aligned with sorted cell_data)
    vals_vec <- cell_data[[var_name]]

    # Attach neighbor values to edge table
    edges[, nval := vals_vec[neighbor_row]]

    # Aggregate: max, min, mean per focal_row, dropping NAs
    agg <- edges[!is.na(nval),
                 .(vmax  = max(nval),
                   vmin  = min(nval),
                   vmean = mean(nval)),
                 by = focal_row]

    # Initialize result columns with NA
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)

    # Fill in aggregated values
    set(cell_data, i = agg$focal_row, j = col_max,  value = agg$vmax)
    set(cell_data, i = agg$focal_row, j = col_min,  value = agg$vmin)
    set(cell_data, i = agg$focal_row, j = col_mean, value = agg$vmean)

    rm(agg)
  }

  # Clean up the edge table
  edges[, nval := NULL]

  # --- 5. Restore original row order and drop helper columns ----------------
  setorder(cell_data, .row_orig)
  cell_data[, c("sidx", "tidx", ".row_orig") := NULL]

  if (was_df) cell_data <- as.data.frame(cell_data)

  return(cell_data)
}
```

### Drop-in replacement call

```r
# â”€â”€ Before (86+ hours) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# â”€â”€ After (~10-25 minutes) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged â€” only the feature
# construction pipeline is replaced.  Column names and numerical
# values are identical to the original implementation.
```

---

## Memory Budget Check

| Object | Approximate size |
|---|---|
| `cell_data` (6.46M Ã— ~125 cols, numeric) | ~6.5 GB |
| `edges` (1.37M Ã— 28 years Ã— 2 int cols + 1 dbl) | ~0.6 GB |
| Aggregation temporaries | ~0.3 GB |
| **Total peak** | **~7.4 GB** (fits in 16 GB) |

If memory is tighter, the year expansion in step 3 can be chunked (e.g., 7 years at a time) with trivial modification.

---

## Why This Preserves the Original Numerical Estimand

- The edge list is an exact expansion of the `nb` object â€” same directed neighbor pairs.
- For each `(focal_cell, year)`, the same set of `neighbor_row` values is aggregated.
- `max`, `min`, `mean` over the same non-NA values produce identical results.
- Column names follow the same `neighbor_{max,min,mean}_{var}` convention.
- The trained Random Forest model is never touched; it simply receives the same feature columns.
END_RESPONSE>>>
