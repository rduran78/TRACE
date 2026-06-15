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
run_id: Anthropic_F05_T10_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key construction (`paste(..., sep="_")` + named-vector lookup) is **not merely a local inefficiency â€” it is a symptom of a fundamentally row-wise algorithmic pattern** applied to a 6.46-million-row dataset.

Here is the full cost breakdown:

### 1. `build_neighbor_lookup` â€” O(N Ã— K) string operations inside `lapply`
- For each of the ~6.46M rows, the code calls `paste()` to build neighbor keys and performs named-vector lookups via `idx_lookup[neighbor_keys]`.
- With ~4 rook neighbors per cell on average, this is ~25.8M `paste()` calls **plus** ~25.8M hash lookups, all inside a sequential R `lapply`.
- The `idx_lookup` named vector (6.46M entries) is itself built via a single `paste()`, but then queried 25.8M times â€” each query traversing R's internal hash on character strings.

### 2. `compute_neighbor_stats` â€” Lightweight but called 5 times redundantly
- Each call to `compute_neighbor_stats` re-traverses the 6.46M-element `neighbor_lookup` list. This is acceptable once the lookup is built, but the real bottleneck is step 1.

### 3. The deeper structural issue
The entire neighbor-lookup problem is actually a **sparse-matrix join on (cell, year)**, not a string-matching problem. Every cell's neighbors are fixed across years. The data is (presumably) sorted or groupable by year. This means:

- You can compute a **year-invariant neighbor index mapping** (cell â†’ neighbor cells) once, as integer indices into `id_order`.
- Then for each year, you simply offset into the data using the year's block of rows.
- This eliminates **all** string construction and hash-based lookup â€” replacing it with integer arithmetic.

### Cost estimate of current approach
- ~6.46M iterations in R-level `lapply`, each doing `paste` + hash lookup â†’ ~86+ hours is plausible.

### Cost estimate of vectorized approach
- One-time integer index construction: milliseconds.
- Neighbor stats via sparse matrix multiplication or vectorized integer indexing: seconds to low minutes.

---

## Optimization Strategy

**Key insight:** If the data is arranged so that all cells for a given year appear in a contiguous block in the same cell order, then the neighbor relationship (which is purely spatial) can be expressed as fixed integer offsets within each year-block.

**Steps:**

1. **Sort data** by `(year, id)` â€” guaranteeing each year-block has cells in identical order.
2. **Build a cell-index-to-row-offset mapping** â€” within each year-block, cell `j` in `id_order` sits at a known position.
3. **Express rook neighbors as integer index pairs** â€” convert the `nb` object into a two-column integer edge list (from_cell_idx, to_cell_idx) once.
4. **For each year**, translate cell-level edge list to row-level indices via simple offset arithmetic, then compute stats in a fully vectorized way using `data.table` or sparse-matrix grouping.
5. **Compute all 5 variables' neighbor stats** in one pass per variable using vectorized operations.

This completely eliminates `paste()`, named-vector lookups, and row-wise `lapply`.

---

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature construction.
#' Preserves the exact numerical estimand (max, min, mean of non-NA neighbor values).
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and all var_names
#' @param id_order        integer vector of cell IDs in the order used by the nb object
#' @param rook_nb         spdep nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names to compute neighbor stats for
#' @return data.table with original columns plus neighbor feature columns appended

build_neighbor_features_optimized <- function(cell_data,
                                               id_order,
                                               rook_nb,
                                               neighbor_source_vars) {

  # --- Step 0: Convert to data.table (non-destructive copy) ---
  dt <- as.data.table(cell_data)

  # --- Step 1: Sort by (year, id) and record sort order for restoring ---
  # Add an original row index so we can restore original order at the end
  dt[, .orig_row := .I]

  # Build a mapping: cell id -> position in id_order (1-based)
  id_to_cellidx <- setNames(seq_along(id_order), as.character(id_order))

  # Assign each row its cell index within id_order
  dt[, .cell_idx := id_to_cellidx[as.character(id)]]

  # Sort by year then cell_idx to create uniform year-blocks
  setorder(dt, year, .cell_idx)

  # --- Step 2: Build directed edge list from nb object (cell-index space) ---
  # from_cellidx -> to_cellidx for all rook neighbor pairs
  n_cells_in_nb <- length(rook_nb)
  edge_from <- rep(seq_len(n_cells_in_nb),
                   times = lengths(rook_nb))
  edge_to   <- unlist(rook_nb, use.names = FALSE)

  # Remove any 0-length entries (islands) â€” they simply won't appear
  # edge_from[i] is the focal cell index, edge_to[i] is the neighbor cell index

  cat(sprintf("Edge list built: %d directed edges\n", length(edge_from)))

  # --- Step 3: Identify year-blocks ---
  # After sorting by (year, cell_idx), each year forms a contiguous block.
  # But not every cell appears in every year, so we need a within-year position map.
  #
  # For each year-block we build a mapping: cell_idx -> row position in dt

  years <- sort(unique(dt$year))
  n_years <- length(years)

  # Pre-identify the start/end row of each year block and the cell_idx values present

  dt[, .row_in_dt := .I]  # current row index after sorting
  year_info <- dt[, .(start = min(.row_in_dt), end = max(.row_in_dt)), by = year]
  setorder(year_info, year)

  # Also build per-year: a vector mapping cell_idx -> row_in_dt (NA if absent)
  # We'll store this as a list of integer vectors of length n_cells_in_nb
  cat("Building per-year cell-index-to-row maps...\n")

  # Efficient approach: for each year, the cell_idx values and their dt row positions
  year_maps <- vector("list", n_years)
  names(year_maps) <- as.character(years)

  for (yi in seq_len(n_years)) {
    yr <- years[yi]
    info <- year_info[year == yr]
    rows_this_year <- info$start:info$end
    cidxs <- dt$.cell_idx[rows_this_year]

    # Map: cell_idx -> row in dt. Use a pre-allocated integer vector.
    map_vec <- rep(NA_integer_, n_cells_in_nb)
    map_vec[cidxs] <- rows_this_year
    year_maps[[yi]] <- map_vec
  }

  # --- Step 4: For each variable, compute neighbor max/min/mean vectorized ---
  cat("Computing neighbor statistics...\n")

  # Pre-allocate result columns
  n_rows <- nrow(dt)
  for (var_name in neighbor_source_vars) {
    dt[, paste0(var_name, "_neighbor_max")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_min")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_mean") := NA_real_]
  }

  # Strategy: for each year, use the edge list to gather all neighbor values,

  # then compute grouped stats. This is fully vectorized within each year.

  for (yi in seq_len(n_years)) {
    yr <- years[yi]
    info <- year_info[year == yr]
    rows_this_year <- info$start:info$end
    n_this_year <- length(rows_this_year)
    year_map <- year_maps[[yi]]

    # Translate edge list to dt-row space for this year
    # focal row  = year_map[edge_from]
    # neighbor row = year_map[edge_to]
    focal_rows    <- year_map[edge_from]
    neighbor_rows <- year_map[edge_to]

    # Keep only edges where both focal and neighbor exist this year
    valid <- !is.na(focal_rows) & !is.na(neighbor_rows)
    f_rows <- focal_rows[valid]
    n_rows_vec <- neighbor_rows[valid]

    if (length(f_rows) == 0L) next

    # Build a data.table of edges for grouped aggregation
    # We'll reuse this structure for each variable to avoid re-filtering
    edge_dt <- data.table(.focal_row = f_rows, .nbr_row = n_rows_vec)

    for (var_name in neighbor_source_vars) {
      # Get neighbor values
      vals_vec <- dt[[var_name]]
      edge_dt[, .nbr_val := vals_vec[.nbr_row]]

      # Drop NA neighbor values
      valid_edges <- edge_dt[!is.na(.nbr_val)]

      if (nrow(valid_edges) == 0L) next

      # Grouped aggregation: max, min, mean by focal row
      agg <- valid_edges[, .(
        nb_max  = max(.nbr_val),
        nb_min  = min(.nbr_val),
        nb_mean = mean(.nbr_val)
      ), by = .focal_row]

      # Write results back into dt
      col_max  <- paste0(var_name, "_neighbor_max")
      col_min  <- paste0(var_name, "_neighbor_min")
      col_mean <- paste0(var_name, "_neighbor_mean")

      set(dt, i = agg$.focal_row, j = col_max,  value = agg$nb_max)
      set(dt, i = agg$.focal_row, j = col_min,  value = agg$nb_min)
      set(dt, i = agg$.focal_row, j = col_mean, value = agg$nb_mean)
    }

    if (yi %% 5 == 0 || yi == n_years) {
      cat(sprintf("  Completed year %d (%d/%d)\n", yr, yi, n_years))
    }
  }

  # --- Step 5: Restore original row order ---
  setorder(dt, .orig_row)

  # Clean up helper columns
  dt[, c(".orig_row", ".cell_idx", ".row_in_dt") := NULL]

  cat("Done.\n")
  return(dt)
}
```

### Drop-in replacement for the outer loop:

```r
# ---- BEFORE (86+ hours) ----
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ---- AFTER (minutes) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_optimized(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data is now a data.table with the same neighbor feature columns appended.
# Convert back to data.frame if downstream code requires it:
# cell_data <- as.data.frame(cell_data)

# The trained Random Forest model is untouched â€” only feature construction changed.
# Numerical results are identical: max, min, mean of non-NA rook-neighbor values per cell-year.
```

---

## Complexity Comparison

| Aspect | Original | Optimized |
|---|---|---|
| **String construction** | ~32M `paste()` calls (6.46M for lookup + 25.8M in lapply) | **Zero** |
| **Hash lookups** | ~25.8M character hash probes on a 6.46M-entry table | **Zero** |
| **R-level loop iterations** | 6.46M (one per row in `lapply`) | 28 (one per year) |
| **Neighbor stat computation** | 5 Ã— 6.46M list traversals in `lapply` | 5 Ã— 28 vectorized `data.table` grouped aggregations |
| **Memory pattern** | 6.46M-element R list of integer vectors | One integer edge list (~2.7M pairs) reused across years |
| **Estimated runtime** | 86+ hours | **~2â€“10 minutes** |
| **Numerical output** | max, min, mean of non-NA neighbor values | **Identical** |
| **RF model impact** | â€” | **None (untouched)** |

The core algorithmic insight is that **spatial neighbor relationships are year-invariant**. The original code rebuilt the cellâ†’row mapping via string hashing 6.46 million times inside a sequential loop. The optimized code exploits the invariant structure: build an integer edge list once, then for each of 28 years, translate it to row-space via a simple array index (`year_map[edge_from]`) and compute grouped statistics with `data.table`'s optimized C-level aggregation.
END_RESPONSE>>>
