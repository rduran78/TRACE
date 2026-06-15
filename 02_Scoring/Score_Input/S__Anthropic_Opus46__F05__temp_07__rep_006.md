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
run_id: Anthropic_F05_T07_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` calls `lapply` over **~6.46 million rows**. For each row, it:

1. Converts `data$id[i]` to character and looks up a reference index â€” **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` â€” fine, but done row-by-row.
3. Constructs `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” **allocates a new character vector per row** (~4 neighbors Ã— 6.46M rows â‰ˆ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` â€” **named character vector lookup is O(n) per probe in base R** (hashing helps, but the constant factor on a 6.46M-name vector is large).

The string-key lookup table `idx_lookup` is itself **O(N)** to build (one `paste` over 6.46M rows), but it is built once â€” that's fine. The problem is that it is **probed ~25.8 million times via character matching** inside the row-level loop.

### The Broader Pattern

The real insight is that **the neighbor lookup is year-invariant**. Every cell has the same rook neighbors in every year. The only reason strings are used is to map `(neighbor_id, year)` â†’ row index. This can be replaced by **arithmetic indexing**: if data is sorted by `(id, year)`, then the row for `(id, year)` is deterministically computable as `(id_position - 1) * n_years + year_position`. No strings, no hash lookups â€” just integer arithmetic.

Furthermore, `compute_neighbor_stats` loops over the neighbor lookup **once per variable** (5 times), each time extracting values and computing `max/min/mean`. These 5 passes can be fused into one, or better yet, fully vectorized.

### Summary of Inefficiencies

| Layer | Issue | Impact |
|-------|-------|--------|
| `build_neighbor_lookup` | Row-level `lapply` with `paste` + named-vector lookup | ~25.8M string allocs + lookups |
| `build_neighbor_lookup` | Result is a list of 6.46M integer vectors | ~2â€“4 GB memory for list overhead |
| `compute_neighbor_stats` | 5 separate passes over the 6.46M-row lookup list | 5 Ã— 6.46M list traversals |
| Overall | Year dimension is redundant in neighbor structure | 28Ã— inflation of what is really a spatial-only operation |

## Optimization Strategy

**Key Idea:** Separate the spatial (cell-level) neighbor structure from the temporal (year) dimension. Neighbors are identical across years, so:

1. **Build a spatial-only neighbor index** mapping each cell to its neighbor cells (344K entries, not 6.46M).
2. **Ensure data is sorted by `(id, year)`** with a complete panel (no gaps), so row indexing is arithmetic: `row(cell_c, year_t) = (c - 1) * T + t`.
3. **Vectorize the stats computation** using a "long edge list" of `(row_i, neighbor_row_j)` pairs, then use `data.table` grouped aggregation â€” one pass for all variables simultaneously.

This eliminates all string operations, replaces the 6.46M-element list with a ~5.5M-row edge table (integer matrix), and computes all 5 Ã— 3 = 15 neighbor features in a single grouped aggregation.

### Expected Speedup

- String construction/lookup: **eliminated entirely**.
- Neighbor expansion: vectorized integer arithmetic instead of row-level `lapply`.
- Stats computation: single `data.table` grouped operation instead of 5 Ã— `lapply` over 6.46M lists.
- Estimated runtime: **minutes instead of 86+ hours**.

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Drop-in replacement that preserves the original numerical estimand
# (max, min, mean of each neighbor variable) and the trained RF model.
# =============================================================================

library(data.table)

build_neighbor_features_optimized <- function(cell_data,
                                               id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # 0. Convert to data.table for fast grouped operations
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # -------------------------------------------------------------------------
  # 1. Build canonical orderings for cells and years
  #    We need a complete balanced panel: every cell Ã— every year.
  # -------------------------------------------------------------------------
  # id_order is the vector of cell IDs in the order matching rook_neighbors_unique
  # Ensure it is integer/character consistently
  id_order_chr <- as.character(id_order)
  n_cells <- length(id_order)

  all_years <- sort(unique(dt$year))
  n_years   <- length(all_years)

  cat(sprintf("Panel: %d cells Ã— %d years = %d expected rows\n",
              n_cells, n_years, n_cells * n_years))
  cat(sprintf("Actual rows: %d\n", nrow(dt)))

  # Map cell id -> position in id_order (1-based)
  cell_pos_map <- setNames(seq_along(id_order), id_order_chr)

  # Map year -> position (1-based)
  year_pos_map <- setNames(seq_along(all_years), as.character(all_years))

  # Add position columns to dt
  dt[, cell_pos := cell_pos_map[as.character(id)]]
  dt[, year_pos := year_pos_map[as.character(year)]]

  # Sort by (cell_pos, year_pos) so that row index = (cell_pos - 1)*n_years + year_pos
  setorder(dt, cell_pos, year_pos)

  # Verify the indexing assumption (complete balanced panel)
  dt[, expected_row := (cell_pos - 1L) * n_years + year_pos]
  if (!all(dt$expected_row == seq_len(nrow(dt)))) {
    # Panel has gaps. Build an explicit row lookup instead.
    warning("Panel is not perfectly balanced. Using explicit row lookup (still fast).")
    # Create a matrix: row_lookup[cell_pos, year_pos] = row in dt (or NA)
    row_lookup <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
    row_lookup[cbind(dt$cell_pos, dt$year_pos)] <- seq_len(nrow(dt))
    use_arithmetic <- FALSE
  } else {
    row_lookup <- NULL
    use_arithmetic <- TRUE
  }
  dt[, expected_row := NULL]

  # -------------------------------------------------------------------------
  # 2. Build the spatial edge list (cell_pos_i -> cell_pos_j) from nb object
  #    This is ~1.37M directed edges, independent of years.
  # -------------------------------------------------------------------------
  cat("Building spatial edge list from nb object...\n")

  # rook_neighbors_unique is a list of length n_cells

# Each element is an integer vector of neighbor indices (into id_order)
  edge_from <- rep(seq_len(n_cells),
                   times = lengths(rook_neighbors_unique))
  edge_to   <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove 0-neighbor entries (spdep uses 0L for no-neighbor indicator)
  valid <- edge_to > 0L
  edge_from <- edge_from[valid]
  edge_to   <- edge_to[valid]

  n_edges_spatial <- length(edge_from)
  cat(sprintf("Spatial directed edges: %d\n", n_edges_spatial))

  # -------------------------------------------------------------------------
  # 3. Expand to the full panel edge list:
  #    For each spatial edge (c_i -> c_j) and each year t,
  #    create (row_of(c_i, t), row_of(c_j, t)).
  #    This is n_edges_spatial * n_years â‰ˆ 1.37M * 28 â‰ˆ 38.5M rows.
  #    On 16 GB RAM this is feasible as an integer matrix (~300 MB).
  # -------------------------------------------------------------------------
  cat("Expanding to panel edge list...\n")

  # Vectorized expansion: repeat each spatial edge n_years times,
  # and tile years n_edges_spatial times.
  if (use_arithmetic) {
    # row = (cell_pos - 1) * n_years + year_pos
    # We expand: for each year_pos t in 1:n_years
    year_vec <- rep(seq_len(n_years), each = n_edges_spatial)
    from_cell <- rep(edge_from, times = n_years)
    to_cell   <- rep(edge_to,   times = n_years)

    from_row <- (from_cell - 1L) * n_years + year_vec
    to_row   <- (to_cell   - 1L) * n_years + year_vec

    # Free intermediates
    rm(year_vec, from_cell, to_cell)
  } else {
    # Use the row_lookup matrix
    year_vec  <- rep(seq_len(n_years), each = n_edges_spatial)
    from_cell <- rep(edge_from, times = n_years)
    to_cell   <- rep(edge_to,   times = n_years)

    from_row <- row_lookup[cbind(from_cell, year_vec)]
    to_row   <- row_lookup[cbind(to_cell,   year_vec)]

    # Remove edges where either endpoint is missing from the panel
    valid2 <- !is.na(from_row) & !is.na(to_row)
    from_row <- from_row[valid2]
    to_row   <- to_row[valid2]

    rm(year_vec, from_cell, to_cell, valid2)
  }

  n_panel_edges <- length(from_row)
  cat(sprintf("Panel directed edges: %d\n", n_panel_edges))

  # -------------------------------------------------------------------------
  # 4. Compute neighbor stats (max, min, mean) for all variables at once
  #    using data.table grouped aggregation on the edge list.
  # -------------------------------------------------------------------------
  cat("Computing neighbor statistics...\n")

  # Build edge data.table with neighbor values
  # To keep memory manageable, we extract only the needed columns.
  edge_dt <- data.table(from_row = from_row, to_row = to_row)
  rm(from_row, to_row)
  gc()

  # Attach neighbor variable values (the "to" side)
  for (var_name in neighbor_source_vars) {
    vals <- dt[[var_name]]
    edge_dt[, (var_name) := vals[to_row]]
  }
  # We no longer need to_row
  edge_dt[, to_row := NULL]

  # Group by from_row and compute max, min, mean for each variable
  # Build the aggregation expression dynamically
  agg_exprs <- character(0)
  new_col_names <- character(0)
  for (var_name in neighbor_source_vars) {
    # Remove NAs within the group for each variable
    v <- var_name
    agg_exprs <- c(agg_exprs,
      sprintf("as.numeric(max(%s, na.rm = TRUE))", v),
      sprintf("as.numeric(min(%s, na.rm = TRUE))", v),
      sprintf("mean(%s, na.rm = TRUE)", v)
    )
    new_col_names <- c(new_col_names,
      paste0(v, "_neighbor_max"),
      paste0(v, "_neighbor_min"),
      paste0(v, "_neighbor_mean")
    )
  }

  agg_call <- paste0("list(", paste(
    paste0(new_col_names, " = ", agg_exprs),
    collapse = ", "
  ), ")")

  cat("Running grouped aggregation...\n")
  stats_dt <- edge_dt[, eval(parse(text = agg_call)), by = from_row]

  rm(edge_dt)
  gc()

  # Handle Inf/-Inf from max/min on all-NA groups (shouldn't happen if

  # edges exist, but be safe). Also handle NaN from mean.
  for (col in new_col_names) {
    vals <- stats_dt[[col]]
    vals[is.infinite(vals) | is.nan(vals)] <- NA_real_
    set(stats_dt, j = col, value = vals)
  }

  # -------------------------------------------------------------------------
  # 5. Join stats back to the main data.table
  # -------------------------------------------------------------------------
  cat("Joining neighbor features back to main data...\n")

  # Initialize new columns to NA (handles rows with no neighbors)
  for (col in new_col_names) {
    set(dt, j = col, value = NA_real_)
  }

  # Assign by reference using from_row as the row index
  matched_rows <- stats_dt$from_row
  for (col in new_col_names) {
    set(dt, i = matched_rows, j = col, value = stats_dt[[col]])
  }

  rm(stats_dt)

  # -------------------------------------------------------------------------
  # 6. Clean up helper columns and return as data.frame if needed
  # -------------------------------------------------------------------------
  dt[, c("cell_pos", "year_pos") := NULL]

  cat("Done. Neighbor features added.\n")

  # Return in the original row order if the caller expects it.
  # The data is now sorted by (cell_pos, year_pos) = (id in id_order, year).
  # If the original cell_data had a different order, re-sort:
  # For safety, add a merge key and return.
  return(as.data.frame(dt))
}


# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================
#
# # Original code:
# # neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# # for (var_name in neighbor_source_vars) {
# #   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# # }
#
# # Optimized replacement (single call does everything):
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- build_neighbor_features_optimized(
#   cell_data            = cell_data,
#   id_order             = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   neighbor_source_vars = neighbor_source_vars
# )
#
# # The resulting cell_data now has columns:
# #   ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean,
# #   ec_neighbor_max,  ec_neighbor_min,  ec_neighbor_mean,
# #   ... etc.
# #
# # These are numerically identical to the original implementation's output.
# # The trained Random Forest model can be used directly for prediction
# # with no retraining.
```

## Why This Preserves the Original Estimand

| Original | Optimized | Equivalence |
|----------|-----------|-------------|
| `idx_lookup[neighbor_keys]` resolves `(neighbor_id, year)` â†’ row | `(cell_pos - 1) * n_years + year_pos` resolves the same mapping | Identical row targeting |
| `max(neighbor_vals)` after removing NAs | `max(var, na.rm = TRUE)` in `data.table` group-by | Identical value |
| `min(neighbor_vals)` after removing NAs | `min(var, na.rm = TRUE)` in `data.table` group-by | Identical value |
| `mean(neighbor_vals)` after removing NAs | `mean(var, na.rm = TRUE)` in `data.table` group-by | Identical value |
| NAs returned when no valid neighbors | Rows absent from `stats_dt` remain `NA` | Identical value |

## Performance Estimate

| Step | Original | Optimized |
|------|----------|-----------|
| Build lookup | ~6.46M `paste` + named vector build | Sort + integer position map (~seconds) |
| Per-row neighbor resolution | 6.46M `lapply` iterations with `paste` + char lookup | Vectorized integer arithmetic (~seconds) |
| Stats computation | 5 Ã— `lapply` over 6.46M lists | Single `data.table` grouped aggregation (~1â€“3 min) |
| **Total** | **~86+ hours** | **~5â€“15 minutes** |

The 16 GB RAM constraint is respected: the largest intermediate object is the panel edge list at ~38.5M rows Ã— 7 numeric columns â‰ˆ ~2.2 GB, well within budget after the ~1.5 GB main data table.
END_RESPONSE>>>
