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
run_id: Anthropic_F08_T07_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup for all ~6.46 million cell-year rows, even though the neighbor *topology* is identical across all 28 years. Specifically:

1. **Redundant topology resolution:** The function iterates over every row (`6.46M` iterations), looks up the cell's neighbors from the `nb` object, then paste-matches them to year-specific row indices. But the neighbor graph is the same for every year â€” only the *values* change. This means the function does 28Ã— the work it needs to discover which cells are neighbors.

2. **String-key lookups are slow:** `paste(id, year)` keys and named-vector lookups (`idx_lookup[neighbor_keys]`) are O(n) hash-table operations repeated billions of times in aggregate.

3. **Row-level `lapply` over 6.46M rows:** Both `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over millions of elements, producing millions of small R objects (lists/vectors). This is extremely slow due to R's per-call overhead and memory allocation pressure.

4. **The actual computation is trivial:** For each cell, compute max/min/mean of ~4 neighbor values for 5 variables. The bottleneck is entirely in the lookup/indexing infrastructure, not arithmetic.

**Key insight:** The neighbor relationship is a **static graph property** of the 344,208 cells. The variable values are a **dynamic panel property** that changes by year. These should be separated: build the graph once over cells, then for each year, use fast vectorized/matrix operations to compute neighbor statistics.

## Optimization Strategy

1. **Build a cell-level neighbor structure once** (344K cells, not 6.46M rows). Convert the `nb` object into a sparse adjacency representation â€” specifically, two integer vectors (`from`, `to`) representing directed edges â€” computed once.

2. **Process year-by-year using vectorized matrix indexing.** For each year, subset the data, extract variable columns as vectors, and use the edge list to gather neighbor values. Then compute grouped max/min/mean using fast grouped operations (`data.table` or `collapse`).

3. **Use `data.table` for fast grouped aggregation.** For each variable and each year: create a table of `(cell_index, neighbor_value)` from the edge list, then aggregate with `max`, `min`, `mean` by cell â€” all vectorized C-level operations.

4. **Avoid creating millions of small R list elements.** Everything stays in columnar vectors and data.table operations.

**Expected speedup:** From ~86+ hours to **minutes**. The edge list has ~1.37M entries; per year per variable, we do ~1.37M lookups and a grouped aggregation over 344K groups â€” trivial for `data.table`. Across 28 years Ã— 5 variables = 140 such passes, each taking a fraction of a second.

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build the static edge list ONCE from the nb object (344K cells)
# ==============================================================================
build_static_edge_list <- function(nb_obj) {
  # nb_obj is a list of length N_cells; nb_obj[[i]] gives integer indices

# of neighbors of cell i (in the id_order ordering).
  # Returns a data.table with columns: from_ref, to_ref (both are integer
# indices into id_order, i.e., cell reference indices 1..N_cells).
  from_vec <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_vec   <- unlist(nb_obj, use.names = FALSE)
  # Remove the spdep "no neighbor" sentinel (0)
  valid <- to_vec > 0L
  data.table(from_ref = from_vec[valid], to_ref = to_vec[valid])
}

# ==============================================================================
# STEP 2: Compute neighbor stats for all variables, all years
# ==============================================================================
compute_all_neighbor_features <- function(cell_data, id_order, nb_obj,
                                          neighbor_source_vars) {
  # --- Convert to data.table if needed ---
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # --- Build static edge list (done once) ---
  edge_list <- build_static_edge_list(nb_obj)
  cat("Edge list built:", nrow(edge_list), "directed edges\n")

  # --- Build cell-id to cell-reference-index mapping (done once) ---
  # id_order[ref_idx] == cell_id
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # --- Add cell reference index to data (done once) ---
  cell_data[, cell_ref := id_to_ref[as.character(id)]]

  # --- Get unique years ---
  years <- sort(unique(cell_data$year))
  cat("Processing", length(years), "years x", length(neighbor_source_vars),
      "variables =", length(years) * length(neighbor_source_vars), "passes\n")

  # --- Pre-allocate output columns ---
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("n_max_", var_name)
    col_min  <- paste0("n_min_", var_name)
    col_mean <- paste0("n_mean_", var_name)
    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)
  }

  # --- Process each year ---
  # For fast row-lookup within each year, we key by (year, cell_ref)
  # But it's simpler and fast enough to subset per year.

  for (yr in years) {
    # Row indices in cell_data for this year
    yr_rows <- which(cell_data$year == yr)

    # Build a lookup: cell_ref -> row index in cell_data for this year
    yr_cell_refs <- cell_data$cell_ref[yr_rows]

    # Map from cell_ref (1..N_cells) to the row index in cell_data
    # Use a pre-allocated vector for O(1) lookup
    n_cells <- length(id_order)
    ref_to_row <- integer(n_cells)
    ref_to_row[yr_cell_refs] <- yr_rows
    # Cells not present this year remain 0

    # For each edge, find the row of the "from" cell and the row of the "to" cell
    from_rows <- ref_to_row[edge_list$from_ref]
    to_rows   <- ref_to_row[edge_list$to_ref]

    # Keep only edges where both endpoints exist this year
    valid_edges <- from_rows > 0L & to_rows > 0L
    from_rows_v <- from_rows[valid_edges]
    to_rows_v   <- to_rows[valid_edges]

    for (var_name in neighbor_source_vars) {
      col_max  <- paste0("n_max_", var_name)
      col_min  <- paste0("n_min_", var_name)
      col_mean <- paste0("n_mean_", var_name)

      # Get neighbor values (the "to" end of each edge)
      neighbor_vals <- cell_data[[var_name]][to_rows_v]

      # Build a small data.table for grouped aggregation
      # "from_rows_v" identifies which row in cell_data each edge's focal cell is
      agg_dt <- data.table(
        focal_row = from_rows_v,
        nval      = neighbor_vals
      )

      # Remove edges where the neighbor value is NA
      agg_dt <- agg_dt[!is.na(nval)]

      if (nrow(agg_dt) > 0L) {
        # Grouped aggregation â€” very fast in data.table
        stats <- agg_dt[, .(
          nmax  = max(nval),
          nmin  = min(nval),
          nmean = mean(nval)
        ), by = focal_row]

        # Write results back into cell_data
        set(cell_data, i = stats$focal_row, j = col_max,  value = stats$nmax)
        set(cell_data, i = stats$focal_row, j = col_min,  value = stats$nmin)
        set(cell_data, i = stats$focal_row, j = col_mean, value = stats$nmean)
      }
    }

    cat("  Year", yr, "done\n")
  }

  # Clean up helper column
  cell_data[, cell_ref := NULL]

  return(cell_data)
}

# ==============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  nb_obj               = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now has columns: n_max_ntl, n_min_ntl, n_mean_ntl, etc.
# Feed directly into the pre-trained Random Forest predict() call â€” no retraining.
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same numerical estimand** | For each cell-year, the same set of rook neighbors is identified, the same variable values are gathered, and `max`, `min`, `mean` are computed identically (NA-excluding). The output columns have the same names and semantics. |
| **Trained RF model untouched** | No model object is modified. The code only prepares predictor columns, then you call `predict(rf_model, newdata = cell_data)` as before. |
| **Column naming convention** | Output columns are `n_max_{var}`, `n_min_{var}`, `n_mean_{var}` â€” adjust the prefix to match whatever the original `compute_and_add_neighbor_features` used, so the RF model's expected feature names match exactly. |

## Performance Estimate

| Component | Count | Cost |
|---|---|---|
| Edge list construction | 1Ã— | < 1 second |
| Per-year subset + ref_to_row | 28Ã— | ~0.5 sec each â‰ˆ 14 sec |
| Per-year-variable aggregation | 140Ã— | ~0.3 sec each â‰ˆ 42 sec |
| **Total** | | **~1â€“2 minutes** |

This replaces the estimated 86+ hours with roughly **1â€“2 minutes** on the same 16 GB laptop, by exploiting the static-vs-changing distinction: the neighbor graph is resolved once over 344K cells, and only the value lookups + aggregations are repeated per year.
END_RESPONSE>>>
