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
run_id: Anthropic_F12_T03_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not ideal, it is a one-shot operation per variable (5 times total) and each invocation merely stacks pre-allocated 3-element vectors. This is a minor cost.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's the evidence from the code:

1. **Massive character key construction and lookup via named vectors:** `build_neighbor_lookup` creates `idx_lookup`, a named vector of ~6.46 million elements keyed by `paste(id, year, sep="_")`. Then, *for every single row* (6.46 million iterations), it:
   - Calls `as.character()` on the id.
   - Looks up `id_to_ref` by character name (linear-ish hash lookup on a 344K-element named vector).
   - Extracts neighbor cell IDs from the `nb` object.
   - Calls `paste()` to build neighbor key strings.
   - Performs named-vector lookup into `idx_lookup` (a 6.46M-element named vector) **for each neighbor key**.

2. **Redundant recomputation across years:** Every cell has the same neighbors in every year. With 344,208 cells Ã— 28 years, the neighbor topology is recomputed 28 times per cell. The `paste`/lookup work is repeated for every year even though only the year suffix changes.

3. **`lapply` over 6.46 million rows with per-element string operations** is the dominant wall-clock cost â€” not the downstream `do.call(rbind, ...)`.

**Quantitative reasoning:** ~6.46M iterations, each doing multiple `paste()` calls and named-vector lookups into a 6.46M-length vector. Named vector lookup in R is O(1) amortized via hashing, but the constant factor is large, and doing it ~6.46M Ã— ~4 neighbors â‰ˆ 25.8 million hash lookups with string allocation is extremely expensive in interpreted R. This dwarfs the one-time `do.call(rbind, ...)` cost.

## Optimization Strategy

1. **Separate topology from time:** Build the neighbor lookup at the **cell level** (344K entries), not the **cell-year level** (6.46M entries). Since rook neighbors don't change across years, compute neighbor indices once per cell, then expand to cell-year rows using vectorized integer arithmetic.

2. **Use integer indexing throughout:** Replace all `paste()`-based named-vector lookups with direct integer-indexed operations. Create a matrix/mapping from `(cell_index, year_index)` â†’ row number using a pre-allocated integer matrix.

3. **Vectorize `compute_neighbor_stats`:** Replace the per-row `lapply` with a single vectorized grouped operation using `data.table` or pre-allocated matrix operations.

4. **Preserve the trained RF model and numerical estimand:** The output columns are identical (max, min, mean of each neighbor variable), just computed faster.

## Working R Code

```r
library(data.table)

optimize_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {

  # Convert to data.table for speed (non-destructive; preserves all columns)
  dt <- as.data.table(cell_data)

  # ---------------------------------------------------------------
  # STEP 1: Build cell-level neighbor lookup (344K entries, not 6.46M)
  # ---------------------------------------------------------------
  n_cells <- length(id_order)
  # Map cell id -> cell index (1-based position in id_order)
  id_to_cidx <- setNames(seq_along(id_order), as.character(id_order))

  # For each cell index, get neighbor cell indices (topology only, no years)
  # rook_neighbors_unique is an nb object: list of integer vectors of neighbor positions
  # These positions already reference id_order, so no conversion needed.
  cell_neighbor_cidx <- rook_neighbors_unique  # list of length n_cells, each element = integer vector of neighbor cell indices

  # ---------------------------------------------------------------
  # STEP 2: Build (cell_index, year_index) -> row mapping via integer matrix
  # ---------------------------------------------------------------
  years_all <- sort(unique(dt$year))
  n_years <- length(years_all)
  year_to_yidx <- setNames(seq_along(years_all), as.character(years_all))

  # Compute cell index and year index for every row
  dt[, c_cidx := id_to_cidx[as.character(id)]]
  dt[, c_yidx := year_to_yidx[as.character(year)]]

  # Build row-lookup matrix: row_matrix[cell_index, year_index] = row number in dt
  # Initialize with NA
  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_matrix[cbind(dt$c_cidx, dt$c_yidx)] <- seq_len(nrow(dt))

  # ---------------------------------------------------------------
  # STEP 3: Expand cell-level neighbors to cell-year neighbor row indices
  #         using vectorized integer arithmetic
  # ---------------------------------------------------------------
  # For each row i, its neighbors are:
  #   neighbor cell indices = cell_neighbor_cidx[[ dt$c_cidx[i] ]]
  #   neighbor rows = row_matrix[ neighbor_cell_indices, dt$c_yidx[i] ]
  #
  # We vectorize this by building an edge list at the cell level,
  # then expanding across years.

  # Build cell-level edge list: (focal_cidx, neighbor_cidx)
  focal_cidx_cell <- rep(seq_len(n_cells),
                         times = lengths(cell_neighbor_cidx))
  neighbor_cidx_cell <- unlist(cell_neighbor_cidx, use.names = FALSE)

  # Remove 0-neighbor entries from nb object (nb uses 0L for no neighbors)
  valid <- neighbor_cidx_cell != 0L
  focal_cidx_cell <- focal_cidx_cell[valid]
  neighbor_cidx_cell <- neighbor_cidx_cell[valid]

  n_edges_cell <- length(focal_cidx_cell)

  # Expand across all years: each cell-level edge appears once per year
  # Total directed edges across all years: n_edges_cell * n_years
  focal_cidx_exp   <- rep(focal_cidx_cell, times = n_years)
  neighbor_cidx_exp <- rep(neighbor_cidx_cell, times = n_years)
  yidx_exp          <- rep(seq_len(n_years), each = n_edges_cell)

  # Map to row numbers
  focal_row    <- row_matrix[cbind(focal_cidx_exp, yidx_exp)]
  neighbor_row <- row_matrix[cbind(neighbor_cidx_exp, yidx_exp)]

  # Remove edges where either focal or neighbor row doesn't exist
  valid_edges <- !is.na(focal_row) & !is.na(neighbor_row)
  focal_row    <- focal_row[valid_edges]
  neighbor_row <- neighbor_row[valid_edges]

  # Free large temporaries
  rm(focal_cidx_exp, neighbor_cidx_exp, yidx_exp, valid_edges,
     focal_cidx_cell, neighbor_cidx_cell)
  gc()

  # ---------------------------------------------------------------
  # STEP 4: Compute neighbor stats per variable using data.table grouping
  # ---------------------------------------------------------------
  # Build edge data.table once
  edges_dt <- data.table(focal_row = focal_row, neighbor_row = neighbor_row)
  rm(focal_row, neighbor_row)
  gc()

  n_rows <- nrow(dt)

  for (var_name in neighbor_source_vars) {
    message("Processing neighbor stats for: ", var_name)

    # Extract neighbor values via integer indexing (fully vectorized)
    edges_dt[, nval := dt[[var_name]][neighbor_row]]

    # Remove NA neighbor values
    edges_valid <- edges_dt[!is.na(nval)]

    # Grouped aggregation: max, min, mean per focal row
    agg <- edges_valid[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_row]

    # Initialize result columns with NA
    max_col  <- rep(NA_real_, n_rows)
    min_col  <- rep(NA_real_, n_rows)
    mean_col <- rep(NA_real_, n_rows)

    # Fill in computed values
    max_col[agg$focal_row]  <- agg$nb_max
    min_col[agg$focal_row]  <- agg$nb_min
    mean_col[agg$focal_row] <- agg$nb_mean

    # Add columns with original naming convention
    set(dt, j = paste0(var_name, "_neighbor_max"),  value = max_col)
    set(dt, j = paste0(var_name, "_neighbor_min"),  value = min_col)
    set(dt, j = paste0(var_name, "_neighbor_mean"), value = mean_col)
  }

  # Clean up temporary columns
  dt[, c("c_cidx", "c_yidx") := NULL]
  edges_dt[, nval := NULL]

  # Return as data.frame to preserve compatibility with downstream RF predict()
  as.data.frame(dt)
}

# ---------------------------------------------------------------
# USAGE (drop-in replacement for the original pipeline)
# ---------------------------------------------------------------
# cell_data <- optimize_neighbor_pipeline(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
#
# # The trained Random Forest model is untouched:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup scope** | 6.46M row-level `lapply` with `paste()` + named-vector lookup | 344K cell-level topology, expanded via integer matrix indexing |
| **String operations** | ~25.8M `paste()` calls + hash lookups | Zero string operations in hot path |
| **Stats computation** | 6.46M-iteration `lapply` + `do.call(rbind, ...)` | Vectorized `data.table` grouped aggregation on edge list |
| **Estimated runtime** | 86+ hours | **Minutes** (dominated by the `data.table` grouped aggregation on ~38.4M edges Ã— 5 variables) |
| **Memory** | Stores 6.46M-element list of integer vectors | Edge list ~300MB + row matrix ~38MB; peak ~1-2GB, fits in 16GB |
| **RF model** | â€” | Preserved, untouched |
| **Numerical output** | max/min/mean per neighbor set | Identical estimand preserved |

The key insight: the bottleneck was never `do.call(rbind, ...)` â€” it was the **6.46 million iterations of string-based key construction and named-vector lookup** in `build_neighbor_lookup()`, redundantly recomputing time-invariant spatial topology for every year. Separating topology from time and replacing string operations with integer matrix indexing eliminates the bottleneck entirely.
END_RESPONSE>>>
