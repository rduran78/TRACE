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
run_id: Anthropic_F04_T03_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Looks up the cell's reference index via character coercion and named-vector lookup.
2. Retrieves neighbor cell IDs.
3. Constructs character paste keys (`id_year`) for every neighbor.
4. Matches those keys against a named character vector (`idx_lookup`).

Character-based key construction (`paste`) and named-vector lookup (`idx_lookup[neighbor_keys]`) are both **O(n)** or worse per call when done millions of times. With ~6.46M rows and an average of ~8 neighbors per cell (1,373,394 directed relationships / ~344K cells â‰ˆ 4 per cell, but bidirectional gives ~8 lookups per row), this produces roughly **50+ million `paste` and named-vector match operations**. Named vector lookup in R is hash-based but still carries significant per-call overhead in an interpreted loop. The 86+ hour estimate is consistent with this.

**`compute_neighbor_stats`** is a secondary bottleneck: another `lapply` over 6.46M rows doing subsetting and summary stats, but it is less severe because it operates on integer indices into a numeric vector.

The Random Forest inference itself is comparatively fast (a single `predict` call on a pre-trained model).

## Optimization Strategy

**Core insight:** The neighbor topology is *time-invariant*. A cell's neighbors are the same in every year. Therefore, we should:

1. **Build the neighbor lookup once at the cell level (344K entries), not at the cell-year level (6.46M entries).** For each cell, store its neighbor cell IDs. Then, for each year, use vectorized operations to gather neighbor values.

2. **Replace the row-level `lapply` with vectorized/matrix operations.** Reshape the variable of interest into a matrix of `cells Ã— years`, use the cell-level neighbor list to index into columns, and compute `max`, `min`, `mean` across neighbors using vectorized code.

3. **Use `data.table` for fast joins and grouping** instead of named-vector lookups and `paste` keys.

4. **Optionally parallelize** across the 5 source variables or across years using `parallel` or `future.apply`.

This reduces the problem from 6.46M interpreted-loop iterations to ~344K (for the neighbor list) plus fully vectorized matrix operations across 28 years.

## Optimized Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build cell-level neighbor lookup (once, 344K entries)
# ============================================================
# rook_neighbors_unique: spdep nb object indexed by position in id_order
# id_order: vector of cell IDs in the same order as the nb object

build_cell_neighbor_lookup <- function(id_order, neighbors) {
  # Returns a list: for each cell index i in id_order,

  # the integer positions (in id_order) of its neighbors.
  # This is essentially what the nb object already is,
  # but we make it explicit and clean.
  n <- length(id_order)
  lapply(seq_len(n), function(i) {
    nb_i <- neighbors[[i]]
    nb_i[nb_i > 0L]
  })
}

# ============================================================
# STEP 2: Vectorized neighbor stats via cell x year matrix
# ============================================================
compute_neighbor_features_fast <- function(cell_dt, id_order, cell_neighbor_lookup, var_name) {
  # cell_dt: data.table with columns id, year, <var_name>
  # id_order: vector of cell IDs defining row order of the matrix
  # cell_neighbor_lookup: list of length(id_order), each element = integer vector of neighbor positions
  
  n_cells <- length(id_order)
  years <- sort(unique(cell_dt$year))
  n_years <- length(years)
  
  # Create a mapping from cell id to matrix row index
  id_to_row <- setNames(seq_along(id_order), as.character(id_order))
  
  # Create a mapping from year to matrix column index
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  # Build the cell x year matrix for this variable
  # Use data.table for fast ordered access
  setkey(cell_dt, id, year)
  
  val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Vectorized fill: map each row of cell_dt to matrix position
  row_idx <- id_to_row[as.character(cell_dt$id)]
  col_idx <- year_to_col[as.character(cell_dt$year)]
  val_mat[cbind(row_idx, col_idx)] <- cell_dt[[var_name]]
  
  # Pre-compute the CSR-like structure for neighbor indexing
  # For each cell, we know its neighbors. We want, for each cell,
  # the max/min/mean of neighbor values, separately for each year (column).
  
  # Strategy: build a "neighbor row index" vector and a "source cell" grouping
  # vector, then use vectorized group operations.
  
  # Expand neighbor pairs: (cell_i, neighbor_cell_j)
  group_ids <- rep(seq_len(n_cells), times = lengths(cell_neighbor_lookup))
  neighbor_ids <- unlist(cell_neighbor_lookup, use.names = FALSE)
  
  # If a cell has no neighbors, it won't appear in group_ids â€” handle later.
  
  # For each year (column), extract neighbor values and compute grouped stats
  # This is 28 iterations (one per year), each fully vectorized over ~1.37M pairs.
  
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Pre-allocate a data.table for grouped operations
  # This avoids re-creating it 28 times
  pair_dt <- data.table(
    cell = group_ids,
    neighbor = neighbor_ids
  )
  
  for (t in seq_len(n_years)) {
    # Get neighbor values for this year
    nb_vals <- val_mat[neighbor_ids, t]
    
    pair_dt[, val := nb_vals]
    
    # Remove NAs before aggregation
    valid <- !is.na(nb_vals)
    if (sum(valid) == 0L) next
    
    stats <- pair_dt[valid, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = cell]
    
    max_mat[stats$cell, t]  <- stats$nb_max
    min_mat[stats$cell, t]  <- stats$nb_min
    mean_mat[stats$cell, t] <- stats$nb_mean
  }
  
  # Now flatten back to the original cell_dt row order
  out_max  <- max_mat[cbind(row_idx, col_idx)]
  out_min  <- min_mat[cbind(row_idx, col_idx)]
  out_mean <- mean_mat[cbind(row_idx, col_idx)]
  
  list(
    max  = out_max,
    min  = out_min,
    mean = out_mean
  )
}

# ============================================================
# STEP 3: Main pipeline (drop-in replacement for outer loop)
# ============================================================
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  cell_dt <- as.data.table(cell_data)
  
  # Step 1: cell-level neighbor lookup (344K entries, runs in seconds)
  cell_neighbor_lookup <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)
    
    stats <- compute_neighbor_features_fast(
      cell_dt, id_order, cell_neighbor_lookup, var_name
    )
    
    # Add columns with same naming convention as original code
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    
    cell_dt[, (max_col)  := stats$max]
    cell_dt[, (min_col)  := stats$min]
    cell_dt[, (mean_col) := stats$mean]
  }
  
  as.data.frame(cell_dt)
}

# ============================================================
# Usage (replaces the original outer loop):
# ============================================================
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Expected Performance Improvement

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `paste` + named-vector lookups | 344K simple integer list copies |
| **Stats computation** | 6.46M `lapply` iterations, each with subsetting | 28 vectorized `data.table` grouped aggregations over ~1.37M pairs |
| **Character operations** | ~50M+ `paste` calls | ~6.46M for matrix fill (one-time), zero during stats |
| **Estimated runtime** | 86+ hours | **~5â€“15 minutes** on 16 GB RAM laptop |
| **Peak memory** | Moderate (list of 6.46M entries) | ~1.5 GB (three 344KÃ—28 matrices + pair data.table of ~1.37M rows) |
| **Numerical output** | max, min, mean per cell-year | Identical max, min, mean per cell-year |
| **RF model** | Unchanged | Unchanged (not retrained) |

**Why this works:** The fundamental insight is that the spatial topology is static across years. The original code redundantly reconstructed and string-matched neighbor relationships for every cell-year row (6.46M times). By factoring the problem into a cell-level topology (344K) crossed with a year loop (28), and using matrix indexing plus `data.table` grouped aggregation, we eliminate virtually all interpreted-loop and string-operation overhead. The numerical results (max, min, mean of neighbor values) are identical, preserving the trained Random Forest model's expected input features.
END_RESPONSE>>>
