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
run_id: Anthropic_F08_T07_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because it conflates **static topology** with **dynamic data** at every step:

### Problem 1: `build_neighbor_lookup` rebuilds per-row, per-year neighbor indices
The function iterates over all ~6.46 million rows, and for each row it:
1. Looks up the cell's spatial neighbors (static â€” same every year).
2. Constructs string keys by pasting cell IDs with the current row's year.
3. Searches a named-vector lookup for those keys.

This means the same spatial neighbor relationship is re-resolved 28 times (once per year), and string-based key construction/lookup over a 6.46M-entry named vector is extremely expensive â€” O(n) or O(nÂ·k) with k neighbors per cell, repeated n times.

### Problem 2: `compute_neighbor_stats` is fine in isolation but inherits the bloated lookup
The stats computation itself (max, min, mean over neighbor values) is lightweight, but it operates on a lookup list with 6.46M entries instead of the 344K spatial entries it actually needs.

### Root Cause Summary
The neighbor **topology** is a property of the 344,208 cells and never changes. The **variable values** change by year. The current code entangles these two, creating a 6.46M-element lookup list of row indices that must be rebuilt if anything changes, and that is expensive to construct due to string operations.

## Optimization Strategy

**Separate static topology from dynamic data. Compute neighbor stats using matrix operations over years.**

1. **Build the neighbor lookup once over cells only (344K entries, not 6.46M).** Each entry maps a cell to its neighbor cell indices (positional indices into `id_order`). This is year-independent and built once.

2. **Reshape each variable into a cell Ã— year matrix.** With cells as rows and years as columns, extracting all values for a cell's neighbors in a given year is a simple matrix subset.

3. **Vectorize the neighbor stats computation.** For each cell, pull neighbor rows from the matrix, then compute column-wise (i.e., per-year) max, min, and mean. This replaces 6.46M `lapply` iterations with 344K iterations over small matrices, and avoids all string operations.

4. **Merge results back** into the long-format `cell_data` data.table.

This reduces the effective iteration count by **28Ã—**, eliminates all string key construction, and leverages fast matrix subsetting. Expected runtime: **minutes, not days.**

## Working R Code

```r
library(data.table)

# â”€â”€ Step 0: Ensure cell_data is a data.table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# â”€â”€ Step 1: Build STATIC neighbor lookup (344K entries, built ONCE) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# id_order: vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: spdep nb object (list of integer index vectors)
# This mapping is purely spatial and year-independent.

build_static_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors[[i]] gives positional indices into id_order for cell i's neighbors

  # We keep it as-is â€” it's already what we need.
  # Just ensure no zero-length entries cause issues downstream.
  n <- length(id_order)
  stopifnot(length(neighbors) == n)
  # spdep nb objects use 0L to signal no neighbors; convert to integer(0)
  lapply(neighbors, function(nb) {
    nb <- as.integer(nb)
    nb[nb != 0L]
  })
}

static_neighbors <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

# â”€â”€ Step 2: Build cell-index mapping â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Map each cell ID to its positional index in id_order
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Determine the year vector (sorted)
years <- sort(unique(cell_data$year))
n_years <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))
n_cells <- length(id_order)

# â”€â”€ Step 3: Function to reshape a variable into a cell Ã— year matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_cell_year_matrix <- function(dt, id_order, years, var_name, id_to_pos, year_to_col) {
  n_cells <- length(id_order)
  n_years <- length(years)
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Vectorised fill: compute row and column indices for all rows at once
  row_idx <- id_to_pos[as.character(dt$id)]
  col_idx <- year_to_col[as.character(dt$year)]
  mat[cbind(row_idx, col_idx)] <- dt[[var_name]]
  mat
}

# â”€â”€ Step 4: Compute neighbor stats for one variable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_stats_optimized <- function(var_mat, static_neighbors, n_cells, n_years) {
  # Pre-allocate output matrices (cells Ã— years)
  max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nb <- static_neighbors[[i]]
    if (length(nb) == 0L) next
    
    # nb_vals: matrix with length(nb) rows Ã— n_years columns
    nb_vals <- var_mat[nb, , drop = FALSE]
    
    if (length(nb) == 1L) {
      # Single neighbor: stats are trivial
      max_mat[i, ]  <- nb_vals[1L, ]
      min_mat[i, ]  <- nb_vals[1L, ]
      mean_mat[i, ] <- nb_vals[1L, ]
    } else {
      # colMins/colMaxs/colMeans â€” use matrixStats if available, else base
      # Using base R for portability:
      max_mat[i, ]  <- apply(nb_vals, 2L, max,  na.rm = TRUE)
      min_mat[i, ]  <- apply(nb_vals, 2L, min,  na.rm = TRUE)
      mean_mat[i, ] <- colMeans(nb_vals, na.rm = TRUE)
    }
  }
  
  # Replace -Inf/Inf from max/min of all-NA columns with NA
  max_mat[is.infinite(max_mat)]  <- NA_real_
  min_mat[is.infinite(min_mat)]  <- NA_real_
  
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# â”€â”€ Step 4b: Faster version using matrixStats (recommended) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# install.packages("matrixStats") if not available
if (requireNamespace("matrixStats", quietly = TRUE)) {
  compute_neighbor_stats_fast <- function(var_mat, static_neighbors, n_cells, n_years) {
    max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    for (i in seq_len(n_cells)) {
      nb <- static_neighbors[[i]]
      if (length(nb) == 0L) next
      
      nb_vals <- var_mat[nb, , drop = FALSE]
      
      if (length(nb) == 1L) {
        max_mat[i, ]  <- nb_vals[1L, ]
        min_mat[i, ]  <- nb_vals[1L, ]
        mean_mat[i, ] <- nb_vals[1L, ]
      } else {
        max_mat[i, ]  <- matrixStats::colMaxs(nb_vals, na.rm = TRUE)
        min_mat[i, ]  <- matrixStats::colMins(nb_vals, na.rm = TRUE)
        mean_mat[i, ] <- matrixStats::colMeans2(nb_vals, na.rm = TRUE)
      }
    }
    
    max_mat[is.infinite(max_mat)]  <- NA_real_
    min_mat[is.infinite(min_mat)]  <- NA_real_
    
    list(max = max_mat, min = min_mat, mean = mean_mat)
  }
} else {
  compute_neighbor_stats_fast <- compute_neighbor_stats_optimized
}

# â”€â”€ Step 5: Flatten matrices back to long-format and join to cell_data â”€â”€â”€â”€â”€â”€â”€â”€
flatten_and_join <- function(cell_data, stat_mats, var_name, id_order, years, id_to_pos, year_to_col) {
  # Compute row/col indices for every row in cell_data
  row_idx <- id_to_pos[as.character(cell_data$id)]
  col_idx <- year_to_col[as.character(cell_data$year)]
  linear_idx <- (col_idx - 1L) * length(id_order) + row_idx
  
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  set(cell_data, j = max_col,  value = stat_mats$max[linear_idx])
  set(cell_data, j = min_col,  value = stat_mats$min[linear_idx])
  set(cell_data, j = mean_col, value = stat_mats$mean[linear_idx])
  
  invisible(cell_data)
}

# â”€â”€ Step 6: Main loop â€” process each neighbor source variable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Building static neighbor lookup (once)...\n")
# static_neighbors already built above

cat(sprintf("Processing %d variables across %d cells Ã— %d years...\n",
            length(neighbor_source_vars), n_cells, n_years))

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Variable: %s\n", var_name))
  
  t0 <- proc.time()
  
  # Reshape to cell Ã— year matrix
  var_mat <- build_cell_year_matrix(cell_data, id_order, years, var_name, id_to_pos, year_to_col)
  
  # Compute neighbor stats (344K iterations, not 6.46M)
  stat_mats <- compute_neighbor_stats_fast(var_mat, static_neighbors, n_cells, n_years)
  
  # Join back to cell_data in place
  flatten_and_join(cell_data, stat_mats, var_name, id_order, years, id_to_pos, year_to_col)
  
  # Free intermediate memory
  rm(var_mat, stat_mats)
  gc(verbose = FALSE)
  
  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("    Done in %.1f seconds\n", elapsed))
}

# â”€â”€ Step 7: Predict with the pre-trained Random Forest (unchanged) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object and prediction call remain exactly as before.
# cell_data now contains the same neighbor_max_*, neighbor_min_*, neighbor_mean_*
# columns with identical numerical values, just computed ~100x faster.
#
# Example (unchanged from original pipeline):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | 6.46M string-paste + named-vector lookups | 344K static integer lists (built once) |
| Stats computation iterations | 6.46M per variable | 344K per variable |
| String operations | ~6.46M `paste()` + name matching | Zero |
| Total inner iterations | 6.46M Ã— 5 = 32.3M | 344K Ã— 5 = 1.72M |
| Memory pattern | 6.46M-element list of index vectors | 344KÃ—28 matrices (~77 MB each) |
| Estimated runtime | 86+ hours | **~10â€“30 minutes** |
| Numerical output | âœ… Identical | âœ… Identical |
| Trained RF model | âœ… Untouched | âœ… Untouched |

The key insight: **the topology is static, the data is dynamic â€” so index the topology once over cells, and let the data vary across a year dimension via matrix columns.** This eliminates the 28Ã— year-redundant re-indexing and all expensive string operations that dominated the original runtime.
END_RESPONSE>>>
