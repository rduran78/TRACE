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
run_id: Anthropic_F08_T10_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) neighbor indices.** It iterates over all ~6.46 million rows, performing string-pasting and hash lookups for every row. But the neighbor *relationships* among cells are **time-invariant** â€” cell A is a rook-neighbor of cell B regardless of year. The lookup should be built once over 344,208 cells, not over 6.46 million cell-year rows.

2. **`compute_neighbor_stats` operates on the full stacked data frame row-by-row.** It calls an `lapply` over 6.46M elements, each of which subsets a vector by index. While each individual operation is small, the sheer volume of R-level loop iterations is crushing.

3. **String key construction (`paste(id, year, sep="_")`) and named-vector lookup (`idx_lookup[neighbor_keys]`)** are O(n) in memory and slow in R for millions of entries. This is entirely unnecessary since the neighbor topology doesn't change.

4. **The data layout is row-stacked (long panel).** To exploit the static topology, we should reshape to a **cell Ã— year matrix** for each variable, apply neighbor aggregation as matrix operations over the cell dimension (using the static neighbor list), and then slot results back.

### Why it matters

| Aspect | Current | Proposed |
|---|---|---|
| Neighbor lookup size | 6.46M entries | 344,208 entries (one per cell) |
| Stats computation loops | 6.46M Ã— 5 vars = 32.3M iterations | 344,208 Ã— 5 vars = 1.72M iterations (vectorized across years) |
| String hashing | ~19M `paste` + named lookups | Zero |
| Memory pattern | Random access into 6.46M-row frame | Column-wise matrix ops (cache-friendly) |

---

## Optimization Strategy

**Core insight:** Separate the *static graph structure* (which cells are neighbors) from the *dynamic variable values* (which change by year). Then vectorize the year dimension.

### Steps

1. **Build a cell-level neighbor lookup once** â€” a simple list of length 344,208, where element `i` contains the integer indices of cell `i`'s neighbors. This is just `rook_neighbors_unique` itself (an `nb` object), possibly re-indexed.

2. **For each variable, reshape to a matrix of dimension `n_cells Ã— n_years`.** Each column is one year's values for all cells.

3. **Compute neighbor max/min/mean as matrix operations.** For each cell `i`, gather its neighbors' rows from the matrix, and compute columnwise (i.e., year-wise) max, min, mean. This turns 6.46M R-level iterations into 344,208 iterations, each of which is a small vectorized matrix operation across 28 years simultaneously.

4. **Melt the result matrices back to long format** and join onto the original data frame.

5. **Pass to the pre-trained Random Forest unchanged** â€” the numerical results are identical (same max, min, mean of the same neighbor values), preserving the original estimand.

### Expected speedup

- Loop iterations reduced by factor of ~18.8Ã— (28 years collapsed).
- Each iteration is vectorized (matrix column ops) instead of scalar.
- No string operations. No hash lookups.
- Estimated runtime: **~2â€“10 minutes** instead of 86+ hours.

---

## Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Exploits the static-topology / dynamic-variable distinction
# ==============================================================================

compute_all_neighbor_features <- function(cell_data, 
                                           id_order, 
                                           rook_neighbors_unique, 
                                           neighbor_source_vars) {
  # ------------------------------------------------------------------
  # STEP 1: Establish stable cell ordering and year ordering

  # ------------------------------------------------------------------
  
  # id_order is the vector of cell IDs in the same order as rook_neighbors_unique.
  # We create a mapping from cell ID -> position index in id_order.
  n_cells <- length(id_order)
  id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))
  
  # Identify unique years and sort them
  years <- sort(unique(cell_data$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_len(n_years), as.character(years))
  
  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(cell_data)))
  
  # ------------------------------------------------------------------
  # STEP 2: Compute row position of each cell_data row in the 
  #         (cell_pos, year_col) matrix layout
  #
  #   We need this to (a) fill matrices from long data, and 
  #   (b) write results back to the correct rows.
  # ------------------------------------------------------------------
  
  cell_data_pos <- id_to_pos[as.character(cell_data$id)]   # cell position for each row
  cell_data_col <- year_to_col[as.character(cell_data$year)] # year column for each row
  
  # Linear index into an n_cells x n_years matrix (column-major)
  linear_idx <- cell_data_pos + (cell_data_col - 1L) * n_cells
  
  # ------------------------------------------------------------------
  # STEP 3: The static neighbor list is rook_neighbors_unique itself.
  #         It is already a list of length n_cells where element [[i]]

  #         gives the indices (into id_order) of cell i's neighbors.
  #         spdep::nb objects use integer index vectors. We just use it
  #         directly, filtering out the 0-neighbor sentinel if present.
  # ------------------------------------------------------------------
  
  # spdep nb objects encode "no neighbors" as a single integer 0.
  # We convert to a clean list of integer vectors.
  nb_list <- lapply(rook_neighbors_unique, function(x) {
    x <- as.integer(x)
    x[x > 0L]
  })
  
  # ------------------------------------------------------------------
  # STEP 4: For each variable, build matrix, compute neighbor stats,
  #         and write results back to cell_data
  # ------------------------------------------------------------------
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing neighbor stats for: %s\n", var_name))
    
    # 4a. Build n_cells x n_years matrix from long data
    vals_vec <- cell_data[[var_name]]
    mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mat[linear_idx] <- vals_vec
    
    # 4b. Compute neighbor max, min, mean â€” loop over cells only (344K),
    #     vectorized across years (28 columns at once)
    
    nb_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    for (i in seq_len(n_cells)) {
      nbrs <- nb_list[[i]]
      if (length(nbrs) == 0L) next
      
      # Submatrix: rows = neighbors, cols = years
      # For a typical rook neighborhood, this is 2-4 rows Ã— 28 cols â€” tiny.
      nb_mat <- mat[nbrs, , drop = FALSE]
      
      if (nrow(nb_mat) == 1L) {
        # Single neighbor: stats are trivial
        nb_max[i, ]  <- nb_mat[1L, ]
        nb_min[i, ]  <- nb_mat[1L, ]
        nb_mean[i, ] <- nb_mat[1L, ]
      } else {
        # Multiple neighbors: column-wise operations
        # Handle NAs the same way as the original code (remove NAs per group)
        # colMeans/colMaxs etc. â€” we use matrixStats if available, else base
        
        # Using base R for portability:
        nb_max[i, ]  <- apply(nb_mat, 2L, function(v) {
          v <- v[!is.na(v)]
          if (length(v) == 0L) NA_real_ else max(v)
        })
        nb_min[i, ]  <- apply(nb_mat, 2L, function(v) {
          v <- v[!is.na(v)]
          if (length(v) == 0L) NA_real_ else min(v)
        })
        nb_mean[i, ] <- apply(nb_mat, 2L, function(v) {
          v <- v[!is.na(v)]
          if (length(v) == 0L) NA_real_ else mean(v)
        })
      }
    }
    
    # 4c. Write results back to cell_data in the correct row positions
    max_col_name  <- paste0(var_name, "_neighbor_max")
    min_col_name  <- paste0(var_name, "_neighbor_min")
    mean_col_name <- paste0(var_name, "_neighbor_mean")
    
    cell_data[[max_col_name]]  <- nb_max[linear_idx]
    cell_data[[min_col_name]]  <- nb_min[linear_idx]
    cell_data[[mean_col_name]] <- nb_mean[linear_idx]
    
    cat(sprintf("    -> Added %s, %s, %s\n", max_col_name, min_col_name, mean_col_name))
  }
  
  return(cell_data)
}

# ==============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The pre-trained Random Forest model is used unchanged downstream:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Further Acceleration (Optional): Use `matrixStats` to Eliminate Inner `apply`

The `apply(..., 2, ...)` inside the cell loop processes only 2â€“4 rows Ã— 28 columns, so it's already fast. But if you want to squeeze out more performance, replace the inner apply calls with `matrixStats::colMaxs`, `colMins`, `colMeans2`:

```r
# Install once: install.packages("matrixStats")
library(matrixStats)

# Replace the multi-neighbor branch with:
if (nrow(nb_mat) == 1L) {
  nb_max[i, ]  <- nb_mat[1L, ]
  nb_min[i, ]  <- nb_mat[1L, ]
  nb_mean[i, ] <- nb_mat[1L, ]
} else {
  nb_max[i, ]  <- colMaxs(nb_mat,  na.rm = TRUE)
  nb_min[i, ]  <- colMins(nb_mat,  na.rm = TRUE)
  nb_mean[i, ] <- colMeans2(nb_mat, na.rm = TRUE)
}
# Note: colMaxs/colMins return -Inf/Inf for all-NA columns.
# Post-process if needed:
# nb_max[i, ][is.infinite(nb_max[i, ])] <- NA_real_
# nb_min[i, ][is.infinite(nb_min[i, ])] <- NA_real_
```

This version with `matrixStats` is the fastest and should bring total runtime to **under 5 minutes** on a 16 GB laptop.

---

## Summary

| Dimension | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-key entries | 344K integer-index entries (reuse `nb` directly) |
| **Main loop iterations** | 6.46M Ã— 5 = 32.3M | 344K Ã— 5 = 1.72M |
| **Per-iteration work** | Scalar (1 year) | Vectorized (28 years simultaneously) |
| **String operations** | ~19M `paste` calls | Zero |
| **Hash table lookups** | ~19M named-vector lookups | Zero |
| **Estimated runtime** | 86+ hours | ~2â€“10 minutes |
| **Numerical output** | Original estimand | **Preserved identically** |
| **RF model** | Pre-trained | **Unchanged, not retrained** |
END_RESPONSE>>>
