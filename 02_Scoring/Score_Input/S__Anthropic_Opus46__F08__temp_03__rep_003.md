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
run_id: Anthropic_F08_T03_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F08_Anthropic_temp_03_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance bottleneck is in `build_neighbor_lookup`. It constructs a **row-level** lookup of length ~6.46 million, where each entry maps a cell-year row to the row indices of its neighbors *in that same year*. But the neighbor topology is **static** â€” it depends only on spatial cell identity, not on year. The current code ignores this invariance and redundantly recomputes the same neighbor cell-ID sets 28 times (once per year per cell), and performs ~6.46 million string-paste + hash-lookup operations to resolve year-specific row indices.

Specifically:

1. **`build_neighbor_lookup`** creates ~6.46M string keys (`id_year`), builds a named lookup vector of the same length, then for each of the 6.46M rows, pastes neighbor IDs with the row's year and looks them up. This is O(NÃ—k) string operations where N â‰ˆ 6.46M and k â‰ˆ average neighbor count (~4 for rook). The named-vector lookup on character keys is slow in R (hash-based but with overhead per call). Estimated: billions of character operations â†’ 86+ hours.

2. **`compute_neighbor_stats`** is called 5 times (once per variable) and iterates over the 6.46M-length list. Each call is O(NÃ—k). This part is comparatively fast once the lookup exists, but the list-of-vectors structure is memory-heavy.

3. The entire design treats the problem as a flat row-level operation, missing the **separability** between the static spatial graph and the year-varying data.

## Optimization Strategy

**Key insight:** Factor the computation into:

- **Static (compute once):** A cell-level neighbor index map â€” for each of the 344,208 cells, store which other cells are its neighbors. This is just the `rook_neighbors_unique` nb object itself (or a cleaned integer-vector version).
- **Dynamic (compute per year):** For each year, extract the column of variable values for all cells in that year, then use the static cell-level neighbor map to compute max/min/mean via vectorized matrix indexing.

**Concrete plan:**

1. Ensure `cell_data` is sorted by `(year, id)` with a consistent cell ordering within each year. This lets us use a simple integer matrix (344,208 rows Ã— 28 columns) for each variable, where row = cell index, column = year index.
2. Convert the `nb` object to a padded neighbor-index matrix (344,208 Ã— max_neighbors), enabling fully vectorized row-subsetting.
3. For each variable, build the cellÃ—year matrix, then for each year-column, gather neighbor values via matrix indexing and compute max/min/mean with vectorized operations â€” no R-level loops over 6.46M rows.

**Expected speedup:** From ~86 hours to **minutes**. The dominant operation becomes matrix indexing and `rowMeans`/`pmax`/`pmin` over ~344K cells Ã— 28 years Ã— 5 variables â€” all vectorized C-level operations in R.

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR STATS COMPUTATION
# Exploits: static neighbor topology + year-varying variables
# Preserves: trained RF model, original numerical estimand
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {
  
  # -------------------------------------------------------------------------
  # STEP 0: Convert to data.table for fast manipulation
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # -------------------------------------------------------------------------
  # STEP 1: BUILD STATIC CELL-LEVEL NEIGHBOR STRUCTURE (done once)
  # -------------------------------------------------------------------------
  # id_order: vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique: nb object (list of integer vectors indexing into id_order)
  
  n_cells <- length(id_order)
  
  # Determine max number of neighbors (for rook on a grid, typically â‰¤ 4)
  max_k <- max(vapply(rook_neighbors_unique, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  # Build padded neighbor-index matrix: n_cells Ã— max_k

  # Each row i contains the cell-order indices of neighbors of cell i,

  # padded with NA

  nb_mat <- matrix(NA_integer_, nrow = n_cells, ncol = max_k)
  nb_count <- integer(n_cells)  # number of actual neighbors per cell
  
  for (i in seq_len(n_cells)) {
    nbrs <- rook_neighbors_unique[[i]]
    # spdep::nb uses 0 to indicate no neighbors
    if (length(nbrs) == 1L && nbrs[1] == 0L) {
      nb_count[i] <- 0L
    } else {
      nb_count[i] <- length(nbrs)
      nb_mat[i, seq_along(nbrs)] <- nbrs
    }
  }
  
  cat("Static neighbor matrix built:", n_cells, "cells, max", max_k, "neighbors\n")
  
  # -------------------------------------------------------------------------
  # STEP 2: ESTABLISH CONSISTENT CELL ORDERING WITHIN EACH YEAR
  # -------------------------------------------------------------------------
  # Create a cell-index column: maps each cell ID to its position in id_order
  id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))
  dt[, cell_idx := id_to_idx[as.character(id)]]
  
  # Get sorted unique years
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_len(n_years), as.character(years))
  dt[, year_idx := year_to_col[as.character(year)]]
  
  # Sort for consistent matrix filling
  setkey(dt, cell_idx, year_idx)
  
  # Verify completeness: we expect a balanced panel (n_cells Ã— n_years)
  # If unbalanced, we handle via the matrix approach (NAs for missing cell-years)
  expected_rows <- n_cells * n_years
  is_balanced <- (nrow(dt) == expected_rows)
  if (!is_balanced) {
    cat("Panel is unbalanced. Using safe indexing.\n")
  } else {
    cat("Panel is balanced:", n_cells, "cells Ã—", n_years, "years =", expected_rows, "rows\n")
  }
  
  # -------------------------------------------------------------------------
  # STEP 3: FOR EACH VARIABLE, COMPUTE NEIGHBOR MAX, MIN, MEAN
  # -------------------------------------------------------------------------
  # Strategy: build a cell Ã— year matrix for the variable, then for each year
  # use the static nb_mat to gather neighbor values and compute stats vectorially.
  
  # Pre-allocate output columns in dt
  for (var_name in neighbor_source_vars) {
    max_col <- paste0("neighbor_max_", var_name)
    min_col <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    dt[, (max_col) := NA_real_]
    dt[, (min_col) := NA_real_]
    dt[, (mean_col) := NA_real_]
  }
  
  # Helper: compute neighbor stats for one variable using vectorized matrix ops
  compute_neighbor_stats_fast <- function(dt, var_name, nb_mat, nb_count, 
                                          n_cells, years, year_to_col, max_k) {
    
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    # Build cell Ã— year value matrix
    val_mat <- matrix(NA_real_, nrow = n_cells, ncol = length(years))
    
    # Fill the matrix from dt (which is keyed by cell_idx, year_idx)
    val_mat[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]
    
    # For each year, compute neighbor stats
    # We process one year at a time to keep memory bounded
    for (yr in years) {
      y_col <- year_to_col[as.character(yr)]
      
      # Current year's values for all cells: length n_cells
      v <- val_mat[, y_col]
      
      # Gather neighbor values into an n_cells Ã— max_k matrix
      # nb_mat[i, j] gives the cell index of the j-th neighbor of cell i
      # v[nb_mat] gives the corresponding values (NA where nb_mat is NA)
      neighbor_vals <- matrix(v[nb_mat], nrow = n_cells, ncol = max_k)
      # Cells with no neighbors: nb_mat row is all NA â†’ neighbor_vals row is all NA â†’ stats = NA
      
      # Compute row-wise max, min, mean ignoring NAs
      # Use matrixStats for speed if available, otherwise base R
      
      # Count non-NA per row
      not_na <- !is.na(neighbor_vals)
      row_n <- rowSums(not_na)
      
      # Mean: rowSums / count
      row_sum <- rowSums(neighbor_vals, na.rm = TRUE)
      n_mean <- ifelse(row_n > 0L, row_sum / row_n, NA_real_)
      
      # Max and Min: use suppressWarnings to handle all-NA rows
      n_max <- suppressWarnings(do.call(pmax, c(as.data.frame(neighbor_vals), na.rm = TRUE)))
      n_min <- suppressWarnings(do.call(pmin, c(as.data.frame(neighbor_vals), na.rm = TRUE)))
      # pmax/pmin return -Inf/Inf for all-NA rows; fix those
      n_max[row_n == 0L] <- NA_real_
      n_min[row_n == 0L] <- NA_real_
      # Also handle Inf/-Inf from all-NA (shouldn't happen with na.rm but be safe)
      n_max[is.infinite(n_max)] <- NA_real_
      n_min[is.infinite(n_min)] <- NA_real_
      
      # Write results back into dt for rows matching this year
      # Rows in dt for this year: cell_idx gives the cell, year_idx == y_col
      row_mask <- dt$year_idx == y_col
      cell_indices_this_year <- dt$cell_idx[row_mask]
      
      set(dt, which(row_mask), max_col,  n_max[cell_indices_this_year])
      set(dt, which(row_mask), min_col,  n_min[cell_indices_this_year])
      set(dt, which(row_mask), mean_col, n_mean[cell_indices_this_year])
    }
    
    invisible(NULL)
  }
  
  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor stats for:", var_name, "...")
    t0 <- proc.time()
    compute_neighbor_stats_fast(dt, var_name, nb_mat, nb_count,
                                n_cells, years, year_to_col, max_k)
    elapsed <- (proc.time() - t0)[3]
    cat(" done in", round(elapsed, 1), "seconds\n")
  }
  
  # -------------------------------------------------------------------------
  # STEP 4: CLEAN UP AND RETURN
  # -------------------------------------------------------------------------
  dt[, c("cell_idx", "year_idx") := NULL]
  
  # Return as data.frame if input was data.frame, else data.table
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}


# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================

# Original:
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }

# Replacement:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique, 
  neighbor_source_vars
)

# Then proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor topology** | Recomputed per row (6.46M string lookups) | Computed once as integer matrix (344K cells) |
| **Per-variable stats** | R-level `lapply` over 6.46M list entries | Vectorized matrix indexing + `pmax`/`pmin`/`rowSums` per year-slice (344K Ã— max_k) |
| **Memory: lookup** | 6.46M-element list of integer vectors | 344K Ã— 4 integer matrix (~5.5 MB) |
| **Time complexity** | ~O(N Ã— k) with slow string ops, N=6.46M | ~O(C Ã— k Ã— Y Ã— V) with fast vectorized ops, C=344K, Y=28, V=5 |
| **Estimated runtime** | 86+ hours | **~5â€“15 minutes** |
| **RF model** | Untouched âœ“ | Untouched âœ“ |
| **Output columns** | `neighbor_max_*`, `neighbor_min_*`, `neighbor_mean_*` | Identical column names and identical numerical values âœ“ |

The key architectural insight: **separate the static spatial graph (compute once, store as integer matrix) from the dynamic year-varying values (process per year-slice with vectorized operations)**. This eliminates all string manipulation, eliminates the 6.46M-element list, and replaces R-level loops with C-level vectorized matrix operations.
END_RESPONSE>>>
