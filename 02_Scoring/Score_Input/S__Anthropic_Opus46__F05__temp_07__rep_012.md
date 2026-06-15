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
run_id: Anthropic_F05_T07_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Local Inefficiency
Inside `build_neighbor_lookup`, the `idx_lookup` named vector (6.46M entries) is created once, but then `paste()` is called **per row** inside `lapply` over 6.46M rows. Each call to `paste()` and the subsequent named-vector lookup (`idx_lookup[neighbor_keys]`) involves string construction, hashing, and matching â€” repeated 6.46M times. With an average of ~4 rook neighbors per cell, that's ~25.8M string constructions and lookups just in this function.

### Broader Algorithmic Problem
More critically, the **entire design is row-centric (long format)** when it should be **year-sliced and integer-indexed**. The key insight is:

1. **Neighbor topology is time-invariant.** Cell A's neighbors are the same in 1992 and 2019. The `nb` object already encodes this as integer indices into the spatial ID vector.
2. **Lookups should be integer-based, not string-based.** Within a single year, the neighbor row indices can be computed by a simple offset from the spatial index â€” no strings needed.
3. **`compute_neighbor_stats` is already vectorized over the lookup list**, but it's called 5 times (once per variable) over the same 6.46M-entry list. The neighbor gather can be done once per variable using matrix operations on year-slices.

The overall complexity of the current approach:
- `build_neighbor_lookup`: O(N_rows Ã— avg_neighbors) string operations â‰ˆ 25.8M string ops
- `compute_neighbor_stats`: Called 5 times, each iterating 6.46M `lapply` calls
- **Total**: ~32M string ops + ~32M R-level list iterations â€” all in interpreted R with no vectorization.

This is why the estimate is 86+ hours.

## Optimization Strategy

1. **Eliminate all string keys.** Work entirely with integer indices.
2. **Slice by year.** Within each year, all 344,208 cells share the same spatial ordering. Build a spatial-index-to-row-offset map once per year (or better, ensure consistent ordering so it's a trivial arithmetic offset).
3. **Vectorize neighbor aggregation using sparse matrices.** Construct a single sparse adjacency matrix (344,208 Ã— 344,208) from the `nb` object. For each year-slice, extract the variable column as a vector and compute `W %*% x` (sum), `rowSums(W != 0)` (count), and use sparse-matrix tricks for min/max â€” or use a grouped approach with `data.table`.
4. **Compute all 5 variables' stats in one pass per year** (or via matrix multiplication across all years at once).

### Expected Speedup
- Sparse matrixâ€“vector multiply for 344K Ã— 344K with ~1.37M nonzeros: milliseconds.
- 28 years Ã— 5 variables Ã— 3 stats = 420 sparse matrix operations, each taking milliseconds.
- **Total: seconds to low minutes** vs. 86+ hours.

## Working R Code

```r
library(Matrix)
library(data.table)

#' Build a sparse row-normalized (or raw) adjacency matrix from an nb object.
#' Returns a dgCMatrix of dimension n_cells x n_cells with 1s for neighbor links.
build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj is a list of length n; nb_obj[[i]] contains integer indices of neighbors of cell i
  # Build COO triplets
  i_idx <- integer(0)
  j_idx <- integer(0)
  
  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    # spdep::nb objects use 0L to indicate no neighbors
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0L) {
      i_idx <- c(i_idx, rep(i, length(nbrs)))
      j_idx <- c(j_idx, nbrs)
    }
  }
  
  sparseMatrix(
    i = i_idx, j = j_idx, x = rep(1, length(i_idx)),
    dims = c(n, n)
  )
}

#' Compute neighbor max, min, mean for a numeric vector x given adjacency matrix W.
#' Returns a 3-column matrix: [max, min, mean], length = length(x).
#' 
#' Strategy: 
#'   - mean = (W %*% x) / (W %*% 1)  (where 1 is a vector of ones, adjusted for NAs)
#'   - For min/max, we use a grouped operation via the sparse structure.
compute_neighbor_stats_sparse <- function(W, x) {
  n <- length(x)
  
  # Handle NAs: create a version of x where NA -> 0 for sum, and a mask
  not_na <- as.numeric(!is.na(x))
  x_safe <- ifelse(is.na(x), 0, x)
  
  # Neighbor count (excluding NAs)
  neighbor_count <- as.numeric(W %*% not_na)
  
  # Neighbor sum (excluding NAs)
  neighbor_sum <- as.numeric(W %*% x_safe)
  
  # Mean
  n_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
  
  # For min and max, we need the actual neighbor values.
  # Sparse matrix approach: iterate over rows of W using its CSC/CSR structure.
  # Convert W to dgRMatrix (row-sparse) for efficient row access, or use dgCMatrix columns of t(W).
  # 
  # More efficient: use data.table on the COO representation.
  
  # Extract COO from W
  W_T <- summary(W)  # gives (i, j, x) triplets
  # W_T$i is the "focal" cell, W_T$j is the "neighbor" cell
  
  dt <- data.table(
    focal = W_T$i,
    neighbor = W_T$j
  )
  
  # Attach neighbor values
  dt[, val := x[neighbor]]
  
  # Remove NA neighbor values
  dt <- dt[!is.na(val)]
  
  # Compute grouped min and max
  agg <- dt[, .(nmax = max(val), nmin = min(val)), by = focal]
  
  # Map back to full vector
  n_max <- rep(NA_real_, n)
  n_min <- rep(NA_real_, n)
  n_max[agg$focal] <- agg$nmax
  n_min[agg$focal] <- agg$nmin
  
  cbind(n_max, n_min, n_mean)
}

#' Main pipeline: compute all neighbor features for the panel dataset.
#' 
#' @param cell_data   data.frame/data.table with columns: id, year, and all var columns.
#'                    Must contain all 6.46M cell-year rows.
#' @param id_order    integer vector of cell IDs in the order matching the nb object.
#'                    Length = 344,208.
#' @param nb_obj      spdep::nb object (rook_neighbors_unique). Length = 344,208.
#' @param neighbor_source_vars character vector of variable names.
#' @return cell_data with new neighbor feature columns appended.
add_all_neighbor_features <- function(cell_data, id_order, nb_obj, neighbor_source_vars) {
  
  n_cells <- length(id_order)
  
  message("Building sparse adjacency matrix...")
  W <- build_adjacency_matrix(nb_obj, n_cells)
  
  # Create mapping from cell id to spatial index (position in id_order)
  id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Convert to data.table for efficiency
  was_df <- !is.data.table(cell_data)
  if (was_df) cell_data <- as.data.table(cell_data)
  
  # Add spatial index column
  cell_data[, spatial_idx := id_to_spatial_idx[as.character(id)]]
  
  # Pre-extract COO triplets once (for min/max computation)
  W_summary <- summary(W)
  coo_dt <- data.table(focal = W_summary$i, neighbor = W_summary$j)
  
  # Precompute the "not-NA neighbor count" denominator helper
  # W %*% not_na per year â€” we need this per variable per year
  
  years <- sort(unique(cell_data$year))
  
  # Pre-allocate result columns
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    cell_data[, (col_max)  := NA_real_]
    cell_data[, (col_min)  := NA_real_]
    cell_data[, (col_mean) := NA_real_]
  }
  
  message("Processing ", length(years), " years x ", length(neighbor_source_vars), " variables...")
  
  for (yr in years) {
    # Extract the year-slice, ordered by spatial index
    year_mask <- cell_data$year == yr
    year_data <- cell_data[year_mask]
    
    # Ensure we have a full spatial grid for this year.
    # Build a vector of length n_cells, indexed by spatial_idx.
    # If some cells are missing for a year, they'll be NA.
    
    setkey(year_data, spatial_idx)
    spatial_indices_present <- year_data$spatial_idx
    
    for (var_name in neighbor_source_vars) {
      # Build full-length spatial vector (NA for missing cells)
      x_full <- rep(NA_real_, n_cells)
      x_full[spatial_indices_present] <- year_data[[var_name]]
      
      # --- Neighbor mean via sparse matrix multiplication ---
      not_na <- as.numeric(!is.na(x_full))
      x_safe <- ifelse(is.na(x_full), 0, x_full)
      
      neighbor_count <- as.numeric(W %*% not_na)
      neighbor_sum   <- as.numeric(W %*% x_safe)
      
      n_mean_full <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
      
      # --- Neighbor min/max via data.table grouped aggregation ---
      # Attach values to COO neighbor list
      coo_dt[, val := x_full[neighbor]]
      agg <- coo_dt[!is.na(val), .(nmax = max(val), nmin = min(val)), by = focal]
      
      n_max_full <- rep(NA_real_, n_cells)
      n_min_full <- rep(NA_real_, n_cells)
      n_max_full[agg$focal] <- agg$nmax
      n_min_full[agg$focal] <- agg$nmin
      
      # --- Write results back into cell_data for this year-slice ---
      col_max  <- paste0("neighbor_max_", var_name)
      col_min  <- paste0("neighbor_min_", var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      
      # Map from spatial_idx back to the rows in cell_data
      cell_data[year_mask, (col_max)  := n_max_full[spatial_idx]]
      cell_data[year_mask, (col_min)  := n_min_full[spatial_idx]]
      cell_data[year_mask, (col_mean) := n_mean_full[spatial_idx]]
    }
    
    message("  Year ", yr, " done.")
  }
  
  # Clean up helper column
  cell_data[, spatial_idx := NULL]
  
  if (was_df) cell_data <- as.data.frame(cell_data)
  
  return(cell_data)
}

# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================

# Original code (86+ hours):
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# Optimized replacement (minutes):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique, 
  neighbor_source_vars
)

# The trained Random Forest model is untouched â€” only the feature-construction
# pipeline is replaced. The output columns (neighbor_max_*, neighbor_min_*, 
# neighbor_mean_*) are numerically identical to the original.
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Key construction** | 25.8M `paste()` string ops | Zero â€” integer indices only |
| **Lookup mechanism** | Named character vector (hash per lookup) | Sparse matrix multiply + `data.table` grouped agg |
| **Neighbor list build** | 6.46M R-level `lapply` iterations | Eliminated entirely |
| **Stats computation** | 5 Ã— 6.46M `lapply` iterations | 28 years Ã— 5 vars Ã— {1 SpMV for mean + 1 `data.table` agg for min/max} = 420 vectorized ops |
| **Estimated runtime** | 86+ hours | **2â€“10 minutes** |
| **RAM** | ~6.46M-entry list of integer vectors | Sparse matrix (1.37M nonzeros â‰ˆ 33 MB) + `data.table` COO (1.37M rows â‰ˆ 22 MB) |
| **Numerical result** | Original estimand | **Identical** â€” same max/min/mean over same neighbor sets |
| **RF model** | â€” | **Untouched** |

The core algorithmic insight: **the neighbor topology is spatial (time-invariant), so factor the problem into spatial-adjacency Ã— year-slices, and use sparse linear algebra instead of per-row string lookups.**
END_RESPONSE>>>
