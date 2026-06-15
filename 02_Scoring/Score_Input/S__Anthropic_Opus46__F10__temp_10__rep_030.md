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
run_id: Anthropic_F10_T10_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The original implementation has three critical bottlenecks that compound to produce the ~86+ hour runtime:

### 1. **O(NÂ²)-scale `build_neighbor_lookup` construction**
The function iterates over **~6.46 million rows**, and for each row it performs `paste` string construction and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a vector of 6.46M entries. This single step alone is catastrophically slow: ~6.46M Ã— k lookups into a 6.46M-length named vector.

### 2. **`lapply` over 6.46M rows in `compute_neighbor_stats`**
Each call to `compute_neighbor_stats` runs an R-level `lapply` across all 6.46M rows, performing subsetting, `NA` removal, and three aggregation functions. This is repeated **5 times** (once per source variable), yielding ~32.3 million R-level function invocations.

### 3. **Redundant topology reconstruction per row-year**
The neighbor graph topology is **year-invariant** â€” cell A is a rook neighbor of cell B in every year. Yet the lookup is built at the row (cell-year) level, expanding 1.37M directed edges into ~6.46M Ã— k index lists by duplicating the same topology 28 times.

### Key Insight
The adjacency structure has only **344,208 nodes and ~1.37M edges**. The yearly expansion is purely an indexing concern. By separating topology from temporal indexing, we can operate on the **sparse graph** directly using vectorized/compiled operations.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Build topology once** | Construct a sparse adjacency matrix (344,208 Ã— 344,208) from `rook_neighbors_unique` once. |
| **Sparse matrixâ€“vector multiplication for mean** | `neighbor_mean = (A %*% x) / (A %*% 1_{!NA})` â€” fully vectorized via `Matrix` package C code. |
| **Vectorized min/max via row-wise sparse ops** | Replace per-row `lapply` with a grouped operation using `data.table` keyed joins, or a custom sparse row sweep. |
| **Year-parallel processing** | Since neighbors only exist within the same year, process each year independently (344K rows Ã— k neighbors), which fits easily in memory. |
| **Eliminate all `paste`/string operations** | Use integer-indexed mappings exclusively. |
| **Single pass per variable** | Compute max, min, mean simultaneously in one pass. |

**Expected speedup**: From ~86 hours to **~2â€“10 minutes** (dominated by sparse matrix operations on 344KÃ—344K matrix, 28 years Ã— 5 variables = 140 sparse matmuls plus grouped min/max).

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# Requirements: data.table, Matrix
# Preserves numerical equivalence with original compute_neighbor_stats output.
# Does NOT retrain the Random Forest model.
# =============================================================================

library(data.table)
library(Matrix)

#' Build a sparse adjacency matrix from an spdep nb object.
#' Constructed ONCE and reused across all years and variables.
#'
#' @param nb_obj  An spdep nb object (list of integer neighbor index vectors).
#'                rook_neighbors_unique â€” indexed into id_order.
#' @param n       Number of spatial cells (length of id_order).
#' @return A dgCMatrix (sparse column-compressed) adjacency matrix, n x n.
build_adjacency_matrix <- function(nb_obj, n) {
  # Pre-calculate total number of edges for pre-allocation
  edge_counts <- vapply(nb_obj, function(x) {
    x <- x[x > 0L]  # spdep nb uses 0 for no-neighbor sentinel
    length(x)
  }, integer(1))
  total_edges <- sum(edge_counts)
  
  # Pre-allocate vectors for triplet construction
  from_idx <- integer(total_edges)
  to_idx   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs > 0L]
    k <- length(nbrs)
    if (k > 0L) {
      from_idx[pos:(pos + k - 1L)] <- i
      to_idx[pos:(pos + k - 1L)]   <- nbrs
      pos <- pos + k
    }
  }
  
  # Sparse matrix: A[i,j] = 1 means j is a neighbor of i
  # So row i contains the neighbors of node i
  sparseMatrix(
    i = from_idx, j = to_idx,
    x = 1, dims = c(n, n),
    repr = "C"  # CSC format, efficient for %*%
  )
}

#' Compute neighbor max, min, mean for one variable across all cell-years.
#' Uses sparse matrix ops for mean and data.table grouped ops for min/max.
#'
#' @param dt          data.table with columns: cell_idx (integer 1..n), year, <var_name>
#' @param var_name    Character: name of the variable column.
#' @param adj_matrix  Sparse adjacency matrix (n x n), row i has neighbors of cell i.
#' @param n_cells     Number of spatial cells.
#' @param years       Sorted integer vector of unique years.
#' @return data.table with columns: cell_idx, year, <var>_max, <var>_min, <var>_mean
compute_neighbor_features_sparse <- function(dt, var_name, adj_matrix, n_cells, years) {
  
  max_col  <- paste0("n_", var_name, "_max")
  min_col  <- paste0("n_", var_name, "_min")
  mean_col <- paste0("n_", var_name, "_mean")
  
  # Pre-extract the adjacency structure in row-oriented form for min/max
  # This is computed once per call but could be hoisted out; however it's
  # only ~1.37M entries so fast to extract.
  adj_t <- summary(adj_matrix)  # data.frame with i, j, x columns
  # adj_t$i = row (focal cell), adj_t$j = column (neighbor cell)
  edge_dt <- data.table(focal = adj_t$i, neighbor = adj_t$j)
  
  # Process year by year â€” each year is independent (neighbors only within same year)
  result_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    
    # Extract the variable values for this year, indexed by cell_idx
    # Create a full-length vector (1..n_cells), NA for missing cells
    yr_rows <- dt[year == yr]
    vals_vec <- rep(NA_real_, n_cells)
    vals_vec[yr_rows$cell_idx] <- yr_rows[[var_name]]
    
    # --- MEAN via sparse matrix-vector product ---
    # Replace NA with 0 for the sum, track non-NA counts separately
    not_na <- as.numeric(!is.na(vals_vec))
    vals_zero <- vals_vec
    vals_zero[is.na(vals_zero)] <- 0
    
    # neighbor_sum[i] = sum of neighbor values of cell i (treating NA as 0)
    neighbor_sum   <- as.numeric(adj_matrix %*% vals_zero)
    # neighbor_count[i] = number of non-NA neighbors of cell i
    neighbor_count <- as.numeric(adj_matrix %*% not_na)
    
    neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # --- MIN and MAX via data.table grouped operation ---
    # Attach neighbor values to edge list
    edge_yr <- copy(edge_dt)
    edge_yr[, val := vals_vec[neighbor]]
    # Remove edges where neighbor value is NA
    edge_yr <- edge_yr[!is.na(val)]
    
    # Grouped aggregation
    if (nrow(edge_yr) > 0) {
      agg <- edge_yr[, .(nmax = max(val), nmin = min(val)), by = focal]
      
      # Initialize full vectors
      neighbor_max <- rep(NA_real_, n_cells)
      neighbor_min <- rep(NA_real_, n_cells)
      neighbor_max[agg$focal] <- agg$nmax
      neighbor_min[agg$focal] <- agg$nmin
    } else {
      neighbor_max <- rep(NA_real_, n_cells)
      neighbor_min <- rep(NA_real_, n_cells)
    }
    
    # Build result for this year â€” only for cells that exist in the data
    res <- data.table(
      cell_idx = yr_rows$cell_idx,
      year     = yr
    )
    res[[max_col]]  <- neighbor_max[yr_rows$cell_idx]
    res[[min_col]]  <- neighbor_min[yr_rows$cell_idx]
    res[[mean_col]] <- neighbor_mean[yr_rows$cell_idx]
    
    result_list[[yi]] <- res
  }
  
  rbindlist(result_list)
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique, rf_model) {
  
  n_cells <- length(id_order)
  cat("Building sparse adjacency matrix for", n_cells, "cells...\n")
  
  # --- Step 1: Build adjacency matrix ONCE ---
  adj_matrix <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
  cat("Adjacency matrix:", nrow(adj_matrix), "x", ncol(adj_matrix),
      "with", nnzero(adj_matrix), "non-zero entries\n")
  
  # --- Step 2: Convert to data.table with integer cell index ---
  dt <- as.data.table(cell_data)
  
  # Map cell IDs to integer indices (1..n_cells) matching id_order
  id_map <- data.table(id = id_order, cell_idx = seq_len(n_cells))
  dt <- merge(dt, id_map, by = "id", sort = FALSE)
  setkey(dt, year, cell_idx)
  
  years <- sort(unique(dt$year))
  cat("Processing", length(years), "years,", nrow(dt), "total rows\n")
  
  # --- Step 3: Compute neighbor features for each source variable ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor features for:", var_name, "...\n")
    
    feat_dt <- compute_neighbor_features_sparse(
      dt, var_name, adj_matrix, n_cells, years
    )
    
    # Merge features back into main table
    max_col  <- paste0("n_", var_name, "_max")
    min_col  <- paste0("n_", var_name, "_min")
    mean_col <- paste0("n_", var_name, "_mean")
    
    # Join on cell_idx and year
    setkeyv(feat_dt, c("cell_idx", "year"))
    setkeyv(dt, c("cell_idx", "year"))
    
    dt[feat_dt, (max_col)  := get(paste0("i.", max_col))]
    dt[feat_dt, (min_col)  := get(paste0("i.", min_col))]
    dt[feat_dt, (mean_col) := get(paste0("i.", mean_col))]
    
    cat("  Done:", var_name, "\n")
  }
  
  # --- Step 4: Remove helper column, restore original structure ---
  dt[, cell_idx := NULL]
  
  # --- Step 5: Apply the pre-trained Random Forest model (NO retraining) ---
  cat("Generating predictions with pre-trained Random Forest...\n")
  
  # Ensure column order/names match what the RF model expects
  # The model object carries its expected variable names in rf_model$forest$independent.variable.names
  # or rf_model$xNames (depending on package: ranger vs randomForest)
  predictions <- predict(rf_model, data = dt)
  
  # Handle different RF package output formats
  if (is.list(predictions) && !is.null(predictions$predictions)) {
    dt[, prediction := predictions$predictions]
  } else {
    dt[, prediction := as.numeric(predictions)]
  }
  
  cat("Pipeline complete.\n")
  return(dt)
}

# =============================================================================
# USAGE (drop-in replacement for the original pipeline)
# =============================================================================
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = trained_rf_model
# )
```

---

## Why This Is Numerically Equivalent

| Original | Optimized | Equivalence |
|---|---|---|
| `mean(neighbor_vals)` | `(A %*% x) / (A %*% 1_notNA)` | Identical: sum of non-NA values divided by count of non-NA values. |
| `max(neighbor_vals)` | `data.table [, max(val), by=focal]` after removing NA edges | Identical: same set of non-NA neighbor values, same `max`. |
| `min(neighbor_vals)` | `data.table [, min(val), by=focal]` after removing NA edges | Identical: same set, same `min`. |
| NA when no valid neighbors | `ifelse(count>0, ..., NA)` and missing rows in grouped agg â†’ NA fill | Identical NA propagation. |

## Performance Summary

| Component | Original | Optimized |
|---|---|---|
| **Topology construction** | 6.46M string pastes + named vec lookup | One sparse matrix build (~1.37M entries): **~2 seconds** |
| **Mean computation (per var-year)** | 6.46M R-level `lapply` calls | One sparse matmul (344K Ã— 344K, 1.37M nnz): **~0.1 seconds** |
| **Min/Max (per var-year)** | Same `lapply`, 3 aggregates per row | `data.table` grouped agg on ~1.37M edges: **~0.3 seconds** |
| **Total (5 vars Ã— 28 years)** | ~86+ hours | **~2â€“10 minutes** |
| **Peak memory** | ~6.46M-length named vector + list of 6.46M index vectors (~several GB) | Sparse matrix (~30 MB) + per-year vectors (~5 MB): **< 2 GB** |
END_RESPONSE>>>
