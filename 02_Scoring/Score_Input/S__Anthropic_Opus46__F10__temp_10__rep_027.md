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
run_id: Anthropic_F10_T10_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each produced by an `lapply` call that performs character string pasting, named-vector lookups, and `NA` filtering. This is O(n) with a very high constant factor due to R's interpreted overhead for string operations.

2. **`compute_neighbor_stats` iterates over ~6.46 million list entries** per variable, extracting subsets from a numeric vector. With 5 variables, this is ~32.3 million R-level loop iterations, each allocating small vectors.

3. **The neighbor lookup is cell-yearâ€“specific but the graph topology is year-invariant.** Rook neighbors don't change across yearsâ€”only the attribute values do. The current code redundantly encodes the year dimension into the lookup, inflating it from ~344K cell entries to ~6.46M cell-year entries.

**Key insight:** The rook adjacency graph has 344,208 nodes and ~1.37M directed edges. This is a **sparse spatial graph** that is constant across all 28 years. The neighbor aggregation (max, min, mean) can be computed as **sparse matrixâ€“vector operations** (one per year Ã— variable), which are orders of magnitude faster than R-level list iteration.

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from the `nb` object (~344K Ã— 344K, ~1.37M nonzeros). This is the graph topology.

2. **Reshape each variable into a 344,208 Ã— 28 matrix** (cells Ã— years). This is the node-attribute matrix.

3. **Compute neighbor statistics via sparse matrix operations:**
   - **Mean:** `A %*% X / degree` (sparse matrixâ€“dense matrix multiply, then element-wise division by row-degree).
   - **Max and Min:** Use a loop over the sparse matrix rows via `dgCMatrix` internals, or use `{MatrixExtra}` / manual CSR iteration. Alternatively, use `data.table` grouped operations on the edge list.

4. **Join results back** to the panel `data.table` and run the pre-trained Random Forest for prediction (no retraining).

For **max** and **min**, sparse matrix multiplication doesn't directly apply. The most efficient R approach is to represent the edge list explicitly and use `data.table` grouped aggregation, which is heavily optimized in C.

## Optimized R Code

```r
# ==============================================================================
# Optimized Neighbor Aggregation Pipeline
# Preserves numerical equivalence with original compute_neighbor_stats
# ==============================================================================

library(data.table)
library(Matrix)
library(randomForest)  # or ranger, whichever was used

# --------------------------------------------------------------------------
# STEP 0: Ensure cell_data is a data.table with columns: id, year, ntl, ec, ...
# --------------------------------------------------------------------------
setDT(cell_data)

# Inputs assumed available:
#   cell_data            : data.table with columns id, year, and all predictor vars
#   id_order             : integer/numeric vector of cell IDs in the nb object order
#   rook_neighbors_unique: nb object (list of integer index vectors, 1-based into id_order)
#   rf_model             : pre-trained Random Forest model

# --------------------------------------------------------------------------
# STEP 1: Build the directed edge list ONCE from the nb object
#          This encodes the full graph topology.
# --------------------------------------------------------------------------
build_edge_list <- function(id_order, nb_obj) {
  # nb_obj[[i]] gives the neighbor indices (into id_order) for cell id_order[i]
  n <- length(nb_obj)
  
  # Pre-calculate total edges for pre-allocation
  n_edges <- sum(vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  from_cell <- integer(n_edges)
  to_cell   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep::nb uses 0 to indicate no neighbors
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    k <- length(nbrs)
    from_cell[pos:(pos + k - 1L)] <- i
    to_cell[pos:(pos + k - 1L)]   <- nbrs
    pos <- pos + k
  }
  
  # Return edge list using actual cell IDs
  data.table(
    from_idx = from_cell[1:(pos - 1L)],
    to_idx   = to_cell[1:(pos - 1L)],
    from_id  = id_order[from_cell[1:(pos - 1L)]],
    to_id    = id_order[to_cell[1:(pos - 1L)]]
  )
}

cat("Building edge list from nb object...\n")
system.time({
  edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
})
cat(sprintf("Edge list: %d directed edges\n", nrow(edge_dt)))

# --------------------------------------------------------------------------
# STEP 2: Build sparse adjacency matrix for MEAN computation
#          A[i,j] = 1 if j is a rook neighbor of i (i.e., edge from i to j)
#          Then neighbor_mean(i) = (A %*% x)[i] / degree[i]
# --------------------------------------------------------------------------
cat("Building sparse adjacency matrix...\n")
n_cells <- length(id_order)

# Map cell IDs to matrix indices (1..n_cells)
id_to_midx <- setNames(seq_along(id_order), as.character(id_order))

A <- sparseMatrix(
  i = edge_dt$from_idx,
  j = edge_dt$to_idx,
  x = 1,
  dims = c(n_cells, n_cells)
)

# Row degrees (number of neighbors per cell)
degree <- diff(A@p)  # for dgCMatrix stored column-wise; need row sums
# Actually for CSC format, rowSums is fine:
degree_vec <- rowSums(A)  # numeric vector, length n_cells

# --------------------------------------------------------------------------
# STEP 3: Reshape cell_data into cell Ã— year matrices for each variable
# --------------------------------------------------------------------------

# Create a cell-index and year-index mapping
cell_data[, cell_idx := id_to_midx[as.character(id)]]

years <- sort(unique(cell_data$year))
n_years <- length(years)
year_to_colidx <- setNames(seq_along(years), as.character(years))
cell_data[, year_idx := year_to_colidx[as.character(year)]]

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# --------------------------------------------------------------------------
# STEP 4: For each variable, compute max, min, mean across neighbors
#          using edge-list + data.table for max/min,
#          and sparse matrix multiply for mean.
# --------------------------------------------------------------------------

compute_all_neighbor_features <- function(cell_data, edge_dt, A, degree_vec,
                                           neighbor_source_vars, id_to_midx,
                                           n_cells, years) {
  n_years <- length(years)
  
  # Pre-index: for the edge-list approach, we need to look up variable values
  # by (cell_idx, year). Build a matrix per variable: n_cells x n_years.
  # Then do the aggregation in matrix space.
  
  # Key cell_data for fast lookups
  setkeyv(cell_data, c("cell_idx", "year_idx"))
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))
    t0 <- proc.time()
    
    # Build cell Ã— year matrix
    val_vec <- cell_data[[var_name]]
    cidx    <- cell_data$cell_idx
    yidx    <- cell_data$year_idx
    
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(cidx, yidx)] <- val_vec
    
    # --- MEAN via sparse matrix multiply ---
    # AX[i, t] = sum of neighbor values for cell i at year t
    AX <- as.matrix(A %*% X)  # n_cells x n_years dense matrix
    
    # Divide by degree; where degree == 0, result should be NA
    mean_mat <- AX / degree_vec  # recycling: degree_vec has length n_cells
    mean_mat[degree_vec == 0, ] <- NA_real_
    
    # --- MAX and MIN via edge list aggregation ---
    # For each edge (from_idx -> to_idx), the neighbor value at year t
    # is X[to_idx, t]. We need max and min grouped by (from_idx, year).
    #
    # Vectorized approach: expand edge list across years using matrix indexing.
    
    # Get neighbor values for all edges: matrix n_edges x n_years
    neighbor_X <- X[edge_dt$to_idx, , drop = FALSE]  # n_edges x n_years
    
    # Now for each (from_idx), compute columnwise max and min of neighbor_X
    # This is a grouped operation. Use data.table for speed.
    
    # We'll process year by year to manage memory, or use rowsum-like tricks.
    # Actually, the most memory-efficient way: iterate over years (28 iterations).
    
    max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    from_idx_vec <- edge_dt$from_idx  # length = n_edges, reused every year
    
    for (t in seq_len(n_years)) {
      nvals <- neighbor_X[, t]
      
      # Remove NAs: create a mask
      valid <- !is.na(nvals)
      if (!any(valid)) next
      
      dt_t <- data.table(
        from = from_idx_vec[valid],
        val  = nvals[valid]
      )
      
      agg <- dt_t[, .(mx = max(val), mn = min(val)), by = from]
      
      max_mat[agg$from, t] <- agg$mx
      min_mat[agg$from, t] <- agg$mn
    }
    
    # Also need to set mean to NA where all neighbor values were NA
    # The sparse multiply treats NA as 0 by default in R. Fix this.
    # Recompute mean properly: count non-NA neighbors per cell-year.
    
    # X_notna: 1 if not NA, 0 if NA
    X_notna <- matrix(0, nrow = n_cells, ncol = n_years)
    X_notna[!is.na(X)] <- 1
    
    X_zero <- X
    X_zero[is.na(X_zero)] <- 0
    
    AX_fixed   <- as.matrix(A %*% X_zero)     # sum of non-NA neighbor values
    A_count    <- as.matrix(A %*% X_notna)     # count of non-NA neighbors
    
    mean_mat <- AX_fixed / A_count
    mean_mat[A_count == 0] <- NA_real_
    
    # --- Write results back to cell_data ---
    max_colname  <- paste0("neighbor_max_", var_name)
    min_colname  <- paste0("neighbor_min_", var_name)
    mean_colname <- paste0("neighbor_mean_", var_name)
    
    cell_data[, (max_colname)  := max_mat[cbind(cell_idx, year_idx)]]
    cell_data[, (min_colname)  := min_mat[cbind(cell_idx, year_idx)]]
    cell_data[, (mean_colname) := mean_mat[cbind(cell_idx, year_idx)]]
    
    elapsed <- (proc.time() - t0)[3]
    cat(sprintf("  Done in %.1f seconds\n", elapsed))
  }
  
  cell_data
}

cat("Computing neighbor features...\n")
system.time({
  cell_data <- compute_all_neighbor_features(
    cell_data, edge_dt, A, degree_vec,
    neighbor_source_vars, id_to_midx,
    n_cells, years
  )
})

# --------------------------------------------------------------------------
# STEP 5: Apply the pre-trained Random Forest model (NO retraining)
# --------------------------------------------------------------------------
cat("Generating predictions with pre-trained Random Forest...\n")

# Ensure predictor columns match what the model expects
# (The neighbor feature columns follow the naming convention of the original code.)
# Adjust column names if the original code used different naming, e.g.:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, ...

# If the original pipeline used names like ntl_neighbor_max, remap:
# original_name_pattern: depends on compute_and_add_neighbor_features
# Adjust this mapping as needed to match the trained model's expected feature names.

cell_data[, prediction := predict(rf_model, newdata = cell_data)]

# Clean up temporary index columns
cell_data[, c("cell_idx", "year_idx") := NULL]

cat("Pipeline complete.\n")
```

## Performance Analysis

| Component | Original (est.) | Optimized (est.) | Speedup |
|---|---|---|---|
| **Neighbor lookup** | ~20â€“30 hrs (string ops on 6.46M rows) | ~2 sec (edge list from nb, integer ops on 1.37M edges) | ~40,000Ã— |
| **Neighbor stats** (5 vars) | ~50â€“60 hrs (R-level list iteration, 32.3M calls) | ~3â€“5 min (sparse matmul for mean + data.table grouped agg for max/min, 28 year iterations) | ~800Ã— |
| **Memory** | ~6.46M lists with variable-length integer vectors | ~1.37M Ã— 3 edge table + 344K Ã— 28 dense matrices (~77 MB each) | Much lower |
| **Total** | ~86+ hrs | **~5â€“10 minutes** | ~500â€“1000Ã— |

## Why This Preserves Numerical Equivalence

1. **Mean:** `A %*% X_zero` computes the exact sum of non-NA neighbor values (NAs replaced with 0 are excluded by dividing by `A_count`, the count of non-NA neighbors). This is algebraically identical to `mean(neighbor_vals[!is.na(neighbor_vals)])`.

2. **Max/Min:** The `data.table` grouped aggregation computes `max(val)` and `min(val)` over exactly the same set of non-NA neighbor values that the original code uses, just via C-optimized grouped operations instead of R-level `lapply`.

3. **The graph topology is identical:** The edge list is constructed from the same `rook_neighbors_unique` nb object, producing the same directed edges. Expanding per-year via the cellÃ—year matrix is equivalent to the original `paste(id, year)` key matching.

4. **The Random Forest model is loaded and used for `predict()` onlyâ€”no retraining occurs.**
END_RESPONSE>>>
