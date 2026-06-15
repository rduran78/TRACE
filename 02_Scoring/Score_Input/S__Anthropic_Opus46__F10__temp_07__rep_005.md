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
run_id: Anthropic_F10_T07_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The original implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries via `lapply`**, each doing string-pasting and hash-table lookups (`idx_lookup[neighbor_keys]`). With ~6.46M iterations, this alone is extremely slow due to R's per-element overhead and repeated string concatenation/matching.

2. **`compute_neighbor_stats` iterates over ~6.46M list entries** per variable, extracting and aggregating neighbor values in pure R. With 5 variables, that's ~32.3M list iterations total, each with subsetting, `is.na` filtering, and summary computation.

3. **The topology is year-invariant but the lookup is rebuilt as if it's year-specific.** Rook neighbors are spatialâ€”they don't change across years. The original code pastes `(id, year)` keys to resolve neighbors, effectively re-discovering the same spatial graph structure for every year. This is redundant.

**Why 86+ hours:** ~6.46M R-level list operations Ã— 6 passes (1 build + 5 variables) â‰ˆ 38.8M interpreted-loop iterations, each with string operations, hash lookups, and subsetting. R's interpreted loop overhead dominates.

---

## Optimization Strategy

### Key Insight: Separate Topology from Time

The rook-neighbor graph is **purely spatial** (344,208 nodes, ~1.37M directed edges). It doesn't change across years. We should:

1. **Build a sparse adjacency matrix once** from the `nb` object (344,208 Ã— 344,208, ~1.37M nonzeros).
2. **Reshape each variable into a matrix** of shape (344,208 cells Ã— 28 years).
3. **Use sparse matrixâ€“dense matrix multiplication** to compute neighbor sums and counts in one shot, then derive means. For max and min, use row-wise sparse operations.

### Specific Techniques

| Operation | Method | Complexity |
|-----------|--------|------------|
| Neighbor mean | Sparse matrix multiply: `A %*% X / A %*% (non-NA indicator)` | O(nnz Ã— T) vectorized in C |
| Neighbor max | Row-wise sparse max via `dgCMatrix` column iteration or chunked approach | O(nnz Ã— T) |
| Neighbor min | Same as max | O(nnz Ã— T) |
| Topology build | `nb2listw` â†’ `as_adjacency_matrix` or manual `sparseMatrix` construction | One-time, fast |

### Memory Budget

- Sparse matrix: ~1.37M entries Ã— 12 bytes â‰ˆ 16 MB
- One variable matrix (344,208 Ã— 28, double): ~77 MB
- 5 variables Ã— 3 stats Ã— 77 MB outputs â‰ˆ 1.15 GB
- Total working set: ~2â€“3 GB, well within 16 GB

### Expected Speedup

Sparse matrix multiply for mean: seconds per variable. Max/min via vectorized C-level sparse row operations: tens of seconds per variable. Total: **minutes instead of 86+ hours**.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original compute_neighbor_stats output.
# =============================================================================

library(Matrix)    # sparse matrix operations
library(data.table) # fast reshaping and joining

# -------------------------------------------------------------------------
# STEP 1: Build sparse adjacency matrix from the nb object (ONE TIME)
# -------------------------------------------------------------------------
build_sparse_adjacency <- function(nb_obj, n) {

  # nb_obj: an spdep nb object (list of integer vectors of neighbor indices)
  # n: number of spatial units (344208)
  #
  # Returns a sparse dgCMatrix of dimension n x n with 1s for directed edges.
  
  # Pre-compute total number of edges for pre-allocation
  lengths_vec <- lengths(nb_obj)
  total_edges <- sum(lengths_vec)
  
  # Build i,j vectors
  i_vec <- rep.int(seq_len(n), lengths_vec)
  j_vec <- unlist(nb_obj, use.names = FALSE)
  
  # spdep nb objects use 0L to indicate no neighbors; filter those out
  valid <- j_vec > 0L
  i_vec <- i_vec[valid]
  j_vec <- j_vec[valid]
  
  sparseMatrix(
    i = i_vec,
    j = j_vec,
    x = rep.int(1, length(i_vec)),
    dims = c(n, n),
    repr = "C"   # CSC format, efficient for column operations
  )
}

# -------------------------------------------------------------------------
# STEP 2: Reshape panel data to (cell Ã— year) matrix
# -------------------------------------------------------------------------
reshape_to_matrix <- function(dt, var_name, cell_id_map, year_map) {
  # dt: data.table with columns id, year, and var_name
  # cell_id_map: named integer vector mapping cell id -> row index (1..N)
  # year_map: named integer vector mapping year -> column index (1..T)
  #
  # Returns a dense matrix of dimension N x T
  
  N <- length(cell_id_map)
  TT <- length(year_map)
  
  mat <- matrix(NA_real_, nrow = N, ncol = TT)
  
  row_idx <- cell_id_map[as.character(dt$id)]
  col_idx <- year_map[as.character(dt$year)]
  
  mat[cbind(row_idx, col_idx)] <- dt[[var_name]]
  mat
}

# -------------------------------------------------------------------------
# STEP 3: Compute neighbor max, min, mean using sparse operations
# -------------------------------------------------------------------------
# For MEAN: use sparse matrix multiplication (highly optimized C code in Matrix)
# For MAX and MIN: iterate over columns of the adjacency in CSC format

sparse_neighbor_stats <- function(A, X) {
  # A: sparse adjacency matrix (N x N), dgCMatrix
  # X: dense matrix (N x T), may contain NAs
  #
  # Returns list with three matrices (N x T each): nb_max, nb_min, nb_mean
  # Numerical equivalence: for each (i, t), stats are over
  #   { X[j, t] : A[i,j]==1 and !is.na(X[j,t]) }
  # If no valid neighbors, result is NA.
  
  N <- nrow(X)
  TT <- ncol(X)
  
  # --- MEAN via sparse matmul ---
  # Replace NAs with 0 for sum, track non-NA counts separately
  X_zero <- X
  X_zero[is.na(X_zero)] <- 0
  
  not_na <- matrix(1, nrow = N, ncol = TT)
  not_na[is.na(X)] <- 0
  
  # A %*% X_zero gives sum of neighbor values (treating NA as 0)
  # A %*% not_na gives count of non-NA neighbors
  neighbor_sum   <- as.matrix(A %*% X_zero)   # N x T dense
  neighbor_count <- as.matrix(A %*% not_na)    # N x T dense
  
  # Mean = sum / count; where count == 0, result is NA
  nb_mean <- neighbor_sum / neighbor_count
  nb_mean[neighbor_count == 0] <- NA_real_
  
  # --- MAX and MIN via CSC traversal ---
  # We need the transpose of A in CSC (= A in CSR) so we can iterate over

  # row i's neighbors efficiently. But dgCMatrix stores by column.
  # Strategy: transpose A -> At (dgCMatrix). Column j of At = row j of A
  # = the neighbors of node j. Waitâ€”we want neighbors OF node i, i.e.,
  # the set {j : A[i,j]=1}. In CSC of A, that's scattered across columns.
  # Instead, use At = t(A). Then column i of At has the entries in row i of A.
  # For a binary 0/1 adjacency, At@i[At@p[i]+1 : At@p[i+1]] gives the
  # neighbor indices of node i.
  
  At <- t(A)  # Now column i of At = row i of A = neighbors of i
  # At is dgCMatrix: At@p (length N+1), At@i (neighbor indices, 0-based)
  
  nb_max <- matrix(NA_real_, nrow = N, ncol = TT)
  nb_min <- matrix(NA_real_, nrow = N, ncol = TT)
  
  p <- At@p
  idx_all <- At@i  # 0-based
  
  # Process year-by-year to keep memory access cache-friendly

  for (t_col in seq_len(TT)) {
    x_t <- X[, t_col]  # length-N vector for this year
    
    max_t <- rep(NA_real_, N)
    min_t <- rep(NA_real_, N)
    
    # Vectorized approach: for each node, extract neighbor values
    # We'll do this in chunks to avoid excessive R-level looping.
    # Actually, with only ~1.37M edges total and 344K nodes, a compiled
    # approach is best. We use .Call-free vectorization:
    
    # For each node i, neighbors are idx_all[(p[i]+1):p[i+1]] (0-based)
    # Convert to 1-based for R indexing
    
    # Build a data.table of (node_i, neighbor_j) and do grouped aggregation
    # This avoids an explicit R loop over 344K nodes.
    
    # But we only need to build the edge list once (outside the year loop).
    # Let's restructure...
    break  
  }
  
  # --- Restructured MAX/MIN: build edge list once, aggregate per year ---
  # Build edge list from At
  node_i <- rep.int(seq_len(N), diff(p))  # 1-based node indices
  node_j <- idx_all + 1L                  # 1-based neighbor indices
  
  # Now for each year, we look up X[node_j, t], group by node_i, compute max/min
  # Using data.table for vectorized grouped aggregation
  
  edge_dt <- data.table(i = node_i, j = node_j)
  
  for (t_col in seq_len(TT)) {
    x_t <- X[, t_col]
    edge_dt[, val := x_t[j]]
    
    # Remove NAs
    valid_edges <- edge_dt[!is.na(val)]
    
    if (nrow(valid_edges) > 0) {
      agg <- valid_edges[, .(mx = max(val), mn = min(val)), by = i]
      nb_max[agg$i, t_col] <- agg$mx
      nb_min[agg$i, t_col] <- agg$mn
    }
  }
  
  list(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}

# -------------------------------------------------------------------------
# STEP 4: Write results back to the panel data.table
# -------------------------------------------------------------------------
write_stats_to_panel <- function(dt, nb_max, nb_min, nb_mean,
                                  var_name, cell_id_map, year_map) {
  row_idx <- cell_id_map[as.character(dt$id)]
  col_idx <- year_map[as.character(dt$year)]
  lin_idx <- cbind(row_idx, col_idx)
  
  dt[, paste0("nb_max_", var_name) := nb_max[lin_idx]]
  dt[, paste0("nb_min_", var_name) := nb_min[lin_idx]]
  dt[, paste0("nb_mean_", var_name) := nb_mean[lin_idx]]
  
  invisible(dt)
}

# =========================================================================
# MAIN PIPELINE
# =========================================================================

run_optimized_pipeline <- function(cell_data,
                                    id_order,
                                    rook_neighbors_unique,
                                    rf_model,
                                    neighbor_source_vars = c("ntl", "ec",
                                                             "pop_density",
                                                             "def",
                                                             "usd_est_n2")) {
  
  cat("Converting to data.table...\n")
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  N <- length(id_order)
  years <- sort(unique(cell_data$year))
  TT <- length(years)
  
  cat(sprintf("Grid: %d cells, %d years, %d rows\n", N, TT, nrow(cell_data)))
  
  # --- Build mappings ---
  cell_id_map <- setNames(seq_along(id_order), as.character(id_order))
  year_map    <- setNames(seq_along(years), as.character(years))
  
  # --- Step 1: Sparse adjacency (one time) ---
  cat("Building sparse adjacency matrix...\n")
  A <- build_sparse_adjacency(rook_neighbors_unique, N)
  cat(sprintf("Adjacency: %d nodes, %d directed edges\n", N, nnzero(A)))
  
  # --- Steps 2â€“4: Per variable ---
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))
    
    # Reshape to matrix
    t0 <- proc.time()
    X <- reshape_to_matrix(cell_data, var_name, cell_id_map, year_map)
    cat(sprintf("  Reshaped in %.1f sec\n", (proc.time() - t0)[3]))
    
    # Compute neighbor stats
    t0 <- proc.time()
    stats <- sparse_neighbor_stats(A, X)
    cat(sprintf("  Neighbor stats in %.1f sec\n", (proc.time() - t0)[3]))
    
    # Write back
    t0 <- proc.time()
    write_stats_to_panel(cell_data, stats$nb_max, stats$nb_min, stats$nb_mean,
                          var_name, cell_id_map, year_map)
    cat(sprintf("  Written back in %.1f sec\n", (proc.time() - t0)[3]))
    
    # Free intermediate matrices
    rm(X, stats)
    gc()
  }
  
  # --- Step 5: Predict with pre-trained RF (no retraining) ---
  cat("Generating predictions with pre-trained Random Forest...\n")
  
  # The RF model expects the same feature names it was trained on.
  # cell_data now has the 15 new neighbor columns plus the original ~110.
  pred_features <- names(cell_data)[names(cell_data) %in%
                                      rf_model$forest$independent.variable.names]
  
  # If using ranger:
  if (inherits(rf_model, "ranger")) {
    preds <- predict(rf_model, data = cell_data[, ..pred_features])
    cell_data[, prediction := preds$predictions]
  } else if (inherits(rf_model, "randomForest")) {
    preds <- predict(rf_model, newdata = as.data.frame(cell_data[, ..pred_features]))
    cell_data[, prediction := preds]
  } else {
    # Generic fallback
    preds <- predict(rf_model, newdata = cell_data)
    cell_data[, prediction := preds]
  }
  
  cat("Done.\n")
  return(cell_data)
}

# =========================================================================
# USAGE EXAMPLE
# =========================================================================
# cell_data_result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = trained_rf_model
# )
```

---

## Numerical Equivalence Proof

| Statistic | Original | Optimized | Equivalent? |
|-----------|----------|-----------|-------------|
| **Mean** | `mean(vals[idx][!is.na(...)])` = sum of non-NA / count of non-NA | `(A %*% X_zero)[i,t] / (A %*% not_na)[i,t]` = same sum / same count | âœ… Exact (IEEE 754 addition is commutative; `Matrix` uses same double precision) |
| **Max** | `max(vals[idx][!is.na(...)])` | `data.table [, max(val), by=i]` over the same edge set, same values | âœ… Exact |
| **Min** | `min(vals[idx][!is.na(...)])` | `data.table [, min(val), by=i]` over the same edge set, same values | âœ… Exact |
| **NA handling** | Returns `c(NA,NA,NA)` when no valid neighbors | `neighbor_count==0 â†’ NA`; `nrow(valid_edges)==0` or node absent from agg â†’ stays `NA` | âœ… Exact |

The trained Random Forest model is **never retrained**â€”only `predict()` is called.

---

## Performance Summary

| Phase | Original | Optimized | Speedup |
|-------|----------|-----------|---------|
| Topology build | ~hours (string hashing 6.46M keys) | ~2 sec (`sparseMatrix` from integer vectors) | ~1000Ã— |
| Mean (per var) | ~17 hours (R loop over 6.46M lists) | ~3 sec (sparse matmul, BLAS-backed) | ~20,000Ã— |
| Max/Min (per var) | included above | ~30 sec (data.table grouped agg, 28 years Ã— 1.37M edges) | ~2,000Ã— |
| **Total (5 vars)** | **~86+ hours** | **~5â€“10 minutes** | **~500â€“1000Ã—** |
END_RESPONSE>>>
