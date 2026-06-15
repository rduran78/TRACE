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
run_id: Anthropic_F10_T07_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per probe in the worst case, and doing this 6.46M times with string concatenation is catastrophically slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in an R-level `lapply`**, extracting subsets of a numeric vector and computing `max/min/mean`. This is repeated 5 times (once per variable), yielding ~32.3M R-level loop iterations with per-element subsetting.

3. **The neighbor topology is year-invariant** (rook neighbors don't change across years), but the lookup is rebuilt as if each cell-year has a unique neighbor set. The 344,208 cells have ~1.37M directed edges â€” this topology is reused identically across all 28 years, yet the implementation materializes it into 6.46M cell-year-specific index vectors.

**Root cause**: The implementation conflates the **spatial graph topology** (344K nodes, 1.37M edges) with the **panel structure** (28 years). It should build the spatial adjacency once and apply it per-year as a sparse matrixâ€“vector product.

## Optimization Strategy

1. **Build a sparse adjacency matrix once** (344,208 Ã— 344,208 with ~1.37M nonzeros) from the `nb` object.
2. **Reshape each variable into a 344,208 Ã— 28 matrix** (cells Ã— years).
3. **Compute neighbor aggregates via sparse matrix operations**:
   - **Mean**: `A_norm %*% X` where `A_norm` is the row-normalized adjacency.
   - **Max / Min**: Use a single pass over the CSC/CSR structure â€” unavoidable for max/min, but done at C level via `data.table` grouped operations or a direct sparse-matrix walk.
4. **Flatten results back** into the original data frame column order.
5. **Feed into the pre-trained Random Forest** without retraining.

Sparse matrixâ€“dense matrix multiplication for mean is O(nnz Ã— 28) â‰ˆ 38.5M flops â€” trivial. Max/min require a grouped operation but can be done efficiently with `data.table` keyed joins on ~38.5M edge-year pairs (1.37M edges Ã— 28 years), which `data.table` handles in seconds.

**Expected speedup**: From 86+ hours to **under 5 minutes**.

## Working R Code

```r
# =============================================================================
# Optimized Neighborhood Aggregation Pipeline
# Preserves numerical equivalence with the original compute_neighbor_stats
# =============================================================================

library(Matrix)
library(data.table)

optimize_neighbor_pipeline <- function(cell_data, 
                                        id_order, 
                                        rook_neighbors_unique,
                                        neighbor_source_vars = c("ntl", "ec", "pop_density", 
                                                                  "def", "usd_est_n2")) {
  
  # -------------------------------------------------------------------------
  # 0. Convert to data.table for speed; record original row order
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .rowid := .I]  # preserve original row order
  
  n_cells <- length(id_order)
  
  # Map cell id -> integer index 1..n_cells
  id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))
  
  # -------------------------------------------------------------------------
  # 1. Build sparse adjacency COO from the nb object (topology, built once)
  #    Edge (i -> j) means j is a rook neighbor of i.
  #    We store (i, j) so that row i aggregates over its neighbors in column j.
  # -------------------------------------------------------------------------
  message("Building sparse adjacency from nb object...")
  
  # Preallocate edge list
  edge_from <- vector("integer", 0)
  edge_to   <- vector("integer", 0)
  
  # nb objects: rook_neighbors_unique[[i]] is an integer vector of neighbor 
  # indices into id_order (with 0L meaning no neighbors)
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    nb_i <- nb_i[nb_i != 0L]  # spdep uses 0 for no-neighbor cards
    if (length(nb_i) > 0L) {
      edge_from <- c(edge_from, rep.int(i, length(nb_i)))
      edge_to   <- c(edge_to, nb_i)
    }
  }
  
  n_edges <- length(edge_from)
  message(sprintf("  %d cells, %d directed edges", n_cells, n_edges))
  
  # Sparse adjacency matrix (n_cells x n_cells), binary
  # A[i,j] = 1 means j is a neighbor of i => row i aggregates columns j
  A <- sparseMatrix(i = edge_from, j = edge_to, x = 1, 
                    dims = c(n_cells, n_cells), repr = "C")  # CSC
  
  # Row-normalized version for mean computation
  row_deg <- diff(A@p)  # for dgCMatrix this doesn't work directly; use rowSums
  # Actually, for a dgCMatrix, we need the row-sparse form for rowSums
  A_r <- as(A, "RsparseMatrix")  # dgRMatrix: row-oriented
  deg  <- tabulate(edge_from, nbins = n_cells)  # degree of each node
  
  # -------------------------------------------------------------------------
  # 2. Build edge table (data.table) for max/min computation
  # -------------------------------------------------------------------------
  edges_dt <- data.table(from_idx = edge_from, to_idx = edge_to)
  rm(edge_from, edge_to)
  
  # -------------------------------------------------------------------------
  # 3. Map cell_data rows to (cell_idx, year)
  # -------------------------------------------------------------------------
  dt[, cell_idx := id_to_idx[as.character(id)]]
  
  # Get sorted unique years
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_len(n_years), as.character(years))
  dt[, year_col := year_to_col[as.character(year)]]
  
  # -------------------------------------------------------------------------
  # 4. For each variable, compute neighbor max, min, mean
  # -------------------------------------------------------------------------
  
  for (var_name in neighbor_source_vars) {
    
    message(sprintf("Processing variable: %s", var_name))
    
    # 4a. Pivot variable into a cells x years matrix
    #     X[cell_idx, year_col] = value
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(dt$cell_idx, dt$year_col)] <- dt[[var_name]]
    
    # -------------------------------------------------------------------
    # 4b. MEAN via sparse matrix multiplication
    #     For each cell i and year t:
    #       mean_val = sum_j A[i,j]*X[j,t] / deg[i]
    #     This is (A %*% X) / deg, where deg is broadcast per row.
    # -------------------------------------------------------------------
    AX <- as.matrix(A %*% X)  # n_cells x n_years dense matrix
    
    # Where deg == 0, result should be NA (no neighbors)
    mean_mat <- AX
    has_neighbors <- deg > 0L
    mean_mat[has_neighbors, ] <- AX[has_neighbors, ] / deg[has_neighbors]
    mean_mat[!has_neighbors, ] <- NA_real_
    
    # Handle cells that have neighbors but all neighbor values are NA for a year:
    # A %*% X treats NA as... actually, standard matrix multiply propagates NA.
    # We need to handle NAs properly: compute sum of non-NA and count of non-NA.
    
    # Create a non-NA indicator matrix
    notNA <- matrix(0, nrow = n_cells, ncol = n_years)
    notNA[!is.na(X)] <- 1
    X_zero <- X
    X_zero[is.na(X_zero)] <- 0  # replace NA with 0 for summation
    
    sum_mat   <- as.matrix(A %*% X_zero)    # sum of non-NA neighbor values
    count_mat <- as.matrix(A %*% notNA)     # count of non-NA neighbor values
    
    mean_mat <- ifelse(count_mat > 0, sum_mat / count_mat, NA_real_)
    
    # -------------------------------------------------------------------
    # 4c. MAX and MIN via edge-table grouped aggregation
    #     For each edge (from_idx -> to_idx), look up X[to_idx, year_col]
    #     for all years, then group by (from_idx, year) and take max/min.
    #     
    #     Key insight: instead of expanding edges Ã— years (38.5M rows),
    #     we work column-by-column (year-by-year) over the edge table.
    # -------------------------------------------------------------------
    
    max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    for (t in seq_len(n_years)) {
      # Get neighbor values for this year via edge table
      nb_vals <- X[edges_dt$to_idx, t]
      
      # Build a temporary data.table for grouped max/min
      # Only keep non-NA values
      valid <- !is.na(nb_vals)
      if (!any(valid)) next
      
      tmp <- data.table(from = edges_dt$from_idx[valid], val = nb_vals[valid])
      
      agg <- tmp[, .(mx = max(val), mn = min(val)), by = from]
      
      max_mat[agg$from, t] <- agg$mx
      min_mat[agg$from, t] <- agg$mn
    }
    
    # -------------------------------------------------------------------
    # 4d. Write results back to dt in original row order
    # -------------------------------------------------------------------
    idx_mat <- cbind(dt$cell_idx, dt$year_col)
    
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    
    set(dt, j = col_max,  value = max_mat[idx_mat])
    set(dt, j = col_min,  value = min_mat[idx_mat])
    set(dt, j = col_mean, value = mean_mat[idx_mat])
    
    rm(X, X_zero, notNA, sum_mat, count_mat, AX, mean_mat, max_mat, min_mat)
    gc()
  }
  
  # -------------------------------------------------------------------------
  # 5. Restore original order and return as data.frame
  # -------------------------------------------------------------------------
  setorder(dt, .rowid)
  dt[, c(".rowid", "cell_idx", "year_col") := NULL]
  
  return(as.data.frame(dt))
}


# =============================================================================
# USAGE
# =============================================================================

# # Load pre-existing objects
# load("cell_data.RData")           # cell_data data.frame
# load("rook_neighbors.RData")      # rook_neighbors_unique (nb object)
# load("id_order.RData")            # id_order vector
# load("trained_rf_model.RData")    # rf_model (pre-trained Random Forest)
# 
# # Run optimized pipeline
# cell_data <- optimize_neighbor_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   neighbor_source_vars   = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
# 
# # Predict using the pre-trained Random Forest (NO retraining)
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized | Equivalence |
|-----------|----------|-----------|-------------|
| **Max** | `max(vals[idx])` after removing NAs | `data.table` `max(val)` grouped by `from`, after filtering NAs | Identical â€” same `max()` function on same values |
| **Min** | `min(vals[idx])` after removing NAs | `data.table` `min(val)` grouped by `from`, after filtering NAs | Identical â€” same `min()` function on same values |
| **Mean** | `mean(vals[idx])` after removing NAs | `sum(non-NA values) / count(non-NA values)` via sparse matmul | Identical â€” `mean(x) = sum(x)/length(x)` for non-NA elements; sparse matmul with NAâ†’0 substitution + separate count matrix reproduces this exactly |
| **NA handling** | Returns `NA` if no neighbors or all neighbor values NA | `count_mat == 0` â†’ NA; `max_mat`/`min_mat` default NA | Identical |

## Performance Estimate

| Step | Complexity | Est. Time |
|------|-----------|-----------|
| Build sparse matrix | O(1.37M) | ~1 sec |
| Mean (sparse matmul, per var) | O(1.37M Ã— 28) Ã— 3 matrices | ~3 sec |
| Max/Min (data.table, per var per year) | O(1.37M) Ã— 28 years | ~15 sec |
| Total for 5 variables | | **~2â€“3 minutes** |

**Speedup: ~1,700Ã— over the original 86+ hour estimate.**
END_RESPONSE>>>
