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
run_id: Anthropic_F10_T07_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **O(nÂ²)-scale neighbor lookup construction**: `build_neighbor_lookup` creates a lookup for every row (~6.46M rows) by string-pasting cell IDs and years into keys, then doing named-vector lookups. Named vector lookups in R are hash-based but the sheer volume (~6.46M string keys, queried ~6.46M Ã— avg_neighbors times) is extremely expensive.

2. **Row-wise `lapply` over 6.46M rows in `compute_neighbor_stats`**: Each call extracts a small vector, removes NAs, and computes three summary statistics. The per-call overhead of R function dispatch, subsetting, and `is.na` checks, repeated ~6.46M times per variable (Ã—5 variables), dominates runtime.

3. **Redundant topology recomputation per year**: The rook-neighbor graph is purely spatial â€” it doesn't change across years. Yet the lookup embeds year into every key, effectively rebuilding the topology 28 times (once per year) inside a single monolithic structure.

**Memory**: The 6.46M-element list of integer vectors for `neighbor_lookup` alone consumes several GB due to R list overhead (each list element has ~100+ bytes of overhead).

## Optimization Strategy

1. **Separate topology from time**: Build a sparse adjacency structure once over the 344,208 cells (not 6.46M cell-years). Represent it as a CSR (Compressed Sparse Row) sparse matrix or, equivalently, two integer vectors (`p` and `j` from a `dgCMatrix`).

2. **Sparse matrixâ€“vector multiplication for mean**: For each variable-year, extract the column vector of values, multiply by the row-normalized adjacency matrix â†’ gives neighbor means in one vectorized operation. This replaces 344K R-level loops with a single C-level sparse matrix multiply.

3. **Sparse matrix operations for max and min**: Use the `Matrix` package or a custom grouped operation. Construct a binary sparse adjacency matrix `A`. For each variable-year, use the adjacency structure to do grouped max/min via data.table or vectorized segment operations.

4. **Year-parallel processing**: Process each year independently (only ~344K rows), which fits comfortably in memory and can be parallelized.

5. **Estimated speedup**: From ~86 hours to ~5â€“15 minutes.

## Working R Code

```r
# =============================================================================
# Optimized Neighborhood Aggregation Pipeline
# Numerically equivalent to the original, orders of magnitude faster.
# =============================================================================

library(Matrix)
library(data.table)

# ---- Step 1: Build sparse adjacency matrix ONCE (topology only) ----

build_adjacency_matrix <- function(id_order, rook_neighbors) {
  # id_order: vector of cell IDs (length N = 344,208)
  # rook_neighbors: spdep nb object (list of length N, each element = integer
  #                 indices into id_order of rook neighbors; 0L means none)
  
  N <- length(id_order)
  
  # Build COO (coordinate) triplets
  from_list <- vector("list", N)
  to_list   <- vector("list", N)
  
  for (i in seq_len(N)) {
    nb_i <- rook_neighbors[[i]]
    nb_i <- nb_i[nb_i != 0L]  # spdep uses 0 for no-neighbor sentinel
    if (length(nb_i) > 0L) {
      from_list[[i]] <- rep.int(i, length(nb_i))
      to_list[[i]]   <- nb_i
    }
  }
  
  from_vec <- unlist(from_list, use.names = FALSE)
  to_vec   <- unlist(to_list, use.names = FALSE)
  
  # Binary adjacency matrix: A[i,j] = 1 means j is a rook neighbor of i
  # Row i contains the neighbors of cell i
  A <- sparseMatrix(
    i = from_vec,
    j = to_vec,
    x = rep.int(1, length(from_vec)),
    dims = c(N, N),
    repr = "C"   # CSC, will transpose for CSR-like row access
  )
  
  return(A)
}

# ---- Step 2: Compute neighbor degree and row-normalized matrix for mean ----

prepare_aggregation_matrices <- function(A) {
  # Degree = number of non-NA neighbors (will adjust per variable-year for NAs)
  # A_norm: row-normalized adjacency for computing means (base case, no NAs)
  # We handle NAs explicitly below, so we just store A here.
  return(A)
}

# ---- Step 3: Compute neighbor stats for one variable, all years at once ----

compute_neighbor_features_fast <- function(cell_dt, A, var_name, id_order) {
  # cell_dt: data.table with columns: id, year, <var_name>
  # A: sparse adjacency matrix (N x N) over id_order
  # id_order: vector of cell IDs defining row/col order of A
  
  N <- length(id_order)
  
  # Create a map from cell ID to matrix index
  id_to_idx <- setNames(seq_len(N), as.character(id_order))
  
  # Add matrix index to data
  cell_dt[, .mat_idx := id_to_idx[as.character(id)]]
  
  # Output columns
  max_col  <- paste0("max_", var_name)
  min_col  <- paste0("min_", var_name)
  mean_col <- paste0("mean_", var_name)
  
  # Pre-allocate output
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  # Extract CSR structure from A (or work with dgCMatrix column-wise on t(A))
  # For dgCMatrix A: A@p, A@i, A@x give CSC. 
  # Row i's neighbors = columns j where A[i,j] != 0.
  # It's easier to work with t(A) in CSC = A in CSR.
  At <- t(A)  # Now At is dgCMatrix; column j of At = row j of A = neighbors of j
  
  years <- sort(unique(cell_dt$year))
  
  for (yr in years) {
    # Extract the value vector for this year, ordered by matrix index
    # Build a full-length vector (N), fill with NA for cells not present this year
    vals_full <- rep(NA_real_, N)
    
    yr_rows <- cell_dt[year == yr]
    vals_full[yr_rows$.mat_idx] <- yr_rows[[var_name]]
    
    # ---- Neighbor MEAN (NA-aware) ----
    # Replace NAs with 0 for sum, track non-NA with indicator
    not_na <- as.double(!is.na(vals_full))
    vals_zero <- vals_full
    vals_zero[is.na(vals_zero)] <- 0
    
    # Sparse matrix-vector multiply: neighbor sums and neighbor counts
    # A %*% vals_zero  -> for each row i, sum of vals of neighbors of i (NA->0)
    # A %*% not_na     -> for each row i, count of non-NA neighbors
    neighbor_sum   <- as.numeric(A %*% vals_zero)
    neighbor_count <- as.numeric(A %*% not_na)
    
    neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # ---- Neighbor MAX and MIN (requires explicit grouped operation) ----
    # Use the CSC structure of At: column j of At lists the neighbors of cell j
    # We do a grouped max/min over neighbor values.
    
    p_vec <- At@p        # length N+1, column pointers (0-based)
    i_vec <- At@i + 1L   # row indices (convert to 1-based) = neighbor indices
    
    # Indices of cells present this year
    idx_present <- yr_rows$.mat_idx
    
    # For efficiency, compute max/min only for present cells using C-level vectorization
    # via data.table's grouped operations on the edge list
    
    # Build edge table for present cells only
    # For cell j (0-based col in At), neighbors are i_vec[(p_vec[j]+1):p_vec[j+1]]
    
    # Vectorized extraction of all (cell, neighbor_value) pairs for present cells
    # Using the CSC pointers:
    
    n_present <- length(idx_present)
    
    # Number of neighbors per present cell
    nn <- p_vec[idx_present + 1L] - p_vec[idx_present]  # 0-based pointers
    
    if (sum(nn) > 0) {
      # Build the expanded cell index and neighbor value vectors
      cell_rep <- rep.int(idx_present, nn)
      
      # Gather all neighbor pointers
      ptr_starts <- p_vec[idx_present] + 1L  # convert to 1-based
      ptr_seq <- sequence(nn, from = ptr_starts)
      
      nb_indices <- i_vec[ptr_seq]
      nb_vals    <- vals_full[nb_indices]
      
      # Use data.table for ultra-fast grouped max/min
      edge_dt <- data.table(cell = cell_rep, val = nb_vals)
      edge_dt <- edge_dt[!is.na(val)]
      
      if (nrow(edge_dt) > 0) {
        agg <- edge_dt[, .(nb_max = max(val), nb_min = min(val)), by = cell]
        
        # Map back: create full vectors
        nb_max_full <- rep(NA_real_, N)
        nb_min_full <- rep(NA_real_, N)
        nb_max_full[agg$cell] <- agg$nb_max
        nb_min_full[agg$cell] <- agg$nb_min
      } else {
        nb_max_full <- rep(NA_real_, N)
        nb_min_full <- rep(NA_real_, N)
      }
    } else {
      nb_max_full <- rep(NA_real_, N)
      nb_min_full <- rep(NA_real_, N)
    }
    
    # Write results back into cell_dt for this year's rows
    row_indices <- which(cell_dt$year == yr)
    mat_indices <- cell_dt$.mat_idx[row_indices]
    
    set(cell_dt, i = row_indices, j = max_col,  value = nb_max_full[mat_indices])
    set(cell_dt, i = row_indices, j = min_col,  value = nb_min_full[mat_indices])
    set(cell_dt, i = row_indices, j = mean_col, value = neighbor_mean[mat_indices])
  }
  
  # Clean up temp column
  cell_dt[, .mat_idx := NULL]
  
  return(cell_dt)
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique, rf_model) {
  
  cat("Converting to data.table...\n")
  cell_dt <- as.data.table(cell_data)
  
  # ---- Build adjacency matrix ONCE ----
  cat("Building sparse adjacency matrix (344,208 x 344,208)...\n")
  t0 <- Sys.time()
  A <- build_adjacency_matrix(id_order, rook_neighbors_unique)
  cat("  Adjacency matrix built in", round(difftime(Sys.time(), t0, units = "secs"), 1), "seconds\n")
  cat("  Non-zeros:", nnzero(A), "\n")
  
  # ---- Compute neighbor features for each source variable ----
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor features for:", var_name, "\n")
    t1 <- Sys.time()
    cell_dt <- compute_neighbor_features_fast(cell_dt, A, var_name, id_order)
    cat("  Done in", round(difftime(Sys.time(), t1, units = "secs"), 1), "seconds\n")
  }
  
  # ---- Predict with pre-trained Random Forest (no retraining) ----
  cat("Generating predictions with pre-trained Random Forest...\n")
  
  # Convert back to data.frame for predict() compatibility if needed
  pred_data <- as.data.frame(cell_dt)
  
  # Get the predictor variable names the model expects
  # (the model object stores this internally)
  predictions <- predict(rf_model, newdata = pred_data)
  
  cell_dt[, prediction := predictions]
  
  cat("Pipeline complete.\n")
  return(cell_dt)
}

# =============================================================================
# USAGE (assumes objects are already loaded in the R session):
#
#   result <- run_optimized_pipeline(
#     cell_data              = cell_data,
#     id_order               = id_order,
#     rook_neighbors_unique  = rook_neighbors_unique,
#     rf_model               = rf_model
#   )
# =============================================================================
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string keys, named vector hash lookup | One 344KÃ—344K sparse matrix (CSR), ~1.37M non-zeros |
| **Mean computation** | `lapply` over 6.46M rows, per-row subsetting | Sparse matrixâ€“vector multiply (`A %*% x`), C-level BLAS |
| **Max/Min computation** | `lapply` over 6.46M rows, per-row `max`/`min` | Vectorized CSC pointer expansion â†’ `data.table` grouped aggregation |
| **Topology reuse** | Implicitly rebuilt for every row (year baked into key) | Built once, reused across 28 years and 5 variables |
| **Memory** | 6.46M-element R list (multi-GB overhead) | ~22MB sparse matrix + temporary vectors |
| **Estimated runtime** | ~86+ hours | ~5â€“15 minutes |
| **Numerical equivalence** | Baseline | Preserved: identical `max`, `min`, `mean` with same NA handling |
| **RF model** | N/A | Untouched; `predict()` only |

**Key insight**: The neighbor graph is a spatial invariant. By factoring out time, the topology shrinks from ~6.46M nodes to ~344K nodes. Sparse linear algebra (`Matrix` package) then replaces millions of R-level function calls with a handful of optimized C-level operations per year-variable combination.
END_RESPONSE>>>
