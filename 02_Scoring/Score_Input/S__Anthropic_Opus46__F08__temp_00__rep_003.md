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
run_id: Anthropic_F08_T00_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) neighbor indices.** It creates a list of 6.46 million elements, each requiring string-pasting, hash lookups, and NA filtering. But the neighbor *relationships* between cells are purely spatial and **never change across years**. The only thing that changes is the variable values attached to those cells each year.

2. **String-key hashing is expensive at scale.** The function creates `paste(id, year)` keys for all 6.46M rows, then for each row, pastes neighbor IDs with that row's year, and looks them up. This is O(n Ã— k) string operations where n = 6.46M and k = average neighbor count (~4 for rook).

3. **`compute_neighbor_stats` iterates over 6.46M list elements with `lapply`.** Each call extracts neighbor values, removes NAs, and computes max/min/mean â€” all in an interpreted R loop.

4. **The entire process repeats for each of 5 variables**, but the neighbor lookup (the expensive part) is the same every time.

### The Key Insight

- **Static:** The neighbor graph (which cells are neighbors of which) is fixed across all 28 years.
- **Dynamic:** The variable values (ntl, ec, pop_density, def, usd_est_n2) change by year.

Therefore, we should:
- Build the neighbor lookup **once at the cell level** (344K cells, not 6.46M cell-years).
- For each variable, split by year, compute neighbor stats using the **cell-level** neighbor structure via vectorized/matrix operations, then reassemble.

---

## Optimization Strategy

### 1. Cell-Level Neighbor Index (build once, ~344K entries)

Build a simple list mapping each cell's positional index (1..344208) to its neighbors' positional indices. This is a one-time O(344K) operation using the existing `rook_neighbors_unique` nb object directly â€” no string hashing needed.

### 2. Sparse Adjacency Matrix (build once)

Convert the nb object to a sparse adjacency matrix (`dgCMatrix`). This enables **vectorized** neighbor aggregation via sparse matrixâ€“vector multiplication for the mean, and analogous operations for max and min.

### 3. Year-Sliced Vectorized Computation

For each year and each variable:
- Extract the variable as a vector aligned to cell order.
- **Mean:** One sparse matrixâ€“vector multiply + divide by neighbor count.
- **Max/Min:** Use the sparse matrix structure to compute row-wise max/min efficiently in C++ via a small Rcpp function, or use a grouped approach.

### 4. Complexity Reduction

| Aspect | Before | After |
|---|---|---|
| Lookup construction | 6.46M string-key lookups | 1 sparse matrix (344K Ã— 344K) |
| Stats computation | 6.46M Ã— 5 R-level lapply iterations | 28 Ã— 5 sparse mat-vec ops |
| Estimated time | 86+ hours | **~2â€“5 minutes** |

---

## Working R Code

```r
library(Matrix)
library(spdep)
library(data.table)

# ==============================================================================
# STEP 1: Build a sparse adjacency matrix from the nb object (ONCE, static)
# ==============================================================================

build_sparse_neighbor_matrix <- function(nb_obj) {
  # nb_obj: spdep nb object, length = number of cells (344,208)
  n <- length(nb_obj)
  
  # Build COO triplets
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # Remove zero-neighbor placeholders (spdep uses integer(0) or 0L)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  # Sparse binary adjacency matrix (rows = focal cell, cols = neighbor cell)
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  return(W)
}

# ==============================================================================
# STEP 2: Compute neighbor max, min, mean for one variable across all years
#          using the sparse matrix
# ==============================================================================

compute_neighbor_features_sparse <- function(DT, var_name, W, id_order) {
  # DT:       data.table with columns: id, year, <var_name>
  # W:        sparse adjacency matrix (n_cells x n_cells)
  # id_order: vector of cell IDs in the order matching W's row/col indices
  
  n_cells <- length(id_order)
  
  # Map cell id -> matrix row index (positional index in id_order)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add positional index to DT
  DT[, cell_pos := id_to_pos[as.character(id)]]
  
  # Precompute neighbor counts per cell (static)
  neighbor_count <- as.numeric(rowSums(W))  # length n_cells
  
  # Column names for output
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Initialize output columns
  DT[, (col_max)  := NA_real_]
  DT[, (col_min)  := NA_real_]
  DT[, (col_mean) := NA_real_]
  
  # Get the sparse structure for row-wise max/min computation
  # W is in dgCMatrix (column-compressed) format; convert to dgRMatrix or
  # use the dgCMatrix directly. For row operations, dgRMatrix is better.
  # However, we can also work with dgCMatrix by transposing.
  # We'll extract the row-pointer structure.
  
  Wt <- t(W)  # Now Wt is dgCMatrix; column j of Wt = neighbors of cell j
  # Wt@p: column pointers, Wt@i: row indices (= neighbor cell indices)
  
  years <- sort(unique(DT$year))
  
  for (yr in years) {
    # Extract rows for this year
    yr_mask <- DT$year == yr
    
    # Build a full-length vector for this variable, indexed by cell_pos
    # (some cells may be missing for a year; they get NA)
    val_vec <- rep(NA_real_, n_cells)
    
    yr_pos  <- DT$cell_pos[yr_mask]
    yr_vals <- DT[[var_name]][yr_mask]
    val_vec[yr_pos] <- yr_vals
    
    # --- MEAN: sparse matrix-vector multiply ---
    # W %*% val_vec gives sum of neighbor values for each cell
    # Handle NAs: replace NA with 0 for sum, track non-NA count
    val_nona <- val_vec
    val_nona[is.na(val_nona)] <- 0
    
    is_valid <- as.numeric(!is.na(val_vec))
    
    neighbor_sum     <- as.numeric(W %*% val_nona)
    neighbor_nvalid  <- as.numeric(W %*% is_valid)
    
    neighbor_mean_vec <- ifelse(neighbor_nvalid > 0,
                                neighbor_sum / neighbor_nvalid,
                                NA_real_)
    
    # --- MAX and MIN: iterate over sparse structure ---
    # Use Wt (dgCMatrix). For each cell j, neighbors are in 
    # Wt@i[(Wt@p[j]+1):Wt@p[j+1]] (0-based indices)
    neighbor_max_vec <- rep(NA_real_, n_cells)
    neighbor_min_vec <- rep(NA_real_, n_cells)
    
    p_ptr <- Wt@p   # length n_cells + 1, 0-based
    i_idx <- Wt@i   # 0-based row indices
    
    # Vectorized approach: for each cell, gather neighbor values
    # We can do this efficiently by working on the full neighbor value vector
    # and using grouping.
    
    # Build a "neighbor values" vector aligned to the sparse entries
    # i_idx contains the neighbor cell indices (0-based)
    all_neighbor_vals <- val_vec[i_idx + 1L]  # +1 for R's 1-based indexing
    
    # Build a grouping vector: which focal cell does each entry belong to?
    # Cell j owns entries from p_ptr[j]+1 to p_ptr[j+1] (1-based: p_ptr[j+1]+1 to p_ptr[j+1+1])
    n_neighbors_per_cell <- diff(p_ptr)  # length n_cells
    focal_cell_group <- rep(seq_len(n_cells), times = n_neighbors_per_cell)
    
    # Now compute grouped max and min using data.table for speed
    if (length(all_neighbor_vals) > 0) {
      tmp_dt <- data.table(
        focal = focal_cell_group,
        nval  = all_neighbor_vals
      )
      
      # Remove NA neighbor values before aggregation
      tmp_dt <- tmp_dt[!is.na(nval)]
      
      if (nrow(tmp_dt) > 0) {
        agg <- tmp_dt[, .(nmax = max(nval), nmin = min(nval)), by = focal]
        neighbor_max_vec[agg$focal] <- agg$nmax
        neighbor_min_vec[agg$focal] <- agg$nmin
      }
    }
    
    # Write results back to DT for this year's rows
    set(DT, which = yr_mask, j = col_max,  value = neighbor_max_vec[yr_pos])
    set(DT, which = yr_mask, j = col_min,  value = neighbor_min_vec[yr_pos])
    set(DT, which = yr_mask, j = col_mean, value = neighbor_mean_vec[yr_pos])
  }
  
  # Clean up helper column
  DT[, cell_pos := NULL]
  
  return(DT)
}

# ==============================================================================
# STEP 3: Full pipeline â€” drop-in replacement for the original outer loop
# ==============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # Convert to data.table for performance (if not already)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  cat("Building sparse adjacency matrix (one-time, static topology)...\n")
  W <- build_sparse_neighbor_matrix(rook_neighbors_unique)
  cat(sprintf("  Sparse matrix: %d x %d, %d non-zero entries\n",
              nrow(W), ncol(W), nnzero(W)))
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor features for '%s'...\n", var_name))
    t0 <- proc.time()
    cell_data <- compute_neighbor_features_sparse(
      cell_data, var_name, W, id_order
    )
    elapsed <- (proc.time() - t0)[3]
    cat(sprintf("  Done in %.1f seconds.\n", elapsed))
  }
  
  # Convert back to data.frame if downstream code expects it
  # cell_data <- as.data.frame(cell_data)  # uncomment if needed
  
  return(cell_data)
}

# ==============================================================================
# USAGE (replaces the original outer loop)
# ==============================================================================

cell_data <- run_neighbor_feature_pipeline(
  cell_data, id_order, rook_neighbors_unique
)

# Then proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Requirement | Guarantee |
|---|---|
| **Same numerical estimand** | Max, min, and mean are computed over exactly the same neighbor sets and the same variable values. The sparse matrix encodes the identical rook-neighbor graph. NA handling is preserved. |
| **Trained RF model untouched** | No model retraining occurs. The output columns have the same names and semantics. The prediction call is unchanged. |
| **Static vs. dynamic separation** | The sparse matrix `W` is built **once** from the nb object (static topology). Variable values are sliced **per year** (dynamic data). This is the core redesign. |

## Performance Summary

| Operation | Count | Cost |
|---|---|---|
| Sparse matrix construction | 1 | ~1 second |
| Sparse mat-vec multiply (mean) | 28 years Ã— 5 vars = 140 | ~0.1s each |
| Grouped max/min via data.table | 140 | ~0.3s each |
| **Total estimated wall time** | | **~2â€“5 minutes** |

This represents a speedup of roughly **1,000â€“2,500Ã—** over the original 86+ hour implementation, achieved entirely by separating the static neighbor topology from the dynamic yearly variable values.
END_RESPONSE>>>
