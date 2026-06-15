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
run_id: Anthropic_F10_T00_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each constructed via string-pasting, named-vector lookup, and `NA` filtering. This is an O(n) loop over millions of rows with expensive string operations (`paste`, named vector indexing). The string-keyed lookup `idx_lookup` is a named character vector of length ~6.46M â€” each lookup into it is O(n) in the worst case because R named vectors use linear hashing that degrades at scale.

2. **`compute_neighbor_stats` iterates over ~6.46M list elements** per variable, subsetting a numeric vector and computing `max/min/mean`. With 5 variables, that's ~32.3 million R-level list iterations with repeated allocation.

3. **The neighbor lookup is row-indexed (cell-year level)** when the topology is actually **year-invariant**. The rook-neighbor graph is purely spatial â€” it doesn't change across years. Yet the code rebuilds the full cell-year neighbor mapping, inflating the problem by 28Ã—.

**Key insight:** The adjacency structure is defined over 344,208 cells, not 6.46M cell-years. The neighbor aggregation for year `t` only uses neighbor values from year `t`. Therefore, we should:
- Build the sparse adjacency matrix **once** over cells (344K Ã— 344K).
- For each year, slice the relevant rows, perform sparse matrixâ€“vector multiplication (or grouped operations) to get `max`, `min`, `mean`.

Sparse matrix multiplication directly gives `sum` (and from that, `mean`). For `max` and `min`, we need grouped operations, but we can use the CSR structure of the sparse matrix to do this efficiently in vectorized C-level code via `data.table` or direct sparse-matrix tricks.

---

## Optimization Strategy

1. **Build a sparse adjacency matrix `A`** (344,208 Ã— 344,208) from `rook_neighbors_unique` once. This is ~1.37M non-zero entries â€” trivially small.

2. **For `mean`:** For each year, extract the variable column as a vector `x` of length 344,208 (ordered by cell). Then `A %*% x` gives the sum of neighbor values. Divide element-wise by the number of neighbors (row sums of `A`) to get the mean. This is a single sparse matrixâ€“vector multiply â€” microseconds.

3. **For `max` and `min`:** Convert `A` to a `data.table` edge list once. For each year-variable, do a keyed join and grouped `max`/`min`. With `data.table` this is highly optimized C-level code.

4. **Process year-by-year** to keep memory bounded (only ~344K rows in memory per year-slice), or process all at once if memory allows.

5. **Estimated speedup:** From ~86 hours to **minutes**. The sparse matrix multiply for 5 variables Ã— 28 years = 140 sparse mat-vec operations on a 344K-dimension matrix with 1.37M nonzeros â€” essentially instantaneous. The `data.table` grouped `max`/`min` over 1.37M edges per year-variable is also very fast.

---

## Optimized R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build sparse adjacency matrix and edge list ONCE
# ==============================================================================

build_sparse_adjacency <- function(id_order, nb_object) {
  # nb_object: spdep nb object (list of integer neighbor index vectors)
  # id_order: vector of cell IDs in the order matching nb_object
  
  n <- length(id_order)
  stopifnot(length(nb_object) == n)
  
  # Build COO triplets
  from_idx <- rep(seq_len(n), times = vapply(nb_object, length, integer(1)))
  to_idx   <- unlist(nb_object, use.names = FALSE)
  
  # Remove any 0-neighbor sentinel (spdep uses integer(0) or 0L for no neighbors)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  # Sparse adjacency matrix (rows = focal cell, cols = neighbor cell)
  A <- sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = 1,
    dims = c(n, n),
    dimnames = list(as.character(id_order), as.character(id_order))
  )
  
  # Number of neighbors per cell (for computing mean)
  n_neighbors <- diff(A@p)  # CSC column counts if transposed; use rowSums instead
  n_neighbors_vec <- as.numeric(rowSums(A))  # guaranteed correct
  
  # Edge list as data.table for max/min operations
  edge_dt <- data.table(
    from = from_idx,
    to   = to_idx
  )
  setkey(edge_dt, from)
  
  list(
    A              = A,
    n_neighbors    = n_neighbors_vec,
    edge_dt        = edge_dt,
    id_order       = id_order,
    n              = n
  )
}

# ==============================================================================
# STEP 2: Compute neighbor stats for all years, all variables
# ==============================================================================

compute_all_neighbor_features <- function(cell_data, neighbor_source_vars, 
                                           id_order, nb_object) {
  
  # Convert to data.table for speed
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # Build adjacency structures once
  cat("Building sparse adjacency structures...\n")
  adj <- build_sparse_adjacency(id_order, nb_object)
  A            <- adj$A
  n_neighbors  <- adj$n_neighbors
  edge_dt      <- adj$edge_dt
  n_cells      <- adj$n
  
  # Create a mapping from cell id to positional index (matching nb_object order)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Get sorted unique years
  years <- sort(unique(cell_data$year))
  
  # Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)
  }
  
  # Index cell_data rows by (year, cell position) for fast assignment
  # Add positional index column
  cell_data[, .pos := id_to_pos[as.character(id)]]
  
  # Process year by year
  cat(sprintf("Processing %d years x %d variables...\n", 
              length(years), length(neighbor_source_vars)))
  
  for (yr in years) {
    
    # Row indices in cell_data for this year
    yr_rows <- which(cell_data$year == yr)
    
    if (length(yr_rows) == 0L) next
    
    # Extract the sub-table for this year, keyed by cell position
    yr_dt <- cell_data[yr_rows, c("id", ".pos", neighbor_source_vars), with = FALSE]
    setkey(yr_dt, .pos)
    
    # Build a fast lookup: position -> value vector (length n_cells, NA for missing)
    # This ensures alignment with the adjacency matrix
    
    for (var_name in neighbor_source_vars) {
      
      col_max  <- paste0("neighbor_max_", var_name)
      col_min  <- paste0("neighbor_min_", var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      
      # Create dense vector aligned to adjacency matrix positions
      x <- rep(NA_real_, n_cells)
      x[yr_dt$.pos] <- yr_dt[[var_name]]
      
      # --- MEAN via sparse matrix-vector multiply ---
      # A %*% x gives sum of neighbor values (NA treated as 0 by Matrix)
      # We need to handle NAs carefully to preserve numerical equivalence
      
      # Replace NA with 0 for sum computation, track valid counts
      x_nona <- x
      x_nona[is.na(x_nona)] <- 0
      x_valid <- as.numeric(!is.na(x))  # 1 if valid, 0 if NA
      
      neighbor_sum   <- as.numeric(A %*% x_nona)
      neighbor_count <- as.numeric(A %*% x_valid)
      
      neighbor_mean <- ifelse(neighbor_count > 0, 
                              neighbor_sum / neighbor_count, 
                              NA_real_)
      # Cells with no neighbors at all -> NA
      neighbor_mean[n_neighbors == 0] <- NA_real_
      
      # --- MAX and MIN via data.table grouped operations ---
      # Get neighbor values via edge list
      edge_vals <- edge_dt[, .(from, val = x[to])]
      # Remove NA neighbor values
      edge_valid <- edge_vals[!is.na(val)]
      
      if (nrow(edge_valid) > 0) {
        agg <- edge_valid[, .(nmax = max(val), nmin = min(val)), by = from]
        
        neighbor_max <- rep(NA_real_, n_cells)
        neighbor_min <- rep(NA_real_, n_cells)
        neighbor_max[agg$from] <- agg$nmax
        neighbor_min[agg$from] <- agg$nmin
      } else {
        neighbor_max <- rep(NA_real_, n_cells)
        neighbor_min <- rep(NA_real_, n_cells)
      }
      
      # Map results back to cell_data rows for this year
      # yr_dt$.pos tells us which adjacency-matrix position each row corresponds to
      positions <- yr_dt$.pos
      
      set(cell_data, i = yr_rows, j = col_max,  value = neighbor_max[positions])
      set(cell_data, i = yr_rows, j = col_min,  value = neighbor_min[positions])
      set(cell_data, i = yr_rows, j = col_mean, value = neighbor_mean[positions])
    }
    
    if (yr %% 5 == 0 || yr == years[1] || yr == years[length(years)]) {
      cat(sprintf("  Year %d done.\n", yr))
    }
  }
  
  # Clean up temporary column
  cell_data[, .pos := NULL]
  
  cat("Neighbor feature computation complete.\n")
  return(cell_data)
}

# ==============================================================================
# STEP 3: Full pipeline â€” compute features, then predict
# ==============================================================================

run_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                         trained_rf_model) {
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  # Compute all neighbor features
  cell_data <- compute_all_neighbor_features(
    cell_data, 
    neighbor_source_vars, 
    id_order, 
    rook_neighbors_unique
  )
  
  # Apply pre-trained Random Forest (no retraining)
  # Extract the predictor columns the model expects
  pred_vars <- trained_rf_model$forest$independent.variable.names  # ranger
  # If using randomForest package instead:
  # pred_vars <- rownames(importance(trained_rf_model))
  
  cat(sprintf("Predicting with %d predictor variables...\n", length(pred_vars)))
  
  predictions <- predict(trained_rf_model, data = cell_data[, ..pred_vars])
  cell_data[, prediction := predictions$predictions]
  
  return(cell_data)
}

# ==============================================================================
# USAGE
# ==============================================================================

# # Load pre-existing objects
# cell_data              <- readRDS("cell_data.rds")
# id_order               <- readRDS("id_order.rds")
# rook_neighbors_unique  <- readRDS("rook_neighbors_unique.rds")
# trained_rf_model       <- readRDS("trained_rf_model.rds")
# 
# result <- run_pipeline(cell_data, id_order, rook_neighbors_unique, trained_rf_model)
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + named-vector lookups | Single sparse matrix build (344K Ã— 344K, 1.37M nnz) |
| **Mean computation** | 6.46M R list iterations per variable | Sparse matrixâ€“vector multiply (~milliseconds) |
| **Max/Min computation** | 6.46M R list iterations per variable | `data.table` grouped aggregation over 1.37M edges |
| **Topology reuse** | Rebuilt per cell-year (28Ã— redundant) | Built once, reused across all 28 years |
| **Memory** | ~6.46M-element list of integer vectors | ~1.37M-entry sparse matrix + edge `data.table` |
| **Estimated runtime** | 86+ hours | **~5â€“15 minutes** |
| **Numerical equivalence** | Baseline | âœ… Preserved (same max, min, mean with NA handling) |
| **Trained RF model** | â€” | âœ… Unchanged, predict-only |

**Key correctness details preserved:**
- NAs in neighbor values are excluded before computing `max`, `min`, `mean` (matching `neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]`).
- Cells with zero valid neighbors get `NA` for all three stats (matching `return(c(NA, NA, NA))`).
- The sparse matrix `mean` uses `A %*% x_valid` to count only non-NA neighbors in the denominator, exactly replicating `mean(neighbor_vals)` after NA removal.
END_RESPONSE>>>
