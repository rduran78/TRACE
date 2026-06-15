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
run_id: Anthropic_F08_T10_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) relationships.** It creates a list of length ~6.46 million, where each element maps a cell-year row to its neighbor cell-year rows. But the neighbor *topology* is identical across all 28 years â€” only the variable *values* change. This means the function does **28Ã— redundant work** discovering the same spatial relationships, and creates a lookup structure **28Ã— larger than necessary**.

2. **String-key hashing is expensive.** The function pastes `id_year` strings and indexes into a named character vector (`idx_lookup`) for every single neighbor edge, across every year. With ~1.37M directed edges Ã— 28 years, this produces ~38.5 million string-match lookups â€” all to rediscover the same cellâ†’cell mapping that is already encoded in `rook_neighbors_unique`.

3. **`compute_neighbor_stats` operates on the bloated 6.46M-element lookup.** Each call iterates over all cell-year rows rather than operating on a per-year matrix slice, defeating cache locality and vectorization opportunities.

4. **Memory pressure.** The 6.46M-element list of integer vectors consumes substantial RAM and puts GC pressure on a 16 GB laptop.

### The Key Insight

> **Neighbor topology is static (cell-to-cell); only the attached variables are dynamic (year-varying).**

The neighbor list `rook_neighbors_unique` already defines, for each of the 344,208 cells, which other cells are its neighbors. This never changes. The only thing that changes across years is the *value* of variables like `ntl`, `ec`, etc. Therefore:

- Build the cell-to-cell neighbor index **once** (344K entries, not 6.46M).
- For each variable, reshape values into a **cells Ã— years matrix**, then compute neighbor max/min/mean using vectorized operations on matrix columns (years), reusing the single static neighbor index.

This reduces complexity by a factor of ~28Ã— in lookup construction and enables fully vectorized stats computation.

---

## Optimization Strategy

| Aspect | Current (Slow) | Redesigned (Fast) |
|---|---|---|
| Lookup granularity | cellÃ—year (6.46M entries) | cell only (344K entries) |
| Lookup keys | String paste + named vector match | Direct integer indexing from `nb` object |
| Stats computation | `lapply` over 6.46M rows per variable | Vectorized matrix operations over 344K cells Ã— 28 years |
| Total lookup builds | 1 (but huge) | 1 (small) |
| Stats calls | 5 Ã— 6.46M = 32.3M `lapply` iterations | 5 variables Ã— 28 years, each vectorized over 344K cells |
| Estimated speedup | Baseline (~86+ hrs) | **~20â€“60 minutes** (conservative) |

### Algorithmic Steps

1. **Build a static cell-level neighbor index** from `rook_neighbors_unique`: a list of 344K integer vectors mapping each cell's positional index to its neighbors' positional indices.

2. **Establish a stable cell ordering** so that cell `i` in the neighbor index corresponds to row `i` in a values matrix.

3. **Reshape each variable into a 344,208 Ã— 28 matrix** (cells Ã— years).

4. **For each variable, compute neighbor stats in vectorized fashion:**
   - For each cell, gather neighbor values from the matrix (all years at once via matrix row-subsetting).
   - Use optimized row-wise max/min/mean across neighbor sets.

5. **Write results back** into the original `cell_data` data frame in the same column format as before, preserving the exact numerical estimand for the pre-trained Random Forest.

---

## Working R Code

```r
# =============================================================================
# REDESIGNED NEIGHBOR STATS COMPUTATION
# Exploits: static topology (cell-to-cell) + dynamic values (year-varying)
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique, 
                                           neighbor_source_vars) {
  
  # -------------------------------------------------------------------------
  # STEP 0: Convert to data.table for speed (non-destructive)
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # -------------------------------------------------------------------------
  # STEP 1: Build STATIC cell-level neighbor index (done ONCE)
  #
  # rook_neighbors_unique is an spdep::nb object: a list of length N where

  # element [[i]] contains integer indices of neighbors of cell i, referencing
  # positions in id_order. A value of 0L means no neighbors.
  # We convert this to a clean list of integer vectors.
  # -------------------------------------------------------------------------
  message("Building static cell-level neighbor index (", 
          format(length(id_order), big.mark = ","), " cells)...")
  
  n_cells <- length(id_order)
  
  # spdep::nb stores 0L for no-neighbor cells; clean those out
  static_neighbor_idx <- lapply(rook_neighbors_unique, function(nb) {
    nb <- nb[nb != 0L]
    as.integer(nb)
  })
  # Now static_neighbor_idx[[i]] gives the positional indices (in id_order)
  # of the neighbors of the i-th cell in id_order.
  
  message("  Static neighbor index built. Length: ", length(static_neighbor_idx))
  
  # -------------------------------------------------------------------------
  # STEP 2: Establish stable cell ordering and year ordering
  #
  # We need a mapping: cell id -> position in id_order (1..N)
  # And: year -> column index in our values matrix
  # -------------------------------------------------------------------------
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  message("  Cells: ", format(n_cells, big.mark = ","), 
          ", Years: ", n_years, " (", min(years), "-", max(years), ")")
  
  # -------------------------------------------------------------------------
  # STEP 3: Map each row in dt to (cell_position, year_column)
  # -------------------------------------------------------------------------
  dt[, cell_pos := id_to_pos[as.character(id)]]
  dt[, year_col := year_to_col[as.character(year)]]
  
  # Verify complete mapping
  stopifnot(!anyNA(dt$cell_pos), !anyNA(dt$year_col))
  
  # Create a row-index matrix: row_index_mat[cell_pos, year_col] = row in dt
  # This lets us write results back to the correct rows.
  row_index_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_index_mat[cbind(dt$cell_pos, dt$year_col)] <- seq_len(nrow(dt))
  
  # -------------------------------------------------------------------------
  # STEP 4: For each variable, reshape â†’ compute neighbor stats â†’ write back
  # -------------------------------------------------------------------------
  
  for (var_name in neighbor_source_vars) {
    
    message("Processing neighbor stats for: ", var_name, " ...")
    t_start <- proc.time()
    
    # 4a. Reshape variable into cells Ã— years matrix
    val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    val_mat[cbind(dt$cell_pos, dt$year_col)] <- dt[[var_name]]
    
    # 4b. Compute neighbor max, min, mean for all cells Ã— all years
    #     Result matrices: same dimension as val_mat
    nb_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # Pre-compute: for each cell, extract the neighbor rows from val_mat
    # This is the inner loop over 344K cells (vectorized across years)
    for (i in seq_len(n_cells)) {
      nb_idx <- static_neighbor_idx[[i]]
      if (length(nb_idx) == 0L) next
      
      # nb_vals is a (num_neighbors Ã— n_years) matrix
      # Each column is one year; each row is one neighbor's value that year
      if (length(nb_idx) == 1L) {
        # Single neighbor: matrix with 1 row
        nb_vals <- matrix(val_mat[nb_idx, ], nrow = 1L, ncol = n_years)
      } else {
        nb_vals <- val_mat[nb_idx, , drop = FALSE]
      }
      
      # Vectorized column-wise (i.e., per-year) stats using colMins etc.
      # Using matrixStats if available, otherwise base R
      # For robustness, use base R apply or manual colwise ops:
      nb_max[i, ]  <- apply(nb_vals, 2L, max,  na.rm = TRUE)
      nb_min[i, ]  <- apply(nb_vals, 2L, min,  na.rm = TRUE)
      nb_mean[i, ] <- colMeans(nb_vals, na.rm = TRUE)
    }
    
    # Handle -Inf/Inf from max/min on all-NA columns â†’ convert to NA
    nb_max[is.infinite(nb_max)]  <- NA_real_
    nb_min[is.infinite(nb_min)]  <- NA_real_
    nb_mean[is.nan(nb_mean)]     <- NA_real_
    
    # 4c. Write results back to data.table using the row index mapping
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    # Flatten matrices back to vector aligned with dt rows
    valid <- !is.na(row_index_mat)
    dt_rows   <- row_index_mat[valid]
    max_vals  <- nb_max[valid]
    min_vals  <- nb_min[valid]
    mean_vals <- nb_mean[valid]
    
    set(dt, i = dt_rows, j = max_col,  value = max_vals)
    set(dt, i = dt_rows, j = min_col,  value = min_vals)
    set(dt, i = dt_rows, j = mean_col, value = mean_vals)
    
    elapsed <- (proc.time() - t_start)["elapsed"]
    message("  Done: ", var_name, " in ", round(elapsed / 60, 1), " minutes.")
  }
  
  # -------------------------------------------------------------------------
  # STEP 5: Clean up helper columns, return as data.frame
  # -------------------------------------------------------------------------
  dt[, c("cell_pos", "year_col") := NULL]
  
  message("All neighbor features computed.")
  return(as.data.frame(dt))
}


# =============================================================================
# OPTIONAL: Faster version using matrixStats (if installed)
# Replaces the inner apply() calls with compiled C routines
# =============================================================================

compute_all_neighbor_features_fast <- function(cell_data, id_order, 
                                                rook_neighbors_unique, 
                                                neighbor_source_vars) {
  
  if (!requireNamespace("matrixStats", quietly = TRUE)) {
    message("matrixStats not found; falling back to base R version.")
    return(compute_all_neighbor_features(cell_data, id_order, 
                                          rook_neighbors_unique, 
                                          neighbor_source_vars))
  }
  
  library(data.table)
  library(matrixStats)
  
  dt <- as.data.table(cell_data)
  n_cells <- length(id_order)
  
  # Static neighbor index
  static_neighbor_idx <- lapply(rook_neighbors_unique, function(nb) {
    nb <- nb[nb != 0L]
    as.integer(nb)
  })
  
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  dt[, cell_pos := id_to_pos[as.character(id)]]
  dt[, year_col := year_to_col[as.character(year)]]
  
  row_index_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_index_mat[cbind(dt$cell_pos, dt$year_col)] <- seq_len(nrow(dt))
  
  # Pre-compute CSR-like structure for batch processing
  message("Building CSR neighbor structure for vectorized access...")
  nb_lengths <- vapply(static_neighbor_idx, length, integer(1L))
  nb_flat    <- unlist(static_neighbor_idx, use.names = FALSE)
  nb_offsets <- c(0L, cumsum(nb_lengths))
  # nb_offsets[i]+1 .. nb_offsets[i+1] gives positions in nb_flat for cell i
  
  for (var_name in neighbor_source_vars) {
    
    message("Processing: ", var_name, " ...")
    t_start <- proc.time()
    
    val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    val_mat[cbind(dt$cell_pos, dt$year_col)] <- dt[[var_name]]
    
    nb_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # Process in chunks to manage memory and allow progress reporting
    chunk_size <- 10000L
    n_chunks <- ceiling(n_cells / chunk_size)
    
    for (ch in seq_len(n_chunks)) {
      i_start <- (ch - 1L) * chunk_size + 1L
      i_end   <- min(ch * chunk_size, n_cells)
      
      for (i in i_start:i_end) {
        n_nb <- nb_lengths[i]
        if (n_nb == 0L) next
        
        from <- nb_offsets[i] + 1L
        to   <- nb_offsets[i + 1L]
        nb_idx <- nb_flat[from:to]
        
        if (n_nb == 1L) {
          # Single neighbor: just copy that row
          nb_max[i, ]  <- val_mat[nb_idx, ]
          nb_min[i, ]  <- val_mat[nb_idx, ]
          nb_mean[i, ] <- val_mat[nb_idx, ]
        } else {
          nb_vals <- val_mat[nb_idx, , drop = FALSE]
          nb_max[i, ]  <- colMaxs(nb_vals,  na.rm = TRUE)
          nb_min[i, ]  <- colMins(nb_vals,  na.rm = TRUE)
          nb_mean[i, ] <- colMeans2(nb_vals, na.rm = TRUE)
        }
      }
      
      if (ch %% 5 == 0 || ch == n_chunks) {
        message("  Chunk ", ch, "/", n_chunks, " done.")
      }
    }
    
    nb_max[is.infinite(nb_max)]  <- NA_real_
    nb_min[is.infinite(nb_min)]  <- NA_real_
    nb_mean[is.nan(nb_mean)]     <- NA_real_
    
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    valid     <- !is.na(row_index_mat)
    dt_rows   <- row_index_mat[valid]
    set(dt, i = dt_rows, j = max_col,  value = nb_max[valid])
    set(dt, i = dt_rows, j = min_col,  value = nb_min[valid])
    set(dt, i = dt_rows, j = mean_col, value = nb_mean[valid])
    
    elapsed <- (proc.time() - t_start)["elapsed"]
    message("  ", var_name, ": ", round(elapsed / 60, 1), " min")
  }
  
  dt[, c("cell_pos", "year_col") := NULL]
  return(as.data.frame(dt))
}


# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Replace the original 3-step process:
#   neighbor_lookup <- build_neighbor_lookup(...)
#   for (var_name in ...) { cell_data <- compute_and_add_neighbor_features(...) }
#
# With this single call:
cell_data <- compute_all_neighbor_features_fast(
  cell_data, 
  id_order, 
  rook_neighbors_unique, 
  neighbor_source_vars
)

# Then proceed to prediction with the pre-trained Random Forest as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` computed on exactly the same neighbor sets with `na.rm = TRUE`, producing identical values to the original. The `is.infinite` â†’ `NA` and `is.nan` â†’ `NA` cleanup matches the original's `return(c(NA, NA, NA))` for all-NA or empty neighbor cases. |
| **Column naming** | Produces `neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>` â€” same as the original `compute_and_add_neighbor_features`. |
| **Pre-trained RF preserved** | No model retraining. The function only computes features; prediction proceeds unchanged afterward. |
| **Memory on 16 GB laptop** | Peak: one `344,208 Ã— 28` matrix (~77 MB) for values + three for results (~231 MB) + the data.table (~5-6 GB) + static neighbor index (~50 MB). Well within 16 GB. |
| **Expected runtime** | ~15â€“45 minutes total (vs. 86+ hours), depending on disk I/O and GC. The inner loop iterates 344K cells (not 6.46M rows), and each iteration is vectorized across 28 years via `colMaxs`/`colMins`/`colMeans2`. |
END_RESPONSE>>>
