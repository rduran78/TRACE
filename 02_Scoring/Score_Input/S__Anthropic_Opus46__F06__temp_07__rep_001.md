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
run_id: Anthropic_F06_T07_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) element-wise `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's spatial neighbors (fast).
2. Constructs character key strings via `paste()` for every neighbor Ã— year combination.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) using character matching against a 6.46M-element named vector.

Character-keyed lookups on a vector of length 6.46M are **O(n)** per probe in the worst case (R named vectors use hashing, but the overhead of creating ~6.46M character keys and hashing them is enormous). This loop runs 6.46 million iterations, each doing multiple `paste` + hash-lookup operations. This alone accounts for many hours.

### Bottleneck 2: `compute_neighbor_stats` â€” Repeated per-row `lapply` over 6.46M rows, called 5 times

For each of the 5 variables, another `lapply` iterates over 6.46M rows, subsetting and computing `max`, `min`, `mean`. The subsetting is fast (integer indexing), but the R-level loop overhead across 6.46M iterations Ã— 5 variables â‰ˆ 32.3M R function calls is substantial.

### Why raster focal/kernel operations don't directly apply

Focal operations assume a regular grid with a fixed rectangular kernel. Here the data is a panel (cell Ã— year) stored as a long data frame, and the neighbor structure is an irregular `spdep::nb` object (not necessarily a regular lattice). Focal operations would require reshaping into a raster stack per year and would not generalize to irregular grids. The comment in the prompt is a red herring â€” the correct approach is vectorized sparse-matrix multiplication.

---

## Optimization Strategy

### Key Insight: Neighbor summary statistics are sparse-matrix operations

If we construct a **sparse adjacency matrix W** (cells Ã— cells) from the `spdep::nb` object, then for any variable vector **v** (one year at a time, or reshaped), the neighbor **mean** is simply:

```
W_rowstandardized %*% v
```

And neighbor **max** and **min** can be computed via row-wise operations on a sparse matrix of neighbor values.

However, since we need max, min, AND mean, and sparse matrix algebra gives us sum/mean directly but not max/min, we use a **hybrid approach**:

1. **Replace `build_neighbor_lookup`** with a single vectorized construction using `data.table` joins â€” O(n) with hash joins instead of O(n) character-vector probes per row.
2. **Replace `compute_neighbor_stats`** with sparse matrix operations for **mean** (via `Matrix` package) and vectorized grouped operations for **max/min** (via `data.table`).
3. Process **year-by-year** to keep memory bounded and enable vectorized operations within each year-slice.

### Expected speedup

| Component | Before | After |
|---|---|---|
| `build_neighbor_lookup` | ~40+ hours (character key loop) | ~30 seconds (data.table join) |
| `compute_neighbor_stats` Ã— 5 | ~40+ hours (R-level lapply) | ~2â€“5 minutes (sparse matrix + data.table) |
| **Total** | **~86+ hours** | **~3â€“8 minutes** |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# 
# Requirements: data.table, Matrix, spdep (already used)
# Preserves: trained Random Forest model (untouched)
# Preserves: original numerical estimand (max, min, mean of rook neighbors)
# =============================================================================

library(data.table)
library(Matrix)

compute_all_neighbor_features <- function(cell_data, 
                                           id_order, 
                                           rook_neighbors_unique, 
                                           neighbor_source_vars) {
  # --------------------------------------------------------------------------
  # STEP 1: Build sparse adjacency matrix from spdep::nb object
  # --------------------------------------------------------------------------
  # id_order maps position index (1..N_cells) to cell id.
  # rook_neighbors_unique[[i]] gives integer indices of neighbors of cell i.
  
  n_cells <- length(id_order)
  
  # Build COO (coordinate) representation of adjacency
  from_list <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  to_list   <- unlist(rook_neighbors_unique)
  
  # Remove any zero-length / empty entries (islands with no neighbors)
  valid <- !is.na(to_list) & to_list > 0
  from_list <- from_list[valid]
  to_list   <- to_list[valid]
  
  # Sparse binary adjacency matrix (n_cells x n_cells)
  # W[i,j] = 1 means cell j is a rook neighbor of cell i
  W <- sparseMatrix(
    i = from_list, 
    j = to_list, 
    x = rep(1, length(from_list)),
    dims = c(n_cells, n_cells)
  )
  
  # Row-standardized version for computing means
  row_sums <- rowSums(W)
  row_sums[row_sums == 0] <- 1  # avoid division by zero for islands
  # Diagonal matrix of inverse row sums
  D_inv <- Diagonal(x = 1 / row_sums)
  W_mean <- D_inv %*% W  # W_mean %*% v gives neighbor mean of v
  
  # Number of neighbors per cell (for detecting islands -> NA)
  n_neighbors <- as.integer(rowSums(W))  # original counts before adjustment
  # Recompute from original
  n_neighbors <- lengths(rook_neighbors_unique)
  
  # --------------------------------------------------------------------------
  # STEP 2: Convert to data.table and create cell-index mapping
  # --------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # Map cell id -> spatial index (position in id_order)
  id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))
  dt[, spatial_idx := id_to_spatial_idx[as.character(id)]]
  
  # Get sorted unique years
  years <- sort(unique(dt$year))
  
  # --------------------------------------------------------------------------
  # STEP 3: Initialize output columns
  # --------------------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]
  }
  
  # --------------------------------------------------------------------------
  # STEP 4: Process year-by-year for memory efficiency
  # --------------------------------------------------------------------------
  # For each year, we have at most n_cells rows. We construct a full-length
  # vector (length n_cells) indexed by spatial_idx, then use sparse matrix ops.
  
  setkey(dt, year, spatial_idx)
  
  for (yr in years) {
    # Subset rows for this year
    yr_mask <- dt$year == yr
    yr_spatial_idx <- dt$spatial_idx[yr_mask]
    
    for (var_name in neighbor_source_vars) {
      col_max  <- paste0("neighbor_max_",  var_name)
      col_min  <- paste0("neighbor_min_",  var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      
      # Build a full-length vector for this year (NA for missing cells)
      full_vec <- rep(NA_real_, n_cells)
      full_vec[yr_spatial_idx] <- dt[[var_name]][yr_mask]
      
      # --- MEAN via sparse matrix multiplication ---
      # Replace NAs with 0 for multiplication, but track valid counts
      v <- full_vec
      v_valid <- as.numeric(!is.na(v))
      v[is.na(v)] <- 0
      
      # Sum of neighbor values
      neighbor_sum   <- as.numeric(W %*% v)
      # Count of valid (non-NA) neighbors
      neighbor_count <- as.numeric(W %*% v_valid)
      
      # Mean = sum / count (NA if count == 0)
      neighbor_mean_full <- ifelse(neighbor_count > 0, 
                                   neighbor_sum / neighbor_count, 
                                   NA_real_)
      
      # --- MAX and MIN via grouped operations on sparse structure ---
      # Extract neighbor values using the sparse matrix structure
      # W@i = row indices (0-based), W@j would require conversion
      # Use the COO representation we already have, filtered to this year
      
      # For each cell i, we need max and min of full_vec[neighbors of i]
      # We already have from_list, to_list from the adjacency construction
      
      # Get neighbor values
      neighbor_vals_vec <- full_vec[to_list]  # value of each neighbor
      
      # Use data.table for grouped max/min (very fast)
      edge_dt <- data.table(
        cell = from_list,
        nval = neighbor_vals_vec
      )
      
      # Remove edges where neighbor value is NA
      edge_dt <- edge_dt[!is.na(nval)]
      
      if (nrow(edge_dt) > 0) {
        stats_dt <- edge_dt[, .(nmax = max(nval), nmin = min(nval)), 
                            by = cell]
        
        neighbor_max_full <- rep(NA_real_, n_cells)
        neighbor_min_full <- rep(NA_real_, n_cells)
        neighbor_max_full[stats_dt$cell] <- stats_dt$nmax
        neighbor_min_full[stats_dt$cell] <- stats_dt$nmin
      } else {
        neighbor_max_full <- rep(NA_real_, n_cells)
        neighbor_min_full <- rep(NA_real_, n_cells)
      }
      
      # Also set to NA for cells with no neighbors at all
      no_neighbors <- n_neighbors == 0
      neighbor_max_full[no_neighbors]  <- NA_real_
      neighbor_min_full[no_neighbors]  <- NA_real_
      neighbor_mean_full[no_neighbors] <- NA_real_
      
      # Write back only for cells present this year
      set(dt, which(yr_mask), col_max,  neighbor_max_full[yr_spatial_idx])
      set(dt, which(yr_mask), col_min,  neighbor_min_full[yr_spatial_idx])
      set(dt, which(yr_mask), col_mean, neighbor_mean_full[yr_spatial_idx])
    }
    
    if (yr %% 5 == 0) message("Completed year: ", yr)
  }
  
  # --------------------------------------------------------------------------
  # STEP 5: Clean up and return
  # --------------------------------------------------------------------------
  dt[, spatial_idx := NULL]
  
  # Return as data.frame if input was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is NOT modified.
# Proceed directly to prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Statistic | Original | Optimized | Equivalence |
|---|---|---|---|
| **max** | `max(vals[idx])` after removing NAs | `data.table` grouped `max` over the same neighbor pairs, same NA removal | **Exact** â€” same values, same edges |
| **min** | `min(vals[idx])` after removing NAs | `data.table` grouped `min` over the same neighbor pairs, same NA removal | **Exact** |
| **mean** | `mean(vals[idx])` after removing NAs | `sum / count_valid` via sparse matrix, excluding NAs | **Exact** â€” algebraically identical |

The sparse matrix `W` is constructed from the identical `rook_neighbors_unique` nb object. The directed edges `(from_list, to_list)` encode exactly the same neighbor relationships. No approximation, interpolation, or kernel smoothing is introduced. The trained Random Forest model is never retouched â€” only the input feature matrix is computed faster.

### Performance Summary

| | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~40+ hrs (6.46M character-key lookups) | Eliminated (sparse matrix built in <1 sec) |
| `compute_neighbor_stats` | ~40+ hrs (6.46M Ã— 5 R-level loops) | ~3â€“8 min (28 years Ã— 5 vars, vectorized) |
| Peak RAM | ~8â€“12 GB (6.46M-element list of lists) | ~4â€“6 GB (sparse matrix + data.table) |
| **Total wall time** | **86+ hours** | **~3â€“8 minutes** |
END_RESPONSE>>>
