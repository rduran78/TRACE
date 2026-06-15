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
run_id: Anthropic_F06_T07_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up which spatial cells are rook neighbors (via the `nb` object).
2. Constructs string keys like `"cellid_year"` for each neighbor in each year.
3. Looks up those keys in a named character vector (`idx_lookup`).

This means **~6.46 million iterations**, each doing string concatenation (`paste`), named-vector lookups (which are hash-table lookups but still slow in R loops), and `NA` filtering. The string key approach is inherently slow because:
- `paste()` is called millions of times inside a loop.
- Named vector subsetting with character keys has overhead per call.
- The entire structure is a **list of 6.46M integer vectors**, which is memory-heavy.

### Bottleneck B: `compute_neighbor_stats` â€” O(n) `lapply` over 6.46M rows, repeated 5 times

For each of the 5 source variables, the function iterates over all 6.46M rows, subsets the variable values by neighbor indices, removes NAs, and computes `max`, `min`, `mean`. This is called 5 times, so effectively **~32.3 million row-level R function calls**.

### Why raster focal/kernel operations are a useful *analogy* but not the right implementation

Focal operations (e.g., `terra::focal`) apply a moving-window function over a regular grid. The panel data here *is* on a regular spatial grid with a time dimension, so conceptually the neighbor aggregation is a 2D spatial focal operation applied independently to each year-slice. However:

- The grid may have irregular boundaries, missing cells, or an `nb` object that doesn't correspond to a simple rectangular raster.
- The `nb` object (`rook_neighbors_unique`) is precomputed and serialized â€” it encodes the exact neighbor relationships. Reimplementing via raster focal would require reconstructing the grid geometry and verifying equivalence, risking subtle mismatches at boundaries.
- **To preserve the original numerical estimand exactly**, we must use the same neighbor relationships.

Therefore: **use the `nb` object directly, but replace R-level loops with vectorized/compiled operations**.

---

## 2. Optimization Strategy

### Strategy: Sparse-matrix multiplication replaces both functions

The key insight: computing `mean` of neighbor values is a **sparse matrixâ€“vector product**. Computing `max` and `min` can be done via sparse-matrix tricks or vectorized group operations.

**Step-by-step:**

1. **Build a sparse neighbor matrix once** (344,208 Ã— 344,208 spatial adjacency matrix from the `nb` object), then expand it to the cell-year level (6.46M Ã— 6.46M) â€” but this is too large. Instead, operate **per-year** on the spatial dimension only (344,208 Ã— 344,208), which is very manageable.

2. **Per year, per variable**: use the sparse adjacency matrix to compute:
   - `neighbor_mean` = (W %*% x) / (W %*% non_na_indicator) â€” weighted by number of non-NA neighbors.
   - `neighbor_max` and `neighbor_min` â€” use grouped operations via the sparse matrix's row structure.

3. This replaces ~6.46M R-level iterations with ~28 sparse matrix multiplications per variable (one per year), each on a 344K-length vector. Sparse matrixâ€“vector products are **compiled C code** in the `Matrix` package.

4. For `max` and `min`, we use `data.table` grouped operations on an edge list derived from the sparse matrix, which is also highly optimized.

**Expected speedup**: from 86+ hours to **minutes**.

---

## 3. Working R Code

```r
library(Matrix)
library(data.table)
library(spdep)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build spatial sparse adjacency matrix from the nb object
#         (done once; 344,208 Ã— 344,208, very sparse)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_spatial_adjacency <- function(nb_obj, id_order) {
  # nb_obj: spdep nb object (list of integer vectors of neighbor indices)
  # id_order: vector of cell IDs in the order matching nb_obj
  n <- length(nb_obj)
  stopifnot(n == length(id_order))
  
  # Build COO (coordinate) representation
  from <- integer(0)
  to   <- integer(0)
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0) {
      from <- c(from, rep(i, length(nbrs)))
      to   <- c(to, nbrs)
    }
  }
  
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  
  list(W = W, id_order = id_order, n = n)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Build an edge-list data.table for max/min operations
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_dt <- function(W) {
  # Extract the (i, j) pairs from the sparse matrix
  W_coo <- summary(W)  # returns a data.frame with i, j, x columns
  data.table(from = W_coo$i, to = W_coo$j)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Compute neighbor stats for all variables, all years
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_all_neighbor_features <- function(cell_data, id_order, nb_obj,
                                          neighbor_source_vars) {
  
  # Convert to data.table for speed (non-destructive copy)
  dt <- as.data.table(cell_data)
  
  # Build spatial adjacency
  message("Building spatial adjacency matrix...")
  adj <- build_spatial_adjacency(nb_obj, id_order)
  W   <- adj$W
  n_cells <- adj$n
  
  # Build mapping from cell id -> spatial index (position in id_order)
  id_to_sidx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add spatial index to dt
  dt[, spatial_idx := id_to_sidx[as.character(id)]]
  
  # Build edge list for max/min
  message("Building edge list...")
  edge_dt <- build_edge_dt(W)
  
  # Get unique years
  years <- sort(unique(dt$year))
  
  # Pre-allocate result columns
  for (var_name in neighbor_source_vars) {
    dt[, paste0("n_max_", var_name) := NA_real_]
    dt[, paste0("n_min_", var_name) := NA_real_]
    dt[, paste0("n_mean_", var_name) := NA_real_]
  }
  
  # Key the data.table for fast lookups
  setkey(dt, year, spatial_idx)
  
  message("Computing neighbor features...")
  
  for (yr in years) {
    # Extract the year-slice, ordered by spatial_idx
    yr_mask <- dt$year == yr
    yr_dt   <- dt[yr_mask]
    setorder(yr_dt, spatial_idx)
    
    # Build a full-length vector for each variable (indexed by spatial_idx)
    # Some spatial cells may be missing in some years; handle that.
    present_sidx <- yr_dt$spatial_idx
    
    for (var_name in neighbor_source_vars) {
      
      # Full-length vector (NA for cells not present this year)
      x_full <- rep(NA_real_, n_cells)
      x_full[present_sidx] <- yr_dt[[var_name]]
      
      # --- MEAN via sparse matrix-vector product ---
      # Sum of neighbor values (NA treated as 0 for the product, corrected below)
      x_for_sum <- x_full
      x_for_sum[is.na(x_for_sum)] <- 0
      
      neighbor_sum <- as.numeric(W %*% x_for_sum)
      
      # Count of non-NA neighbors
      non_na_indicator <- as.numeric(!is.na(x_full))
      neighbor_count   <- as.numeric(W %*% non_na_indicator)
      
      neighbor_mean <- ifelse(neighbor_count > 0,
                              neighbor_sum / neighbor_count,
                              NA_real_)
      
      # --- MAX and MIN via edge list grouped operations ---
      # For each "from" node, gather all neighbor ("to") values and compute max/min
      edge_vals <- x_full[edge_dt$to]
      
      # Temporary data.table: from, val
      tmp <- data.table(from = edge_dt$from, val = edge_vals)
      # Remove edges where neighbor value is NA
      tmp <- tmp[!is.na(val)]
      
      if (nrow(tmp) > 0) {
        agg <- tmp[, .(nmax = max(val), nmin = min(val)), by = from]
        
        # Initialize full-length vectors
        neighbor_max <- rep(NA_real_, n_cells)
        neighbor_min <- rep(NA_real_, n_cells)
        neighbor_max[agg$from] <- agg$nmax
        neighbor_min[agg$from] <- agg$nmin
      } else {
        neighbor_max <- rep(NA_real_, n_cells)
        neighbor_min <- rep(NA_real_, n_cells)
      }
      
      # --- Write results back into dt for cells present this year ---
      max_col  <- paste0("n_max_", var_name)
      min_col  <- paste0("n_min_", var_name)
      mean_col <- paste0("n_mean_", var_name)
      
      set(dt, which(yr_mask),  max_col, neighbor_max[dt$spatial_idx[yr_mask]])
      set(dt, which(yr_mask),  min_col, neighbor_min[dt$spatial_idx[yr_mask]])
      set(dt, which(yr_mask), mean_col, neighbor_mean[dt$spatial_idx[yr_mask]])
    }
    
    message(sprintf("  Year %d done.", yr))
  }
  
  # Remove helper column
  dt[, spatial_idx := NULL]
  
  return(dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Run it
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data       = cell_data,
  id_order        = id_order,
  nb_obj          = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data is now a data.table with the 15 new columns:
#   n_max_ntl, n_min_ntl, n_mean_ntl,
#   n_max_ec,  n_min_ec,  n_mean_ec,
#   ... etc.

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Predict with the pre-trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# The trained model object (e.g., `rf_model`) is loaded from disk as-is.
# Prediction proceeds exactly as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M R-level iterations with `paste`/string hashing | Single sparse matrix built from `nb` object; reused for all years and variables |
| **Mean computation** | Per-row `mean()` in `lapply` Ã— 5 vars Ã— 6.46M rows | Sparse matrixâ€“vector product (`W %*% x`), compiled C code, per year-slice (~344K) |
| **Max/Min computation** | Per-row `max()`/`min()` in `lapply` | `data.table` grouped aggregation on edge list (~1.37M edges per year) |
| **Total R-level loop iterations** | ~32.3M (inner) + 6.46M (lookup build) | 28 years Ã— 5 vars = 140 sparse-mat operations + 140 grouped aggregations |
| **Estimated runtime** | 86+ hours | **~5â€“15 minutes** on a 16 GB laptop |
| **Numerical equivalence** | Baseline | âœ… Identical â€” same neighbor relationships from `nb` object, same `max`/`min`/`mean` |
| **Trained RF model** | â€” | âœ… Preserved, not retrained |
| **Memory** | ~6.46M-element list of integer vectors | Sparse matrix (~1.37M non-zeros) + data.table; well within 16 GB |
END_RESPONSE>>>
