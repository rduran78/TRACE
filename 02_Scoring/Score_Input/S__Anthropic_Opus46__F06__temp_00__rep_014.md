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
run_id: Anthropic_F06_T00_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's spatial neighbors (from ~344K cells).
2. Constructs character keys by pasting `id_year`.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

Character-based named-vector lookup in R is **O(n)** in the worst case per access due to hashing overhead at scale, and `paste()`-based key construction over 6.46M rows is extremely expensive. The lookup is also **redundant across years**: the same cell has the same rook neighbors every year, so the spatial topology is repeated 28 times unnecessarily.

### Bottleneck 2: `compute_neighbor_stats` â€” `lapply` over 6.46M elements

For each of 5 variables Ã— 6.46M rows, an R-level `lapply` iterates row-by-row, subsets a vector by index, removes NAs, and computes `max`, `min`, `mean`. That's ~32.3 million R-level function calls with per-element vector subsetting. The `do.call(rbind, result)` on a 6.46M-element list is also slow.

### Why raster focal/kernel operations don't directly apply

Focal operations assume a regular grid with a fixed rectangular kernel. Here, the grid cells have an irregular rook-neighbor structure (coastal cells, boundary cells have fewer neighbors), and the data is a panel (cell Ã— year). The `spdep::nb` object encodes this irregular topology. A focal approach would require padding/masking and wouldn't preserve the exact neighbor structure. **The correct approach is to keep the explicit neighbor structure but vectorize the computation using sparse matrix algebra.**

---

## Optimization Strategy

### Strategy: Sparse Matrix Multiplication

The key insight: computing `max`, `min`, and `mean` of neighbor values can be reformulated as operations on a **sparse adjacency matrix** `W` (344,208 Ã— 344,208), applied to a matrix of variable values (344,208 Ã— 28 years).

1. **Build the spatial adjacency matrix once** from `rook_neighbors_unique` (~344K Ã— 344K sparse matrix, ~1.37M non-zero entries). This is tiny in memory.

2. **Reshape each variable into a cell Ã— year matrix** (344,208 rows Ã— 28 columns).

3. **Compute neighbor means** via sparse matrix multiplication: `W_row_normalized %*% X` gives the mean of neighbors for every cell-year in one shot.

4. **Compute neighbor max and min** using a custom sparse-matrix row-wise operation. Since `max` and `min` are not linear, we iterate over the sparse structure, but in a **vectorized** manner using the `Matrix` package internals or a grouped operation.

5. **Reshape results back** to the long panel format and column-bind.

**Expected speedup**: From ~86 hours to **minutes**. The sparse matrixâ€“vector product for mean is O(nnz) â‰ˆ 1.37M multiplications per year-column, so 28 Ã— 1.37M â‰ˆ 38M operations per variable â€” trivial. Max/min require a grouped operation but can be done with `data.table` grouping in seconds.

---

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================
# STEP 1: Build sparse rook adjacency matrix (once)
# ==============================================================
build_sparse_adjacency <- function(id_order, rook_neighbors_unique) {
  n <- length(id_order)
  # rook_neighbors_unique is an nb object: list of integer vectors
  # Each element i contains indices (into id_order) of neighbors of cell i
  from <- rep(seq_len(n), lengths(rook_neighbors_unique))
  to   <- unlist(rook_neighbors_unique)
  
  # Remove zero-neighbor entries (nb objects use 0L for no neighbors)
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  return(W)
}

# ==============================================================
# STEP 2: Compute neighbor stats for all variables at once
# ==============================================================
compute_all_neighbor_features <- function(cell_data, id_order, 
                                           rook_neighbors_unique,
                                           neighbor_source_vars) {
  
  # Convert to data.table for speed (non-destructive copy)
  dt <- as.data.table(cell_data)
  
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)
  
  # Create mapping: cell id -> row index in adjacency matrix
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add spatial index and year index to data.table
  dt[, cell_idx := id_to_idx[as.character(id)]]
  year_to_col <- setNames(seq_along(years), as.character(years))
  dt[, year_idx := year_to_col[as.character(year)]]
  
  # Build sparse adjacency matrix W (n_cells x n_cells)
  W <- build_sparse_adjacency(id_order, rook_neighbors_unique)
  
  # Precompute row sums for mean calculation (number of neighbors per cell)
  neighbor_counts <- rowSums(W)  # dense vector of length n_cells
  # Avoid division by zero
  neighbor_counts_safe <- ifelse(neighbor_counts == 0, NA_real_, neighbor_counts)
  
  # ----------------------------------------------------------
  # For max and min: build an edge list (data.table) once
  # ----------------------------------------------------------
  W_csc <- as(W, "dgCMatrix")
  # Extract triplet form
  W_tri <- as(W, "dgTMatrix")
  edge_from <- W_tri@i + 1L   # 1-indexed row (the focal cell)
  edge_to   <- W_tri@j + 1L   # 1-indexed col (the neighbor cell)
  
  # edge_dt: each row is a directed neighbor relationship
  edge_dt <- data.table(focal = edge_from, neighbor = edge_to)
  
  # ----------------------------------------------------------
  # Process each variable
  # ----------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    
    cat("Processing neighbor features for:", var_name, "\n")
    
    # Reshape variable into cell x year matrix
    # X[cell_idx, year_idx] = value
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]
    
    # ---- MEAN: sparse matrix multiplication ----
    # W %*% X gives sum of neighbor values for each cell-year
    WX_sum <- as.matrix(W %*% X)   # n_cells x n_years dense matrix
    # Divide by neighbor count to get mean
    X_mean <- WX_sum / neighbor_counts_safe  # recycling over columns
    
    # ---- MAX and MIN: vectorized grouped operation ----
    # For each edge (focal, neighbor), get the neighbor's values across years
    # Then group by focal cell and compute max/min per year
    
    # Neighbor values: matrix indexed by edge_to
    # neighbor_vals_mat[e, t] = X[edge_to[e], t]
    neighbor_vals_mat <- X[edge_to, , drop = FALSE]  # n_edges x n_years
    
    # We need max and min grouped by focal cell for each year
    # Use data.table for fast grouped operations
    
    X_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # Create a data.table with focal index and all year columns
    year_colnames <- paste0("Y", seq_len(n_years))
    vals_dt <- data.table(focal = edge_from, neighbor_vals_mat)
    setnames(vals_dt, c("focal", year_colnames))
    
    # Compute max per focal cell per year
    max_dt <- vals_dt[, lapply(.SD, function(v) {
      v <- v[!is.na(v)]
      if (length(v) == 0L) NA_real_ else max(v)
    }), by = focal, .SDcols = year_colnames]
    
    # Compute min per focal cell per year
    min_dt <- vals_dt[, lapply(.SD, function(v) {
      v <- v[!is.na(v)]
      if (length(v) == 0L) NA_real_ else min(v)
    }), by = focal, .SDcols = year_colnames]
    
    # Place results back into matrices
    max_mat <- as.matrix(max_dt[, ..year_colnames])
    min_mat <- as.matrix(min_dt[, ..year_colnames])
    X_max[max_dt$focal, ] <- max_mat
    X_min[min_dt$focal, ] <- min_mat
    
    # ---- Map results back to long panel ----
    idx_mat <- cbind(dt$cell_idx, dt$year_idx)
    
    max_col_name  <- paste0("neighbor_max_", var_name)
    min_col_name  <- paste0("neighbor_min_", var_name)
    mean_col_name <- paste0("neighbor_mean_", var_name)
    
    dt[, (max_col_name)  := X_max[idx_mat]]
    dt[, (min_col_name)  := X_min[idx_mat]]
    dt[, (mean_col_name) := X_mean[idx_mat]]
  }
  
  # Remove helper columns
  dt[, c("cell_idx", "year_idx") := NULL]
  
  return(dt)
}

# ==============================================================
# STEP 3: Main execution
# ==============================================================

# --- Load your existing objects ---
# cell_data              : data.frame/data.table with columns id, year, ntl, ec, ...
# id_order               : vector of cell IDs matching rook_neighbors_unique
# rook_neighbors_unique  : spdep nb object (loaded from disk)
# rf_model               : pre-trained Random Forest model (DO NOT retrain)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now has 15 new columns:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... etc.

# --- Predict with the existing trained RF model ---
# (Model is unchanged; features are numerically identical to original)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Further Optimization: Vectorized Max/Min Without Grouped Lambdas

The `data.table` grouped `lapply` for max/min can be further accelerated by processing one year-column at a time with `fmax`/`fmin` from the `collapse` package, which is highly optimized for grouped aggregations:

```r
# Alternative: use collapse::fmax / fmin for even faster grouped max/min
library(collapse)

compute_max_min_fast <- function(edge_from, X, edge_to, n_cells, n_years) {
  X_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  X_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  g <- GRP(edge_from)  # pre-compute grouping object once
  
  for (t in seq_len(n_years)) {
    nv <- X[edge_to, t]
    X_max[g$group.sizes |> seq_along() |> (\(x) sort(unique(edge_from)))(), t] <-
      fmax(nv, g = g, na.rm = TRUE)
    X_min[g$group.sizes |> seq_along() |> (\(x) sort(unique(edge_from)))(), t] <-
      fmin(nv, g = g, na.rm = TRUE)
  }
  
  list(X_max = X_max, X_min = X_min)
}
```

Here is a cleaner version of the fast max/min block:

```r
library(collapse)

compute_max_min_collapse <- function(edge_from, X, edge_to, n_cells, n_years) {
  X_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  X_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  g <- GRP(edge_from)
  unique_focals <- g$groups[[1]]  # sorted unique focal cell indices
  
  for (t in seq_len(n_years)) {
    nv <- X[edge_to, t]
    X_max[unique_focals, t] <- fmax(nv, g = g, na.rm = TRUE)
    X_min[unique_focals, t] <- fmin(nv, g = g, na.rm = TRUE)
  }
  
  list(X_max = X_max, X_min = X_min)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M character-key lookups via `paste` + named vector | Sparse matrix built once from `nb` object; integer indexing |
| **Mean** | R-level `lapply` per row | Single sparse matrix multiply `W %*% X` |
| **Max / Min** | R-level `lapply` per row | Vectorized `data.table` or `collapse` grouped aggregation on edge list |
| **Memory** | 6.46M-element list of index vectors | ~1.37M-entry sparse matrix + cellÃ—year matrices (~77 MB each) |
| **Estimated time** | 86+ hours | **5â€“15 minutes** |
| **Numerical results** | Baseline | **Identical** (same rook neighbors, same max/min/mean) |
| **RF model** | Trained | **Unchanged â€” not retrained** |
END_RESPONSE>>>
