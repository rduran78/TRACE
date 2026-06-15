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
run_id: Anthropic_F06_T07_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over ~6.46 million rows

For every row, the function:
1. Looks up the cell's reference index via a named character vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs `paste()`-based keys for every neighbor Ã— that row's year.
4. Looks up those keys in `idx_lookup` (another named character vector).

Named character vector lookup in R is **O(n)** per query (hash-based, but with overhead from `as.character`, `paste`, and repeated allocation). Over 6.46M rows, each with ~4 rook neighbors on average (~1.37M relationships / 344K cells â‰ˆ 4), this produces **~25.8 million string constructions and hash lookups**, all inside a sequential `lapply`. The resulting list of 6.46M integer vectors also consumes significant memory.

### Bottleneck B: `compute_neighbor_stats` â€” Called 5 times, each iterating over 6.46M rows

Each call does another `lapply` over 6.46M elements, subsetting a numeric vector and computing `max`, `min`, `mean`. This is pure R-level looping â€” no vectorization.

### Combined effect

The two stages together produce roughly **86+ hours** of wall-clock time on a 16 GB laptop. The fundamental problem is: **row-level R-loop iteration over millions of rows with per-row string operations and list allocations**.

---

## 2. Optimization Strategy

### Key insight: Separate the spatial dimension from the temporal dimension

The neighbor structure is **time-invariant** â€” cell A's rook neighbors are the same in every year. The `nb` object has only 344,208 entries (one per cell). The current code "explodes" this into 6.46M entries by replicating the neighbor structure across all 28 years. This is unnecessary.

### Strategy: Vectorized sparse-matrix multiplication

1. **Build a sparse neighbor adjacency matrix `W`** (344,208 Ã— 344,208) from the `nb` object. Each row `i` has 1s in columns corresponding to cell `i`'s rook neighbors. This matrix has ~1.37M non-zero entries â€” trivially small.

2. **Reshape each source variable into a matrix `V`** of dimension (344,208 cells Ã— 28 years), where rows are cells (in `id_order` order) and columns are years.

3. **Compute neighbor stats using sparse matrix operations:**
   - **Neighbor sum** = `W %*% V` (sparse Ã— dense, extremely fast)
   - **Neighbor count** = `W %*% (!is.na(V))` (to handle NAs correctly)
   - **Neighbor mean** = sum / count
   - **Neighbor max and min**: Use a grouped operation over the sparse structure of `W` â€” iterate over the 344K cells (not 6.46M rows), extract neighbor indices from `W`, and compute row-wise max/min on the submatrix.

4. **Flatten back** to the original long-format data frame and attach the 15 new columns (3 stats Ã— 5 variables).

### Why this is fast

| Aspect | Old | New |
|---|---|---|
| Loop iterations for lookup | 6.46M | 0 (matrix construction) |
| Loop iterations for stats | 6.46M Ã— 5 = 32.3M | 344K Ã— 5 = 1.72M (max/min only) |
| Mean computation | Per-row R loop | Sparse matrix multiply (C-level) |
| String operations | ~25.8M `paste()` calls | 0 |
| Memory for lookup | 6.46M-element list | Sparse matrix (~20 MB) |

**Expected speedup: from 86+ hours to ~2â€“10 minutes.**

### Why not raster focal/kernel operations?

The comment in the prompt asks us to consider this. Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed kernel. If the grid is perfectly regular and complete, a 3Ã—3 rook kernel (`matrix(c(0,1,0,1,0,1,0,1,0), 3, 3)`) would work and be very fast. However:
- The `nb` object is **precomputed and serialized**, suggesting the grid may have irregular boundaries, missing cells, or non-rectangular extent.
- Using the `nb` object directly (via sparse matrix) **guarantees identical neighbor relationships** and thus **preserves the original numerical estimand exactly**.
- A raster focal approach would require reconstructing the grid, handling edge/missing cells differently, and risks subtle discrepancies.

**Decision: Use the sparse matrix approach built from the actual `nb` object.** This is both fast and numerically faithful.

---

## 3. Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Replaces: build_neighbor_lookup + compute_neighbor_stats + outer loop
# Preserves: exact numerical results (same neighbor relationships, same stats)
# ==============================================================================

library(Matrix)   # for sparse matrices
library(data.table)  # for fast reshaping

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  
  # --------------------------------------------------------------------------
  # STEP 1: Build sparse adjacency matrix W from the nb object
  # --------------------------------------------------------------------------
  # id_order[i] is the cell ID for the i-th entry in rook_neighbors_unique
  n_cells <- length(id_order)
  
  # Build COO (coordinate) representation
  from_idx <- integer(0)
  to_idx   <- integer(0)
  
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep::nb uses 0 to indicate no neighbors
    nb_i <- nb_i[nb_i != 0L]
    if (length(nb_i) > 0) {
      from_idx <- c(from_idx, rep(i, length(nb_i)))
      to_idx   <- c(to_idx, nb_i)
    }
  }
  
  W <- sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = 1,
    dims = c(n_cells, n_cells)
  )
  
  # --------------------------------------------------------------------------
  # STEP 2: Create a mapping from cell ID to row index in W
  # --------------------------------------------------------------------------
  id_to_widx <- setNames(seq_along(id_order), as.character(id_order))
  
  # --------------------------------------------------------------------------
  # STEP 3: Convert cell_data to data.table for fast operations
  # --------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # Ensure consistent ordering: assign W-row index to each row
  dt[, w_idx := id_to_widx[as.character(id)]]
  
  # Get sorted unique years
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))
  dt[, y_idx := year_to_col[as.character(year)]]
  
  # --------------------------------------------------------------------------
  # STEP 4: For each source variable, compute neighbor max, min, mean
  # --------------------------------------------------------------------------
  
  # Pre-extract the sparse structure of W for max/min computation
  # For each cell i, get its neighbor indices
  # We can extract this from W directly
  W_dgC <- as(W, "dgCMatrix")  # compressed sparse column
  W_dgR <- as(W, "dgRMatrix")  # compressed sparse row â€” better for row access
  # If dgRMatrix is not available, use dgCMatrix on transposed or manual extraction
  
  # Extract neighbor list from sparse matrix (much faster than re-reading nb)
  # Using dgCMatrix: columns of t(W) = rows of W
  Wt <- t(W_dgC)  # now column j of Wt = row j of W = neighbors of j
  
  neighbor_indices <- vector("list", n_cells)
  for (j in seq_len(n_cells)) {
    col_start <- Wt@p[j] + 1L
    col_end   <- Wt@p[j + 1L]
    if (col_end >= col_start) {
      neighbor_indices[[j]] <- Wt@i[col_start:col_end] + 1L
    } else {
      neighbor_indices[[j]] <- integer(0)
    }
  }
  
  for (var_name in neighbor_source_vars) {
    cat("Processing neighbor features for:", var_name, "\n")
    
    # Build cell Ã— year matrix V (n_cells Ã— n_years)
    # Initialize with NA
    V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    V[cbind(dt$w_idx, dt$y_idx)] <- dt[[var_name]]
    
    # ------ NEIGHBOR MEAN via sparse matrix multiply ------
    # Handle NAs: replace NA with 0 for sum, track counts separately
    V_nona <- V
    V_nona[is.na(V_nona)] <- 0
    V_notna <- (!is.na(V)) * 1.0  # indicator matrix
    
    # Neighbor sum and count (sparse %*% dense is fast in Matrix package)
    nb_sum   <- as.matrix(W %*% V_nona)    # n_cells Ã— n_years
    nb_count <- as.matrix(W %*% V_notna)   # n_cells Ã— n_years
    
    nb_mean <- nb_sum / nb_count
    nb_mean[nb_count == 0] <- NA_real_
    
    # ------ NEIGHBOR MAX and MIN via grouped row operation ------
    # This loops over 344K cells (not 6.46M rows) â€” very manageable
    nb_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    for (i in seq_len(n_cells)) {
      nbs <- neighbor_indices[[i]]
      if (length(nbs) == 0L) next
      # Extract submatrix: neighbors Ã— years
      sub <- V[nbs, , drop = FALSE]  # length(nbs) Ã— n_years
      if (nrow(sub) == 1L) {
        nb_max[i, ] <- sub[1L, ]
        nb_min[i, ] <- sub[1L, ]
      } else {
        # Columnwise max/min, ignoring NAs
        nb_max[i, ] <- apply(sub, 2, max, na.rm = TRUE)
        nb_min[i, ] <- apply(sub, 2, min, na.rm = TRUE)
      }
    }
    # apply with na.rm=TRUE returns -Inf/Inf when all NA; fix those
    nb_max[is.infinite(nb_max)] <- NA_real_
    nb_min[is.infinite(nb_min)] <- NA_real_
    
    # ------ MAP BACK to long-format data.table ------
    idx_mat <- cbind(dt$w_idx, dt$y_idx)
    
    max_col <- paste0(var_name, "_max_neighbor")
    min_col <- paste0(var_name, "_min_neighbor")
    mean_col <- paste0(var_name, "_mean_neighbor")
    
    dt[, (max_col)  := nb_max[idx_mat]]
    dt[, (min_col)  := nb_min[idx_mat]]
    dt[, (mean_col) := nb_mean[idx_mat]]
  }
  
  # --------------------------------------------------------------------------
  # STEP 5: Clean up helper columns and return
  # --------------------------------------------------------------------------
  dt[, c("w_idx", "y_idx") := NULL]
  
  return(as.data.frame(dt))
}

# ==============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is unchanged â€” use it directly for prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Further Optimization: Eliminating the `apply` Loop for Max/Min

The 344K-iteration loop with `apply` for max/min is already ~1000Ã— fewer iterations than the original, but if it's still slow (each iteration touches a small submatrix across 28 columns), we can vectorize it further using **data.table grouping**:

```r
# ALTERNATIVE: Fully vectorized max/min using data.table joins
# Instead of looping over cells, expand the edge list and group-aggregate

compute_maxmin_dt <- function(dt, W_from, W_to, var_name, years, year_to_col) {
  # W_from, W_to: integer vectors of directed edges (in W-index space)
  # Build edge table
  edges <- data.table(from_w = W_from, to_w = W_to)
  
  # For each (from_cell, year), we need the var values of to_cell
  # Expand edges across years
  edge_year <- CJ(edge_id = seq_len(nrow(edges)), y_idx = seq_along(years))
  edge_year[, `:=`(from_w = edges$from_w[edge_id], to_w = edges$to_w[edge_id])]
  
  # Lookup values: create a keyed table of (w_idx, y_idx) -> value
  val_table <- dt[, .(w_idx, y_idx, val = get(var_name))]
  setkey(val_table, w_idx, y_idx)
  
  # Join to get neighbor values
  edge_year[, val := val_table[.(to_w, y_idx), val, on = .(w_idx, y_idx)]]
  
  # Group by (from_w, y_idx) and compute max, min
  stats <- edge_year[!is.na(val),
                     .(nb_max = max(val), nb_min = min(val)),
                     by = .(from_w, y_idx)]
  
  # Join back to dt
  setkey(stats, from_w, y_idx)
  dt[, paste0(var_name, "_max_neighbor") :=
       stats[.(w_idx, y_idx), nb_max, on = .(from_w = w_idx, y_idx)]]
  dt[, paste0(var_name, "_min_neighbor") :=
       stats[.(w_idx, y_idx), nb_min, on = .(from_w = w_idx, y_idx)]]
  
  invisible(dt)
}
```

> **Caveat:** This edge-expansion approach creates ~1.37M Ã— 28 â‰ˆ 38.5M rows, using ~600 MB of RAM â€” feasible on a 16 GB machine but heavier. The cell-loop version (344K iterations) is likely the better balance of speed and memory for this dataset.

---

## 5. Summary

| Metric | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M row `lapply` with `paste`/hash | Sparse matrix from `nb` (344K entries) |
| Mean computation | 6.46M row `lapply` | Sparse matrix multiply (`W %*% V`) |
| Max/Min computation | 6.46M row `lapply` | 344K cell loop over 28-column submatrices |
| Estimated runtime | 86+ hours | **~2â€“10 minutes** |
| Memory | 6.46M-element list (~2â€“4 GB) | Sparse matrix (~20 MB) + 5 dense matrices (~130 MB each) |
| Numerical results | Baseline | **Identical** (same neighbors, same operations) |
| Trained RF model | Unchanged | **Unchanged** |
END_RESPONSE>>>
