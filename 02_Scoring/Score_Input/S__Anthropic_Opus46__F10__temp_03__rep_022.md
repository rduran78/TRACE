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
run_id: Anthropic_F10_T03_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **O(nÂ²)-scale `build_neighbor_lookup`**: For each of the ~6.46M rows, it performs string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but the sheer volume (6.46M Ã— avg ~4 neighbors) combined with string construction creates enormous overhead. The `lapply` over 6.46M rows with string operations is the dominant cost.

2. **`compute_neighbor_stats` uses R-level `lapply` over 6.46M rows**: Each iteration subsets a numeric vector, removes NAs, and computes max/min/mean. This is ~6.46M R function calls with small-vector operations â€” death by a thousand cuts.

3. **The neighbor lookup encodes year-specific row indices, but the topology is year-invariant**: Rook neighbors don't change across years. The current code rebuilds year-aware keys for every row, even though the spatial adjacency is static. This conflates graph topology with temporal indexing.

**Key insight**: The adjacency graph has 344,208 nodes and ~1.37M directed edges â€” this is *fixed across all 28 years*. Each year is an independent slice with identical topology. We should:
- Build a sparse adjacency matrix **once** (344K Ã— 344K).
- For each year-slice (344K rows), use sparse matrixâ€“vector multiplication to compute neighbor sums and counts, then derive max/min/mean via grouped operations.

However, sparse matrix multiplication gives us **sum** and **count** (hence mean), but **not max and min**. For max/min, we need grouped operations over the edge list.

---

## Optimization Strategy

1. **Represent topology as a sparse adjacency matrix (`Matrix::sparseMatrix`) and as an edge-list (`data.table`)** â€” built once from the `nb` object.

2. **Sort/index `cell_data` by `(year, id)`** so that each year-slice is a contiguous block of rows, and within each year the cell ordering matches the spatial grid order. This allows direct positional indexing â€” no hash lookups, no string keys.

3. **For each variable and each year**:
   - Extract the variable vector for that year-slice (length 344,208, ordered by cell index).
   - **Mean**: Use sparse matrix multiplication: `A %*% x` gives neighbor sums; divide by `A %*% ones` (precomputed neighbor counts). One matrixâ€“vector multiply per variable-year â€” highly optimized BLAS/CHOLMOD code in the `Matrix` package.
   - **Max/Min**: Use the edge-list in `data.table` with a keyed join to look up neighbor values, then `group by` source node to compute max and min. `data.table` grouped operations over ~1.37M rows are extremely fast.

4. **Vectorize across years**: Loop over 28 years (trivial) Ã— 5 variables = 140 iterations of fast operations, instead of 6.46M R-level iterations.

**Expected speedup**: From 86+ hours to **minutes** (roughly 5â€“15 minutes depending on disk I/O and RAM pressure).

**Numerical equivalence**: Preserved exactly â€” same max, min, and mean of the same non-NA neighbor values.

**RAM**: The sparse matrix is 344K Ã— 344K with ~1.37M non-zeros â‰ˆ ~33 MB. The edge-list `data.table` is ~1.37M Ã— 2 â‰ˆ ~22 MB. Year-slices are 344K Ã— ~110 cols. All fits comfortably in 16 GB.

---

## Optimized R Code

```r
library(Matrix)
library(data.table)

# =============================================================================
# STEP 0: Prepare cell_data as data.table, establish canonical cell ordering
# =============================================================================

# Convert to data.table if not already
cell_data <- as.data.table(cell_data)

# id_order is the vector of cell IDs in the order matching rook_neighbors_unique
# (i.e., rook_neighbors_unique[[i]] contains neighbor indices for id_order[i])
# Build a map from cell ID to its positional index in id_order
n_cells <- length(id_order)
id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

# Add positional index to cell_data
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# Sort by year then cell_pos for contiguous year-slices with canonical ordering
setkey(cell_data, year, cell_pos)

# Verify: each year-slice should have exactly n_cells rows in cell_pos order
# (If some cell-years are missing, we handle that below)
years <- sort(unique(cell_data$year))
n_years <- length(years)

# Check completeness (balanced panel assumed based on problem statement)
stopifnot(nrow(cell_data) == n_cells * n_years)
# After setkey(year, cell_pos), row ((y-1)*n_cells + p) corresponds to
# year=years[y], cell_pos=p. This is critical for direct indexing.

# =============================================================================
# STEP 1: Build sparse adjacency matrix and edge list from nb object (ONCE)
# =============================================================================

build_graph_from_nb <- function(nb_obj) {
  # nb_obj is a list of length n_cells
  # nb_obj[[i]] is an integer vector of neighbor indices (into id_order)
  # Build edge list: from -> to (1-indexed positions)
  from_list <- vector("list", length(nb_obj))
  to_list   <- vector("list", length(nb_obj))
  
  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    nbrs <- nbrs[nbrs != 0L]
    if (length(nbrs) == 0L) next
    from_list[[i]] <- rep.int(i, length(nbrs))
    to_list[[i]]   <- nbrs
  }
  
  from_vec <- unlist(from_list, use.names = FALSE)
  to_vec   <- unlist(to_list, use.names = FALSE)
  
  # Sparse adjacency matrix (rows = source nodes, cols = neighbor nodes)
  # A[i,j] = 1 means j is a neighbor of i
  A <- sparseMatrix(
    i = from_vec, j = to_vec,
    x = 1, dims = c(length(nb_obj), length(nb_obj))
  )
  
  # Edge list as data.table
  edge_dt <- data.table(from_pos = from_vec, to_pos = to_vec)
  
  list(A = A, edges = edge_dt)
}

cat("Building graph topology...\n")
graph <- build_graph_from_nb(rook_neighbors_unique)
A <- graph$A          # sparse Matrix, n_cells x n_cells
edge_dt <- graph$edges # data.table with columns from_pos, to_pos

# Precompute neighbor counts per node (for mean calculation)
# neighbor_counts[i] = number of neighbors of node i
neighbor_counts <- as.numeric(A %*% rep(1, n_cells))  # length n_cells

cat(sprintf("Graph: %d nodes, %d directed edges\n", n_cells, nrow(edge_dt)))

# =============================================================================
# STEP 2: Function to compute neighbor stats for one variable across all years
# =============================================================================

compute_neighbor_features_fast <- function(cell_data, var_name, A, edge_dt,
                                            neighbor_counts, n_cells, years) {
  # Output column names (must match original pipeline)
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate output vectors
  n_total <- nrow(cell_data)
  out_max  <- rep(NA_real_, n_total)
  out_min  <- rep(NA_real_, n_total)
  out_mean <- rep(NA_real_, n_total)
  
  vals_all <- cell_data[[var_name]]
  
  for (yi in seq_along(years)) {
    # Row range for this year (data is sorted by year, cell_pos)
    row_start <- (yi - 1L) * n_cells + 1L
    row_end   <- yi * n_cells
    row_range <- row_start:row_end
    
    # Extract variable values for this year-slice, in cell_pos order
    x <- vals_all[row_range]  # length n_cells, x[p] = value for cell_pos=p
    
    # --- MEAN via sparse matrix multiplication ---
    # Replace NA with 0 for sum, and track non-NA for count
    not_na <- !is.na(x)
    x_zero <- x
    x_zero[!not_na] <- 0
    
    # Neighbor sums (only non-NA values contribute)
    neighbor_sums <- as.numeric(A %*% x_zero)
    
    # Neighbor non-NA counts
    neighbor_nna <- as.numeric(A %*% as.numeric(not_na))
    
    # Mean = sum / count (NA where count == 0)
    yr_mean <- ifelse(neighbor_nna > 0, neighbor_sums / neighbor_nna, NA_real_)
    
    # --- MAX and MIN via edge list + data.table grouped ops ---
    # Look up neighbor values
    nbr_vals <- x[edge_dt$to_pos]
    
    # Build temporary data.table for grouped aggregation
    # Only keep non-NA neighbor values
    valid <- !is.na(nbr_vals)
    if (any(valid)) {
      tmp <- data.table(
        from_pos = edge_dt$from_pos[valid],
        val      = nbr_vals[valid]
      )
      
      agg <- tmp[, .(nmax = max(val), nmin = min(val)), by = from_pos]
      
      # Initialize with NA, then fill
      yr_max <- rep(NA_real_, n_cells)
      yr_min <- rep(NA_real_, n_cells)
      yr_max[agg$from_pos] <- agg$nmax
      yr_min[agg$from_pos] <- agg$nmin
    } else {
      yr_max <- rep(NA_real_, n_cells)
      yr_min <- rep(NA_real_, n_cells)
    }
    
    # Write into output vectors
    out_max[row_range]  <- yr_max
    out_min[row_range]  <- yr_min
    out_mean[row_range] <- yr_mean
  }
  
  # Assign columns to cell_data by reference
  set(cell_data, j = col_max,  value = out_max)
  set(cell_data, j = col_min,  value = out_min)
  set(cell_data, j = col_mean, value = out_mean)
  
  invisible(cell_data)
}

# =============================================================================
# STEP 3: Run for all neighbor source variables
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  t0 <- proc.time()
  
  compute_neighbor_features_fast(
    cell_data, var_name, A, edge_dt, neighbor_counts, n_cells, years
  )
  
  elapsed <- (proc.time() - t0)[3]
  cat(sprintf("    Done in %.1f seconds\n", elapsed))
}

# =============================================================================
# STEP 4: Restore original row order if needed, then predict
# =============================================================================

# If downstream code or the RF model expects original row order, restore it:
# (Assuming cell_data had a natural row index before we re-sorted)
# setkey(cell_data, original_row_id)  # if you stored one

# Apply the pre-trained Random Forest model (do NOT retrain)
# cell_data[, prediction := predict(rf_model, newdata = cell_data)]

cat("Neighbor feature engineering complete.\n")
cat(sprintf("Total rows: %d | Variables added: %d\n",
            nrow(cell_data), 3 * length(neighbor_source_vars)))
```

---

## Summary of Guarantees

| Property | Status |
|---|---|
| **Numerical equivalence** | âœ… Identical max, min, mean of non-NA neighbor values per node-year |
| **Trained RF model preserved** | âœ… No retraining; predict-only |
| **Graph built once** | âœ… Sparse matrix + edge list constructed once from `nb` object |
| **Topology reused across years** | âœ… Same `A` and `edge_dt` for all 28 year-slices |
| **RAM-safe for 16 GB** | âœ… Sparse matrix ~33 MB; edge list ~22 MB; year-slices ~344K rows |
| **Expected runtime** | ~5â€“15 minutes (vs. 86+ hours): sparse mat-vec is O(nnz)â‰ˆ1.37M per multiply; `data.table` grouped max/min over 1.37M edges is sub-second per year-variable |

The key architectural change: **decompose the panel into year-slices sharing a fixed sparse graph**, replacing 6.46M R-level string-key lookups with 140 iterations (28 years Ã— 5 variables) of vectorized sparse linear algebra and `data.table` grouped aggregation.
END_RESPONSE>>>
