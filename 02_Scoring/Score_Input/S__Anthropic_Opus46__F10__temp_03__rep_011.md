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
run_id: Anthropic_F10_T03_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each via an anonymous function call with string-pasting and named-vector lookups. The `paste(id, year)` keying strategy means the lookup table has 6.46M character keys, and for each of the 6.46M rows, it does character matching against this table. This is O(NÂ²)-flavored in practice due to repeated string operations.

2. **`compute_neighbor_stats` iterates row-by-row in pure R** over 6.46M rows, extracting subsets of a numeric vector via list-of-index-vectors. Each call to `max/min/mean` on small vectors has high per-call overhead. With 5 variables Ã— 6.46M rows = 32.3M iterations of the inner function, this dominates runtime.

3. **The neighbor topology is year-invariant but the lookup is rebuilt entangling spatial and temporal dimensions.** Rook neighbors are a fixed spatial property of the grid. The current code re-discovers the same spatial neighbors for every year by embedding year into the key. This inflates the problem from ~344K spatial lookups to ~6.46M spatiotemporal lookups.

**Root cause summary:** The code treats a separable problem (space Ã— time) as a monolithic spatiotemporal problem, and uses pure-R loops with per-element string operations over millions of rows.

---

## Optimization Strategy

### Key Insight: Separate Space from Time

The rook neighbor graph is **purely spatial** â€” it does not change across years. For any variable `v`, the neighbor statistics for cell `i` in year `t` depend only on the values of `v` for cell `i`'s spatial neighbors in the **same year** `t`. This means:

1. **Build the spatial adjacency structure once** as a sparse matrix (344K Ã— 344K), not a 6.46M-entry list.
2. **Reshape each variable into a matrix** of shape (344K cells Ã— 28 years).
3. **Use sparse matrixâ€“dense matrix multiplication** (`A %*% V`) to compute neighbor sums and neighbor counts in one vectorized operation, then derive mean. For max and min, use grouped operations via the sparse structure.

### Specific Optimizations

| Bottleneck | Solution | Speedup Factor |
|---|---|---|
| String-key lookup over 6.46M rows | Integer-indexed sparse matrix, built once | ~100Ã— |
| Row-by-row `lapply` for stats | Vectorized sparse matrix ops for mean; column-parallel grouped ops for max/min | ~50â€“200Ã— |
| Redundant per-year neighbor discovery | Year-invariant adjacency matrix reused across all 28 years | 28Ã— |
| 5 variables processed sequentially with same structure | Same sparse matrix reused for all variables | Marginal but clean |

### Memory Budget

- Sparse adjacency matrix: ~1.37M non-zeros Ã— 12 bytes â‰ˆ 16 MB
- One variable reshaped to (344K Ã— 28) dense matrix: ~77 MB
- Three output matrices (max, min, mean) per variable: ~231 MB
- Peak for one variable pass: ~325 MB
- Total with all 5 variables added to `cell_data`: the original `cell_data` with ~110 columns is ~5.7 GB at 8 bytes/element. Adding 15 columns (3 stats Ã— 5 vars) adds ~775 MB. Fits in 16 GB.

### Numerical Equivalence

The sparse-matrix approach computes **identical** neighbor sets (same rook adjacency, same year matching). The `mean` via `sum/count` is IEEE-754 equivalent when summation order is consistent. For `max` and `min`, we use exact grouped operations. The Random Forest model is loaded and applied unchanged.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Sparse graph neighborhood aggregation for panel grid data
# =============================================================================

library(Matrix)    # sparse matrices
library(data.table) # fast reshaping and joining

# ---- Step 0: Convert cell_data to data.table if not already ----
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ---- Step 1: Build spatial adjacency as a sparse matrix (once) ----
# id_order: vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: spdep nb object (list of integer index vectors)

build_sparse_adjacency <- function(id_order, neighbors) {
  n <- length(id_order)
  # Build COO (coordinate) representation
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) > 0 && !(length(nb_i) == 1 && nb_i[0] == 0L)) {
      # spdep nb objects use 0L to indicate no neighbors
      nb_i <- nb_i[nb_i != 0L]
      if (length(nb_i) > 0) {
        from_list[[i]] <- rep.int(i, length(nb_i))
        to_list[[i]]   <- nb_i
      }
    }
  }
  from_idx <- unlist(from_list, use.names = FALSE)
  to_idx   <- unlist(to_list, use.names = FALSE)

  # Sparse matrix: A[i,j] = 1 means j is a rook neighbor of i
  # So row i contains the neighbors of cell i
  A <- sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = 1,
    dims = c(n, n),
    repr = "C"   # CSC -> will convert to CSR-like via dgRMatrix or use dgCMatrix
  )
  return(A)
}

cat("Building sparse adjacency matrix...\n")
A <- build_sparse_adjacency(id_order, rook_neighbors_unique)
n_cells <- length(id_order)
cat(sprintf("  Adjacency: %d cells, %d directed edges\n", n_cells, nnzero(A)))

# ---- Step 2: Create stable cell-index and year-index mappings ----
# Map cell IDs to row indices in the adjacency matrix
id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))

# Determine sorted unique years
years_unique <- sort(unique(cell_data$year))
n_years <- length(years_unique)
year_to_col <- setNames(seq_along(years_unique), as.character(years_unique))

cat(sprintf("  Panel: %d cells x %d years = %d expected rows\n",
            n_cells, n_years, n_cells * n_years))

# ---- Step 3: Assign spatial and temporal indices to cell_data ----
cell_data[, spatial_idx := id_to_spatial_idx[as.character(id)]]
cell_data[, year_col    := year_to_col[as.character(year)]]

# Verify completeness
stopifnot(all(!is.na(cell_data$spatial_idx)))
stopifnot(all(!is.na(cell_data$year_col)))

# ---- Step 4: Precompute neighbor count matrix for mean calculation ----
# For mean: we need sum of neighbor values / count of non-NA neighbors
# For count of non-NA: A %*% (non-NA indicator matrix)
# For sum: A %*% (value matrix with NA replaced by 0, masked by non-NA)

# ---- Step 5: Function to reshape variable to (n_cells x n_years) matrix ----
reshape_to_matrix <- function(dt, var_name, n_cells, n_years) {
  # Create matrix filled with NA
  M <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  # Fill using spatial_idx and year_col
  M[cbind(dt$spatial_idx, dt$year_col)] <- dt[[var_name]]
  return(M)
}

# ---- Step 6: Compute neighbor stats using sparse matrix operations ----
compute_neighbor_stats_sparse <- function(A, V) {
  # A: n_cells x n_cells sparse adjacency (dgCMatrix)
  # V: n_cells x n_years dense matrix of variable values
  # Returns list with max_mat, min_mat, mean_mat (each n_cells x n_years)

  n_cells <- nrow(V)
  n_years <- ncol(V)

  # --- MEAN via sparse matrix multiplication ---
  # Replace NA with 0 for summation, track non-NA
  not_na <- !is.na(V)
  V_zero <- V
  V_zero[!not_na] <- 0

  # Indicator matrix: 1 where not NA, 0 where NA
  I_mat <- matrix(0, nrow = n_cells, ncol = n_years)
  I_mat[not_na] <- 1

  # Neighbor sum: A %*% V_zero (each row i gets sum of neighbor values)
  neighbor_sum   <- as.matrix(A %*% V_zero)
  # Neighbor count of non-NA: A %*% I_mat
  neighbor_count <- as.matrix(A %*% I_mat)

  # Mean
  mean_mat <- neighbor_sum / neighbor_count
  mean_mat[neighbor_count == 0] <- NA_real_

  # --- MAX and MIN via explicit grouped operations ---
  # Extract the sparse structure once
  # A is dgCMatrix: columns are stored. We need row-wise neighbors.
  # Convert to dgRMatrix or iterate over dgCMatrix columns smartly.
  # Most efficient: use the @i, @p, @x slots of dgCMatrix (CSC format)
  # For row-wise access, transpose to get A^T in CSC = A in CSR
  At <- t(A)  # Now At is dgCMatrix; column j of At = row j of A = neighbors of j

  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # At@p: column pointers (0-indexed), length n_cells+1
  # At@i: row indices (0-indexed) of non-zero entries
  p <- At@p
  row_idx <- At@i  # 0-indexed

  # Process year by year to keep memory bounded
  for (yr in seq_len(n_years)) {
    v <- V[, yr]  # values for this year, length n_cells

    # For each cell i, neighbors are: row_idx[(p[i]+1):p[i+1]] (converting to 1-indexed)
    # Vectorized approach: build neighbor value vector, then use grouping

    # Total number of non-zero entries
    nnz <- length(row_idx)
    if (nnz == 0) next

    # Neighbor values for all edges
    neighbor_vals <- v[row_idx + 1L]  # +1 for 0-indexed to 1-indexed

    # Group IDs: which cell does each edge belong to?
    # Cell i owns entries from index (p[i]+1) to p[i+1] (1-indexed)
    # Build group vector
    group_lengths <- diff(p)  # length n_cells, number of neighbors per cell
    group_id <- rep.int(seq_len(n_cells), times = group_lengths)

    # Remove NA neighbor values
    valid <- !is.na(neighbor_vals)
    if (sum(valid) == 0) next

    nv_valid <- neighbor_vals[valid]
    gid_valid <- group_id[valid]

    # Use data.table for fast grouped max/min
    dt_temp <- data.table(g = gid_valid, val = nv_valid)
    stats <- dt_temp[, .(mx = max(val), mn = min(val)), by = g]

    max_mat[stats$g, yr] <- stats$mx
    min_mat[stats$g, yr] <- stats$mn
  }

  return(list(max_mat = max_mat, min_mat = min_mat, mean_mat = mean_mat))
}

# ---- Step 7: Main loop over variables ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-sort cell_data by (spatial_idx, year_col) for consistent matrix fill
setkey(cell_data, spatial_idx, year_col)

cat("Computing neighbor statistics...\n")
t_start <- proc.time()

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing variable: %s\n", var_name))

  # Reshape to matrix
  V <- reshape_to_matrix(cell_data, var_name, n_cells, n_years)

  # Compute stats
  stats <- compute_neighbor_stats_sparse(A, V)

  # Map back to cell_data rows
  # cell_data already has spatial_idx and year_col
  idx_mat <- cbind(cell_data$spatial_idx, cell_data$year_col)

  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  cell_data[, (max_col)  := stats$max_mat[idx_mat]]
  cell_data[, (min_col)  := stats$min_mat[idx_mat]]
  cell_data[, (mean_col) := stats$mean_mat[idx_mat]]

  # Free memory
  rm(V, stats)
  gc()
}

t_elapsed <- proc.time() - t_start
cat(sprintf("Neighbor stats completed in %.1f seconds\n", t_elapsed[3]))

# ---- Step 8: Clean up helper columns ----
cell_data[, c("spatial_idx", "year_col") := NULL]

# ---- Step 9: Apply pre-trained Random Forest (unchanged) ----
# The model object (e.g., `rf_model`) is already in memory or loaded from disk.
# Predict using the enriched cell_data with all ~125 predictor columns.
# 
# Example:
# library(ranger)  # or randomForest
# predictions <- predict(rf_model, data = cell_data)$predictions
# cell_data[, prediction := predictions]

cat("Pipeline complete. Ready for Random Forest prediction.\n")
```

---

## Performance Analysis

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| Neighbor lookup construction | ~6.46M string-paste + match ops | One sparse matrix build (~1.37M entries) | ~100Ã— |
| Mean computation (per variable) | 6.46M `lapply` calls with `mean()` | Single sparse matrix multiply `A %*% V` (344K Ã— 28) | ~200Ã— |
| Max/Min computation (per variable) | 6.46M `lapply` calls with `max()/min()` | 28 year-passes with `data.table` grouped aggregation | ~30â€“50Ã— |
| Total (5 variables) | ~86+ hours estimated | **~2â€“10 minutes expected** | ~500â€“2500Ã— |

**Why it's faster:**
- **Sparse matrix multiply** (`A %*% V_zero`) is a single BLAS-backed operation that computes all neighbor sums for all cells across all years simultaneously. This replaces ~32M R-level function calls.
- **The adjacency matrix is built once** (16 MB) and reused for all 5 variables and all 28 years.
- **Year-column iteration for max/min** processes only 28 passes (not 6.46M), and within each pass, `data.table` grouped aggregation is C-optimized.
- **Memory peak** stays well under 16 GB: the largest single object is a 344K Ã— 28 dense matrix (~77 MB).

**Numerical equivalence** is preserved because:
- The same rook neighbor sets are encoded in the sparse matrix.
- `mean = sum / count` with NA exclusion matches the original `mean(neighbor_vals[!is.na()])`.
- `max` and `min` are computed on identical neighbor value sets per cell-year.
- The Random Forest model is never retrained â€” only `predict()` is called on the enriched data.
END_RESPONSE>>>
