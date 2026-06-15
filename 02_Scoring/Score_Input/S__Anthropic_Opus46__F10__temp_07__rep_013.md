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
run_id: Anthropic_F10_T07_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each produced by an `lapply` call that performs string pasting, named-vector lookups (`idx_lookup[neighbor_keys]`), and `NA` filtering. Named character vector lookup in R is O(n) per probe in the worst case and extremely slow at this scale. This step alone likely takes many hours.

2. **`compute_neighbor_stats` iterates over 6.46M list entries per variable**, extracting subsets of a numeric vector. With 5 variables, this is ~32.3 million R-level list iterations with repeated subsetting â€” all in pure interpreted R.

3. **The topology is year-invariant but the lookup is rebuilt as if it were year-specific.** Rook neighbors don't change across years. The code pastes `(id, year)` keys to re-discover the same spatial adjacency structure 28 times per cell. This is entirely redundant.

**Root cause:** The algorithm is correct but implemented with the slowest possible R idioms (character key lookups, per-row `lapply`, no vectorization, no use of sparse matrix algebra).

---

## Optimization Strategy

### Key Insight: Separate Topology from Time

The rook neighbor graph is **purely spatial** â€” it doesn't change across years. The panel data is a **cell Ã— year** matrix. Neighbor aggregation is simply **sparse matrixâ€“vector multiplication** (for mean) and analogous sparse operations (for max/min).

### Plan

1. **Build a sparse adjacency matrix `A`** (344,208 Ã— 344,208) once from `rook_neighbors_unique`. This matrix has ~1.37M non-zero entries â€” trivially small for `Matrix::sparseMatrix`.

2. **Reshape each variable into a cell Ã— year matrix** (344,208 Ã— 28).

3. **Compute neighbor statistics using sparse matrix operations:**
   - **Mean:** `A_row_normalized %*% X` (sparse matrix Ã— dense matrix, fully vectorized in C).
   - **Max and Min:** Use a grouped operation over the sparse structure. We iterate over the 28 year-columns (not 6.46M rows), applying vectorized sparse-group max/min per column.

4. **Column-bind results back** to the data.frame.

5. **Run `predict(rf_model, cell_data)`** unchanged.

**Expected speedup:** From 86+ hours to **minutes**. The sparse matrix multiply for mean is a single BLAS call per variable. Max/min require a column-wise loop (28 iterations) with C-level sparse indexing, still extremely fast.

---

## Optimized R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 0: Ensure data is ordered consistently
# ==============================================================================
# cell_data must have columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# id_order: integer vector of all cell IDs in canonical order (length 344,208)
# rook_neighbors_unique: spdep nb object (list of length 344,208)
# rf_model: pre-trained randomForest model â€” NOT modified

# Convert to data.table for speed (non-destructive; same data)
setDT(cell_data)

# Canonical cell index: position in id_order
id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

# Sort data by (id, year) for matrix reshaping
setkey(cell_data, id, year)

# Verify completeness (balanced panel assumed by original code)
n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)
stopifnot(nrow(cell_data) == n_cells * n_years)

# ==============================================================================
# STEP 1: Build sparse adjacency matrix ONCE
# ==============================================================================
# rook_neighbors_unique is an nb object: list of integer vectors of neighbor indices
# into id_order. We build a sparse row-stochastic matrix and a raw adjacency matrix.

build_sparse_adjacency <- function(nb_obj, n) {
  # Preallocate vectors for triplet form
  # Count total edges
  edge_count <- sum(vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))

  row_i <- integer(edge_count)
  col_j <- integer(edge_count)
  pos <- 0L

  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep convention: 0 means no neighbors
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    k <- length(nbrs)
    row_i[(pos + 1L):(pos + k)] <- i
    col_j[(pos + 1L):(pos + k)] <- nbrs
    pos <- pos + k
  }

  # Raw adjacency (values = 1)
  A <- sparseMatrix(
    i = row_i, j = col_j, x = rep(1, edge_count),
    dims = c(n, n), repr = "C"   # CSC -> fast column access; we transpose as needed
  )
  return(A)
}

cat("Building sparse adjacency matrix...\n")
A <- build_sparse_adjacency(rook_neighbors_unique, n_cells)

# Row-normalized version for computing means: each row sums to 1
# (or 0 for isolated nodes, which will produce NaN -> we fix to NA later)
row_degrees <- diff(A@p)  # for dgCMatrix this is column counts; we need row counts
# Actually for dgCMatrix (CSC), row sums:
A_csr <- as(A, "RsparseMatrix")  # dgRMatrix â€” row-oriented
row_deg <- diff(A_csr@p)
row_deg_safe <- ifelse(row_deg == 0L, 1L, row_deg)  # avoid division by zero

# Build row-normalized matrix for mean computation
# D^{-1} A where D is diagonal of row degrees
D_inv <- Diagonal(x = 1 / row_deg_safe, n = n_cells)
A_mean <- D_inv %*% A   # sparse %*% sparse -> sparse, still very fast
# Convert to dgCMatrix for fast dense multiply
A_mean <- as(A_mean, "CsparseMatrix")

# For max/min we need the CSR representation for fast row-wise access
A_csr <- as(A, "RsparseMatrix")

# Precompute which nodes have no neighbors (isolated)
isolated <- (row_deg == 0L)

cat("Adjacency matrix built:", nnzero(A), "non-zero entries.\n")

# ==============================================================================
# STEP 2: Reshape variable into cell Ã— year matrix
# ==============================================================================
# cell_data is keyed by (id, year). We need cell ordering to match id_order.

# Map each row's cell ID to its canonical index
cell_data[, cell_idx := id_to_idx[as.character(id)]]

# Year to column index
year_to_col <- setNames(seq_along(years), as.character(years))
cell_data[, year_idx := year_to_col[as.character(year)]]

# Order by (cell_idx, year_idx) so we can fill matrices column-major
setkey(cell_data, cell_idx, year_idx)

reshape_to_matrix <- function(dt, var_name, n_cells, n_years) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]
  return(mat)
}

# ==============================================================================
# STEP 3: Compute neighbor max, min, mean â€” vectorized
# ==============================================================================
# Mean: A_mean %*% X  (sparse Ã— dense, computed in compiled code)
# Max/Min: row-wise sparse aggregation over X columns

# For max and min we use the CSR structure to iterate in C-like fashion via
# vectorized R. We process one year-column at a time (28 iterations) and use
# the sparse row pointers.

compute_neighbor_max_min <- function(A_csr, X, n_cells, n_years, isolated) {
  # A_csr is dgRMatrix: @p (row pointers, length n+1), @j (column indices), @x (values)
  # We only need the sparsity pattern (all x == 1).
  p <- A_csr@p          # length n_cells + 1
  j <- A_csr@j + 1L     # 0-based -> 1-based column indices (these are neighbor cell indices)

  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (t in seq_len(n_years)) {
    x_col <- X[, t]  # values for this year, indexed by cell

    # For each row i, neighbors are j[(p[i]+1):p[i+1]]
    # We vectorize using a "segment" approach:
    # Expand neighbor values, then use grouping

    # Total number of edges
    # neighbor_vals: value of each neighbor for this year
    neighbor_vals <- x_col[j]  # length = nnz(A)

    # Now we need to aggregate by row. We create a row-id vector.
    # row_id[k] = i means the k-th nonzero belongs to row i
    # We can build this from the row pointers:
    row_id <- rep(seq_len(n_cells), times = diff(p))

    # Remove NAs in neighbor_vals
    valid <- !is.na(neighbor_vals)
    nv <- neighbor_vals[valid]
    ri <- row_id[valid]

    if (length(nv) > 0) {
      # Use data.table for fast grouped max/min
      dt_tmp <- data.table(ri = ri, nv = nv)
      agg <- dt_tmp[, .(mx = max(nv), mn = min(nv)), by = ri]

      max_mat[agg$ri, t] <- agg$mx
      min_mat[agg$ri, t] <- agg$mn
    }
    # Rows not in agg (isolated or all-NA neighbors) remain NA â€” correct.
  }

  list(max = max_mat, min = min_mat)
}

# ==============================================================================
# STEP 4: Main loop â€” 5 variables
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features for", length(neighbor_source_vars), "variables...\n")

for (var_name in neighbor_source_vars) {
  cat("  Processing:", var_name, "... ")
  t0 <- proc.time()

  # Reshape to matrix
  X <- reshape_to_matrix(cell_data, var_name, n_cells, n_years)

  # --- MEAN via sparse matrix multiply ---
  mean_mat <- as.matrix(A_mean %*% X)   # n_cells Ã— n_years dense matrix
  # Fix isolated nodes: A_mean gives 0 for them (since D_inv row = 1, A row = 0),
  # but original code returns NA. Set to NA:
  mean_mat[isolated, ] <- NA_real_

  # Also: if all neighbors had NA for a cell-year, sparse multiply gives 0.
  # We need to detect this. The original code removes NAs then computes mean.
  # Sparse multiply treats NA as 0 (after X has NAs). We must handle this.
  #
  # Correct approach: replace NA with 0 in X, compute sum via A %*% X_zero,
  # compute count of non-NA neighbors via A %*% (non-NA indicator), then mean = sum/count.

  X_zero <- X
  X_zero[is.na(X_zero)] <- 0
  non_na <- (!is.na(X)) * 1.0  # indicator matrix

  sum_mat   <- as.matrix(A %*% X_zero)     # sum of non-NA neighbor values
  count_mat <- as.matrix(A %*% non_na)     # count of non-NA neighbor values

  mean_mat <- ifelse(count_mat > 0, sum_mat / count_mat, NA_real_)
  mean_mat[isolated, ] <- NA_real_

  # --- MAX / MIN ---
  maxmin <- compute_neighbor_max_min(A_csr, X, n_cells, n_years, isolated)
  max_mat <- maxmin$max
  min_mat <- maxmin$min

  # --- Flatten back to data.table column order ---
  # cell_data is keyed by (cell_idx, year_idx), so the linear index is:
  lin_idx <- cbind(cell_data$cell_idx, cell_data$year_idx)

  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  cell_data[, (col_max)  := max_mat[lin_idx]]
  cell_data[, (col_min)  := min_mat[lin_idx]]
  cell_data[, (col_mean) := mean_mat[lin_idx]]

  elapsed <- (proc.time() - t0)[3]
  cat(round(elapsed, 1), "seconds\n")
}

# Clean up helper columns
cell_data[, c("cell_idx", "year_idx") := NULL]

cat("Neighbor feature computation complete.\n")

# ==============================================================================
# STEP 5: Predict with the pre-trained Random Forest (UNCHANGED)
# ==============================================================================
# The RF model is NOT retrained. We simply score.
# Ensure cell_data has all required predictor columns (original ~110 + 15 new neighbor features)

cell_data$prediction <- predict(rf_model, newdata = cell_data)

cat("Prediction complete.\n")
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Topology construction** | 6.46M string-key lookups via named vector | Single sparse matrix build from `nb` object (~1.37M entries) |
| **Mean computation** | 6.46M R-level `lapply` + `mean()` per variable | One sparse `A %*% X_zero` and `A %*% non_na` matrix multiply per variable (BLAS/C level) |
| **Max/Min computation** | 6.46M R-level `lapply` + `max()`/`min()` per variable | Vectorized expand + `data.table` grouped aggregation, 28 iterations per variable |
| **Total R-level iterations** | ~32.3M `lapply` calls + 6.46M key lookups | 5 sparse matmuls + 140 grouped aggregations |
| **Numerical equivalence** | â€” | âœ… Preserved: same NA handling (remove NAs, then aggregate); mean = sum(non-NA) / count(non-NA); max/min over non-NA values only |
| **RF model** | Pre-trained | âœ… Untouched, only `predict()` called |
| **RAM** | ~6.46M-element list of integer vectors | Sparse matrix (~22 MB) + one 344KÃ—28 dense matrix at a time (~77 MB) â€” well within 16 GB |
| **Estimated runtime** | 86+ hours | **~5â€“15 minutes** |

The key transformation: **replace 6.46 million interpreted R list lookups with sparse linear algebra computed in compiled C/Fortran code**, and separate the time-invariant graph topology from the time-varying node attributes.
END_RESPONSE>>>
