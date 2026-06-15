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
run_id: Anthropic_F10_T07_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation suffers from three critical bottlenecks:

### 1. **O(nÂ²)-scale neighbor lookup construction**
`build_neighbor_lookup` iterates over all ~6.46 million rows, and for each row performs string concatenation (`paste`) and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector indexing in R is hash-based but still slow at this scale. The lookup is built once but costs heavily.

### 2. **Per-row `lapply` in `compute_neighbor_stats`**
For each of the ~6.46 million rows, an anonymous function extracts neighbor values, removes NAs, and computes `max/min/mean`. This is called 5 times (once per variable), yielding ~32.3 million R-level function calls with repeated subsetting.

### 3. **Redundant topology recomputation across years**
The rook-neighbor graph is **spatial only** â€” it doesn't change across years. Yet the current code builds a single monolithic lookup over all 6.46M cell-year rows, interleaving spatial topology with temporal matching via string keys. This is wasteful: the same 1.37M directed edges repeat identically for each of 28 years.

### Memory profile
The `neighbor_lookup` list of 6.46M integer vectors, plus the `idx_lookup` named vector of 6.46M entries, plus intermediate string vectors, likely consumes 8â€“12 GB and causes severe GC pressure on a 16 GB machine.

---

## Optimization Strategy

**Core insight:** Separate the spatial graph topology (344K nodes, ~1.37M edges) from the temporal dimension (28 years). Build the sparse adjacency structure once over cells, then for each year, use vectorized sparse-matrix multiplication to compute neighbor aggregates.

### Specific techniques:

1. **Build a sparse adjacency matrix once** from `rook_neighbors_unique` (344,208 Ã— 344,208, ~1.37M nonzeros). This is tiny in memory (~20 MB as a `dgCMatrix`).

2. **For each year, extract the variable column as a dense vector over cells, then compute:**
   - `neighbor_sum = A %*% x` (sparse matrixâ€“vector multiply)
   - `neighbor_count = A %*% (!is.na(x))` (count of non-NA neighbors)
   - `neighbor_mean = neighbor_sum / neighbor_count`
   - For `max` and `min`: use a grouped operation over the edge list (CSC column pointers).

3. **Vectorize max/min** by operating on the CSC structure of the sparse matrix directly, using `vapply` over columns (each column's nonzero entries are the neighbors). Alternatively, use `data.table` grouped operations on the edge list.

4. **Loop over 28 years Ã— 5 variables = 140 iterations**, each operating on a 344K-length vector. Each iteration takes ~0.1â€“0.5 seconds â†’ total ~1â€“2 minutes.

5. **Preserve numerical equivalence:** The sparse matrixâ€“vector product computes exactly the same sums; dividing by the exact non-NA count gives the identical mean. Max and min over the same neighbor sets are identical.

---

## Optimized R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build sparse adjacency matrix from the spdep nb object (ONCE)
# ==============================================================================
build_adjacency_matrix <- function(nb_obj, n) {

  # nb_obj: list of length n, nb_obj[[i]] = integer vector of neighbor indices
  # Builds a sparse n x n adjacency matrix A where A[j, i] = 1 if j is a

  # neighbor of i. This way A %*% x gives the sum of neighbor values for

  # each node.
  
  # Build COO representation
  from_list <- lapply(seq_along(nb_obj), function(i) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) == 0L) return(NULL)
    data.frame(j = nbrs, i = rep.int(i, length(nbrs)))
  })
  
  edges <- rbindlist(from_list)
  
  # A[j, i] = 1 means j is a neighbor of i
  # So A %*% x[i] = sum of x[j] for all j that are neighbors of i
  # Wait â€” we want: for node i, aggregate over its neighbors j.
  # A[j, i] = 1 means column i has nonzeros at rows j (the neighbors of i).
  # Then (A^T %*% x)[i] = sum_{j neighbor of i} x[j]. 
  # OR: build A[i, j] = 1 if j is neighbor of i, then A %*% x directly works.
  
  # Let's build A where A[i, j] = 1 if j is a neighbor of i.
  # Then (A %*% x)[i] = sum of x[j] for j in neighbors(i).
  
  A <- sparseMatrix(
    i = edges$i,
    j = edges$j,
    x = 1,
    dims = c(n, n),
    repr = "C"   # CSC format
  )
  
  return(A)
}

# ==============================================================================
# STEP 2: Compute max and min using the sparse matrix structure
# ==============================================================================
# For max/min we cannot use matrix multiplication. We use the CSR structure.
# In CSR (dgRMatrix), row i's nonzero column indices are in
# j[p[i]+1 : p[i+1]], which correspond to the neighbors of node i.
# We iterate over rows using compiled C-level access via .Call or vectorized R.

compute_sparse_max_min <- function(A_csr, x) {
  # A_csr: dgRMatrix (CSR), x: numeric vector of length ncol(A_csr)
  # Returns matrix of (n x 2): col1 = max, col2 = min over neighbors
  
  n <- nrow(A_csr)
  p <- A_csr@p        # row pointers (0-based), length n+1
  j <- A_csr@j        # column indices (0-based)
  
  # Pre-allocate
  max_vals <- rep(NA_real_, n)
  min_vals <- rep(NA_real_, n)
  
  # Vectorized approach: build a data.table of (row_id, neighbor_value)
  # and do grouped max/min
  
  # Map each nonzero entry to its row
  # Row i (0-based) owns entries from p[i]+1 to p[i+1] (1-based: p[i+1]+1 to p[i+2])
  # Number of nonzeros per row:
  row_counts <- diff(p)  # length n
  
  if (sum(row_counts) == 0L) {
    return(cbind(max_vals, min_vals))
  }
  
  row_ids <- rep.int(seq_len(n), row_counts)
  col_ids <- j + 1L  # convert to 1-based
  
  neighbor_vals <- x[col_ids]
  
  dt <- data.table(row_id = row_ids, val = neighbor_vals)
  dt <- dt[!is.na(val)]
  
  if (nrow(dt) == 0L) {
    return(cbind(max_vals, min_vals))
  }
  
  agg <- dt[, .(mx = max(val), mn = min(val)), by = row_id]
  
  max_vals[agg$row_id] <- agg$mx
  min_vals[agg$row_id] <- agg$mn
  
  cbind(max_vals, min_vals)
}

# ==============================================================================
# STEP 3: Main pipeline
# ==============================================================================
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # Convert to data.table for speed
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  n_cells <- length(id_order)
  cat("Number of cells:", n_cells, "\n")
  
  # --- Build adjacency matrix ONCE ---
  cat("Building sparse adjacency matrix...\n")
  A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
  cat("  Adjacency matrix:", nrow(A), "x", ncol(A), 
      "with", length(A@x), "nonzeros\n")
  
  # Also build CSR version for max/min
  A_csr <- as(A, "RsparseMatrix")
  
  # --- Build cell ID to matrix-row mapping ---
  # id_order[k] is the cell ID for matrix row/col k
  id_to_matidx <- setNames(seq_along(id_order), as.character(id_order))
  
  # --- Get unique years, sorted ---
  years <- sort(unique(cell_data$year))
  cat("Years:", min(years), "-", max(years), "(", length(years), "years)\n")
  
  # --- Map each row of cell_data to its matrix index ---
  cell_data[, mat_idx := id_to_matidx[as.character(id)]]
  
  # --- Ensure cell_data is keyed for fast subsetting ---
  setkey(cell_data, year)
  
  # --- Source variables ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  # Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)
  }
  
  # --- Main loop: iterate over years, then variables ---
  cat("Computing neighbor statistics...\n")
  t0 <- proc.time()
  
  for (yr in years) {
    # Get row indices for this year
    yr_rows <- which(cell_data$year == yr)
    
    # Get the matrix indices for these rows (which cell each row maps to)
    yr_mat_idx <- cell_data$mat_idx[yr_rows]
    
    # Build a dense vector for each variable: x[mat_idx] = value
    # Only cells present in this year get filled; others stay NA
    
    for (var_name in neighbor_source_vars) {
      # Build dense vector over all cells for this year
      x <- rep(NA_real_, n_cells)
      x[yr_mat_idx] <- cell_data[[var_name]][yr_rows]
      
      # --- Neighbor mean via sparse matrix multiply ---
      # Handle NAs: replace NA with 0 for sum, track counts separately
      x_nona <- x
      x_nona[is.na(x_nona)] <- 0
      not_na <- as.numeric(!is.na(x))
      
      neighbor_sum   <- as.numeric(A %*% x_nona)    # length n_cells
      neighbor_count <- as.numeric(A %*% not_na)     # length n_cells
      
      neighbor_mean <- ifelse(neighbor_count > 0, 
                              neighbor_sum / neighbor_count, 
                              NA_real_)
      
      # --- Neighbor max and min via CSR grouped aggregation ---
      maxmin <- compute_sparse_max_min(A_csr, x)
      neighbor_max <- maxmin[, 1]
      neighbor_min <- maxmin[, 2]
      
      # --- Write results back to cell_data for the rows of this year ---
      col_max  <- paste0("neighbor_max_", var_name)
      col_min  <- paste0("neighbor_min_", var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      
      set(cell_data, i = yr_rows, j = col_max,  value = neighbor_max[yr_mat_idx])
      set(cell_data, i = yr_rows, j = col_min,  value = neighbor_min[yr_mat_idx])
      set(cell_data, i = yr_rows, j = col_mean, value = neighbor_mean[yr_mat_idx])
    }
    
    cat("  Year", yr, "done\n")
  }
  
  elapsed <- (proc.time() - t0)[3]
  cat("Neighbor features computed in", round(elapsed / 60, 1), "minutes\n")
  
  # Clean up helper column
  cell_data[, mat_idx := NULL]
  
  return(cell_data)
}

# ==============================================================================
# STEP 4: Run pipeline and predict with existing Random Forest
# ==============================================================================

# --- Load pre-trained model and data ---
# rf_model          <- readRDS("path/to/trained_rf_model.rds")
# cell_data         <- readRDS("path/to/cell_data.rds")
# id_order          <- readRDS("path/to/id_order.rds")
# rook_neighbors_unique <- readRDS("path/to/rook_neighbors_unique.rds")

# --- Compute neighbor features ---
cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)

# --- Predict using the EXISTING pre-trained Random Forest (no retraining) ---
# Ensure column names match what the model expects
# predictions <- predict(rf_model, newdata = cell_data)
# cell_data$prediction <- predictions
```

---

## Why This Is Correct and Numerically Equivalent

| Statistic | Original method | Optimized method | Equivalence |
|-----------|----------------|-----------------|-------------|
| **Mean** | `mean(vals[neighbors])` after removing NAs | `(A %*% x_nona)[i] / (A %*% not_na)[i]` | Identical: same sum divided by same count |
| **Max** | `max(vals[neighbors])` after removing NAs | Grouped `max` over CSR row entries, NAs excluded | Identical: same set of values |
| **Min** | `min(vals[neighbors])` after removing NAs | Grouped `min` over CSR row entries, NAs excluded | Identical: same set of values |
| **NA handling** | Rows with no valid neighbors â†’ `NA` | `neighbor_count == 0` â†’ `NA`; empty groups â†’ `NA` | Identical |

## Performance Analysis

| Aspect | Original | Optimized | Speedup |
|--------|----------|-----------|---------|
| **Lookup build** | ~6.46M string pastes + hash lookups | One `sparseMatrix()` call on 344K nodes | ~1000Ã— |
| **Mean computation** | 6.46M Ã— 5 R function calls | 28 Ã— 5 = 140 sparse mat-vec multiplies | ~500Ã— |
| **Max/Min** | 6.46M Ã— 5 R function calls | 140 `data.table` grouped aggregations | ~200Ã— |
| **Memory** | ~8â€“12 GB (string keys + list of vectors) | ~200 MB (sparse matrix + temp vectors) | ~40Ã— |
| **Estimated wall time** | 86+ hours | **2â€“5 minutes** | ~1000Ã— |

The sparse matrix is ~20 MB. Each year's dense vector is ~2.6 MB. The `data.table` grouped aggregation for max/min processes ~1.37M edges per iteration. Total: 140 iterations Ã— ~0.5s â‰ˆ ~70 seconds, plus overhead. The trained Random Forest model is never modified.
END_RESPONSE>>>
