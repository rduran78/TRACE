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
run_id: Anthropic_F10_T10_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` constructs a per-row (per cell-year) neighbor lookup by iterating over ~6.46 million rows with string-key hashing**, and **`compute_neighbor_stats` iterates over the same ~6.46 million entries per variable with R-level `lapply` loops**. This means:

1. **Redundant topology construction**: The rook-neighbor graph is a *spatial* relationship â€” it is identical across all 28 years. Yet `build_neighbor_lookup` embeds year into every key and re-resolves neighbors for every cell-year row. With ~6.46M rows, this creates ~6.46M list entries, each requiring string concatenation, lookup, and NA filtering. This is O(N_rows Ã— avg_degree) string operations.

2. **R-level loop over millions of rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over ~6.46M elements. R's interpreter overhead per iteration is ~1â€“5 Âµs, so even trivially fast bodies sum to hours.

3. **String-keyed lookups**: Using `paste(id, year, sep="_")` as hash keys and named vector indexing is extremely slow at scale compared to integer indexing.

4. **Per-variable recomputation**: `compute_neighbor_stats` is called 5 times (once per variable), each time iterating over the full 6.46M-row lookup.

**Summary**: The 86+ hour runtime is caused by millions of R-level iterations with string operations, repeated identically for each year and each variable, when the underlying graph topology is year-invariant.

---

## Optimization Strategy

### Key Insight: Separate Topology from Attributes

The rook-neighbor graph is purely spatial (344,208 nodes, ~1.37M directed edges). The yearly panel just replicates node attributes across 28 time slices. Therefore:

1. **Build a sparse adjacency matrix once** from the `nb` object (344,208 Ã— 344,208, ~1.37M nonzeros). This is the graph topology.

2. **Reshape each variable into a cell Ã— year matrix** (344,208 Ã— 28).

3. **Compute neighbor aggregates via sparse matrix operations** â€” sparse matrixâ€“dense matrix multiplication gives neighbor *sums* and neighbor *counts* in one shot, yielding **mean** directly. For **max** and **min**, we use grouped operations over the CSC/CSR structure.

4. **Vectorize everything** â€” eliminate all R-level row iteration.

### Complexity Comparison

| Step | Original | Optimized |
|------|----------|-----------|
| Topology | O(6.46M) string ops | O(1.37M) integer insertions (once) |
| Mean per var | O(6.46M) R iterations | One sparse mat Ã— dense mat multiply |
| Max/Min per var | O(6.46M) R iterations | Vectorized grouped operation over CSR |
| Total R-loop iterations | ~32.3M (5 vars Ã— 6.46M) | **Zero** |

Expected speedup: **~200â€“500Ã—**, bringing runtime to **minutes**.

---

## Optimized R Code

```r
# =============================================================================
# Optimized Neighbor Aggregation Pipeline
# Preserves numerical equivalence with original max/min/mean statistics.
# Preserves the pre-trained Random Forest model (no retraining).
# =============================================================================

library(Matrix)   # sparse matrices
library(data.table)

# ---- 1. Build sparse adjacency matrix ONCE from the nb object ---------------
# rook_neighbors_unique: spdep nb object (list of integer vectors of neighbor indices)
# id_order: vector of cell IDs in the order matching the nb object

build_sparse_adjacency <- function(nb_obj) {
  # nb_obj[[i]] contains integer indices of neighbors of node i
  # We build a CSC sparse matrix (dgCMatrix) of dimension n x n
  n <- length(nb_obj)
  
  # Pre-compute total edges for pre-allocation
  n_edges <- sum(vapply(nb_obj, function(x) {
    # spdep nb objects use 0L to indicate no neighbors
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  # Build COO triplets
  from_idx <- integer(n_edges)
  to_idx   <- integer(n_edges)
  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    k <- length(nbrs)
    from_idx[pos:(pos + k - 1L)] <- i
    to_idx[pos:(pos + k - 1L)]   <- nbrs
    pos <- pos + k
  }
  
  # A[i,j] = 1 means j is a neighbor of i (row i aggregates over its neighbors)
  sparseMatrix(i = from_idx, j = to_idx, x = 1, dims = c(n, n))
}

cat("Building sparse adjacency matrix...\n")
A <- build_sparse_adjacency(rook_neighbors_unique)
n_cells <- nrow(A)
cat(sprintf("  Nodes: %d, Edges (nnz): %d\n", n_cells, nnz(A)))

# ---- 2. Prepare cell_data as data.table for fast reshaping ------------------
# cell_data must have columns: id, year, and the neighbor_source_vars
# id_order defines the mapping from cell id to matrix row index

setDT(cell_data)

# Create integer node index matching the adjacency matrix row order
id_to_node <- setNames(seq_along(id_order), as.character(id_order))
cell_data[, node_idx := id_to_node[as.character(id)]]

# Sorted unique years
years <- sort(unique(cell_data$year))
n_years <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))
cell_data[, year_col := year_to_col[as.character(year)]]

# ---- 3. Functions for neighbor stats via sparse matrix ops -------------------

# For MEAN: A %*% X gives sum of neighbor values for each node.
#           A %*% (ones where X is not NA) gives count of non-NA neighbors.
#           mean = sum / count

# For MAX and MIN: We need grouped operations over the sparse structure.
# We iterate over columns of A (CSC format) or use the explicit edge list.

# Pre-extract the CSR structure for max/min (row-oriented access)
# Convert A to dgRMatrix (CSR) for efficient row-wise access
A_csr <- as(A, "RsparseMatrix")

compute_neighbor_aggregates_matrix <- function(val_vec, node_idx, year_col,
                                                n_cells, n_years, A, A_csr) {
  # Build cell x year matrix (NA for missing)
  X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  X[cbind(node_idx, year_col)] <- val_vec
  
  # --- MEAN via sparse matrix multiplication ---
  not_na <- !is.na(X)
  X_zero <- X
  X_zero[is.na(X_zero)] <- 0  # replace NA with 0 for multiplication
  
  neighbor_sum   <- A %*% X_zero        # n_cells x n_years (Matrix)
  neighbor_count <- A %*% (not_na * 1)  # n_cells x n_years (Matrix)
  
  # Convert to dense
  neighbor_sum   <- as.matrix(neighbor_sum)
  neighbor_count <- as.matrix(neighbor_count)
  
  neighbor_mean <- neighbor_sum / neighbor_count  # NaN where count==0
  neighbor_mean[neighbor_count == 0] <- NA_real_
  
  # --- MAX and MIN via CSR row iteration (vectorized per year) ---
  # Strategy: for each year, use the sparse row pointers to do grouped max/min
  # We use the @p (row pointer) and @j (column index) slots of CSR
  
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # CSR slots: A_csr@p (length n_cells+1), A_csr@j (0-based col indices)
  row_ptr <- A_csr@p   # length n_cells + 1, 0-based
  col_j   <- A_csr@j   # 0-based column indices of nonzeros
  
  for (yr in seq_len(n_years)) {
    x_yr <- X[, yr]  # length n_cells, node values this year
    
    # For each node i, neighbors are col_j[(row_ptr[i]+1):row_ptr[i+1]]  (R 1-based)
    # Vectorized approach: expand neighbor values and use grouped ops
    
    # All neighbor values in edge order
    nbr_vals <- x_yr[col_j + 1L]  # col_j is 0-based, so +1
    
    # Build a group vector: which row (node) does each edge belong to?
    # row_ptr is cumulative count per row
    # Number of neighbors per row:
    row_lengths <- diff(row_ptr)  # length n_cells
    
    # Group index for each edge
    grp <- rep.int(seq_len(n_cells), times = row_lengths)
    
    # Remove NAs
    valid <- !is.na(nbr_vals)
    nbr_vals_v <- nbr_vals[valid]
    grp_v      <- grp[valid]
    
    if (length(nbr_vals_v) > 0) {
      # Use data.table for fast grouped max/min
      dt_tmp <- data.table(g = grp_v, v = nbr_vals_v)
      agg <- dt_tmp[, .(mx = max(v), mn = min(v)), keyby = g]
      
      neighbor_max[agg$g, yr] <- agg$mx
      neighbor_min[agg$g, yr] <- agg$mn
    }
  }
  
  # --- Extract results back to the original row order of cell_data ---
  idx <- cbind(node_idx, year_col)
  list(
    nb_max  = neighbor_max[idx],
    nb_min  = neighbor_min[idx],
    nb_mean = neighbor_mean[idx]
  )
}

# ---- 4. Outer loop: compute and attach neighbor features --------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  
  agg <- compute_neighbor_aggregates_matrix(
    val_vec  = cell_data[[var_name]],
    node_idx = cell_data$node_idx,
    year_col = cell_data$year_col,
    n_cells  = n_cells,
    n_years  = n_years,
    A        = A,
    A_csr    = A_csr
  )
  
  # Assign columns matching the original naming convention
  # (adjust prefixes/suffixes to match what the trained RF expects)
  set(cell_data, j = paste0(var_name, "_nb_max"),  value = agg$nb_max)
  set(cell_data, j = paste0(var_name, "_nb_min"),  value = agg$nb_min)
  set(cell_data, j = paste0(var_name, "_nb_mean"), value = agg$nb_mean)
}

elapsed <- (proc.time() - t0)[3]
cat(sprintf("Done in %.1f seconds.\n", elapsed))

# Clean up helper columns
cell_data[, c("node_idx", "year_col") := NULL]

# ---- 5. Predict with the pre-trained Random Forest (unchanged) --------------
# The trained model object (e.g., rf_model) is loaded from disk.
# Column names of the 110 predictor variables must match exactly.
# No retraining occurs.

# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized | Equivalence |
|-----------|----------|-----------|-------------|
| **mean** | `mean(vals[idx])` where idx are non-NA neighbor rows | `(A %*% X_zero) / (A %*% not_na)` â€” identical sum/count with NA exclusion | âœ… Exact (floating point associativity â‰¤ 1 ULP) |
| **max** | `max(vals[idx])` over non-NA | `data.table [, max(v), by=g]` over identical edge-resolved non-NA values | âœ… Exact |
| **min** | `min(vals[idx])` over non-NA | `data.table [, min(v), by=g]` over identical edge-resolved non-NA values | âœ… Exact |
| **NA handling** | Returns `c(NA,NA,NA)` when no non-NA neighbors | Count==0 â†’ NA for mean; missing group â†’ NA for max/min | âœ… Exact |

## Memory Profile (16 GB laptop)

| Object | Size (approx) |
|--------|---------------|
| Sparse A (dgCMatrix, 1.37M nnz) | ~22 MB |
| A_csr (RsparseMatrix) | ~22 MB |
| X matrix (344K Ã— 28, double) | ~77 MB |
| neighbor_sum, neighbor_count (same) | ~77 MB each |
| neighbor_max, neighbor_min (same) | ~77 MB each |
| cell_data (~6.46M Ã— 110 cols) | ~5.7 GB |
| **Peak total** | **~6.5 GB** âœ… |

## Expected Runtime

- Sparse adjacency build: ~2 seconds
- Per variable (sparse matmul + grouped max/min over 28 years): ~30â€“60 seconds
- **Total for 5 variables: ~3â€“5 minutes** (vs. 86+ hours original)
END_RESPONSE>>>
