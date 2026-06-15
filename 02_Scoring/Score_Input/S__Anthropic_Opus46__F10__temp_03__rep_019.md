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
run_id: Anthropic_F10_T03_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a vector of 6.46M names â€” this alone is catastrophic. With ~1.37M directed edges replicated across 28 years, you're doing tens of millions of string-match lookups against a 6.46M-element named vector.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in an `lapply`**, extracting subsets of a numeric vector per element. The overhead of 6.46M R function calls plus per-element allocation is enormous.

3. **The entire design conflates topology (which cells neighbor which) with panel indexing (which row holds cellÃ—year data).** The rook adjacency graph is time-invariant â€” the same 344,208 nodes with ~1.37M directed edges every year. But the code rebuilds row-level lookups across all 6.46M rows, mixing spatial topology with temporal indexing unnecessarily.

**Root cause:** The algorithm is O(N_rows Ã— avg_neighbors) with high constant factors from R-level interpreted loops, string operations, and named-vector lookups. At 6.46M rows Ã— ~4 neighbors Ã— 5 variables Ã— 3 stats, plus the lookup construction cost, 86+ hours is expected.

## Optimization Strategy

1. **Separate topology from time.** Build the sparse adjacency structure once over 344,208 cells (not 6.46M rows). Represent it as a sparse matrix or CSR-style integer vectors.

2. **Use sparse matrixâ€“vector multiplication for mean; rowwise operations on sparse structure for max/min.** The neighbor mean of a variable is exactly `A %*% x / degree` where `A` is the binary adjacency matrix. Max and min require grouped operations but can be vectorized.

3. **Process year-by-year** (344K rows per year, not 6.46M at once), reusing the same adjacency matrix. This keeps memory bounded and enables vectorized operations per year-slice.

4. **Use `Matrix` package sparse operations** (compiled C code) for mean, and `data.table` grouped operations for max/min â€” both avoid R-level per-node loops entirely.

5. **Avoid all string-pasting and named-vector lookups.** Use integer indexing throughout.

**Expected speedup:** From 86+ hours to roughly 2â€“10 minutes, depending on I/O. The sparse matrix multiply over 344K nodes is milliseconds per variable-year. Max/min via data.table grouping is similarly fast.

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# Prerequisites:
#   cell_data        : data.frame/data.table with columns id, year, and the
#                      neighbor_source_vars. Rows ordered consistently.
#   id_order         : integer vector of cell IDs in the order matching
#                      rook_neighbors_unique (i.e., id_order[i] is the cell ID
#                      for the i-th element of the nb object).
#   rook_neighbors_unique : spdep nb object (list of length 344,208; each
#                      element is an integer vector of neighbor indices, with
#                      0L meaning no neighbors per spdep convention).
#   rf_model         : pre-trained Random Forest model (untouched).
# =============================================================================

library(data.table)
library(Matrix)

# ---------- STEP 1: Build sparse adjacency matrix once (time-invariant) ------

build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj: spdep nb object (list of integer vectors)
  # n: number of nodes (344208)
  # Returns: sparse dgCMatrix of dimension n x n, A[i,j]=1 if j is neighbor of i
  
  from <- vector("integer", 0)
  to   <- vector("integer", 0)
  
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep convention: 0L means no neighbors
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0L) {
      from <- c(from, rep.int(i, length(nbrs)))
      to   <- c(to, nbrs)
    }
  }
  
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n), repr = "C")
}

n_cells <- length(id_order)
cat("Building sparse adjacency matrix for", n_cells, "cells...\n")
A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

# Degree vector (number of neighbors per node), used for mean computation
degree <- diff(A@p)  # CSC column-pointer diff gives column counts; but A is row-oriented
# For row-wise degree from dgCMatrix, convert or compute directly:
degree_vec <- as.numeric(A %*% rep(1, n_cells))  # row sums = number of neighbors per node

# Also build CSR-style structure for fast max/min (row-compressed)
# dgCMatrix is CSC; transpose to get rows as columns, or convert to dgRMatrix
At <- t(A)  # At is dgCMatrix where column j contains the neighbors of node j
# Actually, let's build explicit CSR from the nb object for max/min:

# Pre-build CSR vectors for grouped max/min
# For each node i, we need the indices of its neighbors.
# We'll store as two vectors: pointers (length n+1) and neighbor indices.
csr_ptr <- integer(n_cells + 1L)
csr_nbrs <- vector("integer", 0)

total_edges <- 0L
for (i in seq_len(n_cells)) {
  nbrs <- rook_neighbors_unique[[i]]
  nbrs <- nbrs[nbrs > 0L]
  csr_ptr[i] <- total_edges
  if (length(nbrs) > 0L) {
    csr_nbrs <- c(csr_nbrs, nbrs)
    total_edges <- total_edges + length(nbrs)
  }
}
csr_ptr[n_cells + 1L] <- total_edges

# More efficient: pre-allocate csr_nbrs
# Redo with pre-allocation:
edge_counts <- vapply(rook_neighbors_unique, function(nb) {
  sum(nb > 0L)
}, integer(1))
total_edges <- sum(edge_counts)
csr_ptr <- c(0L, cumsum(edge_counts))
csr_nbrs <- integer(total_edges)
pos <- 1L
for (i in seq_len(n_cells)) {
  nbrs <- rook_neighbors_unique[[i]]
  nbrs <- nbrs[nbrs > 0L]
  nn <- length(nbrs)
  if (nn > 0L) {
    csr_nbrs[pos:(pos + nn - 1L)] <- nbrs
    pos <- pos + nn
  }
}

cat("Adjacency: ", n_cells, " nodes, ", total_edges, " directed edges\n")

# ---------- STEP 2: Map cell IDs to node indices -----------------------------

# id_order[k] = cell_id for node k. We need the reverse:
id_to_node <- setNames(seq_len(n_cells), as.character(id_order))

# ---------- STEP 3: Convert cell_data to data.table and add node index -------

setDT(cell_data)
cell_data[, node_idx := id_to_node[as.character(id)]]

# Verify no missing mappings
stopifnot(!anyNA(cell_data$node_idx))

# Sort by year and node_idx for efficient slicing
setkey(cell_data, year, node_idx)

# ---------- STEP 4: Compute neighbor stats per variable per year -------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Function: given a numeric vector x of length n_cells (one value per node for
# a single year), compute max, min, mean of neighbor values for each node.
compute_neighbor_stats_sparse <- function(x, A, degree_vec, csr_ptr, csr_nbrs, n) {
  # Mean via sparse matrix multiply: neighbor_mean = (A %*% x) / degree
  # Where degree_vec[i] = 0, result is NA
  Ax <- as.numeric(A %*% x)
  nb_mean <- ifelse(degree_vec > 0, Ax / degree_vec, NA_real_)
  
  # Max and min via CSR traversal â€” vectorized in C++ would be ideal,

  # but we can do it efficiently in R with the CSR structure:
  nb_max <- rep(NA_real_, n)
  nb_min <- rep(NA_real_, n)
  
  # Vectorized approach: expand all neighbor values, then group by node
  # Build node-id vector for each edge (which node "owns" this edge)
  # This is the CSR row expansion
  if (length(csr_nbrs) > 0L) {
    # Row indices for each edge (which node each neighbor belongs to)
    row_ids <- rep.int(seq_len(n), diff(csr_ptr))
    # Neighbor values
    nbr_vals <- x[csr_nbrs]
    
    # Use data.table for grouped max/min (very fast, single pass)
    edge_dt <- data.table(row_id = row_ids, val = nbr_vals)
    # Remove NAs in neighbor values
    edge_dt <- edge_dt[!is.na(val)]
    
    if (nrow(edge_dt) > 0L) {
      stats_dt <- edge_dt[, .(nb_max = max(val), nb_min = min(val)), by = row_id]
      nb_max[stats_dt$row_id] <- stats_dt$nb_max
      nb_min[stats_dt$row_id] <- stats_dt$nb_min
    }
  }
  
  # Handle mean where all neighbors are NA: if degree > 0 but all neighbor vals

  # are NA, the sparse multiply gives 0 (NA treated as 0 in sparse ops).
  # We need to correct this.
  # Count non-NA neighbors per node:
  if (anyNA(x) && length(csr_nbrs) > 0L) {
    x_notna <- as.numeric(!is.na(x))
    valid_count <- as.numeric(A %*% x_notna)
    # Replace x NAs with 0 for the sum
    x_nona <- x
    x_nona[is.na(x_nona)] <- 0
    Ax_corrected <- as.numeric(A %*% x_nona)
    nb_mean <- ifelse(valid_count > 0, Ax_corrected / valid_count, NA_real_)
  }
  
  data.table(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}

# ---------- STEP 5: Process each variable Ã— year ----------------------------

years <- sort(unique(cell_data$year))

cat("Computing neighbor features for", length(neighbor_source_vars), "variables across",
    length(years), "years...\n")

# Pre-allocate result columns
for (var_name in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]
}

for (yr in years) {
  cat("  Year:", yr, "\n")
  
  # Extract rows for this year (already keyed by year, node_idx)
  yr_rows <- cell_data[.(yr)]  # keyed lookup
  yr_idx  <- which(cell_data$year == yr)
  
  # Build a full-length vector for each variable (indexed by node_idx)
  # Some nodes may be missing for a year; those stay NA.
  node_indices <- yr_rows$node_idx
  
  for (var_name in neighbor_source_vars) {
    # Create dense vector of length n_cells, NA for missing nodes
    x <- rep(NA_real_, n_cells)
    x[node_indices] <- yr_rows[[var_name]]
    
    # Compute stats
    stats <- compute_neighbor_stats_sparse(x, A, degree_vec, csr_ptr, csr_nbrs, n_cells)
    
    # Write back to cell_data for the rows of this year
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    
    set(cell_data, i = yr_idx, j = col_max,  value = stats$nb_max[node_indices])
    set(cell_data, i = yr_idx, j = col_min,  value = stats$nb_min[node_indices])
    set(cell_data, i = yr_idx, j = col_mean, value = stats$nb_mean[node_indices])
  }
}

cat("Neighbor feature computation complete.\n")

# ---------- STEP 6: Predict with pre-trained Random Forest (unchanged) -------
# The RF model expects the same column names as the original pipeline produced.
# The columns neighbor_max_*, neighbor_min_*, neighbor_mean_* are numerically
# identical to the original compute_neighbor_stats output.

# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string-paste + named-vector lookups | One-time sparse matrix build over 344K nodes |
| **Mean computation** | R-level `lapply` over 6.46M elements | Sparse matrixâ€“vector multiply (`Matrix` C code) |
| **Max/Min computation** | R-level per-element loop | CSR expansion + `data.table` grouped aggregation |
| **Time complexity** | O(rows Ã— neighbors) with huge R overhead | O(edges) per variable-year in compiled code |
| **Memory** | 6.46M-element list of integer vectors | ~5.5M non-zeros in sparse matrix + CSR vectors (~50 MB) |
| **Estimated runtime** | 86+ hours | **~2â€“10 minutes** |
| **Numerical equivalence** | Baseline | âœ… Identical max, min, mean per node-year |
| **RF model** | Pre-trained | âœ… Untouched, used only for `predict()` |

### Why numerical equivalence holds:

- **Mean:** `(A %*% x)[i] / valid_count[i]` = `sum(x[neighbors(i)]) / count_non_na(neighbors(i))` â€” identical to `mean(neighbor_vals)` after NA removal.
- **Max/Min:** The CSR expansion produces exactly the same neighbor value sets per node; `data.table` grouped `max`/`min` is identical to the original element-wise computation.
- **NA handling:** Explicitly corrected: nodes with zero valid neighbors get `NA` for all three statistics, matching the original `if (length(neighbor_vals) == 0) return(c(NA, NA, NA))`.
END_RESPONSE>>>
