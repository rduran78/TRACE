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
run_id: Anthropic_F10_T03_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a ~6.46M-element character vector â€” this alone is catastrophically slow.

2. **`compute_neighbor_stats` iterates over ~6.46 million list elements in a serial `lapply`**, extracting subsets of a numeric vector per element. The overhead of 6.46M R function calls plus per-element `max`/`min`/`mean` is enormous.

3. **The neighbor lookup is year-aware but the graph topology is year-invariant.** The rook-neighbor structure is purely spatial (344,208 cells), yet the code rebuilds it at the cell-year level (6.46M rows), inflating the problem ~28Ã—.

**Root cause summary:** The code treats a *spatial* graph as a *spatiotemporal* graph, uses slow string-keyed lookups, and processes everything in serial R loops.

## Optimization Strategy

1. **Build the sparse adjacency structure once over 344,208 cells** as a `dgCMatrix` (sparse column-compressed matrix from the `Matrix` package). This is the canonical sparse graph representation.

2. **Reshape each variable into a 344,208 Ã— 28 dense matrix** (cells Ã— years). This enables vectorized column-wise (per-year) sparse matrixâ€“vector operations.

3. **Compute neighbor aggregates via sparse matrix algebra:**
   - `neighbor_sum = A %*% X` (sum of neighbor values)
   - `neighbor_count = A %*% (!is.na(X))` (count of non-NA neighbors)
   - `neighbor_mean = neighbor_sum / neighbor_count`
   - For `max` and `min`, use a grouped operation over the sparse structure (no native sparse-matrix operation exists for element-wise max/min, so we use the CSC structure directly in a tight vectorized loop or `data.table` grouped operation).

4. **Avoid any `lapply` over millions of elements.** Everything is either sparse matrix multiplication or `data.table` grouped aggregation.

5. **Memory:** A 344,208 Ã— 28 dense matrix of doubles is ~77 MB. The sparse adjacency matrix with ~1.37M entries is ~16 MB. Total working memory for all variables is well under 2 GB. Fits easily in 16 GB.

6. **Expected speedup:** From 86+ hours to **minutes**. Sparse matrixâ€“dense matrix multiplication for 344K Ã— 28 with ~1.37M nonzeros is trivial. The max/min grouped operations via `data.table` over ~1.37M edges Ã— 28 years are also fast.

## Working R Code

```r
# ==============================================================================
# Optimized neighbor-aggregation pipeline
# Preserves numerical equivalence with the original implementation.
# Requires: Matrix, data.table, ranger (or randomForest â€” whichever was used)
# ==============================================================================

library(Matrix)
library(data.table)

# --------------------------------------------------------------------------
# 1. Build sparse adjacency matrix ONCE (344,208 x 344,208)
#    rook_neighbors_unique: spdep nb object (list of integer neighbor indices)
#    id_order: vector of cell IDs in the order matching the nb object
# --------------------------------------------------------------------------

build_adjacency_matrix <- function(nb_obj, n = length(nb_obj)) {
  # nb_obj[[i]] contains integer indices of neighbors of node i
  # Build COO triplets
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)

  # Remove 0-neighbor placeholders (spdep uses integer(0) or 0L)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]

  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n), repr = "C")
}

A <- build_adjacency_matrix(rook_neighbors_unique)
n_cells <- length(id_order)

# --------------------------------------------------------------------------
# 2. Create a mapping from cell id to spatial index (row in A)
# --------------------------------------------------------------------------

id_to_sidx <- setNames(seq_along(id_order), as.character(id_order))

# --------------------------------------------------------------------------
# 3. Convert cell_data to data.table for fast manipulation
# --------------------------------------------------------------------------

dt <- as.data.table(cell_data)

# Ensure consistent year ordering
years <- sort(unique(dt$year))
n_years <- length(years)

# Spatial index for every row
dt[, sidx := id_to_sidx[as.character(id)]]

# Sort by sidx and year so we can build matrices efficiently
setkey(dt, sidx, year)

# --------------------------------------------------------------------------
# 4. Function: reshape a variable into cells x years matrix
#    Rows = spatial index (1..n_cells), Cols = year index (1..n_years)
# --------------------------------------------------------------------------

year_to_yidx <- setNames(seq_along(years), as.character(years))

reshape_to_matrix <- function(dt, var_name, n_cells, years, year_to_yidx) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = length(years))
  yidx <- year_to_yidx[as.character(dt$year)]
  mat[cbind(dt$sidx, yidx)] <- dt[[var_name]]
  mat
}

# --------------------------------------------------------------------------
# 5. Compute neighbor MEAN via sparse matrix multiplication
#    For a variable matrix X (n_cells x n_years):
#      neighbor_sum   = A %*% X
#      neighbor_count = A %*% (!is.na(X))   [treating TRUE as 1]
#      neighbor_mean  = neighbor_sum / neighbor_count
# --------------------------------------------------------------------------

compute_neighbor_mean_matrix <- function(A, X) {
  # Replace NA with 0 for summation; track non-NA
  X_nona <- X
  X_nona[is.na(X_nona)] <- 0
  not_na <- (!is.na(X)) * 1  # 1/0 numeric matrix

  neighbor_sum   <- as.matrix(A %*% X_nona)
  neighbor_count <- as.matrix(A %*% not_na)

  result <- neighbor_sum / neighbor_count  # yields NaN where count=0
  result[neighbor_count == 0] <- NA_real_
  result
}

# --------------------------------------------------------------------------
# 6. Compute neighbor MAX and MIN via edge-list + data.table grouped ops
#    This is the only part that can't use matrix algebra directly.
#    We extract the edge list once and reuse it for all variables/years.
# --------------------------------------------------------------------------

# Extract edge list from sparse matrix (COO format)
A_T <- summary(A)  # gives i, j, x triplets for nonzero entries
edges <- data.table(from = A_T$i, to = A_T$j)
setkey(edges, to)  # key on 'to' because we look up neighbor values by 'to' node

compute_neighbor_maxmin_matrix <- function(edges, X, n_cells, n_years) {
  # X is n_cells x n_years
  # For each edge (from, to), the neighbor value for node 'from' is X[to, year]
  # We need max and min of these values grouped by 'from' for each year.

  # Strategy: iterate over years (only 28) â€” each year is a fast grouped op
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (y in seq_len(n_years)) {
    vals_y <- X[, y]  # length n_cells
    # Attach neighbor values to edges
    edges[, val := vals_y[to]]
    # Remove NA neighbor values
    valid_edges <- edges[!is.na(val)]
    if (nrow(valid_edges) == 0L) next
    # Grouped max/min by 'from'
    agg <- valid_edges[, .(mx = max(val), mn = min(val)), by = from]
    max_mat[agg$from, y] <- agg$mx
    min_mat[agg$from, y] <- agg$mn
  }

  list(max = max_mat, min = min_mat)
}

# --------------------------------------------------------------------------
# 7. Main loop: compute all neighbor features and attach to data.table
# --------------------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Processing neighbor features for: ", var_name)

  # Reshape variable to matrix
  X <- reshape_to_matrix(dt, var_name, n_cells, years, year_to_yidx)

  # Compute mean (sparse matmul â€” very fast)
  mean_mat <- compute_neighbor_mean_matrix(A, X)

  # Compute max and min (grouped edge ops â€” fast, 28 iterations)
  maxmin <- compute_neighbor_maxmin_matrix(edges, X, n_cells, n_years)

  # Write results back to data.table using the same (sidx, yidx) mapping
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")

  idx <- cbind(dt$sidx, year_to_yidx[as.character(dt$year)])

  dt[, (col_max)  := maxmin$max[idx]]
  dt[, (col_min)  := maxmin$min[idx]]
  dt[, (col_mean) := mean_mat[idx]]

  # Free memory
  rm(X, mean_mat, maxmin)
}

# --------------------------------------------------------------------------
# 8. Convert back and predict with the pre-trained Random Forest
# --------------------------------------------------------------------------

# Drop the helper column
dt[, sidx := NULL]

# Convert back to data.frame if the RF model expects one
cell_data <- as.data.frame(dt)

# Predict using the existing trained model (DO NOT retrain)
# Adjust 'rf_model' to whatever your saved model object is named.
predictions <- predict(rf_model, newdata = cell_data)
```

## Complexity Comparison

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | O(N_rows Ã— avg_string_match) â‰ˆ 6.46M Ã— expensive char lookup | O(nnz) â‰ˆ 1.37M integer ops, **once** |
| Mean computation per variable | O(N_rows Ã— avg_degree) in R loop â‰ˆ 6.46M calls | Sparse matmul: O(nnz Ã— n_years) â‰ˆ 1.37M Ã— 28, **vectorized BLAS** |
| Max/Min per variable | Same R loop as mean | `data.table` grouped agg: 28 iterations Ã— ~1.37M edges, **C-level** |
| Total variables | Ã—5 | Ã—5 |
| **Estimated wall time** | **86+ hours** | **~2â€“10 minutes** |

## Why Numerical Equivalence Is Preserved

- The sparse matrix `A` encodes exactly the same directed rook-neighbor relationships as `rook_neighbors_unique`.
- `A %*% X_nona` computes the exact sum of non-NA neighbor values (same as `sum(neighbor_vals)` in the original).
- `A %*% not_na` computes the exact count (same as `length(neighbor_vals[!is.na(neighbor_vals)])`).
- Division gives the identical mean.
- `data.table` `max()` and `min()` grouped by `from` over the same edge set with the same NA-removal logic produce identical results.
- Cells with zero valid neighbors yield `NA` in all three statistics, matching the original `c(NA, NA, NA)` return.
END_RESPONSE>>>
