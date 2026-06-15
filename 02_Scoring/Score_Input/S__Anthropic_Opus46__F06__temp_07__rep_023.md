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
run_id: Anthropic_F06_T07_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) list construction
`build_neighbor_lookup` iterates over **~6.46 million rows** with `lapply`, and for each row it performs string-pasting, named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R using character keys are hash-table lookups, but doing ~6.46M of them â€” each involving multiple `paste` calls and subsetting â€” is extremely slow. The function builds a **row-level** neighbor lookup (one entry per cell-year), but the spatial neighbor structure is **invariant across years**. This means the same spatial lookup is redundantly recomputed 28 times per cell.

### Bottleneck 2: `compute_neighbor_stats` â€” Row-wise `lapply` over millions of rows
`compute_neighbor_stats` loops over 6.46M entries, each time subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. This is repeated for 5 variables = ~32.3M iterations of small R-level function calls. The overhead of interpreted R loops at this scale is enormous.

### Why 86+ hours?
The combination of ~6.46M Ã— (string operations + list indexing) in the lookup build, followed by ~32.3M small R function calls for stats, dominates. Neither step uses vectorized or matrix operations.

### Why raster focal/kernel operations don't directly apply
Focal operations assume a regular grid with a fixed rectangular kernel. Rook neighbors on an irregular or incomplete grid (with missing cells, coastlines, etc.) won't map cleanly to a fixed-size kernel. The neighbor structure is precomputed as an `nb` object precisely because it's irregular. We must preserve the exact `nb` topology to preserve the numerical estimand.

---

## Optimization Strategy

1. **Separate spatial structure from temporal replication.** The neighbor graph is spatial-only (344,208 cells). Build a sparse adjacency structure once at the cell level, then exploit the fact that year is just an offset.

2. **Use a sparse matrix multiplication approach.** Construct a sparse row-normalized (or raw) adjacency matrix `W` of dimension 344,208 Ã— 344,208. Then for each variable, reshape the data into a 344,208 Ã— 28 matrix. Neighbor means become `W %*% X` (a single sparse matrix multiply). For min and max, use row-wise sparse iteration â€” but in compiled code via the `Matrix` package or `data.table`.

3. **Use `data.table` for fast indexing and column assignment** instead of repeated data.frame copies.

4. **Compute min/max via sparse row iteration in C++ (`Rcpp`)** or via a grouped `data.table` approach using the edge list.

This reduces the problem from ~32M interpreted R iterations to a handful of sparse matrix multiplications and vectorized grouped operations, bringing runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)
library(Matrix)

# â”€â”€ 0. Prepare data.table and stable ordering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# Ensure a canonical ordering: cells in id_order, years 1992-2019
# id_order is the vector of cell IDs matching rook_neighbors_unique
years <- sort(unique(cell_data$year))
n_cells <- length(id_order)
n_years <- length(years)

# Create integer cell index and year index
cell_data[, cell_idx := match(id, id_order)]
cell_data[, year_idx := match(year, years)]

# Sort so that matrix layout is (cell, year) in column-major-friendly order
setkey(cell_data, cell_idx, year_idx)

# Verify completeness (balanced panel assumed from problem statement)
stopifnot(nrow(cell_data) == n_cells * n_years)

# â”€â”€ 1. Build sparse adjacency matrix from nb object â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#    rook_neighbors_unique is an nb object: a list of length n_cells
#    where each element is an integer vector of neighbor indices (into id_order)
#    nb objects use 0L to denote no neighbors.

# Build COO (coordinate) representation
from_idx <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
to_idx   <- unlist(rook_neighbors_unique)

# Remove "no neighbor" entries (coded as 0 in nb objects)
valid <- to_idx > 0L
from_idx <- from_idx[valid]
to_idx   <- to_idx[valid]

# Number of neighbors per cell (for computing means)
n_neighbors <- tabulate(from_idx, nbins = n_cells)
# For cells with 0 neighbors, avoid division by zero later
n_neighbors_safe <- pmax(n_neighbors, 1L)

# Sparse adjacency matrix (binary): W[i,j] = 1 if j is rook neighbor of i
W <- sparseMatrix(
  i = from_idx, j = to_idx,
  x = 1, dims = c(n_cells, n_cells),
  repr = "C"   # CSC format, efficient for %*%
)

# Row-normalized version for means
W_mean <- sparseMatrix(
  i = from_idx, j = to_idx,
  x = 1 / n_neighbors_safe[from_idx],
  dims = c(n_cells, n_cells),
  repr = "C"
)

# â”€â”€ 2. Build edge-list data.table for min/max â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#    This avoids Rcpp and uses data.table's grouped operations
edges_dt <- data.table(from = from_idx, to = to_idx)

# â”€â”€ 3. Reshape each variable into a matrix (n_cells x n_years) â”€â”€â”€â”€â”€â”€â”€â”€
#    Then compute neighbor stats via sparse matrix ops + grouped ops

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  message("Processing neighbor features for: ", var_name)

  # Extract values in (cell_idx, year_idx) order â€” already sorted by setkey
  vals_vec <- cell_data[[var_name]]

  # Reshape to matrix: rows = cells, cols = years
  X <- matrix(vals_vec, nrow = n_cells, ncol = n_years, byrow = FALSE)

  # â”€â”€ 3a. Neighbor MEAN via sparse matrix multiply â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  # W_mean %*% X gives (n_cells x n_years) matrix of neighbor means
  # Cells with no neighbors get 0 from the multiply; we fix to NA below
  mean_mat <- as.matrix(W_mean %*% X)

  # Handle NAs in source data:
  # Where source has NAs, the sparse multiply treated them as 0, which is wrong.
  # We need: for each cell i and year t, mean of non-NA neighbor values.
  # Strategy: compute sum of non-NA values and count of non-NA values separately.

  # Indicator of non-NA
  notNA <- matrix(as.numeric(!is.na(X)), nrow = n_cells, ncol = n_years)

  # Replace NA with 0 for summation
  X_zero <- X
  X_zero[is.na(X_zero)] <- 0

  sum_mat   <- as.matrix(W %*% X_zero)   # sum of non-NA neighbor values

  count_mat <- as.matrix(W %*% notNA)     # count of non-NA neighbor values

  mean_mat <- ifelse(count_mat > 0, sum_mat / count_mat, NA_real_)

  # Cells with no neighbors at all â†’ NA
  no_neighbors <- n_neighbors == 0L
  if (any(no_neighbors)) {
    mean_mat[no_neighbors, ] <- NA_real_
  }

  # â”€â”€ 3b. Neighbor MAX and MIN via edge-list grouped operations â”€â”€â”€â”€â”€
  # For each (from_cell, year), we need max and min of X[to_cell, year]
  # across all neighbors. We do this year by year to manage memory,
  # or we can do it fully vectorized with an expanded edge table.

  # Expand edges across years: each edge appears once per year
  # Total rows: length(from_idx) * n_years â‰ˆ 1.37M * 28 â‰ˆ 38.5M rows
  # Each row is ~24 bytes â†’ ~925 MB. Tight on 16GB but feasible.
  # Alternative: loop over years (28 iterations, fast each).

  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (t in seq_len(n_years)) {
    # Get neighbor values for this year
    neighbor_vals <- X[edges_dt$to, t]

    # Build temporary DT with from-cell and neighbor value
    tmp <- data.table(from = edges_dt$from, val = neighbor_vals)

    # Remove NA neighbor values
    tmp <- tmp[!is.na(val)]

    if (nrow(tmp) > 0) {
      agg <- tmp[, .(max_val = max(val), min_val = min(val)), by = from]
      max_mat[agg$from, t] <- agg$max_val
      min_mat[agg$from, t] <- agg$min_val
    }
  }

  # â”€â”€ 3c. Flatten back to vector (cell_idx, year_idx order) and assign â”€
  # cell_data is keyed by (cell_idx, year_idx), so column-major
  # flattening of the matrix matches row order.
  max_col_name  <- paste0("n_max_", var_name)
  min_col_name  <- paste0("n_min_", var_name)
  mean_col_name <- paste0("n_mean_", var_name)

  cell_data[, (max_col_name)  := as.vector(max_mat)]
  cell_data[, (min_col_name)  := as.vector(min_mat)]
  cell_data[, (mean_col_name) := as.vector(mean_mat)]
}

# â”€â”€ 4. Clean up helper columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data[, c("cell_idx", "year_idx") := NULL]

# â”€â”€ 5. Predict with the pre-trained Random Forest (unchanged) â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The trained model object (e.g., `rf_model`) is used as-is:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + hash lookups | Sparse matrix built once from `nb` object (~1 sec) |
| **Neighbor mean** | 6.46M Ã— 5 R-level loops | 5 sparse matrix multiplications (~seconds each) |
| **Neighbor min/max** | 6.46M Ã— 5 R-level loops | 28 Ã— 5 grouped `data.table` aggregations (~seconds each) |
| **Memory** | Millions of small R lists | Sparse matrices + dense matrices (~2-4 GB peak) |
| **Estimated runtime** | 86+ hours | **~5â€“15 minutes** |
| **Numerical result** | âœ… Preserved | âœ… Identical (same max/min/mean of non-NA rook neighbors) |
| **Trained RF model** | âœ… Unchanged | âœ… Unchanged â€” only `predict()` is called |

The key insight is that **spatial neighbors are time-invariant**, so the lookup should be built once at the cell level (344K entries) rather than at the cell-year level (6.46M entries). The sparse matrix formulation then vectorizes the mean computation into a single matrix multiply, and `data.table` grouped aggregation handles min/max efficiently in compiled C code rather than interpreted R loops.
END_RESPONSE>>>
