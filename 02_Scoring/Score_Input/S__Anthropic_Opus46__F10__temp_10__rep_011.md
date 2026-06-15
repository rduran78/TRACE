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
run_id: Anthropic_F10_T10_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via expensive string-pasting and named-vector lookups (`paste` + `setNames` + named indexing). Named vector lookups in R are O(n) hash probes on a 6.46M-element character vector â€” repeated 6.46M times. This alone is O(nÂ²)-like in practice.

2. **`compute_neighbor_stats` iterates over 6.46M list elements** with `lapply`, extracting subsets of a numeric vector by index, removing NAs, and computing max/min/mean one row at a time. The per-element R overhead (function call, subsetting, `is.na`, three aggregation calls) dominates â€” there is no vectorization.

3. **The neighbor lookup is monolithic across all years.** But the graph topology (which cell neighbors which cell) is *time-invariant*. The 344,208-cell rook adjacency is duplicated across 28 years in the lookup, inflating from ~1.37M edges to ~38.5M index pairs stored in nested lists of lists.

**Net effect:** ~86+ hours driven by R-level loop overhead on millions of small operations, plus massive memory pressure from redundant data structures on a 16 GB laptop.

---

## Optimization Strategy

### Key Insight: Separate Topology from Time

The rook adjacency graph is **static** â€” it does not change across years. We should:

1. **Build the sparse adjacency matrix once** (344,208 Ã— 344,208) from the `nb` object â€” a `dgCMatrix` with ~1.37M nonzero entries (~22 MB).
2. **Reshape each variable into a 344,208 Ã— 28 matrix** (cells Ã— years).
3. **Compute neighbor aggregates via sparse matrixâ€“dense matrix multiplication** and analogous sparse-max/sparse-min operations â€” fully vectorized, year-parallel.

### Specific Techniques

| Operation | Method |
|---|---|
| **Neighbor mean** | `A_norm %*% X` where `A_norm` is the row-normalized adjacency (each row sums to 1, or to 1/degree). This is a single sparse matrix Ã— dense matrix multiply â€” O(nnz Ã— 28). |
| **Neighbor sum & count** | `A %*% X` gives sum; row-degree gives count; mean = sum/count. Handles NA via a parallel mask matrix. |
| **Neighbor max / min** | Iterate over sparse rows using the CSC/CSR structure in compiled code. We use `data.table` grouped operations on the edge list for max and min, which is extremely fast. |

### Complexity Comparison

| | Original | Optimized |
|---|---|---|
| Neighbor lookup build | O(N_rows Ã— k) with string ops | O(nnz) integer sparse matrix construction |
| Mean (per variable) | O(N_rows Ã— k) R-level loops | O(nnz Ã— T) single sparse matmul |
| Max/Min (per variable) | O(N_rows Ã— k) R-level loops | O(nnz Ã— T) vectorized `data.table` grouped ops |
| Total time estimate | 86+ hours | **~2â€“10 minutes** |

### Numerical Equivalence

- Mean is computed as `sum_of_non_NA_neighbors / count_of_non_NA_neighbors` â€” identical to the original.
- Max and min are computed over exactly the same neighbor sets with NA exclusion â€” identical to the original.
- The trained Random Forest model is loaded and applied with `predict()` â€” never retrained.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Sparse graph topology Ã— dense attribute matrices
# Numerically equivalent to the original loop-based implementation
# =============================================================================

library(Matrix)
library(data.table)

# ---- 1. Build sparse adjacency matrix ONCE from the nb object ---------------

build_adjacency_matrix <- function(nb_obj, n) {
 # nb_obj: spdep nb object (list of integer vectors of neighbor indices)
 # n: number of spatial cells (length of nb_obj)
 # Returns: sparse dgCMatrix of dimension n x n (binary adjacency)

 from <- rep(seq_len(n), times = vapply(nb_obj, length, integer(1)))
 to   <- unlist(nb_obj, use.names = FALSE)

 # Remove any 0-length or out-of-range entries
 valid <- to >= 1L & to <= n
 from  <- from[valid]
 to    <- to[valid]

 sparseMatrix(i = from, j = to, x = 1, dims = c(n, n), repr = "C")
}

# ---- 2. Reshape panel data to cell Ã— year matrices --------------------------

reshape_to_matrix <- function(cell_dt, id_order, years, var_name) {
 # cell_dt:  data.table with columns: id, year, <var_name>
 # id_order: integer vector of cell IDs defining row order
 # years:    sorted integer vector of years defining column order
 # Returns:  numeric matrix [n_cells x n_years]

 n_cells <- length(id_order)
 n_years <- length(years)

 # Map cell id -> row index, year -> col index
 id_map   <- setNames(seq_along(id_order), as.character(id_order))
 year_map <- setNames(seq_along(years), as.character(years))

 row_idx <- id_map[as.character(cell_dt$id)]
 col_idx <- year_map[as.character(cell_dt$year)]

 mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
 mat[cbind(row_idx, col_idx)] <- cell_dt[[var_name]]
 mat
}

# ---- 3. Compute neighbor MEAN via sparse matrix multiplication ---------------

compute_neighbor_mean_sparse <- function(A, X) {
 # A: binary sparse adjacency matrix [n x n]
 # X: dense matrix [n x T] (may contain NAs)
 # Returns: dense matrix [n x T] of neighbor means (NA where no valid neighbors)

 # Mask: 1 where X is not NA, 0 where NA
 notNA <- matrix(1, nrow = nrow(X), ncol = ncol(X))
 notNA[is.na(X)] <- 0

 # Replace NAs with 0 for summation
 X0 <- X
 X0[is.na(X0)] <- 0

 # Neighbor sums and counts via sparse matmul
 neighbor_sum   <- A %*% X0       # [n x T]
 neighbor_count <- A %*% notNA    # [n x T]

 # Convert to dense
 neighbor_sum   <- as.matrix(neighbor_sum)
 neighbor_count <- as.matrix(neighbor_count)

 # Mean = sum / count; NA where count == 0
 result <- neighbor_sum / neighbor_count
 result[neighbor_count == 0] <- NA_real_
 result
}

# ---- 4. Compute neighbor MAX and MIN via edge-list + data.table --------------

compute_neighbor_max_min_sparse <- function(A, X) {
 # A: binary sparse adjacency matrix [n x n] (dgCMatrix)
 # X: dense matrix [n x T]
 # Returns: list(max = [n x T], min = [n x T])

 n_cells <- nrow(X)
 n_years <- ncol(X)

 # Extract edge list from sparse matrix (1-indexed)
 A_t <- as(A, "TsparseMatrix")  # triplet form
 from_idx <- A_t@i + 1L
 to_idx   <- A_t@j + 1L
 n_edges  <- length(from_idx)

 # Pre-allocate result matrices
 max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
 min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

 # Process one year-column at a time to control memory
 # (each year: edge-list table with ~1.37M rows â€” trivial for data.table)
 for (t in seq_len(n_years)) {
   neighbor_vals <- X[to_idx, t]

   # Build edge table for this year
   dt <- data.table(
     from = from_idx,
     val  = neighbor_vals
   )

   # Remove edges where neighbor value is NA
   dt <- dt[!is.na(val)]

   if (nrow(dt) == 0L) next

   # Grouped max and min
   agg <- dt[, .(vmax = max(val), vmin = min(val)), by = from]

   max_mat[agg$from, t] <- agg$vmax
   min_mat[agg$from, t] <- agg$vmin
 }

 list(max = max_mat, min = min_mat)
}

# ---- 5. Write results back to the panel data.table --------------------------

write_matrix_to_dt <- function(cell_dt, mat, id_order, years, col_name) {
 # mat: [n_cells x n_years] result matrix
 # Writes values back into cell_dt by matching id and year

 id_map   <- setNames(seq_along(id_order), as.character(id_order))
 year_map <- setNames(seq_along(years), as.character(years))

 row_idx <- id_map[as.character(cell_dt$id)]
 col_idx <- year_map[as.character(cell_dt$year)]

 cell_dt[[col_name]] <- mat[cbind(row_idx, col_idx)]
 invisible(cell_dt)
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================

run_optimized_pipeline <- function(cell_data,
                                   id_order,
                                   rook_neighbors_unique,
                                   rf_model,
                                   neighbor_source_vars = c("ntl", "ec",
                                                            "pop_density",
                                                            "def",
                                                            "usd_est_n2")) {

 cat("Converting to data.table...\n")
 cell_dt <- as.data.table(cell_data)
 setkey(cell_dt, id, year)

 n_cells <- length(id_order)
 years   <- sort(unique(cell_dt$year))
 n_years <- length(years)

 cat(sprintf("Grid: %d cells x %d years = %d rows\n",
             n_cells, n_years, nrow(cell_dt)))

 # ---- Step 1: Build sparse adjacency matrix (once) -------------------------
 cat("Building sparse adjacency matrix...\n")
 A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
 cat(sprintf("Adjacency: %d nonzeros (directed edges)\n", nnzero(A)))

 # ---- Step 2: For each source variable, compute neighbor features ----------
 for (var_name in neighbor_source_vars) {
   cat(sprintf("Processing variable: %s\n", var_name))

   # Reshape to matrix
   X <- reshape_to_matrix(cell_dt, id_order, years, var_name)

   # Neighbor mean (sparse matmul)
   cat("  Computing neighbor mean (sparse matmul)...\n")
   mean_mat <- compute_neighbor_mean_sparse(A, X)

   # Neighbor max and min (edge-list + data.table)
   cat("  Computing neighbor max/min (data.table grouped ops)...\n")
   maxmin <- compute_neighbor_max_min_sparse(A, X)

   # Write back to data.table
   max_col  <- paste0(var_name, "_neighbor_max")
   min_col  <- paste0(var_name, "_neighbor_min")
   mean_col <- paste0(var_name, "_neighbor_mean")

   write_matrix_to_dt(cell_dt, maxmin$max, id_order, years, max_col)
   write_matrix_to_dt(cell_dt, maxmin$min, id_order, years, min_col)
   write_matrix_to_dt(cell_dt, mean_mat,   id_order, years, mean_col)

   # Free memory
   rm(X, mean_mat, maxmin)
   gc(verbose = FALSE)

   cat(sprintf("  Done: added %s, %s, %s\n", max_col, min_col, mean_col))
 }

 # ---- Step 3: Apply pre-trained Random Forest (no retraining) --------------
 cat("Applying pre-trained Random Forest model...\n")

 # Convert back to data.frame for predict() compatibility
 cell_data_out <- as.data.frame(cell_dt)

 predictions <- predict(rf_model, newdata = cell_data_out)
 cell_data_out$rf_prediction <- predictions

 cat("Pipeline complete.\n")
 cell_data_out
}

# =============================================================================
# USAGE EXAMPLE
# =============================================================================
#
# # Load pre-computed objects
# load("cell_data.RData")            # cell_data data.frame
# load("id_order.RData")             # id_order integer vector
# load("rook_neighbors.RData")       # rook_neighbors_unique (nb object)
# load("rf_model.RData")             # rf_model (trained randomForest)
#
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = rf_model
# )
```

---

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized | Equivalence |
|---|---|---|---|
| **Mean** | `mean(vals[!is.na(vals)])` = sum/count | `(A %*% X0) / (A %*% notNA)` where X0 has NAs replaced with 0 | Identical: same sum, same count per node-year |
| **Max** | `max(vals[!is.na(vals)])` | `data.table [!is.na(val), max(val), by=from]` | Identical: same neighbor set, same `max()` call |
| **Min** | `min(vals[!is.na(vals)])` | `data.table [!is.na(val), min(val), by=from]` | Identical: same neighbor set, same `min()` call |
| **NA rule** | Returns `c(NA,NA,NA)` when zero valid neighbors | Matrix cells default to `NA`; `count==0` â†’ `NA`; no `agg$from` entry â†’ row stays `NA` | Identical |

The Random Forest model object is passed through unchanged and only called with `predict()` â€” no retraining.

---

## Performance Estimates

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~hours (string paste on 6.46M rows) | <1 sec (sparse matrix from integer vectors) |
| Neighbor mean (Ã—5 vars) | ~hours (6.46M lapply iterations Ã—5) | ~10 sec (5 sparse matmuls, each 1.37M nnz Ã— 28 cols) |
| Neighbor max/min (Ã—5 vars) | ~hours | ~2 min (5 vars Ã— 28 years Ã— 1.37M edge data.table groupby) |
| RF predict | same | same |
| **Total** | **86+ hours** | **~2â€“10 minutes** |

Memory peak: ~1.5 GB (one 344KÃ—28 dense matrix â‰ˆ 77 MB; sparse matrix â‰ˆ 22 MB; edge data.table per year â‰ˆ 22 MB; original data.frame â‰ˆ ~5 GB with 110 columns).
END_RESPONSE>>>
