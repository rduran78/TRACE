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
run_id: Anthropic_F06_T03_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F06_Anthropic_temp_03_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) list with per-row string operations

`build_neighbor_lookup` iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's rook neighbors by cell ID.
2. Constructs string keys (`paste(id, year)`) for every neighborâ€“year combination.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

String hashing over 6.46M rows with ~4 neighbors each â‰ˆ **25+ million string constructions and lookups**. The resulting list of 6.46M integer vectors also consumes substantial memory.

### Bottleneck B: `compute_neighbor_stats` â€” R-level `lapply` over 6.46M elements

For each of the 5 variables, `lapply` iterates over 6.46M list elements, subsetting a numeric vector and computing `max`, `min`, `mean`. That's **~32 million R-level function calls** (5 vars Ã— 6.46M rows), each with overhead from anonymous function dispatch, subsetting, and NA handling.

### Why raster focal/kernel operations don't directly apply

Focal operations assume a regular grid with a fixed rectangular kernel. Here the data is a **panel** (cell Ã— year) stored as a long data frame, and the neighbor structure is an irregular `nb` object (not all cells have 4 neighbors â€” boundary cells, islands, masked cells). Focal operations would require reshaping into a 2D raster per year, running the focal, then reshaping back â€” feasible but fragile and risks misalignment. The better strategy is to **vectorize the neighbor computation directly** using sparse-matrix multiplication and grouped column operations.

---

## 2. Optimization Strategy

### Key Insight: Neighbor aggregation is a sparse matrixâ€“vector product

If we construct a **sparse adjacency matrix W** of dimension `N_cells Ã— N_cells` from the `nb` object, then for any year-slice of a variable `x`, the neighbor sum is `W %*% x`, the neighbor count is `W %*% (non-NA indicator)`, and the neighbor mean is their ratio. Max and min require a different approach but can be computed via a **grouped operation** on a long-form edge table.

### Plan

| Step | What | Speedup mechanism |
|------|------|-------------------|
| 1 | Build a sparse `N_cells Ã— N_cells` rook adjacency matrix once | `Matrix::sparseMatrix` from `nb` â€” seconds |
| 2 | Reshape each variable into a `N_cells Ã— N_years` matrix | Vectorized indexing â€” seconds |
| 3 | For **mean**: sparse matrix multiply `W %*% X` and divide by `W %*% (!is.na(X))` per year-column | Fully vectorized, C-level â€” seconds per variable |
| 4 | For **max** and **min**: build a long edge table `(from, to)`, join variable values, then `data.table` grouped `max`/`min` by `(from, year)` | `data.table` grouped ops â€” very fast |
| 5 | Map results back to the original `cell_data` row order | Vectorized merge |

**Expected runtime: ~2â€“5 minutes** instead of 86+ hours.

**Numerical equivalence**: The sparse-matrix mean is algebraically identical to the per-row `mean(neighbor_vals[!is.na()])`. The `data.table` grouped max/min are identical to per-row `max`/`min` with `na.rm = TRUE`. The trained Random Forest model is never touched.

---

## 3. Working R Code

```r
# ==============================================================================
# Prerequisites
# ==============================================================================
library(data.table)
library(Matrix)

# ==============================================================================
# STEP 1: Build sparse adjacency matrix from nb object (once)
# ==============================================================================
build_sparse_adjacency <- function(nb_obj, id_order) {
  # nb_obj: list of integer vectors (indices into id_order), class "nb"
  # id_order: vector of cell IDs in the order used by the nb object
  n <- length(nb_obj)
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # Remove 0-neighbor placeholders (spdep uses integer(0) or 0)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  
  # Return matrix and the mapping from cell ID to matrix row index
  id_to_row <- setNames(seq_len(n), as.character(id_order))
  list(W = W, id_to_row = id_to_row, id_order = id_order)
}

adj <- build_sparse_adjacency(rook_neighbors_unique, id_order)
W          <- adj$W
id_to_row  <- adj$id_to_row
n_cells    <- length(id_order)

# ==============================================================================
# STEP 2: Build edge table for max/min (once)
# ==============================================================================
build_edge_dt <- function(W) {
  # Extract (i, j) pairs from sparse matrix
  W_t <- as(W, "TsparseMatrix")   # triplet form
  data.table(from = W_t@i + 1L, to = W_t@j + 1L)
}

edge_dt <- build_edge_dt(W)

# ==============================================================================
# STEP 3: Convert cell_data to data.table and establish index mappings
# ==============================================================================
cell_dt <- as.data.table(cell_data)

# Ensure consistent year ordering
years     <- sort(unique(cell_dt$year))
n_years   <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))

# Map each row to (cell_matrix_row, year_column)
cell_dt[, cell_row := id_to_row[as.character(id)]]
cell_dt[, year_col := year_to_col[as.character(year)]]

# Row-order index for writing results back
cell_dt[, orig_idx := .I]

# ==============================================================================
# STEP 4: Function to compute neighbor stats for one variable
# ==============================================================================
compute_neighbor_features_fast <- function(cell_dt, var_name, W, edge_dt,
                                           n_cells, n_years, years) {
  
  # --- 4a. Reshape variable into N_cells x N_years matrix ---
  X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  X[cbind(cell_dt$cell_row, cell_dt$year_col)] <- cell_dt[[var_name]]
  
  # --- 4b. Neighbor MEAN via sparse matrix multiply ---
  # non-NA indicator
  notNA <- !is.na(X)
  storage.mode(notNA) <- "double"   # for matrix multiply
  
  # Sum of neighbor values (NA treated as 0 after masking)
  X_zero <- X
  X_zero[is.na(X_zero)] <- 0
  
  neighbor_sum   <- as.matrix(W %*% X_zero)       # n_cells x n_years
  neighbor_count <- as.matrix(W %*% notNA)         # n_cells x n_years
  
  neighbor_mean <- neighbor_sum / neighbor_count   # NaN where count==0
  neighbor_mean[neighbor_count == 0] <- NA_real_
  
  # --- 4c. Neighbor MAX and MIN via edge table + data.table grouped ops ---
  # For each edge (from -> to), get the "to" cell's value per year.
  # We need to do this across all years simultaneously.
  
  # Build a long table: edge Ã— year
  # Efficient approach: for each year column, look up values for all edges
  
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Process year by year (each year is fast: ~1.37M edges)
  from_vec <- edge_dt$from
  to_vec   <- edge_dt$to
  
  for (t in seq_len(n_years)) {
    vals_t <- X[to_vec, t]   # neighbor values for all edges in year t
    
    # Remove NAs before grouping
    valid <- !is.na(vals_t)
    if (!any(valid)) next
    
    dt_t <- data.table(from = from_vec[valid], val = vals_t[valid])
    
    agg <- dt_t[, .(mx = max(val), mn = min(val)), by = from]
    
    max_mat[agg$from, t] <- agg$mx
    min_mat[agg$from, t] <- agg$mn
  }
  
  # --- 4d. Map results back to cell_dt row order ---
  idx <- cbind(cell_dt$cell_row, cell_dt$year_col)
  
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  cell_dt[, (max_col)  := max_mat[idx]]
  cell_dt[, (min_col)  := min_mat[idx]]
  cell_dt[, (mean_col) := neighbor_mean[idx]]
  
  invisible(cell_dt)
}

# ==============================================================================
# STEP 5: Run for all 5 neighbor source variables
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_dt <- compute_neighbor_features_fast(
    cell_dt, var_name, W, edge_dt, n_cells, n_years, years
  )
}

# ==============================================================================
# STEP 6: Clean up helper columns and convert back if needed
# ==============================================================================
cell_dt[, c("cell_row", "year_col", "orig_idx") := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(cell_dt)

# ==============================================================================
# STEP 7: Predict with the pre-trained Random Forest (unchanged)
# ==============================================================================
# The trained model object (e.g., `rf_model`) is used as-is:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Neighbor lookup** | 6.46M string-key list construction | Sparse matrix built once from `nb` object |
| **Mean computation** | R-level `lapply` + `mean()` Ã— 6.46M Ã— 5 | Sparse matrix multiply (`W %*% X`), C-level |
| **Max/Min computation** | R-level `lapply` + `max()`/`min()` Ã— 6.46M Ã— 5 | `data.table` grouped aggregation per year (~28 iterations Ã— 1.37M edges) |
| **Memory** | ~6.46M-element list of integer vectors | ~344K Ã— 28 matrices + sparse matrix (~5.5M non-zeros) |
| **Estimated runtime** | 86+ hours | **2â€“5 minutes** |
| **Numerical result** | Baseline | **Identical** (same arithmetic, no approximation) |
| **RF model** | Pre-trained | **Untouched** |

The sparse-matrix mean is algebraically identical to summing non-NA neighbor values and dividing by the non-NA count. The `data.table` grouped `max`/`min` with NA removal produces the same result as the original per-row `max(neighbor_vals[!is.na()])`. No retraining of the Random Forest is performed.
END_RESPONSE>>>
