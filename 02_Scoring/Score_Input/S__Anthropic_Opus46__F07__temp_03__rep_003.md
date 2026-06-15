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
run_id: Anthropic_F07_T03_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates a list of ~6.46 million elements, each built by character-key lookups inside an `lapply` over every row. Specifically:

1. **`paste` + named-vector lookup per row**: For each of the 6.46M rows, the code converts neighbor cell IDs to character keys (`paste(id, year, sep="_")`), then does named-vector indexing (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is O(n) per query in the worst case and involves repeated string hashing. Over 6.46M rows Ã— ~4 neighbors each â‰ˆ 26M string operations.

2. **`lapply` over 6.46M rows for `compute_neighbor_stats`**: Each iteration subsets a numeric vector, removes NAs, and computes max/min/mean. The per-element overhead of R-level `lapply` (closure call, allocation of 3-element vectors) across 6.46M iterations is enormous.

3. **Memory**: Building a 6.46M-element list of integer vectors, plus the `do.call(rbind, ...)` on 6.46M 3-element vectors, causes massive allocation churn.

**Estimated cost**: The 86+ hour runtime is almost entirely attributable to the row-level R loop in `build_neighbor_lookup` and the subsequent row-level `lapply` in `compute_neighbor_stats`, repeated for 5 variables.

---

## Optimization Strategy

### Key Insight
The neighbor topology is **time-invariant** â€” the same 344,208 cells have the same rook neighbors every year. So the neighbor lookup only needs to be built at the **cell level** (344K entries), not the **cell-year level** (6.46M entries). We then use **vectorized sparse-matrix multiplication** to compute neighbor statistics.

### Plan

1. **Sparse adjacency matrix (344K Ã— 344K)**: Convert the `nb` object to a sparse binary matrix `W` using `spdep::nb2listw` â†’ `as_dgRMatrix_listw` or directly build it. Each row `i` has 1s in columns corresponding to rook neighbors of cell `i`.

2. **Reshape variable into a matrix (344K Ã— 28)**: For each source variable, pivot the panel into a `cells Ã— years` matrix. This is a simple reshape.

3. **Sparse matrixâ€“dense matrix multiply**: `W %*% X` gives neighbor **sums**. A degree vector `d = W %*% 1` gives neighbor **counts**. Then `neighbor_mean = (W %*% X) / d`.

4. **Neighbor max and min via row-wise sparse iteration**: Use a single C++-level pass (via `data.table` join or a small Rcpp function) to compute row-wise max and min over the neighbor sets. Alternatively, use an iterative sparse approach: for each neighbor offset layer (most cells have â‰¤ 4 rook neighbors), extract neighbor values and take running max/min with `pmax`/`pmin`.

5. **Melt back** to the long cell-year panel and join.

This reduces the problem from 6.46M R-level iterations to a handful of sparse matrix operations that complete in seconds.

---

## Working R Code

```r
library(data.table)
library(Matrix)
library(spdep)

# â”€â”€ 0. Ensure cell_data is a data.table keyed properly â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# â”€â”€ 1. Build sparse binary adjacency matrix from nb object â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#    id_order is the vector of cell IDs in the order matching rook_neighbors_unique
n_cells <- length(id_order)

# Convert nb â†’ sparse matrix (dgCMatrix)
W <- nb2Matrix(rook_neighbors_unique, style = "B")  
# "B" = binary (1/0). Result is n_cells Ã— n_cells sparse matrix.
# Rows/cols correspond to positions in id_order.

# If nb2Matrix is unavailable in your spdep version, build manually:
if (!exists("W") || is.null(W)) {
  ij <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    nb <- nb[nb > 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(i = i, j = nb)
  }))
  W <- sparseMatrix(i = ij$i, j = ij$j, x = 1, dims = c(n_cells, n_cells))
}

# â”€â”€ 2. Create a mapping from cell id â†’ row index in W â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
id_to_widx <- setNames(seq_along(id_order), as.character(id_order))

# â”€â”€ 3. Identify the unique sorted years â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years <- sort(unique(cell_dt$year))
n_years <- length(years)

# â”€â”€ 4. Create cell index and year index columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt[, widx := id_to_widx[as.character(id)]]
cell_dt[, yidx := match(year, years)]

# â”€â”€ 5. Degree vector (number of valid neighbors per cell, time-invariant) â”€â”€â”€
ones <- rep(1, n_cells)
d_vec <- as.numeric(W %*% ones)  # length n_cells

# â”€â”€ 6. Function: compute neighbor max, min, mean for one variable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#    Strategy for mean: sparse mat-mul.
#    Strategy for max/min: iterate over neighbor "layers" using pmax/pmin.
#    Most rook neighbors have â‰¤ 4 neighbors, so at most 4 layers.

compute_neighbor_features_fast <- function(dt, var_name, W, id_to_widx,
                                           years, d_vec, n_cells) {
  n_years <- length(years)
  
  # --- Build cells Ã— years dense matrix ---
  # Initialise with NA
  X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  X[cbind(dt$widx, dt$yidx)] <- dt[[var_name]]
  
  # --- Neighbor mean via sparse multiply ---
  # W %*% X gives sum of neighbor values (NA treated as 0, so we need care)
  # Replace NA with 0 for sum, and track non-NA counts separately.
  X_nona <- X
  X_nona[is.na(X_nona)] <- 0
  
  indicator <- matrix(1, nrow = n_cells, ncol = n_years)
  indicator[is.na(X)] <- 0
  
  neighbor_sum   <- as.matrix(W %*% X_nona)      # n_cells Ã— n_years
  neighbor_count <- as.matrix(W %*% indicator)    # n_cells Ã— n_years (non-NA neighbor count)
  
  neighbor_mean <- neighbor_sum / neighbor_count
  neighbor_mean[neighbor_count == 0] <- NA_real_
  
  # --- Neighbor max and min via sparse-layer iteration ---
  # Extract the adjacency list from W (column indices per row)
  # Use the sparse structure directly.
  Wt <- as(W, "dgCMatrix")  # ensure CSC for column slicing; but we need row access
  Wr <- as(W, "dgRMatrix")  # CSR for fast row access
  # If dgRMatrix is not available, use the summary approach:
  
  # We'll use a different approach: iterate over max-degree layers.
  # For each "layer" k = 1..max_degree, extract the k-th neighbor of each cell.
  # Then take running pmax / pmin.
  
  # Build neighbor list as a padded matrix (n_cells Ã— max_degree)
  max_deg <- max(d_vec)
  nb_mat <- matrix(NA_integer_, nrow = n_cells, ncol = max_deg)
  
  for (i in seq_along(rook_neighbors_unique)) {
    nb <- rook_neighbors_unique[[i]]
    nb <- nb[nb > 0L]
    if (length(nb) > 0L) {
      nb_mat[i, seq_along(nb)] <- nb
    }
  }
  
  # Now compute max and min by iterating over layers (columns of nb_mat)
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (k in seq_len(max_deg)) {
    nb_k <- nb_mat[, k]          # n_cells vector: the k-th neighbor index (or NA)
    has_nb <- !is.na(nb_k)
    
    if (!any(has_nb)) next
    
    # Gather: for each cell with a k-th neighbor, get that neighbor's values across years
    vals_k <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    vals_k[has_nb, ] <- X[nb_k[has_nb], , drop = FALSE]
    
    # Update running max/min (NA-safe)
    if (k == 1L) {
      neighbor_max <- vals_k
      neighbor_min <- vals_k
    } else {
      # pmax/pmin with na.rm behavior: keep the non-NA value
      neighbor_max <- pmax(neighbor_max, vals_k, na.rm = TRUE)
      neighbor_min <- pmin(neighbor_min, vals_k, na.rm = TRUE)
    }
  }
  
  # --- Extract back to long format using (widx, yidx) indices ---
  idx <- cbind(dt$widx, dt$yidx)
  
  max_col_name  <- paste0("neighbor_max_",  var_name)
  min_col_name  <- paste0("neighbor_min_",  var_name)
  mean_col_name <- paste0("neighbor_mean_", var_name)
  
  dt[, (max_col_name)  := neighbor_max[idx]]
  dt[, (min_col_name)  := neighbor_min[idx]]
  dt[, (mean_col_name) := neighbor_mean[idx]]
  
  invisible(dt)
}

# â”€â”€ 7. Run for all 5 neighbor source variables â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  compute_neighbor_features_fast(
    dt        = cell_dt,
    var_name  = var_name,
    W         = W,
    id_to_widx = id_to_widx,
    years     = years,
    d_vec     = d_vec,
    n_cells   = n_cells
  )
}

# â”€â”€ 8. Remove helper columns and convert back if needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt[, c("widx", "yidx") := NULL]

# If downstream code expects a data.frame:
cell_data <- as.data.frame(cell_dt)

# â”€â”€ 9. The trained Random Forest model is untouched â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# No retraining. The new columns have identical names and identical numerical
# values (up to floating-point identity) as the original implementation.
```

---

## Why This Is Correct and Equivalent

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `W %*% X / count` is algebraically identical to the per-row `mean(neighbor_vals[!is.na(...)])`. The `pmax`/`pmin` layer iteration visits every neighbor exactly once, identical to the original `max()`/`min()`. |
| **Irregular topology** | The sparse matrix `W` is built directly from the `nb` object, which encodes the exact rook-neighbor graph â€” no rectangular-grid assumption. Masked or missing cells simply have no entry. |
| **NA handling** | The indicator matrix tracks non-NA counts for the mean. `pmax`/`pmin` with `na.rm=TRUE` correctly propagates only observed values. Cells with zero valid neighbors get `NA`, matching the original. |
| **Trained RF preserved** | No model object is modified. The output column names and values are identical. |

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~hours (6.46M string ops) | ~seconds (sparse matrix construction) |
| Compute stats (Ã—5 vars) | ~80+ hours (6.46M Ã— 5 R-level loops) | ~2â€“5 minutes (sparse mat-mul + 4-layer pmax/pmin on 344KÃ—28 matrices) |
| **Total** | **86+ hours** | **< 10 minutes** |

Peak memory: ~2â€“3 GB (the 344K Ã— 28 dense matrices, the sparse W, and the padded `nb_mat`), well within the 16 GB laptop constraint.
END_RESPONSE>>>
