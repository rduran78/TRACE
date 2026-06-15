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
run_id: Anthropic_F07_T07_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over ~6.46 million rows with an `lapply` loop, performing per-row character pasting, hash lookups, and NA filtering. This creates ~6.46 million small R vectors and does millions of string allocations. `compute_neighbor_stats` then loops over those 6.46 million entries again per variable. With 5 variables, you get ~32.3 million inner-loop iterations total. The 86+ hour estimate comes from:

1. **O(N) string key construction per row** inside `build_neighbor_lookup` â€” `paste()` and named-vector lookup on 6.46M rows is extremely slow in a serial `lapply`.
2. **Millions of tiny allocations** â€” each row produces a small integer vector; R's memory allocator and garbage collector are hammered.
3. **Redundant work** â€” the neighbor *topology* is year-invariant (same neighbors every year), but the lookup is rebuilt as if it were year-specific. The year dimension is only needed to align values, not to discover neighbors.
4. **`compute_neighbor_stats` is pure R** â€” looping over 6.46M entries calling `max/min/mean` on small vectors has massive interpreter overhead.

## Optimization Strategy

**Key insight:** The neighbor graph is *time-invariant*. Separate the spatial topology from the temporal alignment.

1. **Build a sparse adjacency matrix once** (344,208 Ã— 344,208) from the `nb` object â€” this is standard and instant via `spdep::nb2listw` â†’ `listw2mat` or directly via `Matrix::sparseMatrix`.
2. **Reshape each variable into a matrix** of dimension (344,208 cells Ã— 28 years). Columns are years, rows are cells in `id_order`.
3. **Compute neighbor stats via sparse matrix multiplication and row-wise operations** â€” for the mean, a sparse-matrixâ€“dense-matrix multiply gives neighbor sums; divide by neighbor counts. For max and min, use a grouped C-level operation (via `data.table` or a small Rcpp function over the sparse structure).
4. **Melt back** to the long panel and join.

This eliminates all per-row string work, reduces to vectorized linear-algebra or C-level grouped operations, and runs in **minutes, not days**.

## Working R Code

```r
library(data.table)
library(Matrix)
library(spdep)

# â”€â”€ 0. Ensure cell_data is a data.table keyed properly â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)

# id_order : character/integer vector of cell IDs in the same order as rook_neighbors_unique
# rook_neighbors_unique : an nb object (list of integer index vectors)

n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

# â”€â”€ 1. Build sparse binary adjacency matrix (time-invariant) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
nb_to_sparse <- function(nb_obj) {
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # remove 0-neighbor entries (spdep encodes no-neighbor as 0L)
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(length(nb_obj), length(nb_obj)))
}

W <- nb_to_sparse(rook_neighbors_unique)          # 344208 x 344208, very sparse
neighbor_counts <- rowSums(W)                      # number of rook neighbors per cell

# â”€â”€ 2. Map cell IDs to row indices in the adjacency matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
id_to_row <- setNames(seq_along(id_order), as.character(id_order))

# Add a matrix-row index and a year-column index to cell_data
cell_data[, row_idx := id_to_row[as.character(id)]]
cell_data[, yr_idx  := match(year, years)]

# â”€â”€ 3. Generic function: compute neighbor max, min, mean for one variable â”€â”€â”€
compute_neighbor_features <- function(dt, var_name, W, id_to_row, years,
                                      neighbor_counts, n_cells, n_years) {

  # 3a. Pivot variable into a dense matrix  (cells Ã— years)
  #     Missing cell-years stay NA â†’ filled with NA automatically
  val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  val_mat[cbind(dt$row_idx, dt$yr_idx)] <- dt[[var_name]]

  # 3b. Neighbor MEAN via sparse multiply â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  #     Replace NA with 0 for multiplication; track valid counts separately.
  not_na  <- !is.na(val_mat)                        # logical matrix
  val0    <- val_mat; val0[!not_na] <- 0             # NA â†’ 0

  neigh_sum   <- as.matrix(W %*% val0)              # sum of neighbor values
  neigh_count <- as.matrix(W %*% (not_na * 1.0))    # count of non-NA neighbors
  neigh_mean  <- neigh_sum / neigh_count             # element-wise
  neigh_mean[neigh_count == 0] <- NA_real_

  # 3c. Neighbor MAX and MIN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  #     We iterate over the sparse structure at the C-level via data.table.
  #     Extract (i, j) pairs from W once, then do grouped ops per year column
  #     in a vectorised way.

  W_coo  <- summary(W)  # data.frame with i, j, x columns
  from_v <- W_coo$i
  to_v   <- W_coo$j

  neigh_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neigh_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # Process one year-column at a time (28 iterations â€” trivial overhead)
  for (k in seq_len(n_years)) {
    col_vals <- val_mat[, k]                       # length n_cells
    nv       <- col_vals[to_v]                     # neighbor values along edges

    # data.table grouped max/min â€” very fast C-level grouping
    edge_dt <- data.table(from = from_v, nv = nv)
    edge_dt <- edge_dt[!is.na(nv)]
    if (nrow(edge_dt) == 0L) next

    stats <- edge_dt[, .(mx = max(nv), mn = min(nv)), by = from]
    neigh_max[stats$from, k] <- stats$mx
    neigh_min[stats$from, k] <- stats$mn
  }

  # 3d. Map back to long panel rows â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  idx <- cbind(dt$row_idx, dt$yr_idx)
  dt[, paste0("neighbor_max_",  var_name) := neigh_max[idx]]
  dt[, paste0("neighbor_min_",  var_name) := neigh_min[idx]]
  dt[, paste0("neighbor_mean_", var_name) := neigh_mean[idx]]

  invisible(dt)
}

# â”€â”€ 4. Outer loop over the 5 source variables â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_data <- compute_neighbor_features(
    dt              = cell_data,
    var_name        = var_name,
    W               = W,
    id_to_row       = id_to_row,
    years           = years,
    neighbor_counts = neighbor_counts,
    n_cells         = n_cells,
    n_years         = n_years
  )
}

# â”€â”€ 5. Clean up helper columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data[, c("row_idx", "yr_idx") := NULL]

# cell_data now has 15 new columns (3 stats Ã— 5 vars), numerically identical
# to the original implementation. The trained Random Forest is untouched.
```

## Why This Preserves the Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor graph** | `W` is built from the identical `rook_neighbors_unique` nb object; same directed pairs. |
| **Same NA handling** | Neighbor values that are NA are excluded before computing max/min/mean, exactly as the original code does. Cells with zero valid neighbors get NA. |
| **Same aggregation functions** | `max`, `min`, `mean` â€” no approximation, no sampling. The sparse-multiply path for the mean computes `sum / count`, which is algebraically identical. |
| **Trained RF untouched** | No model retraining; only the feature-engineering step is optimized. |

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build lookup | ~hours (6.46M string ops) | ~seconds (sparse matrix from nb) |
| Compute stats (Ã—5 vars) | ~80+ hours | ~5â€“15 minutes total |
| **Total** | **86+ hours** | **< 20 minutes** |

The sparse matrix `W` consumes ~20 MB. Each dense `n_cells Ã— n_years` matrix is ~77 MB. Peak memory stays well within 16 GB.
END_RESPONSE>>>
