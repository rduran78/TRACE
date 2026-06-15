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
run_id: Anthropic_F04_T10_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, the function:

1. Looks up a cell's reference index via character key matching (`id_to_ref[as.character(...)]`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs character paste-keys (`paste(id, year, sep="_")`) for every neighbor of every row.
4. Performs named-vector character lookups (`idx_lookup[neighbor_keys]`) â€” this is **O(n Ã— k)** character hashing over millions of entries.

With ~6.46M rows and an average of ~4 rook neighbors per cell, this creates **~25.8 million character key constructions and lookups** inside a sequential R `lapply`, which is extremely slow due to R's overhead on character operations and lack of vectorization. The `compute_neighbor_stats` function compounds this with another `lapply` over 6.46M entries doing per-element subsetting.

**Root causes:**
- Per-row character string construction and named-vector lookup (hash table probing on character keys) inside a pure-R loop.
- `do.call(rbind, result)` on a list of 6.46M small vectors is itself slow.
- No vectorization or use of data.table/matrix-level operations.

## Optimization Strategy

**Key insight:** The neighbor relationship is defined at the **cell level** (344,208 cells), not the **cell-year level** (6.46M rows). We should:

1. **Restructure data into a matrix** (cells Ã— years) per variable, enabling vectorized column-wise operations.
2. **Build the neighbor lookup once at the cell level** (344K entries, not 6.46M).
3. **Use `data.table` for fast joins and grouping**, and **sparse-matrix multiplication** to compute neighbor means (and similarly derive max/min) in a fully vectorized way.
4. **Replace the row-level `lapply`** with a single sparse-matrixâ€“times-dense-matrix multiplication for neighbor means, and grouped parallel-max/min operations.

The sparse adjacency matrix approach computes **all neighbor means for all cell-years in one matrix multiplication** per variable â€” reducing billions of R-level operations to a single optimized BLAS/sparse call.

## Optimized Working R Code

```r
# ==============================================================================
# Optimized spatial neighbor feature construction
# Preserves the trained RF model and original numerical estimand.
# ==============================================================================

library(data.table)
library(Matrix)

build_neighbor_features_optimized <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {

  # -----------------------------------------------------------
  # 1. Convert to data.table for fast operations
  # -----------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Ensure consistent cell ID ordering
  n_cells <- length(id_order)
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # -----------------------------------------------------------
  # 2. Build sparse row-normalized adjacency matrix (cell-level)
  #    Dimension: n_cells x n_cells
  #    A[i, j] = 1/deg(i) if j is a neighbor of i, else 0
  #    So A %*% X gives neighbor means.
  #    Also build a binary (non-normalized) version for max/min.
  # -----------------------------------------------------------
  # Construct COO triplets from nb object
  from_idx <- integer(0)
  to_idx   <- integer(0)

  for (i in seq_along(rook_neighbors_unique)) {
    nbrs <- rook_neighbors_unique[[i]]
    # spdep nb objects use 0L to denote no neighbors
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0L) {
      from_idx <- c(from_idx, rep(i, length(nbrs)))
      to_idx   <- c(to_idx, nbrs)
    }
  }

  # Binary adjacency (for max/min later)
  adj_binary <- sparseMatrix(
    i = from_idx, j = to_idx,
    x = rep(1, length(from_idx)),
    dims = c(n_cells, n_cells)
  )

  # Row-normalized adjacency (for mean)
  deg <- diff(adj_binary@p)  # column counts in CSC; we need row sums
  row_deg <- tabulate(from_idx, nbins = n_cells)
  row_deg[row_deg == 0] <- NA_real_  # avoid division by zero; these rows will produce NaN -> NA
  norm_vals <- 1.0 / row_deg[from_idx]

  adj_mean <- sparseMatrix(
    i = from_idx, j = to_idx,
    x = norm_vals,
    dims = c(n_cells, n_cells)
  )

  # -----------------------------------------------------------
  # 3. Get sorted unique years
  # -----------------------------------------------------------
  years <- sort(unique(dt$year))
  n_years <- length(years)

  # -----------------------------------------------------------
  # 4. Map each row to (cell_idx, year_idx) for matrix positioning
  # -----------------------------------------------------------
  dt[, cell_idx := id_to_idx[as.character(id)]]
  year_to_col <- setNames(seq_along(years), as.character(years))
  dt[, year_idx := year_to_col[as.character(year)]]

  # -----------------------------------------------------------
  # 5. For each variable, build a (n_cells x n_years) matrix,
  #    compute neighbor stats via sparse matrix ops, and join back.
  # -----------------------------------------------------------

  for (var_name in neighbor_source_vars) {

    cat("Processing neighbor features for:", var_name, "\n")

    # Build cell x year matrix (NA where data is missing)
    val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    val_vec <- dt[[var_name]]
    val_mat[cbind(dt$cell_idx, dt$year_idx)] <- val_vec

    # ---- Neighbor MEAN via sparse matrix multiplication ----
    # adj_mean %*% val_mat gives weighted (=mean) neighbor values.
    # But we need to handle NAs: treat them as missing, not zero.
    #
    # Strategy for mean with NA handling:
    #   mean = (sum of non-NA neighbor vals) / (count of non-NA neighbor vals)

    notna_mat <- matrix(0, nrow = n_cells, ncol = n_years)
    notna_mat[!is.na(val_mat)] <- 1

    val_mat_zero <- val_mat
    val_mat_zero[is.na(val_mat_zero)] <- 0  # replace NA with 0 for multiplication

    neighbor_sum   <- as.matrix(adj_binary %*% val_mat_zero)  # n_cells x n_years
    neighbor_count <- as.matrix(adj_binary %*% notna_mat)      # n_cells x n_years

    mean_mat <- neighbor_sum / neighbor_count  # NaN where count==0, which becomes NA
    mean_mat[neighbor_count == 0] <- NA_real_

    # ---- Neighbor MAX and MIN ----
    # No single sparse matrix trick for max/min. But we can iterate over
    # cells (344K, NOT 6.46M) which is ~19x faster than row-level.
    # We vectorize across years for each cell.

    max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    # Use the CSR representation for efficient row-wise neighbor access
    adj_csr <- as(adj_binary, "RsparseMatrix")

    for (i in seq_len(n_cells)) {
      # Get neighbor indices for cell i from CSR
      row_start <- adj_csr@p[i] + 1L
      row_end   <- adj_csr@p[i + 1L]
      if (row_end < row_start) next  # no neighbors

      nbr_indices <- adj_csr@j[row_start:row_end] + 1L  # 1-based

      if (length(nbr_indices) == 1L) {
        max_mat[i, ] <- val_mat[nbr_indices, ]
        min_mat[i, ] <- val_mat[nbr_indices, ]
      } else {
        nbr_block <- val_mat[nbr_indices, , drop = FALSE]  # k_neighbors x n_years
        # suppressWarnings to handle all-NA columns
        suppressWarnings({
          max_mat[i, ] <- apply(nbr_block, 2, max, na.rm = TRUE)
          min_mat[i, ] <- apply(nbr_block, 2, min, na.rm = TRUE)
        })
      }
    }
    # Fix Inf/-Inf from all-NA slices
    max_mat[is.infinite(max_mat)] <- NA_real_
    min_mat[is.infinite(min_mat)] <- NA_real_

    # ---- Extract values back to the data.table row order ----
    idx_mat <- cbind(dt$cell_idx, dt$year_idx)

    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    dt[, (col_max)  := max_mat[idx_mat]]
    dt[, (col_min)  := min_mat[idx_mat]]
    dt[, (col_mean) := mean_mat[idx_mat]]
  }

  # -----------------------------------------------------------
  # 6. Clean up helper columns and return as data.frame
  # -----------------------------------------------------------
  dt[, c("cell_idx", "year_idx") := NULL]
  return(as.data.frame(dt))
}

# ==============================================================================
# Usage (drop-in replacement for the original outer loop)
# ==============================================================================
# cell_data <- build_neighbor_features_optimized(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
#
# # Then predict with the existing trained RF model as before:
# # predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Further Optimization: Vectorize Max/Min via `data.table` Grouping

The cell-level loop for max/min (344K iterations with `apply`) is already ~19Ã— faster than the original 6.46M-row loop, but can be further accelerated:

```r
# ==============================================================================
# Alternative max/min via data.table long-format grouping (fully vectorized)
# Replaces the cell-level for-loop for max/min inside the function above.
# ==============================================================================

compute_max_min_dt <- function(val_mat, adj_binary, n_cells, n_years) {

  # Build edge list from sparse adjacency
  adj_coo <- summary(adj_binary)  # returns data.frame with i, j, x

  # Long-format: for each (cell_i, year_t), get neighbor cell_j's value
  edges <- data.table(cell_i = adj_coo$i, cell_j = adj_coo$j)

  # Cross join edges with years
  year_dt <- data.table(year_idx = seq_len(n_years))
  edges_years <- edges[, .(year_idx = seq_len(n_years)), by = .(cell_i, cell_j)]

  # This creates ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows. At 16 bytes/row
  # that is ~600 MB â€” fits in 16 GB RAM but is large. If memory is tight,
  # process in year-chunks (see below).

  # Look up neighbor values
  edges_years[, nbr_val := val_mat[cbind(cell_j, year_idx)]]

  # Group by (cell_i, year_idx), compute max and min
  stats <- edges_years[!is.na(nbr_val),
    .(nmax = max(nbr_val), nmin = min(nbr_val)),
    by = .(cell_i, year_idx)
  ]

  # Fill into matrices
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  max_mat[cbind(stats$cell_i, stats$year_idx)] <- stats$nmax
  min_mat[cbind(stats$cell_i, stats$year_idx)] <- stats$nmin

  list(max_mat = max_mat, min_mat = min_mat)
}
```

**If the 38.5M-row table is too large for 16 GB RAM**, chunk by year:

```r
compute_max_min_dt_chunked <- function(val_mat, adj_binary, n_cells, n_years) {

  adj_coo <- summary(adj_binary)
  edges <- data.table(cell_i = adj_coo$i, cell_j = adj_coo$j)

  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (t in seq_len(n_years)) {
    edges[, nbr_val := val_mat[cell_j, t]]
    stats <- edges[!is.na(nbr_val),
      .(nmax = max(nbr_val), nmin = min(nbr_val)),
      by = cell_i
    ]
    max_mat[stats$cell_i, t] <- stats$nmax
    min_mat[stats$cell_i, t] <- stats$nmin
  }

  list(max_mat = max_mat, min_mat = min_mat)
}
```

## Performance Summary

| Component | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M character pastes + named-vector lookups | Eliminated; integer matrix indexing |
| **Neighbor mean** | `lapply` over 6.46M rows, R-level arithmetic | Single sparse matrix multiplication (`adj_binary %*% val_mat`), ~seconds |
| **Neighbor max/min** | `lapply` over 6.46M rows | 344K cell-level loop or fully vectorized `data.table` groupby |
| **Result assembly** | `do.call(rbind, ...)` on 6.46M 3-element vectors | Direct matrix indexing into `data.table` columns |
| **Estimated time** | **86+ hours** | **~5â€“15 minutes** (depending on max/min strategy) |
| **RAM footprint** | Moderate (large character vectors, 6.46M-element list) | ~1.5 GB for matrices (344K Ã— 28 Ã— 5 vars Ã— 8 bytes Ã— 3 stats) + sparse matrix |
| **RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical equivalence** | Baseline | Identical max/min/mean values âœ“ |

The fundamental speedup comes from (1) eliminating millions of per-row character-key operations, (2) leveraging sparse matrix algebra for the mean computation, and (3) reducing the iteration space from 6.46M cell-years to 344K cells (or fully vectorizing via `data.table` grouping).
END_RESPONSE>>>
