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
run_id: Anthropic_F10_T07_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookups in R are O(n) hash probes per call, and doing this 6.46M times with string concatenation is extremely slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in an `lapply`**, extracting and aggregating neighbor values one node-year at a time. This is pure R-level looping with no vectorization.

3. **The topology is year-invariant but is being resolved per cell-year row.** Every cell has the same rook neighbors every year, so the 344,208-cell adjacency structure is needlessly replicated 28 times into a 6.46M-element lookup.

**Root cause:** The design entangles the spatial topology (344K nodes, ~1.37M edges) with the temporal panel (28 years), producing a 6.46M-node "graph" when the actual graph has only 344K nodes. All neighbor aggregation can be expressed as sparse matrixâ€“vector products, which are highly optimized in C via the `Matrix` package.

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from the `nb` object (344,208 Ã— 344,208, ~1.37M nonzeros). This is the graph topology.
2. **Reshape each variable into a 344,208 Ã— 28 matrix** (rows = cells, columns = years).
3. **Compute neighbor aggregates via sparse matrix operations:**
   - **Mean:** `A_row_normalized %*% X` (one sparse matrix multiply per variable).
   - **Max / Min:** Use a CSC/CSR walk or `data.table` grouped operations on the edge list â€” but the most efficient pure-R approach is to convert the sparse matrix to an edge list once and use `data.table` grouping.
4. **Flatten results back** into the long panel and bind columns.
5. **Predict** with the pre-trained Random Forest.

This reduces runtime from 86+ hours to **minutes** by eliminating all R-level per-row iteration and leveraging compiled sparse linear algebra and `data.table` grouped aggregation.

## Working R Code

```r
# =============================================================================
# Optimized spatialâ€neighbor feature engineering
# Preserves numerical equivalence with the original pipeline
# =============================================================================

library(Matrix)
library(data.table)
library(spdep)      # for nb2listw if needed

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars, rf_model = NULL) {

  # -----------------------------------------------------------
  # 0. Convert to data.table for speed; keep original row order
  # -----------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_order := .I]

  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)

  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(dt)))

  # -----------------------------------------------------------
  # 1. Build sparse adjacency matrix ONCE (344,208 x 344,208)
  #    from the nb object. This encodes the rook graph topology.
  # -----------------------------------------------------------
  cat("Building sparse adjacency matrix from nb object...\n")

  # Map cell IDs to integer indices 1..n_cells
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Build COO (coordinate) representation of adjacency
  from_list <- vector("list", n_cells)
  to_list   <- vector("list", n_cells)
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    if (length(nb_i) > 0 && !(length(nb_i) == 1 && nb_i[0] == 0L)) {
      # nb objects use 0 to indicate no neighbors
      nb_i <- nb_i[nb_i != 0L]
      if (length(nb_i) > 0) {
        from_list[[i]] <- rep.int(i, length(nb_i))
        to_list[[i]]   <- nb_i
      }
    }
  }
  from_vec <- unlist(from_list, use.names = FALSE)
  to_vec   <- unlist(to_list, use.names = FALSE)
  rm(from_list, to_list)

  n_edges <- length(from_vec)
  cat(sprintf("Directed edges in adjacency: %d\n", n_edges))

  # Sparse adjacency matrix (binary)
  A <- sparseMatrix(
    i    = from_vec,
    j    = to_vec,
    x    = rep.int(1, n_edges),
    dims = c(n_cells, n_cells)
  )

  # Row-degree for computing mean (number of neighbors per cell)
  deg <- rowSums(A)  # integer vector length n_cells

  # Row-normalized adjacency for mean computation
  # D^{-1} A where D = diag(deg); handle zero-degree nodes
  deg_inv <- ifelse(deg > 0, 1 / deg, 0)
  A_mean  <- Diagonal(x = deg_inv) %*% A   # still sparse

  # -----------------------------------------------------------
  # 2. Build edge-list data.table for max/min (grouped aggregation)
  # -----------------------------------------------------------
  edge_dt <- data.table(from = from_vec, to = to_vec)
  rm(from_vec, to_vec)
  gc()

  # -----------------------------------------------------------
  # 3. Create cell-index column in dt for fast matrix indexing
  #    Map each row's cell ID to its position in id_order
  # -----------------------------------------------------------
  dt[, cell_idx := id_to_idx[as.character(id)]]

  # Ensure data is sorted by (cell_idx, year) for matrix reshaping

  setkey(dt, cell_idx, year)

  # -----------------------------------------------------------
  # 4. For each variable, compute neighbor max, min, mean
  # -----------------------------------------------------------
  cat("Computing neighbor statistics...\n")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing variable: %s\n", var_name))

    # 4a. Pivot to matrix: rows = cells (1..n_cells), cols = years
    #     Some cells may be missing for some years; handle via NA
    X_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    year_to_col <- setNames(seq_along(years), as.character(years))

    # Vectorized fill
    row_indices <- dt$cell_idx
    col_indices <- year_to_col[as.character(dt$year)]
    X_mat[cbind(row_indices, col_indices)] <- dt[[var_name]]

    # ---------------------------------------------------------
    # 4b. MEAN via sparse matrix multiply: M = A_mean %*% X_mat
    #     This gives the mean of neighbor values for each cell-year.
    #     Cells with zero neighbors get 0; we'll fix to NA below.
    # ---------------------------------------------------------
    # Replace NA with 0 for matrix multiply, but track NA counts
    X_notna  <- ifelse(is.na(X_mat), 0, 1)
    X_filled <- ifelse(is.na(X_mat), 0, X_mat)

    # Number of non-NA neighbors per cell-year
    N_valid <- as.matrix(A %*% X_notna)   # n_cells x n_years
    # Sum of non-NA neighbor values
    S_vals  <- as.matrix(A %*% X_filled)  # n_cells x n_years

    # Mean = sum / count (NA where count == 0)
    mean_mat <- ifelse(N_valid > 0, S_vals / N_valid, NA_real_)

    rm(S_vals)

    # ---------------------------------------------------------
    # 4c. MAX and MIN via edge-list grouped aggregation
    #     For each year-column, look up neighbor values and aggregate.
    #     This is vectorized over all edges simultaneously.
    # ---------------------------------------------------------
    max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (yy in seq_len(n_years)) {
      # Get values for all "to" (neighbor) nodes in this year
      neighbor_vals <- X_mat[edge_dt$to, yy]

      # Build temporary DT with from-node and neighbor value
      tmp <- data.table(
        from = edge_dt$from,
        val  = neighbor_vals
      )
      # Remove edges where neighbor value is NA
      tmp <- tmp[!is.na(val)]

      if (nrow(tmp) > 0) {
        agg <- tmp[, .(mx = max(val), mn = min(val)), by = from]
        max_mat[agg$from, yy] <- agg$mx
        min_mat[agg$from, yy] <- agg$mn
      }
    }

    rm(X_mat, X_filled, X_notna, N_valid)

    # ---------------------------------------------------------
    # 4d. Flatten matrices back to long panel and attach to dt
    # ---------------------------------------------------------
    max_col_name  <- paste0("neighbor_max_", var_name)
    min_col_name  <- paste0("neighbor_min_", var_name)
    mean_col_name <- paste0("neighbor_mean_", var_name)

    dt[, (max_col_name)  := max_mat[cbind(cell_idx, year_to_col[as.character(year)])]]
    dt[, (min_col_name)  := min_mat[cbind(cell_idx, year_to_col[as.character(year)])]]
    dt[, (mean_col_name) := mean_mat[cbind(cell_idx, year_to_col[as.character(year)])]]

    rm(max_mat, min_mat, mean_mat)
    gc()
  }

  # -----------------------------------------------------------
  # 5. Restore original row order and clean up
  # -----------------------------------------------------------
  setorder(dt, .row_order)
  dt[, c("cell_idx", ".row_order") := NULL]

  cat("Neighbor feature computation complete.\n")

  # -----------------------------------------------------------
  # 6. Predict with pre-trained Random Forest (if provided)
  # -----------------------------------------------------------
  if (!is.null(rf_model)) {
    cat("Generating predictions with pre-trained Random Forest...\n")
    dt[, prediction := predict(rf_model, newdata = dt)]
  }

  return(as.data.frame(dt))
}

# =============================================================================
# Usage
# =============================================================================
# result <- optimize_neighbor_features(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   neighbor_source_vars   = c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
#   rf_model               = trained_rf_model
# )
```

## Why This Is Numerically Equivalent

| Original | Optimized | Equivalence |
|---|---|---|
| `max(vals[neighbor_indices])` | `data.table` grouped `max` over edge list per year | Identical: same set of non-NA neighbor values, same `max` function |
| `min(vals[neighbor_indices])` | `data.table` grouped `min` over edge list per year | Identical |
| `mean(vals[neighbor_indices])` | `sum(non-NA neighbor vals) / count(non-NA neighbor vals)` via sparse matrix multiply | Identical: `A %*% X_filled / A %*% X_notna` equals `mean` of non-NA neighbors |
| NA handling: skip NA neighbors, return NA if zero valid neighbors | Replicated exactly via `N_valid > 0` guard and `!is.na(val)` filter | Identical |

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| Topology resolution | 6.46M string-paste + named lookups | One sparse matrix build (344K Ã— 344K, 1.37M nnz) |
| Mean computation | 6.46M R-level `lapply` iterations | 5 sparse matrix multiplies (~0.5s each) |
| Max/Min computation | 6.46M R-level loops | 5 vars Ã— 28 years = 140 `data.table` grouped aggregations on ~1.37M-row edge list |
| Memory | ~6.46M-element list of integer vectors | ~50 MB sparse matrix + ~77 MB per variable matrix (344K Ã— 28) |
| **Estimated time** | **86+ hours** | **~5â€“15 minutes** |

The key insight: **the graph topology is time-invariant**. By separating the spatial adjacency (sparse matrix, built once) from the temporal panel (matrix columns), we convert millions of R-level iterations into a handful of sparse linear algebra operations and vectorized `data.table` aggregations â€” all executed in compiled C/C++ code.
END_RESPONSE>>>
