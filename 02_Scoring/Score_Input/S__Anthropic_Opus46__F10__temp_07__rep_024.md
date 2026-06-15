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
run_id: Anthropic_F10_T07_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). String operations on millions of keys are extremely slow in R. The lookup is O(n) per row with expensive character matching.

2. **`compute_neighbor_stats` iterates over 6.46 million list elements** in a serial `lapply`, extracting subsets of a numeric vector for each node-year. This is repeated 5 times (once per variable), totaling ~32.3 million list iterations.

3. **The neighbor topology is year-invariant but the lookup conflates topology and time.** Every cell has the same rook neighbors every year, yet the lookup rebuilds index references per cell-year row, inflating the problem by 28Ã—.

**Memory:** Storing 6.46M list entries (each a small integer vector) consumes substantial RAM and causes GC pressure. String keys (`paste(id, year)`) add further overhead.

**Net effect:** ~86+ hours on a 16 GB laptop.

## Optimization Strategy

1. **Separate topology from time.** Build a sparse adjacency structure once over 344,208 cells (not 6.46M cell-years). Rook neighbors don't change across years.

2. **Use sparse matrixâ€“dense matrix multiplication for aggregation.** Construct a row-normalized (or raw) sparse adjacency matrix `A` (344,208 Ã— 344,208) with ~1.37M non-zero entries. For each year, extract the variable column as a dense vector, then:
   - `A %*% x` gives neighbor sums; dividing by neighbor counts gives **mean**.
   - For **max** and **min**, use grouped operations via `data.table` with the edge list.

3. **Use `data.table` throughout** for fast grouped operations, avoiding `lapply` over millions of rows.

4. **Process year-by-year within the sparse matrix framework** to keep memory bounded, or process all years at once via a reshaped matrix if RAM allows.

5. **Preserve numerical equivalence:** The same neighbors, same values, same `max`/`min`/`mean` â€” just computed via vectorized/matrix operations.

## Optimized R Code

```r
library(data.table)
library(Matrix)

optimize_neighbor_pipeline <- function(cell_data_df, id_order, rook_neighbors_unique,
                                        neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                  "def", "usd_est_n2")) {

  # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

  # 1. Convert to data.table and set up cell/year indexing

  # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  cat("Step 1: Setting up data.table and indexing...\n")
  dt <- as.data.table(cell_data_df)

  # Ensure id_order is a plain vector
  id_order <- as.vector(id_order)
  n_cells  <- length(id_order)

  # Map cell IDs to integer indices 1..n_cells
  id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))

  # Add cell index to dt
  dt[, cell_idx := id_to_idx[as.character(id)]]

  # Sort by cell_idx and year for fast matrix reshaping
  setkey(dt, cell_idx, year)

  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))

  cat(sprintf("  %d cells, %d years, %d rows\n", n_cells, n_years, nrow(dt)))

  # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  # 2. Build sparse adjacency edge list from nb object (once)
  # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  cat("Step 2: Building edge list from nb object...\n")

  # rook_neighbors_unique is a list of length n_cells;

  # rook_neighbors_unique[[i]] contains integer indices (into id_order)
  # of neighbors of cell i.
  from_vec <- integer(0)
  to_vec   <- integer(0)

  # Pre-count edges for allocation
  edge_counts <- vapply(rook_neighbors_unique, function(nb) {
    nb <- nb[nb != 0L]  # spdep nb uses 0 for no-neighbor regions
    length(nb)
  }, integer(1))
  total_edges <- sum(edge_counts)

  cat(sprintf("  Total directed edges: %d\n", total_edges))

  from_vec <- integer(total_edges)
  to_vec   <- integer(total_edges)
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb <- rook_neighbors_unique[[i]]
    nb <- nb[nb != 0L]
    k  <- length(nb)
    if (k > 0L) {
      from_vec[pos:(pos + k - 1L)] <- i
      to_vec[pos:(pos + k - 1L)]   <- nb
      pos <- pos + k
    }
  }

  edge_dt <- data.table(from = from_vec, to = to_vec)
  rm(from_vec, to_vec)

  # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  # 3. Build sparse matrix for mean (neighbor sum / neighbor count)
  # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  cat("Step 3: Building sparse adjacency matrix...\n")

  # Sparse adjacency matrix: A[i,j] = 1 if j is a neighbor of i
  A <- sparseMatrix(
    i = edge_dt$from,
    j = edge_dt$to,
    x = rep(1, nrow(edge_dt)),
    dims = c(n_cells, n_cells)
  )

  # Neighbor counts per cell (for mean calculation)
  neighbor_counts <- as.vector(rowSums(A))  # integer-valued

  # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  # 4. For each variable, compute max/min/mean across neighbors
  #    Strategy:
  #      - Reshape variable into n_cells Ã— n_years matrix
  #      - MEAN: sparse mat-mul A %*% X / counts
  #      - MAX/MIN: grouped operations on edge_dt
  # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Step 4: Processing variable '%s'...\n", var_name))

    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    # 4a. Build n_cells Ã— n_years matrix of variable values
    #     dt is keyed by (cell_idx, year), so we can reshape efficiently
    vals <- dt[[var_name]]

    # Create matrix: rows = cell_idx, cols = year index
    # dt is sorted by (cell_idx, year), so if balanced panel:
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(dt$cell_idx, year_to_col[as.character(dt$year)])] <- vals

    # 4b. MEAN via sparse matrix multiplication
    #     neighbor_sum = A %*% X  (n_cells Ã— n_years)
    #     neighbor_mean = neighbor_sum / neighbor_counts
    cat("    Computing mean via sparse mat-mul...\n")
    neighbor_sum <- as.matrix(A %*% X)  # n_cells Ã— n_years dense

    # Handle NA propagation: we need mean of non-NA neighbors
    # Count non-NA neighbors per cell-year
    X_notna <- !is.na(X)
    X_zero  <- X
    X_zero[is.na(X_zero)] <- 0

    neighbor_sum_nona  <- as.matrix(A %*% X_zero)
    neighbor_count_nona <- as.matrix(A %*% (X_notna * 1.0))

    mean_mat <- neighbor_sum_nona / neighbor_count_nona
    mean_mat[neighbor_count_nona == 0] <- NA_real_

    # 4c. MAX and MIN via grouped edge-list operations
    cat("    Computing max/min via edge-list grouping...\n")

    max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    # Process year by year to keep memory bounded
    for (yi in seq_along(years)) {
      # Values of neighbors for this year: for each edge (from, to),
      # the neighbor value is X[to, yi]
      neighbor_vals <- X[edge_dt$to, yi]

      # Build temporary data.table for grouped max/min
      tmp <- data.table(from = edge_dt$from, val = neighbor_vals)
      tmp <- tmp[!is.na(val)]

      if (nrow(tmp) > 0L) {
        agg <- tmp[, .(mx = max(val), mn = min(val)), by = from]
        max_mat[agg$from, yi] <- agg$mx
        min_mat[agg$from, yi] <- agg$mn
      }
    }

    # 4d. Write results back to dt
    cat("    Writing results back...\n")
    dt[, (max_col)  := max_mat[cbind(cell_idx, year_to_col[as.character(year)])]]
    dt[, (min_col)  := min_mat[cbind(cell_idx, year_to_col[as.character(year)])]]
    dt[, (mean_col) := mean_mat[cbind(cell_idx, year_to_col[as.character(year)])]]

    # Free per-variable matrices
    rm(X, X_zero, X_notna, neighbor_sum, neighbor_sum_nona,
       neighbor_count_nona, mean_mat, max_mat, min_mat)
    gc()
  }

  # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  # 5. Clean up and return
  # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  dt[, cell_idx := NULL]
  cat("Done.\n")
  return(dt)
}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# USAGE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# # Load data
# load("cell_data.RData")        # -> cell_data (data.frame)
# load("rook_neighbors.RData")   # -> rook_neighbors_unique (nb list), id_order
# load("rf_model.RData")         # -> rf_model (pre-trained randomForest)
#
# # Run optimized pipeline
# cell_data_dt <- optimize_neighbor_pipeline(
#   cell_data_df          = cell_data,
#   id_order              = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   neighbor_source_vars  = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
#
# # Predict with the pre-trained RF (no retraining)
# # Ensure column order matches training data
# pred_vars <- names(rf_model$forest$xlevels)  # or stored feature names
# # If rf_model was trained via randomForest:
# # pred_vars can be extracted from rf_model$importance rownames
# pred_vars <- rownames(rf_model$importance)
#
# cell_data_dt[, prediction := predict(rf_model, newdata = .SD), .SDcols = pred_vars]
```

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + named-vector lookups | One sparse matrix build (344K Ã— 344K, 1.37M nnz) |
| **Mean computation** | 6.46M `lapply` iterations per variable | Single sparse `A %*% X` matrix multiply per variable |
| **Max/Min computation** | 6.46M `lapply` with subsetting per variable | 28 `data.table` grouped aggregations per variable (1.37M rows each) |
| **Total variable passes** | 5 Ã— 6.46M = 32.3M list iterations | 5 Ã— (1 matmul + 28 grouped aggs) |
| **Memory pattern** | ~6.46M list entries + string keys (~4â€“8 GB) | Sparse matrix (~30 MB) + dense 344KÃ—28 matrices (~77 MB each) |
| **Estimated time** | 86+ hours | **~5â€“15 minutes** |

### Why This Is Numerically Equivalent

- **Mean:** `A %*% X_zero / A %*% notNA(X)` computes exactly `sum(non-NA neighbor values) / count(non-NA neighbor values)` â€” identical to the original `mean(neighbor_vals)` after NA removal.
- **Max/Min:** The edge-list grouped aggregation computes `max`/`min` over exactly the same set of neighbor values (excluding NAs) as the original code.
- **Topology:** The sparse matrix and edge list are built from the same `rook_neighbors_unique` nb object with the same `id_order` mapping.
- **The pre-trained Random Forest is loaded and used for `predict()` only â€” never retrained.**
END_RESPONSE>>>
