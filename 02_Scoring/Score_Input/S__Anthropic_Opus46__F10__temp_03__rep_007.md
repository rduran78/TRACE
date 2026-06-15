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
run_id: Anthropic_F10_T03_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each produced by an `lapply` call that performs string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per probe when the vector is large (hash-based, but with overhead). With ~6.46M rows, this alone is extremely expensive.

2. **`compute_neighbor_stats` iterates over ~6.46M list elements** in R-level `lapply`, subsetting a numeric vector by index, removing NAs, and computing max/min/mean. This is called 5 times (once per variable), yielding ~32.3M R-level loop iterations total. Pure R loops over millions of elements are inherently slow.

3. **The neighbor topology is year-invariant but the lookup is rebuilt as if it's year-specific.** Each cell has the same rook neighbors every year. The code pastes `(neighbor_id, year)` keys to find row indices, repeating the same structural work 28 times per cell. This is a 28Ã— redundancy.

**Estimated cost breakdown:**
- `build_neighbor_lookup`: ~6.46M string operations + hash lookups â†’ hours
- `compute_neighbor_stats` Ã— 5 vars: ~32.3M R-level iterations â†’ hours
- Total: 86+ hours is consistent with this analysis

## Optimization Strategy

1. **Build a sparse adjacency structure once using integer arithmetic, not string keys.** The rook neighbor graph has 344,208 nodes and ~1.37M directed edges. Represent this as a CSR (Compressed Sparse Row) format: two integer vectors (`row_ptr` of length 344,209 and `col_idx` of length ~1.37M).

2. **Expand to the panel level using vectorized integer offsets.** Since every cell appears in every year (balanced panel), row `(i, t)` maps to index `(t-1)*N + i` (or `(i-1)*T + t` depending on sort order). Neighbor row indices for cell `i` in year `t` are simply the neighbor cells' indices shifted by the same year offset. This is pure integer vector arithmetic â€” no string operations.

3. **Vectorize the aggregation using sparse matrix multiplication.** Construct a sparse `(N*T) Ã— (N*T)` block-diagonal adjacency matrix (one block per year, all blocks identical topology). Then:
   - `neighbor_max` â†’ use the sparse structure with grouped row operations
   - `neighbor_min` â†’ same
   - `neighbor_mean` â†’ sparse matrix Ã— dense vector, then divide by neighbor count

   For **mean**, this is literally one sparse matrix-vector multiply per variable. For **max** and **min**, we use efficient C-level grouped operations via `data.table` or a custom sparse-row approach.

4. **Use `data.table` for the grouped operations** to get C-level speed, or use the `Matrix` package for sparse matrix-vector products.

5. **The Random Forest model is never retouched** â€” we only reproduce the exact same 15 neighbor-derived columns (5 vars Ã— 3 stats) with identical numerical values.

## Optimized R Code

```r
library(data.table)
library(Matrix)

optimize_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                  "def", "usd_est_n2")) {
  # â”€â”€â”€ 0. Convert to data.table for speed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  N <- length(id_order)                          # 344,208 cells
  years <- sort(unique(cell_data$year))
  T_years <- length(years)                       # 28
  NR <- N * T_years                              # ~6.46M expected rows

  # â”€â”€â”€ 1. Build CSR-style adjacency from the nb object (once) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  # rook_neighbors_unique is a list of length N where element [[i]] contains
  # integer indices (into id_order) of rook neighbors of cell i.
  # We build "from" and "to" vectors in terms of cell position (1..N).

  message("Building sparse adjacency structure...")
  edge_from <- integer(0)
  edge_to   <- integer(0)

  # Pre-allocate by counting total edges
  n_edges <- sum(vapply(rook_neighbors_unique, length, integer(1)))
  edge_from <- integer(n_edges)
  edge_to   <- integer(n_edges)

  pos <- 1L
  for (i in seq_len(N)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep::nb objects use 0L for no-neighbor islands; filter those
    nb_i <- nb_i[nb_i > 0L]
    len <- length(nb_i)
    if (len > 0L) {
      edge_from[pos:(pos + len - 1L)] <- i
      edge_to[pos:(pos + len - 1L)]   <- nb_i
      pos <- pos + len
    }
  }
  # Trim if any islands caused over-allocation
  edge_from <- edge_from[1:(pos - 1L)]
  edge_to   <- edge_to[1:(pos - 1L)]

  n_actual_edges <- length(edge_from)
  message(sprintf("  %d directed edges across %d cells", n_actual_edges, N))

  # â”€â”€â”€ 2. Map cell_data rows to (cell_position, year_position) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  message("Mapping rows to (cell, year) grid positions...")

  # Create cell_id -> position map
  id_to_pos <- integer(0)
  # Use a fast integer match via data.table
  id_map_dt <- data.table(cell_id = as.integer(id_order), pos = seq_len(N))
  setkey(id_map_dt, cell_id)

  year_map_dt <- data.table(year = years, ypos = seq_len(T_years))
  setkey(year_map_dt, year)

  # Add positions to cell_data
  cell_data[, row_orig := .I]
  cell_data[id_map_dt, cell_pos := i.pos, on = .(id = cell_id)]
  cell_data[year_map_dt, year_pos := i.ypos, on = .(year)]

  # â”€â”€â”€ 3. Sort data by (cell_pos, year_pos) for contiguous memory access â”€â”€â”€â”€
  # We'll create a mapping: grid index = (cell_pos - 1) * T_years + year_pos
  # This gives each (cell, year) a unique integer in 1..NR
  cell_data[, grid_idx := (cell_pos - 1L) * T_years + year_pos]

  # Verify completeness (balanced panel)
  if (nrow(cell_data) != NR) {
    message(sprintf("  Warning: expected %d rows, got %d (unbalanced panel)", NR, nrow(cell_data)))
    message("  Handling gracefully with NA fill...")
  }

  # Create a reorder vector: for grid_idx g, which row of cell_data is it?
  # This lets us build dense vectors aligned to the grid.
  grid_to_row <- integer(NR)
  grid_to_row[cell_data$grid_idx] <- cell_data$row_orig

  # â”€â”€â”€ 4. Build the block-diagonal sparse adjacency matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  # For year t (1-indexed), cell i's grid_idx = (i-1)*T + t

  # Edge (i -> j) in year t becomes: row = (i-1)*T + t, col = (j-1)*T + t
  #
  # We replicate the edge list T_years times with appropriate offsets.

  message("Building block-diagonal sparse adjacency matrix...")

  total_panel_edges <- as.numeric(n_actual_edges) * T_years
  sp_i <- integer(total_panel_edges)
  sp_j <- integer(total_panel_edges)

  for (t in seq_len(T_years)) {
    offset <- (0:(N - 1L)) * T_years + t  # grid_idx for each cell in year t
    start <- (t - 1L) * n_actual_edges + 1L
    end   <- t * n_actual_edges
    sp_i[start:end] <- (edge_from - 1L) * T_years + t
    sp_j[start:end] <- (edge_to   - 1L) * T_years + t
  }

  # Sparse adjacency matrix (NR x NR) with 1s on edges
  A <- sparseMatrix(i = sp_i, j = sp_j, x = 1, dims = c(NR, NR))

  # Neighbor count per grid node (for computing mean)
  neighbor_count <- rowSums(A)  # fast for sparse

  # Free large temporaries
  rm(sp_i, sp_j)
  gc()

  message(sprintf("  Sparse matrix: %d x %d with %d non-zeros", NR, NR, length(A@x)))

  # â”€â”€â”€ 5. Build dense variable vectors aligned to grid â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  # For each variable, create a length-NR vector where position g = grid_idx
  # has the value from the corresponding cell_data row.

  build_grid_vector <- function(var_name) {
    v <- rep(NA_real_, NR)
    valid <- grid_to_row > 0L
    v[valid] <- cell_data[[var_name]][grid_to_row[valid]]
    v
  }

  # â”€â”€â”€ 6. Compute neighbor stats per variable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  message("Computing neighbor statistics...")

  # For MEAN: A %*% x gives sum of neighbor values; divide by neighbor_count.
  # For MAX and MIN: we need grouped row-wise max/min over sparse entries.
  #
  # Efficient approach for max/min: use the CSR structure of A directly.
  # A is stored in CSC format (dgCMatrix). We transpose to get row-access.

  At <- t(A)  
  # At is CSC, so columns of At = rows of A.
  # At@p[g]+1 .. At@p[g+1] gives the nonzero row indices in column g of At,
  # which are the neighbor grid indices for node g.

  compute_stats_sparse <- function(vals, At, neighbor_count, NR) {
    # Replace NA with sentinel values for max/min computation
    vals_for_max <- vals
    vals_for_min <- vals
    vals_for_max[is.na(vals_for_max)] <- -Inf
    vals_for_min[is.na(vals_for_min)] <- Inf

    # For sum (to compute mean), treat NA as 0 but track count of valid
    vals_for_sum <- vals
    vals_for_sum[is.na(vals_for_sum)] <- 0

    # Count valid (non-NA) neighbors per node
    valid_indicator <- as.numeric(!is.na(vals))

    # Sparse matrix-vector products (C-level, very fast)
    neighbor_sum       <- as.numeric(A %*% vals_for_sum)
    neighbor_valid_cnt <- as.numeric(A %*% valid_indicator)

    # Mean
    n_mean <- ifelse(neighbor_valid_cnt > 0, neighbor_sum / neighbor_valid_cnt, NA_real_)

    # Max and Min: must iterate over sparse structure, but in C via .Call or
    # use a data.table grouped approach on the edge list.
    # 
    # Fastest pure-R approach: work with At (CSC) directly.
    # At@p is 0-based column pointers, At@i is 0-based row indices.

    p <- At@p
    idx <- At@i + 1L  # convert to 1-based

    n_max <- rep(NA_real_, NR)
    n_min <- rep(NA_real_, NR)

    # Vectorized approach using data.table on the edge list
    # Edge list: for each node g, neighbors are idx[p[g]+1 .. p[g+1]]
    # Build a data.table of (node, neighbor_val) and group by node.

    # Reconstruct edge list from CSC
    node_ids <- rep(seq_len(NR), diff(p))
    neighbor_ids <- idx
    neighbor_vals_raw <- vals[neighbor_ids]

    edge_dt <- data.table(node = node_ids, nval = neighbor_vals_raw)
    # Remove NA neighbor values
    edge_dt <- edge_dt[!is.na(nval)]

    if (nrow(edge_dt) > 0) {
      stats_dt <- edge_dt[, .(nmax = max(nval), nmin = min(nval)), by = node]
      n_max[stats_dt$node] <- stats_dt$nmax
      n_min[stats_dt$node] <- stats_dt$nmin
    }

    cbind(n_max, n_min, n_mean)
  }

  for (var_name in neighbor_source_vars) {
    message(sprintf("  Processing variable: %s", var_name))

    v <- build_grid_vector(var_name)
    stats <- compute_stats_sparse(v, At, neighbor_count, NR)

    # Map results back to cell_data row order
    max_col <- paste0("neighbor_max_", var_name)
    min_col <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    # stats is indexed by grid_idx; map back to original row order
    result_max  <- rep(NA_real_, nrow(cell_data))
    result_min  <- rep(NA_real_, nrow(cell_data))
    result_mean <- rep(NA_real_, nrow(cell_data))

    valid_grid <- cell_data$grid_idx
    result_max  <- stats[valid_grid, 1]
    result_min  <- stats[valid_grid, 2]
    result_mean <- stats[valid_grid, 3]

    set(cell_data, j = max_col,  value = result_max)
    set(cell_data, j = min_col,  value = result_min)
    set(cell_data, j = mean_col, value = result_mean)

    message(sprintf("    Done: %s", var_name))
  }

  # â”€â”€â”€ 7. Clean up helper columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  cell_data[, c("row_orig", "cell_pos", "year_pos", "grid_idx") := NULL]

  message("Neighbor feature computation complete.")
  return(cell_data)
}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# USAGE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#
# cell_data <- optimize_neighbor_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   neighbor_source_vars   = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
#
# # Then predict with the pre-trained Random Forest (unchanged):
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Is Fast: Complexity Comparison

| Component | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | ~6.46M string pastes + hash lookups (~hours) | One sparse matrix construction via integer arithmetic (~seconds) |
| **Mean computation** | 6.46M R-level loops Ã— 5 vars | 5 sparse matrixâ€“vector multiplies (C-level CHOLMOD/CSC, ~seconds each) |
| **Max/Min computation** | 6.46M R-level loops Ã— 5 vars | 5 `data.table` grouped aggregations on ~38.5M-row edge table (C-level, ~seconds each) |
| **Total estimated time** | 86+ hours | **~2â€“5 minutes** |

## Key Design Decisions

1. **Grid indexing `(cell_pos - 1) * T + year_pos`** eliminates all string operations. The mapping is bijective and invertible with pure integer arithmetic.

2. **Block-diagonal sparse matrix** reuses the same topology across all 28 years. The matrix is constructed once and has ~38.5M non-zeros (1.37M edges Ã— 28 years), which fits easily in RAM (~600 MB for the sparse triplet).

3. **`A %*% x` for mean** leverages CHOLMOD's optimized sparse BLAS â€” this is the single biggest speedup, turning 6.46M R-level loops into one C-level matrix-vector product.

4. **`data.table` grouped aggregation for max/min** converts the sparse structure back to an edge list and uses `data.table`'s radix-sort grouped operations, which run at C speed.

5. **Numerical equivalence is preserved exactly**: the same neighbor sets are aggregated with the same `max`, `min`, `mean` functions, and NAs are handled identically (excluded before aggregation; result is NA if all neighbors are NA or no neighbors exist).

6. **The pre-trained Random Forest is untouched** â€” only the input feature columns are computed.

## Memory Profile

| Object | Approximate Size |
|---|---|
| `cell_data` (6.46M Ã— 110 cols) | ~5.7 GB |
| Sparse matrix `A` (6.46M Ã— 6.46M, 38.5M nnz) | ~0.6 GB |
| Transpose `At` | ~0.6 GB |
| Edge `data.table` (38.5M rows Ã— 2 cols) | ~0.6 GB |
| Dense grid vectors (per variable) | ~52 MB |
| **Total peak** | **~8â€“9 GB** (fits in 16 GB) |

If memory is tight, you can drop `At` and rebuild the edge `data.table` once from the CSC structure, then reuse it across all variables:

```r
# Memory-saving variant: extract edge list once, reuse for all variables
p <- At@p
edge_node <- rep(seq_len(NR), diff(p))
edge_neighbor <- At@i + 1L
rm(At)
gc()

# Then for each variable:
edge_dt <- data.table(node = edge_node, nval = v[edge_neighbor])
edge_dt <- edge_dt[!is.na(nval)]
stats_dt <- edge_dt[, .(nmax = max(nval), nmin = min(nval)), by = node]
```

This reduces peak memory by ~0.6 GB.
END_RESPONSE>>>
