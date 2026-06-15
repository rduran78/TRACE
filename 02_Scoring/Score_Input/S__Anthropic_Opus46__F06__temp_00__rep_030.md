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
run_id: Anthropic_F06_T00_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's rook neighbors (fast).
2. Constructs character keys by pasting neighbor IDs with the current row's year (slow â€” character allocation and concatenation at scale).
3. Performs named-vector lookups via `idx_lookup[neighbor_keys]` (slow â€” repeated hash lookups on a 6.46M-entry named vector).

This is **O(N Ã— k)** where N â‰ˆ 6.46M and k â‰ˆ average number of rook neighbors (~4). The constant factor is enormous because of repeated `paste()` and named-vector indexing on character keys.

### Bottleneck B: `compute_neighbor_stats` â€” O(N) `lapply` with per-row subsetting

For each of the 5 variables, another `lapply` over 6.46M rows extracts neighbor values, removes NAs, and computes `max`, `min`, `mean`. This is called 5 times, so ~32.3M R-level function invocations with small-vector operations â€” extremely slow due to R's interpreter overhead.

### Why raster focal/kernel operations are a useful *analogy* but not the right implementation

Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed kernel. If the 344K cells form a complete regular grid, focal operations could work. However:
- The panel has a **time dimension** â€” neighbors are only within the same year.
- The grid may have irregular boundaries, missing cells, or an incomplete lattice.
- The neighbor structure is precomputed as an `spdep::nb` object, which may encode irregular adjacency.

The correct approach is to **vectorize the sparse-neighbor computation** using the same `nb` structure but with matrix/data.table operations instead of row-level `lapply`.

---

## 2. Optimization Strategy

| Step | Current | Optimized | Speedup Factor |
|------|---------|-----------|----------------|
| Neighbor lookup | Character paste + named vector lookup per row | Integer index arithmetic: `(cell_index - 1) Ã— T + year_offset` | ~100â€“500Ã— |
| Neighbor stats | `lapply` over 6.46M rows, 5 times | Sparse matrix multiplication / vectorized `data.table` group-by | ~50â€“200Ã— |
| Overall | ~86+ hours | **~5â€“15 minutes** | ~350â€“1000Ã— |

### Key ideas:

1. **Eliminate `build_neighbor_lookup` entirely.** Instead, exploit the panel's regular structure: every cell appears once per year, so if we sort by `(id, year)`, the row index for cell `c` in year `y` is deterministic. We build a sparse neighbor-row matrix once.

2. **Use sparse matrix multiplication for `mean`**, and vectorized grouped operations for `max` and `min`.

3. **Process all 5 variables in one pass** over the neighbor structure rather than 5 separate passes.

---

## 3. Working R Code

```r
library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {
  
  # -------------------------------------------------------------------
  # STEP 0: Convert to data.table for speed; record original row order
  # -------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .orig_row := .I]
  
  # -------------------------------------------------------------------
  # STEP 1: Build integer mappings
  # -------------------------------------------------------------------
  # Map cell id -> integer index (1..n_cells)
  unique_ids <- as.character(id_order)
  n_cells    <- length(unique_ids)
  id_to_int  <- setNames(seq_len(n_cells), unique_ids)
  
  # Map year -> integer index (1..n_years)
  years      <- sort(unique(dt$year))
  n_years    <- length(years)
  year_to_int <- setNames(seq_len(n_years), as.character(years))
  
  # Assign integer cell and year indices
  dt[, cell_int := id_to_int[as.character(id)]]
  dt[, year_int := year_to_int[as.character(year)]]
  
  # -------------------------------------------------------------------
  # STEP 2: Sort by (cell_int, year_int) so row index is deterministic
  #         row(c, y) = (c - 1) * n_years + y
  # -------------------------------------------------------------------
  setorder(dt, cell_int, year_int)
  dt[, sorted_row := .I]
  
  # Verify the deterministic mapping holds
  expected_row <- (dt$cell_int - 1L) * n_years + dt$year_int
  stopifnot(all(dt$sorted_row == expected_row))
  
  N <- nrow(dt)  # total rows (~6.46M)
  
  # -------------------------------------------------------------------
  # STEP 3: Build sparse adjacency in ROW space (one-time cost)
  #
  # For each cell c with neighbors {n1, n2, ...}, and for each year y,
  # row (c,y) has neighbor rows {(n1,y), (n2,y), ...}.
  # We tile the cell-level adjacency across all years.
  # -------------------------------------------------------------------
  message("Building sparse neighbor matrix...")
  
  # Extract cell-level adjacency as (from, to) integer pairs
  from_cell <- integer(0)
  to_cell   <- integer(0)
  for (c_idx in seq_len(n_cells)) {
    nb <- rook_neighbors_unique[[c_idx]]
    if (length(nb) > 0 && !all(is.na(nb))) {
      nb <- nb[!is.na(nb)]
      from_cell <- c(from_cell, rep(c_idx, length(nb)))
      to_cell   <- c(to_cell, nb)
    }
  }
  
  n_edges <- length(from_cell)
  message(sprintf("  Cell-level edges: %d", n_edges))
  
  # Tile across years: for each year y, create row-level edges
  # row = (cell - 1) * n_years + year
  from_row <- integer(n_edges * n_years)
  to_row   <- integer(n_edges * n_years)
  
  for (y in seq_len(n_years)) {
    offset <- (y - 1L) * n_edges
    from_row[offset + seq_len(n_edges)] <- (from_cell - 1L) * n_years + y
    to_row[offset + seq_len(n_edges)]   <- (to_cell - 1L)   * n_years + y
  }
  
  # Remove any edges pointing to rows that don't exist (boundary/missing cells)
  valid <- from_row >= 1L & from_row <= N & to_row >= 1L & to_row <= N
  from_row <- from_row[valid]
  to_row   <- to_row[valid]
  
  # Sparse adjacency matrix (not row-normalized yet)
  # W[i,j] = 1 means row j is a rook neighbor of row i
  W <- sparseMatrix(
    i = from_row, j = to_row,
    x = rep(1, length(from_row)),
    dims = c(N, N)
  )
  
  # Degree (number of non-NA neighbors per row â€” will adjust for NA vals per variable)
  rm(from_row, to_row, from_cell, to_cell, valid)
  gc()
  
  message("Sparse neighbor matrix built.")
  
  # -------------------------------------------------------------------
  # STEP 4: Compute neighbor stats for each variable
  #
  # For MEAN: use sparse matrix multiplication  W %*% x / degree
  # For MAX and MIN: vectorized grouped operations via the sparse structure
  # -------------------------------------------------------------------
  
  # Pre-extract the adjacency list from the sparse matrix for max/min
  # (CSC format gives us column-wise access; we need row-wise)
  # Convert to dgRMatrix (row-compressed) for efficient row access
  W_row <- as(W, "RsparseMatrix")
  
  # Row pointers and column indices (0-based in internal representation)
  row_ptr  <- W_row@p   # length N+1, 0-based
  col_idx  <- W_row@j   # 0-based column indices
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Processing variable: %s", var_name))
    
    x <- dt[[var_name]]
    
    # --- MEAN via sparse matrix multiplication ---
    # Handle NAs: replace with 0 for sum, track non-NA for count
    not_na   <- as.numeric(!is.na(x))
    x_clean  <- ifelse(is.na(x), 0, x)
    
    neighbor_sum   <- as.numeric(W %*% x_clean)
    neighbor_count <- as.numeric(W %*% not_na)
    
    neighbor_mean <- ifelse(neighbor_count > 0,
                            neighbor_sum / neighbor_count,
                            NA_real_)
    
    # --- MAX and MIN via vectorized row-wise operations ---
    # Use the row-compressed sparse matrix
    neighbor_max <- rep(NA_real_, N)
    neighbor_min <- rep(NA_real_, N)
    
    # Vectorized approach: for each edge, accumulate max/min by "from" row
    # Reconstruct edge list from sparse matrix
    # from = row index, to = col_idx (the neighbor whose value we read)
    edge_from <- rep(seq_len(N), diff(row_ptr))
    edge_to   <- col_idx + 1L  # convert to 1-based
    
    edge_vals <- x[edge_to]
    
    # Remove edges where the neighbor value is NA
    valid_edge <- !is.na(edge_vals)
    edge_from_v <- edge_from[valid_edge]
    edge_vals_v <- edge_vals[valid_edge]
    
    # Use data.table for grouped max/min (very fast)
    if (length(edge_from_v) > 0) {
      edge_dt <- data.table(from = edge_from_v, val = edge_vals_v)
      agg <- edge_dt[, .(nb_max = max(val), nb_min = min(val)), by = from]
      neighbor_max[agg$from] <- agg$nb_max
      neighbor_min[agg$from] <- agg$nb_min
      rm(edge_dt, agg)
    }
    
    rm(edge_from, edge_to, edge_vals, valid_edge, edge_from_v, edge_vals_v)
    
    # --- Assign to data.table ---
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    set(dt, j = max_col,  value = neighbor_max)
    set(dt, j = min_col,  value = neighbor_min)
    set(dt, j = mean_col, value = neighbor_mean)
    
    rm(neighbor_max, neighbor_min, neighbor_mean, neighbor_sum, neighbor_count,
       x, x_clean, not_na)
    gc()
    
    message(sprintf("  Done: %s", var_name))
  }
  
  # -------------------------------------------------------------------
  # STEP 5: Restore original row order and return
  # -------------------------------------------------------------------
  setorder(dt, .orig_row)
  dt[, c(".orig_row", "sorted_row", "cell_int", "year_int") := NULL]
  
  # Return as data.frame if input was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# =====================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =====================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# Then apply the pre-trained Random Forest model as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Detail |
|---|---|
| **Root cause** | Row-level `lapply` over 6.46M rows with character key construction and named-vector lookups â€” pure R interpreter overhead |
| **Key insight** | The panel is regular: every cell Ã— year combination maps to a deterministic integer row index. The cell-level adjacency can be "tiled" across years via integer arithmetic, then encoded as a single sparse matrix. |
| **Mean** | Computed via sparse matrixâ€“vector multiplication (`W %*% x`), which is O(nnz) in optimized C/Fortran â€” handles ~38.5M edges in seconds. |
| **Max / Min** | Edge list extracted once from the sparse matrix; grouped aggregation via `data.table` (radix-sort based, highly optimized). |
| **NA handling** | Identical to original: NAs excluded before computing each statistic; rows with zero valid neighbors get `NA`. |
| **Numerical equivalence** | Preserved â€” same `max`, `min`, `mean` over the same neighbor sets. |
| **RF model** | Untouched â€” only the feature-engineering step is optimized. |
| **Expected runtime** | ~5â€“15 minutes on a 16 GB laptop (dominated by sparse matrix construction and 5 SpMV operations). |
| **Memory** | Sparse matrix: ~38.5M non-zeros Ã— 12 bytes â‰ˆ ~460 MB; well within 16 GB. |
END_RESPONSE>>>
