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
run_id: Anthropic_F12_T07_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized bind operation per variable â€” only 5 calls total. Each inner function returns a fixed-length numeric vector of length 3, so there is no repeated list growth or quadratic copying happening inside `compute_neighbor_stats()`.

**The true bottleneck is `build_neighbor_lookup()`.**

Here is why:

1. **6.46 million `paste()` + named-vector lookups via character indexing.** `idx_lookup` is a named vector of length ~6.46M. For each of the ~6.46M rows, the function does:
   - `as.character(data$id[i])` â€” character conversion per row.
   - `id_to_ref[as.character(...)]` â€” named-vector lookup (linear hash probe on a 344K-element named vector).
   - `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” string construction for every neighbor of every row.
   - `idx_lookup[neighbor_keys]` â€” named-vector lookup against a **6.46M-element** named character vector, repeated for every neighbor key.

2. **This is called inside `lapply` over 6.46M rows**, and each row has on average ~4 rook neighbors. That means roughly **25.8 million character-key lookups against a 6.46M-length named vector**, plus 25.8 million `paste()` calls, all inside an interpreted R loop. Named-vector lookup in R is O(n) worst-case per probe in base R (it uses hashing but with overhead), and the sheer volume makes this catastrophically slow.

3. **`build_neighbor_lookup()` is called once and produces the lookup for all 5 variables**, so its cost is paid once â€” but that single invocation dominates the entire 86+ hour runtime. `compute_neighbor_stats()` by contrast does simple integer-indexed numeric subsetting, which is fast.

**Summary:** The bottleneck is the row-by-row character-key construction and named-vector lookups in `build_neighbor_lookup()`, not the `do.call(rbind, ...)` in `compute_neighbor_stats()`.

---

## Optimization Strategy

1. **Eliminate all character/paste operations.** Replace the character-keyed lookup with pure integer arithmetic. Since every cell appears for every year (a balanced panel), we can compute the row index of any (cell, year) combination arithmetically.

2. **Vectorize the neighbor lookup construction.** Instead of `lapply` over 6.46M rows, expand the neighbor list once at the cell level (344K cells Ã— ~4 neighbors = ~1.37M pairs), then broadcast across all 28 years using vectorized integer operations.

3. **Replace `do.call(rbind, ...)` with pre-allocated matrix indexing** in `compute_neighbor_stats()` for a minor additional speedup.

4. **Preserve the trained Random Forest model** â€” we only change the feature-engineering pipeline, not the model or the numerical values produced.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE
# =============================================================================
# Assumptions (from the problem statement):
#   - cell_data is a data.frame with columns: id, year, ntl, ec, pop_density,
#     def, usd_est_n2, and ~110 other columns.
#   - cell_data is sorted by (id, year) â€” i.e., all years for cell 1, then
#     all years for cell 2, etc. If not, we sort it once.
#   - id_order is the vector of unique cell IDs in the order matching
#     rook_neighbors_unique (the spdep::nb object).
#   - rook_neighbors_unique is the nb object (list of integer neighbor indices).
#   - The panel is balanced: every cell has exactly n_years rows.
# =============================================================================

# ---- 0. Ensure consistent ordering ----------------------------------------
# Sort cell_data by (id, year) so that rows for the same cell are contiguous
# and years are in ascending order within each cell.
cell_data <- cell_data[order(cell_data$id, cell_data$year), ]

n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

stopifnot(nrow(cell_data) == n_cells * n_years)  # balanced panel check

# ---- 1. Build integer ID-to-block mapping ----------------------------------
# After sorting by (id, year), the rows for cell id_order[k] occupy
# positions ((k-1)*n_years + 1) : (k*n_years).
# We need a fast map from cell_data$id values to their block index k.

# Create a map: cell_id -> block index (1-based position in id_order)
id_to_block <- setNames(seq_along(id_order), as.character(id_order))

# Verify the sort matches id_order ordering â€” if not, reorder id_order
# to match the sort, or re-sort cell_data to match id_order.
# We re-sort cell_data to match id_order's ordering:
cell_data$`.block` <- id_to_block[as.character(cell_data$id)]
cell_data <- cell_data[order(cell_data$`.block`, cell_data$year), ]
cell_data$`.block` <- NULL
# Now row ((k-1)*n_years + j) corresponds to id_order[k], years[j].

# ---- 2. Build vectorized neighbor lookup (FAST) ---------------------------
# For each cell k, neighbors are rook_neighbors_unique[[k]] (vector of
# integer indices into id_order). We expand this to row-level indices.

build_neighbor_lookup_fast <- function(n_cells, n_years, neighbors) {
  # Step 1: Expand cell-level neighbor pairs.
  #   For cell k with neighbors {n1, n2, ...}, create pairs (k, n1), (k, n2), ...
  #   This is ~1.37M pairs total.
  
  n_neighbors <- lengths(neighbors)  # integer vector, length n_cells
  
  # Source cell indices (repeated for each neighbor)
  src_cell <- rep(seq_len(n_cells), times = n_neighbors)
  
  # Destination cell indices
  dst_cell <- unlist(neighbors, use.names = FALSE)
  
  # Step 2: For each year j in 1:n_years, the row index of cell k is
  #   (k - 1) * n_years + j.
  # We need, for each ROW r = (src_cell_k - 1)*n_years + j, the set of
  # neighbor rows { (dst_cell - 1)*n_years + j } for each dst_cell in
  # neighbors of src_cell_k.
  #
  # Instead of building a list of length n_cells*n_years, we build a
  # CSR-like structure: two vectors (neighbor_rows, row_ptr).
  
  n_pairs <- length(src_cell)  # ~1.37M
  total_entries <- n_pairs * n_years  # ~38.4M â€” fits in memory easily
  
  # Pre-allocate
  neighbor_row_vec <- integer(total_entries)
  
  # row_ptr: for source row r, its neighbors are in
  #   neighbor_row_vec[ row_ptr[r] : (row_ptr[r+1] - 1) ]
  # n_neighbors_per_row[r] = n_neighbors[ cell_of_row_r ]
  n_neighbors_per_row <- rep(n_neighbors, each = n_years)
  row_ptr <- c(0L, cumsum(n_neighbors_per_row)) + 1L
  # row_ptr has length n_cells*n_years + 1
  
  # Fill neighbor_row_vec using vectorized operations over years
  for (j in seq_len(n_years)) {
    # Source rows for year j: all cells, year j
    # These are rows j, n_years+j, 2*n_years+j, ...
    # But we only care about cells that HAVE neighbors (all of them in src_cell).
    
    # For the pair (src_cell[p], dst_cell[p]), in year j:
    #   source row = (src_cell[p] - 1) * n_years + j
    #   dest   row = (dst_cell[p] - 1) * n_years + j
    
    dst_rows_j <- (dst_cell - 1L) * n_years + j
    
    # Where do these go in neighbor_row_vec?
    # For source row r = (src_cell[p]-1)*n_years + j,
    # the neighbors start at row_ptr[r].
    # Within that block, the p-th neighbor of cell src_cell[p] goes at
    # a specific offset.
    
    # We need the offset of each pair within its cell's neighbor list.
    # pair_offset[p] = cumulative count of pairs with the same src_cell, up to p.
    # Since src_cell is rep(1:n_cells, times=n_neighbors), the pairs for cell k
    # are contiguous. The offset within cell k's block is 0, 1, ..., n_neighbors[k]-1.
    pair_offset <- sequence(n_neighbors) - 1L  # 0-based offset within each cell
    
    # The position in neighbor_row_vec for pair p, year j:
    src_rows_j <- (src_cell - 1L) * n_years + j
    positions <- row_ptr[src_rows_j] + pair_offset
    
    neighbor_row_vec[positions] <- dst_rows_j
  }
  
  list(neighbor_row_vec = neighbor_row_vec,
       row_ptr          = row_ptr,
       n_rows           = n_cells * n_years)
}

cat("Building neighbor lookup (vectorized)...\n")
system.time({
  nb_csr <- build_neighbor_lookup_fast(n_cells, n_years, rook_neighbors_unique)
})

# ---- 3. Compute neighbor stats (FAST) -------------------------------------
compute_neighbor_stats_fast <- function(data, nb_csr, var_name) {
  vals   <- data[[var_name]]
  n_rows <- nb_csr$n_rows
  rp     <- nb_csr$row_ptr
  nrv    <- nb_csr$neighbor_row_vec
  
  out <- matrix(NA_real_, nrow = n_rows, ncol = 3)
  colnames(out) <- paste0("neighbor_", c("max", "min", "mean"), "_", var_name)
  
  for (i in seq_len(n_rows)) {
    start <- rp[i]
    end   <- rp[i + 1L] - 1L
    if (start > end) next
    
    nv <- vals[nrv[start:end]]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    
    out[i, 1L] <- max(nv)
    out[i, 2L] <- min(nv)
    out[i, 3L] <- mean(nv)
  }
  out
}

# Even faster: fully vectorized using data.table for the group-by operation
compute_neighbor_stats_vectorized <- function(data, nb_csr, var_name) {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("Install data.table for the vectorized path.")
  }
  
  vals   <- data[[var_name]]
  n_rows <- nb_csr$n_rows
  rp     <- nb_csr$row_ptr
  nrv    <- nb_csr$neighbor_row_vec
  
  # Build a two-column table: (source_row, neighbor_value)
  # source_row for each entry in nrv
  n_per_row <- diff(rp)  # length n_rows
  src_row   <- rep(seq_len(n_rows), times = n_per_row)
  
  dt <- data.table::data.table(
    src = src_row,
    val = vals[nrv]
  )
  
  # Remove NAs
  dt <- dt[!is.na(val)]
  
  # Group-by aggregation â€” data.table is heavily optimized for this
  agg <- dt[, .(nmax = max(val), nmin = min(val), nmean = mean(val)), by = src]
  
  # Map back to full output matrix
  out <- matrix(NA_real_, nrow = n_rows, ncol = 3)
  colnames(out) <- paste0("neighbor_", c("max", "min", "mean"), "_", var_name)
  out[agg$src, 1L] <- agg$nmax
  out[agg$src, 2L] <- agg$nmin
  out[agg$src, 3L] <- agg$nmean
  
  out
}

# ---- 4. Outer loop â€” add features to cell_data ----------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  cat("  Processing:", var_name, "\n")
  stats <- compute_neighbor_stats_vectorized(cell_data, nb_csr, var_name)
  cell_data[[colnames(stats)[1]]] <- stats[, 1]
  cell_data[[colnames(stats)[2]]] <- stats[, 2]
  cell_data[[colnames(stats)[3]]] <- stats[, 3]
}
cat("Done.\n")

# ---- 5. The trained Random Forest model is untouched -----------------------
# The new columns are numerically identical to the originals, so the
# existing trained RF model can be applied directly:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Performance Analysis

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M `lapply` iterations, each doing `paste()` + character-key lookup against a 6.46M named vector. **Estimated: 70â€“80+ hours.** | Pure integer arithmetic, vectorized across years. ~1.37M neighbor pairs Ã— 28 years filled via vectorized indexing. **Estimated: 30â€“90 seconds.** |
| `compute_neighbor_stats` | `lapply` over 6.46M rows + `do.call(rbind, ...)` on 6.46M 3-element vectors. **Estimated: 5â€“10 minutes per variable.** | `data.table` group-by on ~25.8M rows (single vectorized aggregation). **Estimated: 5â€“15 seconds per variable.** |
| **Total** | **~86+ hours** | **~2â€“5 minutes** |

The key insight: the colleague misidentified a minor cost (`do.call(rbind, ...)` â€” minutes) as the bottleneck, while the true bottleneck was the **O(N Ã— k) character-key construction and lookup** in `build_neighbor_lookup()` where N = 6.46M rows and k â‰ˆ 4 neighbors, all running through R's interpreted `lapply` with expensive string operations against a massive named vector. The fix is to replace all string-based indexing with **integer arithmetic** exploiting the balanced-panel structure, and to replace the row-level R loop in the stats computation with a **vectorized `data.table` group-by**.
END_RESPONSE>>>
