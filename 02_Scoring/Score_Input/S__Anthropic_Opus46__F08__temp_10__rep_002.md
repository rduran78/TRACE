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
run_id: Anthropic_F08_T10_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static spatial topology with dynamic temporal data.**

### Root cause breakdown:

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) indices.** It creates a list of length ~6.46 million, where each element maps a cell-year row to the row indices of its neighbors *in the same year*. This involves 6.46M calls to `paste()`, hash lookups, and subsetting â€” all redundant across years because the neighbor *structure* is identical every year.

2. **String-keyed lookups are expensive.** The function builds `idx_lookup` as a named vector keyed by `"id_year"` strings. Looking up ~1,373,394 Ã— 28 â‰ˆ 38.5M string keys is extremely slow in R.

3. **`compute_neighbor_stats` iterates with `lapply` over 6.46M rows.** Each iteration subsets a numeric vector, removes NAs, and computes max/min/mean. The per-element overhead of 6.46M R-level function calls dominates.

4. **No vectorization or matrix-based computation.** The spatial structure is a fixed adjacency graph. For any given year-slice, neighbor stats are a sparse-matrix operation that can be vectorized entirely.

### The key insight:

| Aspect | Nature | Rebuild per year? |
|---|---|---|
| Which cells neighbor which cells | **Static** (spatial topology) | **No** |
| Variable values attached to cells | **Dynamic** (changes by year) | **Yes** |

The neighbor lookup should be built **once at the cell level** (344K entries), and then for each year, we simply slice the relevant variable column and apply sparse-matrix operations to compute neighbor max, min, and mean.

---

## Optimization Strategy

1. **Build a static cell-level neighbor lookup once** â€” a list of length 344,208 mapping each cell's positional index to its neighbors' positional indices. This is just `rook_neighbors_unique` itself (an `nb` object), possibly reindexed.

2. **Build a sparse adjacency matrix once** from the `nb` object. A sparse `dgCMatrix` of dimension 344,208 Ã— 344,208. This enables vectorized neighbor-mean computation via matrix multiplication.

3. **For each year, reshape the variable into a cell-indexed vector** and compute:
   - **Neighbor mean**: sparse matrix multiply `A %*% x / neighbor_count` (fully vectorized).
   - **Neighbor max and min**: iterate over 344K cells (not 6.46M) using the cell-level neighbor list, or use a grouped operation.

4. **Avoid all string-key lookups.** Use integer positional indexing throughout.

5. **Expected speedup**: From ~86 hours to ~5â€“15 minutes. The bottleneck moves from 6.46M R-level iterations with string hashing to 28 Ã— 5 sparse matrix multiplications on 344K-dimensional vectors, plus a fast grouped max/min pass.

---

## Working R Code

```r
library(Matrix)

# =============================================================================
# STEP 1: Build STATIC spatial structures (done once, independent of year)
# =============================================================================

build_static_neighbor_structures <- function(id_order, neighbors_nb) {
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors_nb: spdep nb object (list of integer neighbor indices)
  #
  # Returns:
  #   adj_matrix:    sparse binary adjacency matrix (344208 x 344208)
  #   neighbor_list: the nb object as a plain list of integer vectors
  #   neighbor_count: integer vector of neighbor counts per cell
  
  n_cells <- length(id_order)
  stopifnot(length(neighbors_nb) == n_cells)
  
  # Build sparse adjacency matrix from nb object
  # Each neighbors_nb[[i]] contains integer indices of neighbors of cell i
  from <- rep(seq_len(n_cells), lengths(neighbors_nb))
  to   <- unlist(neighbors_nb)
  
  adj_matrix <- sparseMatrix(
    i    = from,
    j    = to,
    x    = 1,
    dims = c(n_cells, n_cells)
  )
  
  neighbor_count <- diff(adj_matrix@p)  # CSC column pointer diff = col counts
  # Actually for row-based counts:
  neighbor_count <- as.integer(rowSums(adj_matrix))
  
  list(
    adj_matrix     = adj_matrix,
    neighbor_list  = lapply(neighbors_nb, as.integer),  # plain list
    neighbor_count = neighbor_count,
    id_order       = id_order,
    n_cells        = n_cells
  )
}

# =============================================================================
# STEP 2: For each year Ã— variable, compute neighbor max, min, mean
# =============================================================================

compute_neighbor_stats_for_year <- function(values_vec, static) {
  # values_vec: numeric vector of length n_cells, ordered by id_order
  # static: output of build_static_neighbor_structures
  #
  # Returns: data.frame with columns neighbor_max, neighbor_min, neighbor_mean
  #          length n_cells, ordered by id_order
  
  n_cells       <- static$n_cells
  adj_matrix    <- static$adj_matrix
  neighbor_list <- static$neighbor_list
  neighbor_count <- static$neighbor_count
  

  # --- Neighbor MEAN via sparse matrix multiplication (fully vectorized) ---
  # Replace NA with 0 for the multiply, but track NA counts
  vals <- values_vec
  is_na <- is.na(vals)
  vals[is_na] <- 0
  
  neighbor_sum   <- as.numeric(adj_matrix %*% vals)
  # Count how many non-NA neighbors each cell has
  neighbor_valid <- as.numeric(adj_matrix %*% as.numeric(!is_na))
  
  neighbor_mean <- ifelse(neighbor_valid > 0, neighbor_sum / neighbor_valid, NA_real_)
  
  # --- Neighbor MAX and MIN via C++-speed grouped operation ---
  # We iterate over 344K cells (not 6.46M). 
  # Use vapply for fixed-output-size efficiency.
  
  neighbor_max <- rep(NA_real_, n_cells)
  neighbor_min <- rep(NA_real_, n_cells)
  
  for (i in seq_len(n_cells)) {
    nb_idx <- neighbor_list[[i]]
    if (length(nb_idx) == 0L) next
    nb_vals <- values_vec[nb_idx]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) == 0L) next
    neighbor_max[i] <- max(nb_vals)
    neighbor_min[i] <- min(nb_vals)
  }
  
  data.frame(
    neighbor_max  = neighbor_max,
    neighbor_min  = neighbor_min,
    neighbor_mean = neighbor_mean
  )
}

# =============================================================================
# STEP 3: Faster max/min alternative using data.table + sparse triplet
#         (avoids the 344K R-level loop entirely)
# =============================================================================

compute_neighbor_stats_for_year_fast <- function(values_vec, static) {
  # Fully vectorized version using data.table for grouped max/min
  
  n_cells    <- static$n_cells
  adj_matrix <- static$adj_matrix
  
  vals   <- values_vec
  is_na  <- is.na(vals)
  vals0  <- vals
  vals0[is_na] <- 0
  
  # --- MEAN (sparse matmul) ---
  neighbor_sum   <- as.numeric(adj_matrix %*% vals0)
  neighbor_valid <- as.numeric(adj_matrix %*% as.numeric(!is_na))
  neighbor_mean  <- ifelse(neighbor_valid > 0, neighbor_sum / neighbor_valid, NA_real_)
  
  # --- MAX and MIN (vectorized via sparse triplet expansion) ---
  # Convert adjacency to triplet form: (from_cell, to_cell)
  # For each edge (i, j), the "neighbor value for cell i" is values_vec[j]
  
  if (!requireNamespace("data.table", quietly = TRUE)) {
    # Fallback to loop version
    neighbor_max <- rep(NA_real_, n_cells)
    neighbor_min <- rep(NA_real_, n_cells)
    neighbor_list <- static$neighbor_list
    for (i in seq_len(n_cells)) {
      nb_idx <- neighbor_list[[i]]
      if (length(nb_idx) == 0L) next
      nb_vals <- values_vec[nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0L) next
      neighbor_max[i] <- max(nb_vals)
      neighbor_min[i] <- min(nb_vals)
    }
  } else {
    # Use the pre-computed triplet edges from static structure
    from_idx <- static$edge_from
    to_idx   <- static$edge_to
    
    nb_vals <- values_vec[to_idx]
    
    # Remove edges where the neighbor value is NA
    valid <- !is.na(nb_vals)
    
    dt <- data.table::data.table(
      cell = from_idx[valid],
      val  = nb_vals[valid]
    )
    
    agg <- dt[, .(nmax = max(val), nmin = min(val)), by = cell]
    
    neighbor_max <- rep(NA_real_, n_cells)
    neighbor_min <- rep(NA_real_, n_cells)
    neighbor_max[agg$cell] <- agg$nmax
    neighbor_min[agg$cell] <- agg$nmin
  }
  
  data.frame(
    neighbor_max  = neighbor_max,
    neighbor_min  = neighbor_min,
    neighbor_mean = neighbor_mean
  )
}

# =============================================================================
# STEP 4: Main pipeline â€” replaces the original outer loop
# =============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  library(Matrix)
  library(data.table)
  
  cat("Building static neighbor structures (one-time cost)...\n")
  
  static <- build_static_neighbor_structures(id_order, rook_neighbors_unique)
  
  # Pre-compute edge lists for data.table-based max/min
  adj_t <- as(static$adj_matrix, "TsparseMatrix")
  static$edge_from <- adj_t@i + 1L   # 1-indexed
  static$edge_to   <- adj_t@j + 1L
  
  cat(sprintf("  %d cells, %d directed edges.\n", 
              static$n_cells, length(static$edge_from)))
  
  # Build cell-index mapping: for each cell in id_order, which rows in cell_data?
  # And for each year, which rows?
  cell_dt <- as.data.table(cell_data)
  
  # Ensure id_order alignment: create a mapping from cell ID -> position in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  cell_dt[, cell_pos := id_to_pos[as.character(id)]]
  
  years <- sort(unique(cell_dt$year))
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  # Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    cell_dt[, paste0("neighbor_max_", var_name) := NA_real_]
    cell_dt[, paste0("neighbor_min_", var_name) := NA_real_]
    cell_dt[, paste0("neighbor_mean_", var_name) := NA_real_]
  }
  
  cat(sprintf("Computing neighbor stats for %d variables Ã— %d years...\n",
              length(neighbor_source_vars), length(years)))
  
  for (yr in years) {
    yr_rows <- which(cell_dt$year == yr)
    yr_cell_pos <- cell_dt$cell_pos[yr_rows]
    
    # Build a mapping from cell_pos -> row in cell_dt for this year
    # This handles the case where not all cells appear in every year
    n_cells <- static$n_cells
    pos_to_yr_row <- rep(NA_integer_, n_cells)
    pos_to_yr_row[yr_cell_pos] <- yr_rows
    
    for (var_name in neighbor_source_vars) {
      
      # Build cell-indexed values vector (length = n_cells)
      # Cells not present this year get NA
      values_vec <- rep(NA_real_, n_cells)
      values_vec[yr_cell_pos] <- cell_dt[[var_name]][yr_rows]
      
      # Compute neighbor stats (operates on 344K-length vector)
      stats <- compute_neighbor_stats_for_year_fast(values_vec, static)
      
      # Write results back to the correct rows in cell_dt
      set(cell_dt, 
          i    = yr_rows,
          j    = paste0("neighbor_max_", var_name),
          value = stats$neighbor_max[yr_cell_pos])
      set(cell_dt, 
          i    = yr_rows,
          j    = paste0("neighbor_min_", var_name),
          value = stats$neighbor_min[yr_cell_pos])
      set(cell_dt, 
          i    = yr_rows,
          j    = paste0("neighbor_mean_", var_name),
          value = stats$neighbor_mean[yr_cell_pos])
    }
    
    cat(sprintf("  Year %d complete.\n", yr))
  }
  
  # Remove helper column and convert back to data.frame
  cell_dt[, cell_pos := NULL]
  
  as.data.frame(cell_dt)
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================

# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# # Then proceed with the pre-trained Random Forest prediction (unchanged):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup** | 6.46M-element list, rebuilt with string keys | 344K-element static list + sparse matrix, built **once** |
| **Neighbor mean** | R-level loop over 6.46M rows | Sparse matrix multiplication (vectorized BLAS) |
| **Neighbor max/min** | R-level loop over 6.46M rows | `data.table` grouped aggregation over ~1.37M edge triplets per year |
| **String operations** | ~45M `paste()` + named-vector lookups | **Zero** â€” pure integer positional indexing |
| **Time complexity** | O(rows Ã— avg_neighbors) with high R overhead | O(edges) per yearÃ—variable, vectorized |
| **Estimated runtime** | ~86+ hours | **~5â€“15 minutes** |
| **Numerical result** | max, min, mean of neighbor values | **Identical** â€” same arithmetic, same NA handling |
| **RF model** | Pre-trained | **Untouched** â€” no retraining |
| **RAM footprint** | ~6.46M-element list of integer vectors | ~344K sparse matrix + edge vectors (~22 MB) |
END_RESPONSE>>>
