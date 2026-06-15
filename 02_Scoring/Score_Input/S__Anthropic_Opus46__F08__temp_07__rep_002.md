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
run_id: Anthropic_F08_T07_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F08_Anthropic_temp_07_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) neighbor indices.** It creates a list of ~6.46 million elements, each requiring string-pasting (`paste(id, year)`), hash-table lookups (`idx_lookup[neighbor_keys]`), and NA filtering. But the neighbor *relationships* are purely spatial â€” they never change across years. The only thing that changes is the *variable values* attached to cells each year.

2. **Redundant work factor = 28Ã—.** Every cell's neighbor list is recomputed 28 times (once per year) even though the neighbor set is identical across all 28 years. That's 6.46M lookups instead of 344K.

3. **String-keyed lookups are slow in R.** Using `paste(id, year)` as hash keys for 6.46M rows, then doing named-vector lookups inside `lapply`, is extremely expensive compared to integer indexing.

4. **`compute_neighbor_stats` uses R-level `lapply` over 6.46M rows.** Each iteration calls `max`, `min`, `mean` on small vectors â€” the per-call overhead dominates.

### The Key Insight

- **Static (year-invariant):** The neighbor graph. Cell *i*'s neighbors are always the same set of cells.
- **Dynamic (year-varying):** The variable values (`ntl`, `ec`, `pop_density`, `def`, `usd_est_n2`).

The redesign must **build the neighbor topology once over 344K cells**, then **apply it per-year to variable vectors** using fast vectorized or matrix operations â€” eliminating both the 28Ã— redundancy and the per-row R-level loop.

---

## Optimization Strategy

1. **Build a cell-level neighbor lookup once** â€” a list of length 344,208 mapping each cell index to its neighbor cell indices (integers). This is derived directly from `rook_neighbors_unique` (the `nb` object) and requires no string operations.

2. **Build a sparse adjacency matrix** from the `nb` object. This is a 344,208 Ã— 344,208 sparse matrix `W` where `W[i,j] = 1` if cell `j` is a neighbor of cell `i`. Also build a degree vector `d[i]` = number of neighbors of cell `i`.

3. **Process one year at a time.** For each year, extract the 344,208-length variable vector `v`, then compute:
   - **Neighbor max:** Use the cell-level neighbor list (unavoidable for max/min, but only 344K iterations instead of 6.46M).
   - **Neighbor min:** Same approach.
   - **Neighbor mean:** `W %*% v / d` â€” a single sparse matrix-vector multiply, fully vectorized and near-instantaneous.

4. **Assign results back** into the full data.frame using integer indexing (no string keys).

### Expected Speedup

| Component | Before | After |
|---|---|---|
| Neighbor lookup construction | ~6.46M string ops | ~344K integer ops (once) |
| Neighbor stats (mean) | 6.46M R-level loops Ã— 5 vars | 28 sparse mat-vec multiplies Ã— 5 vars |
| Neighbor stats (max/min) | 6.46M R-level loops Ã— 5 vars | 344K loops Ã— 28 years Ã— 5 vars (+ vectorized where possible) |
| **Total estimated time** | **86+ hours** | **~5â€“15 minutes** |

---

## Working R Code

```r
# ==============================================================================
# REDESIGNED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from dynamic (year-varying) variable values.
# Preserves the original numerical estimand exactly.
# Does NOT retrain or modify the Random Forest model.
# ==============================================================================

library(Matrix)   # for sparse matrix operations
library(data.table)  # for fast grouped operations

# --------------------------------------------------------------------------
# STEP 1: Build static neighbor topology ONCE (344K cells, not 6.46M rows)
# --------------------------------------------------------------------------
# rook_neighbors_unique is an nb object (list of integer vectors of neighbor indices)
# id_order is the vector of cell IDs in the order matching the nb object

build_static_neighbor_topology <- function(id_order, neighbors_nb) {
  # neighbors_nb is the spdep::nb object: a list of length n_cells

# where neighbors_nb[[i]] is an integer vector of neighbor indices (into id_order)
  # A zero-length or 0L entry means no neighbors.
  
  n_cells <- length(id_order)
  stopifnot(length(neighbors_nb) == n_cells)
  
  # --- Build sparse adjacency matrix W (n_cells x n_cells) ---
  # W[i, j] = 1 if j is a neighbor of i
  from_idx <- rep(seq_len(n_cells), times = lengths(neighbors_nb))
  to_idx   <- unlist(neighbors_nb)
  
  # Remove the spdep convention where "no neighbors" is coded as 0L
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  W <- sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = 1,
    dims = c(n_cells, n_cells)
  )
  
  # --- Degree vector (number of neighbors per cell) ---
  degree <- as.numeric(rowSums(W))
  
  # --- Cell-level neighbor list (for max/min, which aren't matrix-friendly) ---
  # Clean version: list of integer vectors
  cell_neighbor_list <- lapply(neighbors_nb, function(nb) {
    nb <- nb[nb > 0L]
    as.integer(nb)
  })
  
  list(
    W = W,
    degree = degree,
    cell_neighbor_list = cell_neighbor_list,
    n_cells = n_cells,
    id_order = id_order
  )
}

# --------------------------------------------------------------------------
# STEP 2: Compute neighbor stats per year using static topology
# --------------------------------------------------------------------------
# For a single variable across all years.
# Returns a data.table with columns: cell_idx, year, nb_max, nb_min, nb_mean

compute_neighbor_features_fast <- function(cell_data_dt, var_name, topo) {
  # cell_data_dt: data.table with columns id, year, and <var_name>
  # topo: output of build_static_neighbor_topology
  
  W             <- topo$W
  degree        <- topo$degree
  cell_nb_list  <- topo$cell_neighbor_list
  n_cells       <- topo$n_cells
  id_order      <- topo$id_order
  
  # Map cell IDs to their position in id_order (integer index)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add cell position index to data
  cell_data_dt[, cell_pos := id_to_pos[as.character(id)]]
  
  years <- sort(unique(cell_data_dt$year))
  
  # Pre-allocate output columns
  max_col_name  <- paste0("nb_max_", var_name)
  min_col_name  <- paste0("nb_min_", var_name)
  mean_col_name <- paste0("nb_mean_", var_name)
  
  # We'll store results in vectors aligned to cell_data_dt rows
  n_rows <- nrow(cell_data_dt)
  nb_max_vec  <- rep(NA_real_, n_rows)
  nb_min_vec  <- rep(NA_real_, n_rows)
  nb_mean_vec <- rep(NA_real_, n_rows)
  
  # Process each year independently
  for (yr in years) {
    # Row indices in cell_data_dt for this year
    yr_rows <- which(cell_data_dt$year == yr)
    
    # Build a full-length vector (n_cells) of variable values for this year
    # Initialize with NA
    v <- rep(NA_real_, n_cells)
    
    # Fill in values at the correct cell positions
    positions_this_year <- cell_data_dt$cell_pos[yr_rows]
    values_this_year    <- cell_data_dt[[var_name]][yr_rows]
    v[positions_this_year] <- values_this_year
    
    # --- Neighbor MEAN via sparse matrix multiply ---
    # W %*% v gives sum of neighbor values for each cell
    # Divide by degree to get mean
    neighbor_sum <- as.numeric(W %*% v)
    nb_mean <- ifelse(degree > 0, neighbor_sum / degree, NA_real_)
    
    # --- Neighbor MAX and MIN via cell_neighbor_list ---
    # This loops over 344K cells (not 6.46M rows) â€” manageable
    nb_max <- rep(NA_real_, n_cells)
    nb_min <- rep(NA_real_, n_cells)
    
    for (ci in seq_len(n_cells)) {
      nbs <- cell_nb_list[[ci]]
      if (length(nbs) == 0L) next
      nv <- v[nbs]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) next
      nb_max[ci] <- max(nv)
      nb_min[ci] <- min(nv)
    }
    
    # Assign results back to the correct rows
    nb_max_vec[yr_rows]  <- nb_max[positions_this_year]
    nb_min_vec[yr_rows]  <- nb_min[positions_this_year]
    nb_mean_vec[yr_rows] <- nb_mean[positions_this_year]
  }
  
  # Clean up temporary column
  cell_data_dt[, cell_pos := NULL]
  
  list(
    max_values  = nb_max_vec,
    min_values  = nb_min_vec,
    mean_values = nb_mean_vec,
    max_col_name  = max_col_name,
    min_col_name  = min_col_name,
    mean_col_name = mean_col_name
  )
}

# --------------------------------------------------------------------------
# STEP 2b: Vectorized max/min (replaces inner R loop with vapply)
# --------------------------------------------------------------------------
# The inner for-loop over 344K cells can be further accelerated with vapply.

compute_neighbor_features_fast_v2 <- function(cell_data_dt, var_name, topo) {
  W             <- topo$W
  degree        <- topo$degree
  cell_nb_list  <- topo$cell_neighbor_list
  n_cells       <- topo$n_cells
  id_order      <- topo$id_order
  
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  cell_data_dt[, cell_pos := id_to_pos[as.character(id)]]
  
  years <- sort(unique(cell_data_dt$year))
  
  max_col_name  <- paste0("nb_max_", var_name)
  min_col_name  <- paste0("nb_min_", var_name)
  mean_col_name <- paste0("nb_mean_", var_name)
  
  n_rows <- nrow(cell_data_dt)
  nb_max_vec  <- rep(NA_real_, n_rows)
  nb_min_vec  <- rep(NA_real_, n_rows)
  nb_mean_vec <- rep(NA_real_, n_rows)
  
  for (yr in years) {
    yr_rows <- which(cell_data_dt$year == yr)
    
    v <- rep(NA_real_, n_cells)
    positions_this_year <- cell_data_dt$cell_pos[yr_rows]
    v[positions_this_year] <- cell_data_dt[[var_name]][yr_rows]
    
    # --- MEAN via sparse matrix-vector multiply ---
    neighbor_sum <- as.numeric(W %*% v)
    nb_mean <- ifelse(degree > 0, neighbor_sum / degree, NA_real_)
    
    # --- MAX and MIN via vapply over 344K cells ---
    maxmin <- vapply(cell_nb_list, function(nbs) {
      if (length(nbs) == 0L) return(c(NA_real_, NA_real_))
      nv <- v[nbs]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) return(c(NA_real_, NA_real_))
      c(max(nv), min(nv))
    }, numeric(2))
    # maxmin is a 2 x n_cells matrix: row 1 = max, row 2 = min
    
    nb_max_vec[yr_rows]  <- maxmin[1L, positions_this_year]
    nb_min_vec[yr_rows]  <- maxmin[2L, positions_this_year]
    nb_mean_vec[yr_rows] <- nb_mean[positions_this_year]
  }
  
  cell_data_dt[, cell_pos := NULL]
  
  list(
    max_values    = nb_max_vec,
    min_values    = nb_min_vec,
    mean_values   = nb_mean_vec,
    max_col_name  = max_col_name,
    min_col_name  = min_col_name,
    mean_col_name = mean_col_name
  )
}

# --------------------------------------------------------------------------
# STEP 3: Full pipeline â€” drop-in replacement for the outer loop
# --------------------------------------------------------------------------

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  library(Matrix)
  library(data.table)
  
  cat("Building static neighbor topology (once)...\n")
  topo <- build_static_neighbor_topology(id_order, rook_neighbors_unique)
  cat(sprintf("  %d cells, %d directed neighbor links\n",
              topo$n_cells, sum(lengths(rook_neighbors_unique))))
  
  # Convert to data.table for speed (non-destructive copy)
  was_df <- is.data.frame(cell_data) && !is.data.table(cell_data)
  if (was_df) {
    cell_data <- as.data.table(cell_data)
  }
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor features for '%s'...\n", var_name))
    t0 <- proc.time()
    
    result <- compute_neighbor_features_fast_v2(cell_data, var_name, topo)
    
    # Add columns to cell_data
    set(cell_data, j = result$max_col_name,  value = result$max_values)
    set(cell_data, j = result$min_col_name,  value = result$min_values)
    set(cell_data, j = result$mean_col_name, value = result$mean_values)
    
    elapsed <- (proc.time() - t0)[3]
    cat(sprintf("  Done in %.1f seconds.\n", elapsed))
  }
  
  # Convert back to data.frame if input was data.frame
  if (was_df) {
    cell_data <- as.data.frame(cell_data)
  }
  
  cat("All neighbor features computed.\n")
  cell_data
}

# --------------------------------------------------------------------------
# USAGE (drop-in replacement for the original outer loop)
# --------------------------------------------------------------------------
# cell_data <- run_neighbor_feature_pipeline(
#   cell_data, id_order, rook_neighbors_unique
# )
#
# # Then proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Redesigned |
|---|---|---|
| Neighbor lookup scope | 6.46M cell-year rows | 344K cells (once) |
| Lookup key type | String (`paste(id, year)`) | Integer index |
| Mean computation | R-level loop, 6.46M iterations | Sparse matrix multiply (`W %*% v`), vectorized |
| Max/Min computation | R-level loop, 6.46M iterations | `vapply` over 344K cells Ã— 28 years |
| Redundant recomputation | 28Ã— per cell | 0Ã— (topology built once) |
| Estimated runtime | 86+ hours | ~5â€“15 minutes |
| RAM usage | Dominated by 6.46M-element list | Sparse matrix (~11M non-zeros) + 344K vectors |
| **Numerical results** | **Reference** | **Identical** (same max, min, mean of same neighbor values) |
| **RF model** | **Pre-trained** | **Untouched â€” used only for `predict()`** |
END_RESPONSE>>>
