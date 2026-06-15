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
run_id: Anthropic_F08_T00_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor *topology* is identical across all 28 years. The function pastes cell IDs and years into string keys, performs string matching via named vectors for every single row, and returns row indices into the full panel. This is O(NÃ—T) in both time and memory, and the string operations dominate runtime.

Specifically:

1. **Redundant topology replication.** The rook-neighbor graph has ~344K cells. The lookup rebuilds this for every cell-year combination (344K Ã— 28 = ~9.6M list elements), even though the neighbor *cell IDs* never change year to year.

2. **Expensive string-key lookups.** `paste(..., sep="_")` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called millions of times inside an `lapply` over ~6.46M rows. Named vector lookup in R is O(n) hash probing per call, repeated billions of times in aggregate.

3. **`compute_neighbor_stats` is fine in principle** â€” it's a simple vectorized extraction â€” but it operates on the bloated per-row lookup, so it inherits the cost.

4. **The outer loop** calls this for 5 variables sequentially, each time traversing the 6.46M-entry lookup.

**Estimated cost breakdown:** Building the lookup â‰ˆ 60â€“70 hours; computing stats â‰ˆ 15â€“20 hours. Total â‰ˆ 80â€“90 hours.

---

## Optimization Strategy

**Key insight:** Separate the *static topology* (which cells are neighbors) from the *dynamic data* (year-varying variable values). Compute neighbor statistics per-year using matrix operations on the static graph.

### Steps:

1. **Build the neighbor graph once** as a sparse adjacency matrix (344K Ã— 344K) from `rook_neighbors_unique`. This is a one-time O(cells) operation.

2. **For each variable and each year**, extract the variable vector for that year's cells, then use **sparse matrixâ€“vector multiplication** to compute neighbor sums and counts. From sum and count, derive mean. For max and min, use a grouped operation over the sparse structure.

3. **Sparse matrix multiplication** gives us `sum` and `count` (for mean) in milliseconds per year. For `max` and `min`, we iterate over the sparse matrix column structure â€” still only ~1.37M directed edges per year, not 6.46M rows.

4. **Total work:** 5 variables Ã— 28 years Ã— ~1.37M edges = ~192M operations, versus the original ~6.46M Ã— average_neighbors Ã— string_ops. Expected runtime: **minutes, not days**.

5. **Numerical equivalence:** The sparse-matrix approach computes exactly the same max, min, and mean of the same neighbor values. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build sparse adjacency matrix ONCE from the static neighbor topology
# ==============================================================================
build_sparse_adjacency <- function(id_order, neighbors) {
  # id_order: vector of cell IDs in the order used by the nb object
  # neighbors: spdep nb object (list of integer index vectors)
  n <- length(id_order)
  
  # Build COO (coordinate) triplets
  from_idx <- integer(0)
  to_idx   <- integer(0)
  
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nb_i <- nb_i[nb_i != 0L]
    if (length(nb_i) > 0L) {
      from_idx <- c(from_idx, rep.int(i, length(nb_i)))
      to_idx   <- c(to_idx, nb_i)
    }
  }
  
  # Sparse matrix: W[i,j] = 1 means cell j is a neighbor of cell i
  # So W %*% x gives, for each cell i, the sum of x over its neighbors
  W <- sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = 1,
    dims = c(n, n)
  )
  
  return(W)
}

# ==============================================================================
# STEP 2: Compute neighbor max, min using sparse structure (per year-variable)
# ==============================================================================
compute_neighbor_max_min_sparse <- function(W, vals) {
  # W: dgCMatrix (CSC format), vals: numeric vector aligned to columns/rows
  # For each row i, compute max and min of vals[j] where W[i,j] == 1
  n <- nrow(W)
  nb_max <- rep(NA_real_, n)
  nb_min <- rep(NA_real_, n)
  
  # Convert to dgRMatrix (CSR) for efficient row-wise access
  W_csr <- as(W, "RsparseMatrix")
  
  # Access the CSR slots: @p (row pointers), @j (column indices)
  p <- W_csr@p
  j <- W_csr@j  # 0-based column indices
  
  for (i in seq_len(n)) {
    start <- p[i] + 1L      # R is 1-based
    end   <- p[i + 1L]
    if (end >= start) {
      col_indices <- j[start:end] + 1L  # convert to 1-based
      neighbor_vals <- vals[col_indices]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0L) {
        nb_max[i] <- max(neighbor_vals)
        nb_min[i] <- min(neighbor_vals)
      }
    }
  }
  
  list(nb_max = nb_max, nb_min = nb_min)
}

# ==============================================================================
# STEP 3: Compute neighbor mean using sparse matrix multiplication
# ==============================================================================
compute_neighbor_mean_sparse <- function(W, vals) {
  # Replace NA with 0 for summation, track non-NA counts
  not_na <- as.numeric(!is.na(vals))
  vals_clean <- vals
  vals_clean[is.na(vals_clean)] <- 0
  
  nb_sum   <- as.numeric(W %*% vals_clean)
  nb_count <- as.numeric(W %*% not_na)
  
  nb_mean <- ifelse(nb_count > 0, nb_sum / nb_count, NA_real_)
  return(nb_mean)
}

# ==============================================================================
# STEP 4: Optimized main pipeline
# ==============================================================================
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # --- Convert to data.table for fast grouped operations ---
  dt <- as.data.table(cell_data)
  
  # --- Ensure consistent cell ordering ---
  # Map each cell ID to its position in id_order (the nb object's ordering)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  dt[, cell_pos := id_to_pos[as.character(id)]]
  
  # --- Build sparse adjacency matrix ONCE (static topology) ---
  message("Building sparse adjacency matrix (one-time)...")
  W <- build_sparse_adjacency(id_order, rook_neighbors_unique)
  n_cells <- length(id_order)
  message(sprintf("  Adjacency matrix: %d x %d, %d non-zero entries",
                   nrow(W), ncol(W), nnzero(W)))
  
  # --- Get sorted unique years ---
  years <- sort(unique(dt$year))
  
  # --- Neighbor source variables ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  # --- Pre-allocate result columns ---
  for (var_name in neighbor_source_vars) {
    dt[, paste0("neighbor_max_", var_name) := NA_real_]
    dt[, paste0("neighbor_min_", var_name) := NA_real_]
    dt[, paste0("neighbor_mean_", var_name) := NA_real_]
  }
  
  # --- Process year by year ---
  for (yr in years) {
    message(sprintf("Processing year %d ...", yr))
    
    # Row indices for this year
    yr_rows <- which(dt$year == yr)
    
    # Cell positions for this year's rows (position in id_order)
    yr_cell_pos <- dt$cell_pos[yr_rows]
    
    # Build a full-length vector (length = n_cells) for each variable,
    # indexed by cell position. Cells not present this year get NA.
    for (var_name in neighbor_source_vars) {
      
      # Scatter this year's values into a vector aligned with id_order
      full_vals <- rep(NA_real_, n_cells)
      full_vals[yr_cell_pos] <- dt[[var_name]][yr_rows]
      
      # --- Neighbor mean via sparse mat-vec (very fast) ---
      nb_mean <- compute_neighbor_mean_sparse(W, full_vals)
      
      # --- Neighbor max and min via CSR row traversal ---
      maxmin <- compute_neighbor_max_min_sparse(W, full_vals)
      
      # --- Write results back to the data.table ---
      set(dt, i = yr_rows, j = paste0("neighbor_max_",  var_name), value = maxmin$nb_max[yr_cell_pos])
      set(dt, i = yr_rows, j = paste0("neighbor_min_",  var_name), value = maxmin$nb_min[yr_cell_pos])
      set(dt, i = yr_rows, j = paste0("neighbor_mean_", var_name), value = nb_mean[yr_cell_pos])
    }
  }
  
  # --- Clean up helper column ---
  dt[, cell_pos := NULL]
  
  message("Neighbor feature computation complete.")
  return(dt)
}

# ==============================================================================
# STEP 5: Drop-in replacement for the outer loop
# ==============================================================================

# Original code replaced:
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }

# New code:
cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)

# Then proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Further Optimization: Vectorized Max/Min (Eliminating the R Loop)

The `compute_neighbor_max_min_sparse` function above still uses an R-level `for` loop over 344K cells. On a 16 GB laptop this runs in ~10â€“30 seconds per variable-year (vs. hours before), but we can push it further with a fully vectorized approach:

```r
compute_neighbor_max_min_vectorized <- function(W, vals) {
  # Fully vectorized using the sparse structure directly
  # W is dgCMatrix (CSC). Convert to dgTMatrix for (i,j) triplets.
  W_t <- as(W, "TsparseMatrix")
  
  row_i <- W_t@i + 1L   # 1-based row indices
  col_j <- W_t@j + 1L   # 1-based col indices
  
  # Get neighbor values for every edge
  edge_vals <- vals[col_j]
  
  # Remove edges where neighbor value is NA
  valid <- !is.na(edge_vals)
  row_i_v <- row_i[valid]
  edge_vals_v <- edge_vals[valid]
  
  n <- nrow(W)
  nb_max <- rep(NA_real_, n)
  nb_min <- rep(NA_real_, n)
  
  if (length(row_i_v) > 0L) {
    # Use data.table for fast grouped max/min
    edge_dt <- data.table(row = row_i_v, val = edge_vals_v)
    agg <- edge_dt[, .(mx = max(val), mn = min(val)), by = row]
    nb_max[agg$row] <- agg$mx
    nb_min[agg$row] <- agg$mn
  }
  
  list(nb_max = nb_max, nb_min = nb_min)
}
```

Replace the call in the main pipeline to use this version for maximum speed. This reduces the max/min computation to a single `data.table` grouped aggregation over ~1.37M edges â€” typically under 1 second per variable-year.

---

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~60â€“70 hours | ~2 seconds (sparse matrix) |
| Neighbor stats (5 vars Ã— 28 yrs) | ~15â€“20 hours | ~2â€“5 minutes total |
| Memory (lookup structure) | ~20â€“40 GB (exceeds 16 GB) | ~200 MB (sparse matrix) |
| **Total** | **~80â€“90 hours** | **< 10 minutes** |

**Numerical equivalence:** The max, min, and mean are computed over exactly the same neighbor sets and values. The trained Random Forest model is loaded and used for prediction without modification.
END_RESPONSE>>>
