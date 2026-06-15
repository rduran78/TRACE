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
run_id: Anthropic_F05_T03_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F05_Anthropic_temp_03_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts `data$id[i]` to character and looks it up in `id_to_ref` â€” **6.46M character coercions + named-vector lookups**.
2. Subsets `id_order[neighbors[[ref_idx]]]` â€” fine, but done row-by-row.
3. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” **allocates a new character vector per row** (~4 neighbor keys Ã— 6.46M rows â‰ˆ 25.8M string constructions).
4. Looks up `idx_lookup[neighbor_keys]` â€” **named-vector lookup via string matching on a 6.46M-element named vector**, done 6.46M times.

The `idx_lookup` named vector has 6.46M entries. R's named-vector lookup is **O(N)** internally (hashing helps, but the constant factor on 6.46M names is brutal). Doing this 6.46M times gives effectively **O(NÂ²)** behavior â€” hence the 86+ hour estimate.

### The Broader Pattern

The neighbor lookup is **year-invariant in structure**: cell A's rook neighbors are the same cells every year. The only thing that changes across years is which rows in the data correspond to those neighbor cells. This means:

- The `neighbors` structure (which cell IDs are neighbors) is **static across years**.
- The mapping from `(cell_id, year)` â†’ row index is a **simple arithmetic mapping** if the data is sorted, or a one-time hash table build.

Yet the current code reconstructs string keys and performs string-based lookups **per row**, completely ignoring this separable structure.

### `compute_neighbor_stats` Is Efficient â€” But Fed by an Expensive Lookup

`compute_neighbor_stats` itself is fine (simple numeric indexing). The entire bottleneck is `build_neighbor_lookup`.

---

## Optimization Strategy

### Key Insight: Separate the Spatial Structure from the Temporal Indexing

Since rook neighbors are time-invariant, we can:

1. **Build a row-index matrix** of shape `(n_cells, n_years)` mapping `(cell_position, year_position)` â†’ row number in `data`. This is a one-time O(N) operation.
2. **For each cell**, its neighbor row-indices in year `t` are simply `row_matrix[neighbor_positions, year_position]`. No strings, no hashing â€” pure integer indexing.
3. **Vectorize the neighbor stats** using matrix operations or `data.table` grouping instead of per-row `lapply`.

### Further: Vectorize Stats Computation with Sparse Matrix Multiplication

The "mean/max/min of neighbor values" can be computed as:
- **Mean**: sparse adjacency matrix Ã— value vector (matrix-vector multiply), divided by neighbor counts.
- **Max/Min**: row-wise sparse operations.

This replaces the entire `lapply` over 6.46M rows with a single sparse matrix operation per variable â€” **O(nnz)** where nnz â‰ˆ 1.37M edges Ã— 28 years â‰ˆ 38.5M, done in optimized C code.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# =============================================================================
# Prerequisites: data.table, Matrix, spdep (already available in your pipeline)
# 
# Inputs expected:
#   cell_data            â€” data.frame/data.table with columns: id, year, and
#                          the 5 neighbor source variables
#   id_order             â€” integer vector of cell IDs in the order matching
#                          rook_neighbors_unique
#   rook_neighbors_unique â€” nb object (list of integer index vectors)
#
# Preserves: all original column values, trained RF model (untouched),
#            and the original numerical estimand (max, min, mean of
#            non-NA neighbor values per cell-year).
# =============================================================================

library(data.table)
library(Matrix)

optimized_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # ------------------------------------------------------------------
  # 0. Convert to data.table for speed; keep original row order
  # ------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_order := .I]
  
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)
  
  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(dt)))
  
  # ------------------------------------------------------------------
  # 1. Build integer mappings (one-time, no strings)
  # ------------------------------------------------------------------
  # Map cell id -> position in id_order (1..n_cells)
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_len(n_cells)
  
  # Map year -> position (1..n_years)
  year_to_pos <- setNames(seq_len(n_years), as.character(years))
  
  # ------------------------------------------------------------------
  # 2. Build row-index matrix: row_idx_mat[cell_pos, year_pos] = row in dt
  #    This replaces ALL string-key lookups. O(N) one-time cost.
  # ------------------------------------------------------------------
  cell_pos_vec <- id_to_pos[dt$id]
  year_pos_vec <- year_to_pos[as.character(dt$year)]
  
  row_idx_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_idx_mat[cbind(cell_pos_vec, year_pos_vec)] <- seq_len(nrow(dt))
  
  cat("Row-index matrix built.\n")
  
  # ------------------------------------------------------------------
  # 3. Build sparse adjacency matrix from nb object (one-time)
  #    A[i,j] = 1 if cell j is a rook neighbor of cell i
  #    Dimensions: n_cells x n_cells
  # ------------------------------------------------------------------
  # Extract COO (coordinate) representation from nb object
  from_list <- lapply(seq_len(n_cells), function(i) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 1L && nb[1] == 0L) {
      # spdep convention: 0 means no neighbors
      return(data.table(i = integer(0), j = integer(0)))
    }
    data.table(i = rep(i, length(nb)), j = nb)
  })
  edges <- rbindlist(from_list)
  
  # Sparse adjacency matrix (n_cells x n_cells)
  adj <- sparseMatrix(
    i = edges$i,
    j = edges$j,
    x = 1,
    dims = c(n_cells, n_cells)
  )
  
  # Neighbor counts per cell (used for mean calculation)
  # This is the structural count; NA handling adjusts it per variable/year.
  cat(sprintf("Adjacency matrix: %d cells, %d directed edges.\n",
              n_cells, length(edges$i)))
  rm(edges, from_list)
  
  # ------------------------------------------------------------------
  # 4. For each variable, compute neighbor max, min, mean per cell-year
  #    using sparse matrix operations â€” one year-slice at a time to

  #    handle NAs correctly and compute max/min (which aren't linear).
  # ------------------------------------------------------------------
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  # Pre-allocate result columns
  for (var_name in neighbor_source_vars) {
    dt[, paste0(var_name, "_neighbor_max")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_min")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_mean") := NA_real_]
  }
  
  # We process year-by-year. For each year, we have a vector of length
  # n_cells (some NA where cells don't appear that year).
  # For MEAN: sparse mat-vec multiply handles the sum; we just need
  #           the count of non-NA neighbors.
  # For MAX/MIN: we use a trick with the sparse matrix.
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing variable: %s\n", var_name))
    
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")
    
    for (yi in seq_len(n_years)) {
      # Row indices in dt for this year
      row_indices <- row_idx_mat[, yi]  # length n_cells, some NA
      
      # Build value vector for this year (length n_cells)
      vals <- rep(NA_real_, n_cells)
      present <- !is.na(row_indices)
      vals[present] <- dt[[var_name]][row_indices[present]]
      
      # --- MEAN via sparse matrix-vector multiply ---
      # Replace NA with 0 for summation, track non-NA mask
      not_na <- !is.na(vals)
      vals_zero <- ifelse(not_na, vals, 0)
      
      # Sum of neighbor values (adj %*% vals_zero)[i] = sum of vals over neighbors of i
      neighbor_sum   <- as.numeric(adj %*% vals_zero)
      # Count of non-NA neighbors
      neighbor_count <- as.numeric(adj %*% as.numeric(not_na))
      
      neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
      
      # --- MAX and MIN via sparse iteration ---
      # For max: replace NA with -Inf, multiply, then fix up.
      # But matrix multiply gives SUM, not MAX. We need a different approach.
      #
      # Efficient approach: use the adjacency list structure we already have
      # (rook_neighbors_unique) but vectorized per year-slice.
      # Since we have vals (length n_cells), we iterate over cells using
      # the nb list â€” but this is only 344K iterations (not 6.46M).
      
      neighbor_max <- rep(NA_real_, n_cells)
      neighbor_min <- rep(NA_real_, n_cells)
      
      # Vectorized approach: expand neighbor pairs, compute, then aggregate
      # We already have adj in sparse format. Extract its structure once
      # (outside the year loop for efficiency â€” we'll restructure below).
      # For now, use the nb list directly â€” 344K iterations is fast.
      
      for (ci in seq_len(n_cells)) {
        nb_idx <- rook_neighbors_unique[[ci]]
        if (length(nb_idx) == 1L && nb_idx[1] == 0L) next
        nb_vals <- vals[nb_idx]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) == 0L) next
        neighbor_max[ci] <- max(nb_vals)
        neighbor_min[ci] <- min(nb_vals)
      }
      
      # Write results back to dt for rows present this year
      target_rows <- row_indices[present]
      cell_positions <- which(present)
      
      set(dt, i = target_rows, j = col_max,  value = neighbor_max[cell_positions])
      set(dt, i = target_rows, j = col_min,  value = neighbor_min[cell_positions])
      set(dt, i = target_rows, j = col_mean, value = neighbor_mean[cell_positions])
    }
  }
  
  # ------------------------------------------------------------------
  # 5. Restore original order and return
  # ------------------------------------------------------------------
  setorder(dt, .row_order)
  dt[, .row_order := NULL]
  
  return(as.data.frame(dt))
}
```

**Wait** â€” the inner `for (ci in seq_len(n_cells))` loop over 344K cells Ã— 28 years Ã— 5 variables is still ~48M R-level loop iterations for max/min. That's much better than 6.46M Ã— 5 but still suboptimal. Let's fully vectorize max/min using the COO expansion approach:

```r
# =============================================================================
# FULLY VECTORIZED VERSION
# =============================================================================

library(data.table)
library(Matrix)

optimized_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {
  
  dt <- as.data.table(cell_data)
  dt[, .row_order := .I]
  
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)
  
  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(dt)))
  
  # ---- Integer mappings ----
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_len(n_cells)
  
  year_to_pos <- setNames(seq_len(n_years), as.character(years))
  
  # ---- Row-index matrix ----
  cell_pos_vec <- id_to_pos[dt$id]
  year_pos_vec <- year_to_pos[as.character(dt$year)]
  
  row_idx_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_idx_mat[cbind(cell_pos_vec, year_pos_vec)] <- seq_len(nrow(dt))
  
  # ---- Build edge table from nb object (one-time) ----
  from_vec <- integer(0)
  to_vec   <- integer(0)
  for (ci in seq_len(n_cells)) {
    nb <- rook_neighbors_unique[[ci]]
    if (length(nb) == 1L && nb[1] == 0L) next
    from_vec <- c(from_vec, rep(ci, length(nb)))
    to_vec   <- c(to_vec, nb)
  }
  # More memory-efficient construction:
  edge_dt <- data.table(from = from_vec, to = to_vec)
  rm(from_vec, to_vec)
  n_edges <- nrow(edge_dt)
  cat(sprintf("Edge table: %d directed edges.\n", n_edges))
  
  # ---- Sparse adjacency for mean (sum + count) ----
  adj <- sparseMatrix(
    i = edge_dt$from,
    j = edge_dt$to,
    x = 1,
    dims = c(n_cells, n_cells)
  )
  
  # ---- Pre-allocate output columns ----
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  for (var_name in neighbor_source_vars) {
    dt[, paste0(var_name, "_neighbor_max")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_min")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_mean") := NA_real_]
  }
  
  # ---- Process each variable Ã— year ----
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Variable: %s ...\n", var_name))
    
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")
    
    for (yi in seq_len(n_years)) {
      
      # Value vector for this year (length n_cells, NA where absent)
      row_indices <- row_idx_mat[, yi]
      present <- !is.na(row_indices)
      
      vals <- rep(NA_real_, n_cells)
      vals[present] <- dt[[var_name]][row_indices[present]]
      
      # ---- MEAN via sparse mat-vec ----
      not_na     <- !is.na(vals)
      vals_zero  <- ifelse(not_na, vals, 0)
      nb_sum     <- as.numeric(adj %*% vals_zero)
      nb_count   <- as.numeric(adj %*% as.numeric(not_na))
      nb_mean    <- ifelse(nb_count > 0, nb_sum / nb_count, NA_real_)
      
      # ---- MAX / MIN via edge-table vectorization ----
      # Get neighbor values for all edges at once
      nb_vals <- vals[edge_dt$to]  # length n_edges
      
      # Build a data.table of (from_cell, neighbor_value), drop NAs, aggregate
      agg_dt <- data.table(from = edge_dt$from, val = nb_vals)
      agg_dt <- agg_dt[!is.na(val)]
      
      if (nrow(agg_dt) > 0) {
        agg <- agg_dt[, .(vmax = max(val), vmin = min(val)), by = from]
        
        nb_max <- rep(NA_real_, n_cells)
        nb_min <- rep(NA_real_, n_cells)
        nb_max[agg$from] <- agg$vmax
        nb_min[agg$from] <- agg$vmin
      } else {
        nb_max <- rep(NA_real_, n_cells)
        nb_min <- rep(NA_real_, n_cells)
      }
      
      # ---- Write back to dt ----
      target_rows    <- row_indices[present]
      cell_positions <- which(present)
      
      set(dt, i = target_rows, j = col_max,  value = nb_max[cell_positions])
      set(dt, i = target_rows, j = col_min,  value = nb_min[cell_positions])
      set(dt, i = target_rows, j = col_mean, value = nb_mean[cell_positions])
    }
  }
  
  setorder(dt, .row_order)
  dt[, .row_order := NULL]
  return(as.data.frame(dt))
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================
# cell_data <- optimized_neighbor_features(cell_data, id_order, rook_neighbors_unique)
#
# The trained Random Forest model is untouched â€” only the feature-construction
# pipeline is replaced. Output columns have identical names and identical
# numerical values (max, min, mean of non-NA rook-neighbor values per cell-year).
```

---

## Complexity Comparison

| Aspect | Original | Optimized |
|---|---|---|
| **String key constructions** | ~6.46M Ã— ~4 neighbors = **~25.8M** `paste()` calls inside `lapply`, plus 6.46M keys for `idx_lookup` | **Zero** â€” pure integer indexing |
| **Named-vector lookups** | 6.46M lookups into a 6.46M-element named vector (**O(N) hash probe Ã— N**) | **Zero** â€” replaced by matrix indexing `row_idx_mat[cell, year]` |
| **R-level loop iterations** | **6.46M** (one per cell-year row) | **0** for mean; max/min use `data.table` grouped aggregation over ~1.37M edges Ã— 28 years = **~38.4M rows** but processed in **vectorized C** via `data.table` |
| **Sparse mat-vec for mean** | N/A | **28 years Ã— 5 vars = 140** sparse matrix-vector multiplies, each O(nnz) â‰ˆ O(1.37M) in compiled code |
| **Estimated wall time** | **86+ hours** | **~5â€“15 minutes** (sparse mat-vec: ~140 Ã— <0.5s; data.table agg: ~140 Ã— ~1s; overhead ~1 min) |
| **Peak RAM** | ~6.46M-element named character vector + lapply overhead | `row_idx_mat`: 344K Ã— 28 Ã— 4B â‰ˆ 38 MB; sparse `adj`: ~5.5 MB; `edge_dt`: ~22 MB; `agg_dt` per slice: ~22 MB. **Total overhead: ~100â€“200 MB** â€” well within 16 GB |

## Why This Preserves the Original Estimand

For each cell-year `(c, t)`:
- **Original**: finds all rook neighbors of `c`, looks up their row in year `t`, extracts non-NA values of `var`, computes `max`, `min`, `mean`.
- **Optimized**: uses the same `rook_neighbors_unique` structure, the same year, the same variable column, drops NAs identically, and computes the same `max`, `min`, `mean`. The sparse matrix mean uses `sum / count_non_na`, which is algebraically identical. The `data.table` grouped `max`/`min` are identical to R's `max`/`min` on the same non-NA subset.

The trained Random Forest model is never touched â€” only the upstream feature matrix construction is accelerated.
END_RESPONSE>>>
