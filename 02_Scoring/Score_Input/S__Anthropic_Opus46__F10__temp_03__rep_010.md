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
run_id: Anthropic_F10_T03_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query on average due to hashing overhead at scale, and doing this 6.46M times with string concatenation is extremely expensive.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a serial `lapply`**, extracting subsets of a numeric vector and computing `max/min/mean`. The per-element R function-call overhead dominates when the inner work is trivial (typically 2â€“4 neighbors per cell).

3. **The topology is year-invariant but the lookup is rebuilt as if it's year-specific.** Rook neighbors don't change across years. The spatial adjacency graph has ~344K nodes and ~1.37M directed edges. The panel simply replicates this graph 28 times. The current code entangles spatial topology with temporal indexing, preventing vectorized computation.

**Root cause:** The algorithm is O(rows Ã— avg_neighbors) in R-interpreted loop iterations rather than in vectorized C-level operations. With ~6.46M rows Ã— ~4 neighbors Ã— 5 variables, that's ~129M R-level element accesses wrapped in `lapply`.

## Optimization Strategy

1. **Build the sparse adjacency structure once as a CSR (Compressed Sparse Row) representation** using integer vectors â€” no strings, no named lookups. Use the `Matrix` package's `dgCMatrix`/`dgRMatrix` or manual CSR vectors.

2. **Exploit year-invariance:** The neighbor graph is purely spatial. For each year-slice, the same spatial adjacency applies. Instead of building a 6.46M-row lookup, build a 344K-node adjacency and apply it 28 times.

3. **Vectorized sparse matrixâ€“vector multiplication for `mean`:** If `A` is the row-normalized adjacency matrix (each row sums to 1 over its neighbors), then `A %*% x = neighbor_mean(x)`. This is a single sparse matrixâ€“vector multiply â€” microseconds in C via `Matrix`.

4. **For `max` and `min`:** Use the CSR structure with a tight `vapply` over only 344K nodes (not 6.46M rows), or use `data.table` grouping on an edge-list. The key insight: group-by on the ~1.37M edge list is far faster than 6.46M `lapply` calls.

5. **Process year-by-year** to keep memory bounded (344K rows per year-slice fits easily in RAM).

6. **Use `data.table` for the final join** back to the panel.

**Expected speedup:** From ~86 hours to ~2â€“5 minutes. The dominant operation becomes 28 years Ã— 5 variables Ã— 3 sparse operations â‰ˆ 420 vectorized operations on 344K-length vectors.

## Optimized R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build the sparse adjacency matrix ONCE (spatial topology only)
# ==============================================================================
# Inputs:
#   id_order             â€” integer vector of length 344,208 (cell IDs in order)
#   rook_neighbors_unique â€” spdep nb object (list of length 344,208)
#
# We build:
#   A_binary  â€” 344208 x 344208 sparse binary adjacency matrix (dgCMatrix)
#   A_norm    â€” row-normalized version (for computing means)
#   edge_dt   â€” data.table with columns (from_idx, to_idx) for max/min

build_spatial_adjacency <- function(id_order, rook_neighbors_unique) {
  n <- length(id_order)
  
  # Build COO (coordinate) representation
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  
  for (i in seq_len(n)) {
    nb <- rook_neighbors_unique[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nb <- nb[nb != 0L]
    if (length(nb) > 0L) {
      from_list[[i]] <- rep.int(i, length(nb))
      to_list[[i]]   <- nb
    }
  }
  
  from_idx <- unlist(from_list, use.names = FALSE)
  to_idx   <- unlist(to_list, use.names = FALSE)
  
  # Binary adjacency matrix (rows = focal node, cols = neighbor node)
  A_binary <- sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = rep(1, length(from_idx)),
    dims = c(n, n)
  )
  
  # Row-normalized adjacency (for mean computation)
  row_sums <- rowSums(A_binary)
  row_sums[row_sums == 0] <- NA_real_  # will produce NA for isolated nodes
  # Diagonal matrix of inverse row sums
  D_inv <- Diagonal(x = ifelse(is.na(row_sums), 0, 1 / row_sums))
  A_norm <- D_inv %*% A_binary
  
  # Edge data.table for max/min (much faster than per-node lapply)
  edge_dt <- data.table(from_idx = from_idx, to_idx = to_idx)
  
  # Degree vector (number of neighbors per node); 0 means isolated
  degree <- as.integer(row_sums)
  degree[is.na(degree)] <- 0L
  
  list(
    A_binary = A_binary,
    A_norm   = A_norm,
    edge_dt  = edge_dt,
    degree   = degree,
    n        = n
  )
}

# ==============================================================================
# STEP 2: Compute neighbor max, min, mean for one variable and one year-slice
# ==============================================================================
# Uses:
#   - Sparse mat-vec for mean
#   - data.table edge-list grouping for max and min

compute_neighbor_stats_fast <- function(values, adj) {
  # values: numeric vector of length adj$n (one per spatial node, one year)
  # adj:    output of build_spatial_adjacency
  
  n <- adj$n
  
  # --- MEAN via sparse matrix-vector multiply ---
  # A_norm %*% values gives neighbor mean; isolated nodes get 0, fix to NA
  nb_mean <- as.numeric(adj$A_norm %*% values)
  nb_mean[adj$degree == 0L] <- NA_real_
  # Handle case where all neighbors are NA: A_norm %*% x will give 0 if 

  # neighbor values are NA (treated as 0 in matrix multiply). We need to 
  # correct this.
  # Count non-NA neighbors per node:
  not_na <- as.numeric(!is.na(values))
  non_na_count <- as.numeric(adj$A_binary %*% not_na)
  
  # Compute sum of non-NA neighbor values
  values_zero <- values
  values_zero[is.na(values_zero)] <- 0
  nb_sum <- as.numeric(adj$A_binary %*% values_zero)
  
  # Corrected mean
  nb_mean <- ifelse(non_na_count > 0, nb_sum / non_na_count, NA_real_)
  nb_mean[adj$degree == 0L] <- NA_real_
  
  # --- MAX and MIN via data.table edge-list grouping ---
  edge_dt <- adj$edge_dt
  # Attach neighbor values to edges
  edge_dt[, val := values[to_idx]]
  
  # Group by focal node, compute max and min (na.rm = TRUE)
  agg <- edge_dt[!is.na(val), .(
    nb_max = max(val),
    nb_min = min(val)
  ), by = from_idx]
  
  # Initialize output
  nb_max <- rep(NA_real_, n)
  nb_min <- rep(NA_real_, n)
  
  nb_max[agg$from_idx] <- agg$nb_max
  nb_min[agg$from_idx] <- agg$nb_min
  
  list(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}

# ==============================================================================
# STEP 3: Main pipeline â€” process all variables across all years
# ==============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # Convert to data.table for speed (non-destructive if already data.table)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # Ensure consistent spatial ordering
  # Create a mapping from cell ID to spatial index (1..N)
  id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Build adjacency ONCE
  message("Building spatial adjacency structure...")
  adj <- build_spatial_adjacency(id_order, rook_neighbors_unique)
  message(sprintf("  Nodes: %d, Edges: %d", adj$n, nrow(adj$edge_dt)))
  
  # Pre-allocate the edge_dt val column (avoids repeated allocation)
  adj$edge_dt[, val := NA_real_]
  
  # Source variables
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  # Get unique years, sorted
  years <- sort(unique(cell_data$year))
  message(sprintf("Processing %d years x %d variables...", length(years), length(neighbor_source_vars)))
  
  # Pre-allocate output columns
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")
    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)
  }
  
  # Add spatial index column for fast subsetting
  cell_data[, spatial_idx := id_to_spatial_idx[as.character(id)]]
  
  # Process year by year
  for (yr in years) {
    message(sprintf("  Year %d...", yr))
    
    # Row indices for this year
    yr_rows <- which(cell_data$year == yr)
    
    # Spatial indices for these rows (which spatial nodes are present this year)
    sp_idx <- cell_data$spatial_idx[yr_rows]
    
    for (var_name in neighbor_source_vars) {
      # Build a full-length spatial vector (NA for missing nodes)
      full_vals <- rep(NA_real_, adj$n)
      full_vals[sp_idx] <- cell_data[[var_name]][yr_rows]
      
      # Compute neighbor stats
      stats <- compute_neighbor_stats_fast(full_vals, adj)
      
      # Write results back to the correct rows
      col_max  <- paste0(var_name, "_neighbor_max")
      col_min  <- paste0(var_name, "_neighbor_min")
      col_mean <- paste0(var_name, "_neighbor_mean")
      
      set(cell_data, i = yr_rows, j = col_max,  value = stats$nb_max[sp_idx])
      set(cell_data, i = yr_rows, j = col_min,  value = stats$nb_min[sp_idx])
      set(cell_data, i = yr_rows, j = col_mean, value = stats$nb_mean[sp_idx])
    }
  }
  
  # Clean up helper column
  cell_data[, spatial_idx := NULL]
  
  message("Neighbor feature computation complete.")
  cell_data
}

# ==============================================================================
# STEP 4: Execute and predict with pre-trained Random Forest
# ==============================================================================

# --- Run the optimized pipeline ---
cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)

# --- Predict using the EXISTING pre-trained Random Forest (no retraining) ---
# rf_model is already loaded in the environment
cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string-paste + named-vector lookups | One-time sparse matrix build (344K nodes) |
| **Mean computation** | 6.46M `lapply` calls with subsetting | Sparse matrixâ€“vector multiply (C-level BLAS) |
| **Max/Min computation** | 6.46M `lapply` calls | `data.table` group-by on 1.37M edge list |
| **Topology reuse** | Rebuilt implicitly per row | Built once, reused 28 Ã— 5 = 140 times |
| **Memory** | ~6.46M-element list of integer vectors | ~1.37M-edge sparse matrix + edge table (~50 MB) |
| **Estimated runtime** | ~86+ hours | **~2â€“5 minutes** |
| **Numerical equivalence** | Baseline | âœ… Preserved (same max/min/mean, same NA handling) |
| **RF model** | Pre-trained | âœ… Unchanged, predict-only |

**Key insight:** The rook-neighbor graph is a **time-invariant spatial topology**. By separating the spatial graph structure from the temporal panel structure, we reduce the problem from 6.46M interpreted-R iterations to 140 vectorized sparse-algebra operations on 344K-length vectors, achieving roughly a **1,000â€“2,500Ã— speedup**.
END_RESPONSE>>>
