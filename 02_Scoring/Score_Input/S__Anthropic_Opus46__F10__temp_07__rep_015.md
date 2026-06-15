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
run_id: Anthropic_F10_T07_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting, named-vector lookups (`idx_lookup[neighbor_keys]`), and `NA` filtering. Named vector lookups in R are O(n) hash probes on a 6.46M-element character vector â€” this alone is catastrophically slow.

2. **`compute_neighbor_stats` iterates row-by-row** over 6.46M rows in an R-level `lapply`, extracting variable subsets and computing `max/min/mean` per row. This is repeated 5 times (once per source variable), yielding ~32.3 million R-level function calls.

3. **The neighbor topology is year-invariant** (rook contiguity depends on spatial position, not time), yet the lookup embeds year into every key, inflating the structure from ~344K spatial relationships to ~6.46M row-level relationships. The topology should be built once at the cell level and reused across all 28 years.

**Estimated cost breakdown of the 86+ hours:**
- ~40% in `build_neighbor_lookup` (string operations, named vector indexing on 6.46M keys)
- ~55% in `compute_neighbor_stats` (R-level row-wise loops Ã— 5 variables)
- ~5% overhead (memory pressure, GC on 16 GB RAM)

## Optimization Strategy

| Principle | Implementation |
|---|---|
| **Separate topology from time** | Build a sparse adjacency structure once over 344K cells, not 6.46M rows. |
| **Use sparse matrix multiplication** | Encode adjacency as a sparse `dgCMatrix`. Neighbor sums and counts become `A %*% X` â€” a single BLAS-level operation. |
| **Vectorize min/max** | Use grouped operations via `data.table` keyed joins or sparse-matrix tricks with sentinel values for min/max. |
| **Process per-year slices** | Each year is independent given the topology. Process 28 year-slices, each 344K rows, avoiding 6.46M-row monolithic operations. |
| **Batch all 5 variables simultaneously** | One matrix multiply per statistic type (sum, min, max) across all 5 variables at once. |
| **Memory-safe** | Peak memory: sparse matrix (~1.4M non-zeros â‰ˆ 33 MB) + one year-slice of 344K Ã— 5 doubles â‰ˆ 13 MB. Well within 16 GB. |

**Expected speedup:** From 86+ hours to approximately **2â€“5 minutes**.

## Optimized R Code

```r
# =============================================================================
# Optimized Neighbor Aggregation Pipeline
# Numerically equivalent to the original build_neighbor_lookup +
# compute_neighbor_stats pipeline.
# =============================================================================

library(data.table)
library(Matrix)

# -------------------------------------------------------------------------
# STEP 1: Build sparse adjacency matrix ONCE (344,208 Ã— 344,208, ~1.37M nnz)
# -------------------------------------------------------------------------
# Inputs:
#   id_order             â€” integer vector of length 344,208 (cell IDs in nb order)
#   rook_neighbors_unique â€” spdep nb object (list of integer index vectors)
#
# The nb object is 1-indexed into id_order. We build a sparse column-major
# matrix A where A[i,j] = 1 means cell j is a rook neighbor of cell i.
# Then: neighbor_sum_of_x[i] = sum_j A[i,j] * x[j]  =  (A %*% x)[i]

build_adjacency_matrix <- function(id_order, nb_obj) {
  n <- length(id_order)
  
  # Pre-calculate total number of edges for pre-allocation
  n_edges <- sum(lengths(nb_obj))
  
  # Pre-allocate vectors
  from_idx <- integer(n_edges)
  to_idx   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0L to denote "no neighbors"
    nbrs <- nbrs[nbrs > 0L]
    nn <- length(nbrs)
    if (nn > 0L) {
      from_idx[pos:(pos + nn - 1L)] <- i
      to_idx[pos:(pos + nn - 1L)]   <- nbrs
      pos <- pos + nn
    }
  }
  
  # Trim if any 0-neighbor nodes caused over-allocation
  if (pos - 1L < n_edges) {
    from_idx <- from_idx[1:(pos - 1L)]
    to_idx   <- to_idx[1:(pos - 1L)]
  }
  
  A <- sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = 1,
    dims = c(n, n),
    repr = "C"          
  )
  
  return(A)
}

# -------------------------------------------------------------------------
# STEP 2: Compute neighbor stats for all variables, all years
# -------------------------------------------------------------------------
# For each cell i and year t, for each variable v, we need:
#   neighbor_max_v  = max of v over rook neighbors of i in year t
#   neighbor_min_v  = min of v over rook neighbors of i in year t
#   neighbor_mean_v = mean of v over rook neighbors of i in year t
#
# Mean = sum / count.  Sum and count are sparse mat-vec products.
# Max and min use sentinel-value tricks:
#   max: replace NA with -Inf, multiply, then fix cells with 0 valid neighbors.
#   min: replace NA with +Inf, multiply, then fix cells with 0 valid neighbors.
#
# Because max(a,b) != linear, we cannot use a single mat-vec for max/min.
# Instead, we use a data.table grouped approach that is still vectorized
# and avoids R-level row-wise loops.

compute_all_neighbor_features <- function(cell_data, id_order, nb_obj,
                                          neighbor_source_vars) {
  
  n_cells <- length(id_order)
  
  # Convert to data.table if not already
  dt <- as.data.table(cell_data)
  
  # --- Build cell-level edge list once ---
  # Edge list: data.table with columns (from_cell_pos, to_cell_pos)
  # where positions are 1-based indices into id_order.
  
  n_edges <- sum(lengths(nb_obj))
  from_pos <- integer(n_edges)
  to_pos   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs > 0L]
    nn <- length(nbrs)
    if (nn > 0L) {
      from_pos[pos:(pos + nn - 1L)] <- i
      to_pos[pos:(pos + nn - 1L)]   <- nbrs
      pos <- pos + nn
    }
  }
  if (pos - 1L < n_edges) {
    from_pos <- from_pos[1:(pos - 1L)]
    to_pos   <- to_pos[1:(pos - 1L)]
  }
  
  edges <- data.table(from_pos = from_pos, to_pos = to_pos)
  
  # --- Map cell IDs to positions ---
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add cell position to dt (preserving original row order)
  dt[, row_orig := .I]
  dt[, cell_pos := id_to_pos[as.character(id)]]
  
  # --- Create a keyed lookup: (cell_pos, year) -> row index in dt ---
  # This allows us to join neighbor values by (cell_pos, year).
  setkey(dt, cell_pos, year)
  
  years <- sort(unique(dt$year))
  
  # Pre-allocate result columns
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
  }
  
  # --- Process year by year ---
  # For each year, we have at most 344,208 rows and ~1.37M edges.
  # We join edges with the year-slice to get neighbor values, then
  # aggregate with data.table (vectorized C-level grouping).
  
  for (yr in years) {
    
    # Extract year slice: cell_pos -> variable values
    yr_rows <- dt[.(unique(dt$cell_pos), yr), nomatch = 0L, which = TRUE]
    
    # More robust: filter directly
    yr_dt <- dt[year == yr, c("cell_pos", neighbor_source_vars), with = FALSE]
    setkey(yr_dt, cell_pos)
    
    # Join edges with neighbor (to_pos) values
    # For each edge (from_pos -> to_pos), get to_pos's variable values
    edge_yr <- copy(edges)
    
    # Join to get neighbor values (the "to" node's attributes)
    edge_yr[yr_dt, on = .(to_pos = cell_pos),
            (neighbor_source_vars) := mget(paste0("i.", neighbor_source_vars))]
    
    # Now aggregate by from_pos: compute max, min, mean for each variable
    # Only over non-NA neighbor values
    
    for (var_name in neighbor_source_vars) {
      max_col  <- paste0("neighbor_max_", var_name)
      min_col  <- paste0("neighbor_min_", var_name)
      mean_col <- paste0("neighbor_mean_", var_name)
      
      # Subset to non-NA values of this variable
      sub <- edge_yr[!is.na(get(var_name)), .(from_pos, val = get(var_name))]
      
      if (nrow(sub) == 0L) next
      
      agg <- sub[, .(
        nmax  = max(val),
        nmin  = min(val),
        nmean = mean(val)
      ), by = from_pos]
      
      # Write back into dt: match on (cell_pos == from_pos, year == yr)
      # Use row indices for direct assignment
      
      # Get the row indices in dt for this year and these cell positions
      idx_dt <- dt[.(agg$from_pos, yr), which = TRUE, nomatch = 0L]
      
      # We need a reliable join. Let's use a keyed merge approach:
      setkey(agg, from_pos)
      
      # Get dt rows for this year
      yr_mask <- which(dt$year == yr)
      yr_sub  <- dt[yr_mask, .(cell_pos, row_pos = yr_mask)]
      # Actually, let's be more direct:
      
      match_dt <- dt[year == yr, .(cell_pos, dt_row = .I)]
      # .I here is relative to the subset â€” we need absolute row indices
      # Use the row_orig column instead
      match_dt <- dt[year == yr, .(cell_pos, dt_row = row_orig)]
      setkey(match_dt, cell_pos)
      
      merged <- agg[match_dt, on = .(from_pos = cell_pos), nomatch = 0L]
      
      if (nrow(merged) > 0L) {
        set(dt, i = merged$dt_row, j = max_col,  value = merged$nmax)
        set(dt, i = merged$dt_row, j = min_col,  value = merged$nmin)
        set(dt, i = merged$dt_row, j = mean_col, value = merged$nmean)
      }
    }
    
    # Progress
    message(sprintf("Year %d done.", yr))
  }
  
  # Restore original row order and drop helper columns
  setorder(dt, row_orig)
  dt[, c("row_orig", "cell_pos") := NULL]
  
  return(dt)
}

# -------------------------------------------------------------------------
# STEP 3: Even faster â€” sparse matrix approach for mean, 
#          data.table for min/max (hybrid)
# -------------------------------------------------------------------------
# This version uses sparse matrix multiplication for mean (the most 
# algebraically natural) and data.table for min/max.

compute_all_neighbor_features_fast <- function(cell_data, id_order, nb_obj,
                                                neighbor_source_vars) {
  
  n_cells <- length(id_order)
  dt <- as.data.table(cell_data)
  dt[, row_orig := .I]
  
  # --- Map cell IDs to nb-object positions ---
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  dt[, cell_pos := id_to_pos[as.character(id)]]
  
  # --- Build edge list once ---
  n_edges_total <- sum(lengths(nb_obj))
  ev_from <- integer(n_edges_total)
  ev_to   <- integer(n_edges_total)
  p <- 1L
  for (i in seq_len(n_cells)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs > 0L]
    nn <- length(nbrs)
    if (nn > 0L) {
      ev_from[p:(p + nn - 1L)] <- i
      ev_to[p:(p + nn - 1L)]   <- nbrs
      p <- p + nn
    }
  }
  actual_edges <- p - 1L
  ev_from <- ev_from[1:actual_edges]
  ev_to   <- ev_to[1:actual_edges]
  
  # Sparse adjacency matrix (for mean computation via mat-vec)
  A <- sparseMatrix(i = ev_from, j = ev_to, x = 1,
                    dims = c(n_cells, n_cells), repr = "C")
  
  # Edge data.table (for min/max computation)
  edges_dt <- data.table(from_pos = ev_from, to_pos = ev_to)
  
  # --- Pre-allocate output columns ---
  out_cols <- character(0)
  for (var_name in neighbor_source_vars) {
    for (stat in c("max", "min", "mean")) {
      col <- paste0("neighbor_", stat, "_", var_name)
      dt[, (col) := NA_real_]
      out_cols <- c(out_cols, col)
    }
  }
  
  years <- sort(unique(dt$year))
  
  # --- Process per year ---
  for (yr in years) {
    
    yr_idx <- which(dt$year == yr)
    yr_sub <- dt[yr_idx, c("cell_pos", neighbor_source_vars), with = FALSE]
    
    # Build a mapping: cell_pos -> position within this year-slice
    # Not all 344K cells may be present in every year
    present_pos <- yr_sub$cell_pos
    pos_to_slice <- integer(n_cells)  # 0 means absent
    pos_to_slice[present_pos] <- seq_along(present_pos)
    
    # For each variable, compute mean via sparse matrix, min/max via data.table
    for (var_name in neighbor_source_vars) {
      
      vals_full <- rep(NA_real_, n_cells)
      vals_full[present_pos] <- yr_sub[[var_name]]
      
      # ---- MEAN via sparse matrix ----
      # Handle NAs: replace with 0 for sum, track non-NA for count
      not_na <- as.double(!is.na(vals_full))
      vals_0 <- ifelse(is.na(vals_full), 0, vals_full)
      
      neighbor_sum   <- as.numeric(A %*% vals_0)
      neighbor_count <- as.numeric(A %*% not_na)
      
      neighbor_mean <- ifelse(neighbor_count > 0,
                              neighbor_sum / neighbor_count,
                              NA_real_)
      
      # ---- MIN / MAX via data.table ----
      # Join neighbor values onto edge list
      edge_vals <- vals_full[ev_to]
      
      # Build a small DT of (from_pos, val) â€” only non-NA
      valid <- !is.na(edge_vals)
      if (any(valid)) {
        agg_dt <- data.table(from_pos = ev_from[valid], val = edge_vals[valid])
        agg <- agg_dt[, .(nmax = max(val), nmin = min(val)), by = from_pos]
        
        # Write max/min into full-length vectors
        neighbor_max <- rep(NA_real_, n_cells)
        neighbor_min <- rep(NA_real_, n_cells)
        neighbor_max[agg$from_pos] <- agg$nmax
        neighbor_min[agg$from_pos] <- agg$nmin
      } else {
        neighbor_max <- rep(NA_real_, n_cells)
        neighbor_min <- rep(NA_real_, n_cells)
      }
      
      # ---- Write back to dt ----
      max_col  <- paste0("neighbor_max_", var_name)
      min_col  <- paste0("neighbor_min_", var_name)
      mean_col <- paste0("neighbor_mean_", var_name)
      
      set(dt, i = yr_idx, j = max_col,  value = neighbor_max[present_pos])
      set(dt, i = yr_idx, j = min_col,  value = neighbor_min[present_pos])
      set(dt, i = yr_idx, j = mean_col, value = neighbor_mean[present_pos])
    }
    
    message(sprintf("Year %d complete.", yr))
  }
  
  # Restore original order, drop helper
  setorder(dt, row_orig)
  dt[, c("row_orig", "cell_pos") := NULL]
  
  return(dt)
}

# =========================================================================
# MAIN EXECUTION
# =========================================================================

library(data.table)
library(Matrix)
library(ranger)  # or randomForest â€” whichever was used

# --- Load pre-existing objects ---
# cell_data              : data.frame/data.table with columns id, year, ntl, ec, 
#                          pop_density, def, usd_est_n2, ... (~6.46M rows)
# id_order               : integer vector of 344,208 cell IDs matching nb object
# rook_neighbors_unique  : spdep nb object (list of length 344,208)
# rf_model               : pre-trained Random Forest model (DO NOT retrain)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# --- Compute all neighbor features (fast hybrid approach) ---
cell_data <- compute_all_neighbor_features_fast(
  cell_data, 
  id_order, 
  rook_neighbors_unique,
  neighbor_source_vars
)

# --- Predict with the pre-trained Random Forest ---
# The model expects the same feature names as in training, including
# neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, etc.
predictions <- predict(rf_model, data = cell_data)

# If using ranger:
# predictions <- predict(rf_model, data = cell_data)$predictions
# If using randomForest:
# predictions <- predict(rf_model, newdata = cell_data)

cell_data$predicted <- predictions
```

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized | Equivalence |
|-----------|----------|-----------|-------------|
| **max** | `max(vals[idx])` after NA removal | `data.table` `max(val)` grouped by `from_pos`, same NA exclusion | Identical â€” same `max()` over same set of non-NA values |
| **min** | `min(vals[idx])` after NA removal | `data.table` `min(val)` grouped by `from_pos`, same NA exclusion | Identical |
| **mean** | `mean(vals[idx])` after NA removal | `sum(non_NA_vals) / count(non_NA_vals)` via sparse `A %*% x` | Identical â€” IEEE 754 addition is the same; `mean()` = `sum()/length()` which is exactly what `A %*% vals_0 / A %*% not_na` computes |

## Performance Analysis

| Component | Original (86+ hrs) | Optimized | Speedup |
|-----------|-------------------|-----------|---------|
| Topology build | ~35 hrs (string hashing on 6.46M keys) | ~5 sec (integer edge list + sparse matrix) | ~25,000Ã— |
| Neighbor stats | ~50 hrs (R-level lapply Ã— 6.46M Ã— 5) | ~2 min (28 sparse mat-vec + 28 data.table aggs Ã— 5) | ~1,500Ã— |
| Memory peak | ~12 GB (6.46M-element list of integer vectors) | ~3 GB (sparse matrix + year slices) | 4Ã— reduction |
| **Total** | **86+ hours** | **~2â€“5 minutes** | **~1,000â€“2,500Ã—** |

The key insight: **the graph topology is time-invariant**. By separating the spatial structure (344K nodes, 1.37M edges) from the temporal dimension (28 years), we reduce the problem from 6.46M row-level lookups to 28 repetitions of a 344K-node sparse operation â€” and sparse matrix-vector multiplication is implemented in optimized C/Fortran (via the `Matrix` package's CHOLMOD backend).
END_RESPONSE>>>
