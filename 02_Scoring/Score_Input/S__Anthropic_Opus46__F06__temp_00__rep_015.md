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
run_id: Anthropic_F06_T00_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F06_Anthropic_temp_00_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's rook neighbors (by cell ID).
2. Constructs string keys (`paste(neighbor_id, year, "_")`).
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

String pasting and named-vector lookup on a 6.46M-element character vector is extremely slow. The named lookup is effectively an O(n) hash probe repeated ~6.46M Ã— ~4 neighbors â‰ˆ 25.8M times. This alone can take many hours.

### Bottleneck B: `compute_neighbor_stats` â€” `lapply` over 6.46M rows returning lists

For each of 5 variables, `lapply` iterates over 6.46M rows, subsets a numeric vector by index, computes `max/min/mean`, and returns a 3-element vector. The `do.call(rbind, result)` on a 6.46M-element list of vectors is also very slow (repeated memory allocation).

### Why raster focal/kernel operations are not directly applicable

The grid cells are irregular (not all cells exist in every year; the neighbor structure comes from a precomputed `spdep::nb` object that may reflect irregular boundaries). Focal operations assume a complete regular raster grid. Using them would risk altering the numerical results for edge cells, missing cells, or irregular geometries. We must preserve the exact rook-neighbor structure and the exact numerical estimand, so we use **sparse-matrix multiplication and vectorized operations** instead, which perfectly replicate the original logic.

---

## 2. Optimization Strategy

### Key Insight: Represent the neighbor structure as a sparse matrix, then compute stats via matrix operations and vectorized grouping.

**Step 1 â€” Eliminate string keys entirely.** Since every cell appears in every year (344,208 cells Ã— 28 years = 9,637,824, but only ~6.46M rows exist), we build an integer lookup from `(cell_id, year)` to row index using `data.table` or a two-column integer match â€” no string pasting.

**Step 2 â€” Build a sparse adjacency matrix (6.46M Ã— 6.46M) at the cell-year level.** Each row `i` has non-zero entries in columns corresponding to its rook neighbors in the same year. With ~6.46M rows and ~4 neighbors each, this matrix has ~25.8M non-zero entries â€” easily fits in memory as a `dgCMatrix` (~600 MB).

**Step 3 â€” Compute neighbor stats vectorially.** For each variable:
- **Mean**: sparse matrix Ã— dense vector, divided by the row-wise count of neighbors. One matrix-vector multiply for all 6.46M rows.
- **Max / Min**: Use `data.table` grouped operations on the edge list representation (source, target) to compute grouped max and min. This avoids the sparse matrix for max/min (which don't distribute over addition) but is still fully vectorized.

**Expected speedup**: From 86+ hours to **~5â€“15 minutes**.

---

## 3. Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# ==============================================================================
# Prerequisites: data.table, Matrix, (optional: collapse for even faster grouped stats)
# install.packages(c("data.table", "Matrix"))

library(data.table)
library(Matrix)

# --------------------------------------------------------------------------
# STEP 1: Build an integer lookup from (cell_id, year) -> row index
#         No string pasting. Pure integer operations.
# --------------------------------------------------------------------------

build_edge_list <- function(cell_data, id_order, rook_neighbors_unique) {
  # cell_data must have columns: id, year (and be a data.frame or data.table)
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]
  
  # Map cell id -> position in id_order (reference index into nb object)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Build a keyed lookup: given (id, year) -> row_idx
  setkey(dt, id, year)
  
  # For each cell in id_order, get its neighbors
  # rook_neighbors_unique is an nb object: a list of integer vectors
  # rook_neighbors_unique[[ref]] gives the ref-indices of neighbors of cell ref
  
  # Build the edge list at the cell level first (ref_from, ref_to)
  n_cells <- length(id_order)
  
  from_ref <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  to_ref   <- unlist(rook_neighbors_unique, use.names = FALSE)
  
  # Remove any 0-entries (spdep uses 0 for "no neighbors" in some representations)
  valid <- to_ref > 0L & to_ref <= n_cells
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]
  
  # Convert ref indices to actual cell IDs
  from_id <- id_order[from_ref]
  to_id   <- id_order[to_ref]
  
  # Now expand over years: each (from_id, to_id) pair exists for every year
  # that BOTH from_id and to_id appear in the data
  
  # Get the set of years per cell
  years_in_data <- sort(unique(dt$year))
  n_years <- length(years_in_data)
  n_edges_cell <- length(from_id)
  
  # Expand: repeat each cell-level edge for each year
  from_id_expanded <- rep(from_id, each = n_years)
  to_id_expanded   <- rep(to_id,   each = n_years)
  year_expanded     <- rep(years_in_data, times = n_edges_cell)
  
  edge_dt <- data.table(
    from_id = from_id_expanded,
    to_id   = to_id_expanded,
    year    = year_expanded
  )
  
  # Join to get row indices for 'from' and 'to'
  # from side
  edge_dt <- merge(edge_dt, dt[, .(id, year, row_idx)],
                   by.x = c("from_id", "year"),
                   by.y = c("id", "year"),
                   all.x = FALSE, sort = FALSE)
  setnames(edge_dt, "row_idx", "from_row")
  
  # to side
  edge_dt <- merge(edge_dt, dt[, .(id, year, row_idx)],
                   by.x = c("to_id", "year"),
                   by.y = c("id", "year"),
                   all.x = FALSE, sort = FALSE)
  setnames(edge_dt, "row_idx", "to_row")
  
  # Keep only the row indices â€” this is our edge list at the cell-year level
  edge_dt <- edge_dt[, .(from_row, to_row)]
  
  return(edge_dt)
}

# --------------------------------------------------------------------------
# STEP 2: Build sparse adjacency matrix (for mean) and keep edge list (for max/min)
# --------------------------------------------------------------------------

build_adjacency_matrix <- function(edge_dt, n_rows) {
  # Sparse matrix: A[i,j] = 1 if j is a rook neighbor of i (same year)
  A <- sparseMatrix(
    i = edge_dt$from_row,
    j = edge_dt$to_row,
    x = 1,
    dims = c(n_rows, n_rows)
  )
  return(A)
}

# --------------------------------------------------------------------------
# STEP 3: Compute neighbor stats vectorially
# --------------------------------------------------------------------------

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {
  
  cat("Building edge list...\n")
  t0 <- Sys.time()
  edge_dt <- build_edge_list(cell_data, id_order, rook_neighbors_unique)
  cat("  Edge list:", nrow(edge_dt), "directed edges. Time:", 
      round(difftime(Sys.time(), t0, units = "mins"), 1), "min\n")
  
  n_rows <- nrow(cell_data)
  
  cat("Building sparse adjacency matrix...\n")
  t0 <- Sys.time()
  A <- build_adjacency_matrix(edge_dt, n_rows)
  # Number of neighbors per row (for computing mean)
  neighbor_count <- as.numeric(A %*% rep(1, n_rows))
  cat("  Time:", round(difftime(Sys.time(), t0, units = "mins"), 1), "min\n")
  
  # Convert cell_data to data.table if not already
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  for (var_name in neighbor_source_vars) {
    cat("Processing variable:", var_name, "\n")
    t0 <- Sys.time()
    
    vals <- cell_data[[var_name]]
    
    # --- Handle NAs: we need to replicate the original behavior ---
    # Original: for each row i, gather neighbor values, remove NAs, then compute.
    # If all neighbor values are NA (or no neighbors), result is NA.
    
    # For MEAN: 
    #   We need sum of non-NA neighbor values / count of non-NA neighbor values.
    #   Replace NA with 0 for the sum, and count non-NA neighbors separately.
    vals_no_na <- ifelse(is.na(vals), 0, vals)
    not_na     <- as.numeric(!is.na(vals))
    
    neighbor_sum     <- as.numeric(A %*% vals_no_na)
    neighbor_count_valid <- as.numeric(A %*% not_na)
    
    neighbor_mean <- ifelse(neighbor_count_valid > 0,
                            neighbor_sum / neighbor_count_valid,
                            NA_real_)
    
    # For MAX and MIN: use data.table grouped operations on the edge list
    # Attach the 'to' values to the edge list
    edge_dt[, val := vals[to_row]]
    
    # Remove edges where the neighbor value is NA
    edge_valid <- edge_dt[!is.na(val)]
    
    # Grouped max and min by from_row
    stats <- edge_valid[, .(nb_max = max(val), nb_min = min(val)), by = from_row]
    
    # Initialize result vectors with NA
    neighbor_max <- rep(NA_real_, n_rows)
    neighbor_min <- rep(NA_real_, n_rows)
    
    neighbor_max[stats$from_row] <- stats$nb_max
    neighbor_min[stats$from_row] <- stats$nb_min
    
    # Also: rows with no neighbors at all should be NA (already handled)
    # Rows with neighbors but all NA should be NA (already handled)
    
    # Add to cell_data using the same naming convention as the original code
    # Original function likely creates: {var}_nb_max, {var}_nb_min, {var}_nb_mean
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    
    cell_data[, (max_col)  := neighbor_max]
    cell_data[, (min_col)  := neighbor_min]
    cell_data[, (mean_col) := neighbor_mean]
    
    cat("  Time:", round(difftime(Sys.time(), t0, units = "mins"), 1), "min\n")
  }
  
  # Clean up temporary column from edge_dt
  edge_dt[, val := NULL]
  
  return(cell_data)
}

# ==============================================================================
# USAGE â€” Drop-in replacement for the original outer loop
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique,
  neighbor_source_vars
)

# Now cell_data has the 15 new columns (5 vars Ã— 3 stats each).
# Proceed with prediction using the pre-trained Random Forest model as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Memory-Constrained Alternative (if the full edge expansion exceeds RAM)

If expanding all cell-level edges across 28 years in one shot causes memory pressure on a 16 GB laptop, process year-by-year:

```r
compute_all_neighbor_features_chunked <- function(cell_data, id_order, 
                                                   rook_neighbors_unique,
                                                   neighbor_source_vars) {
  
  if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)
  cell_data[, row_idx := .I]
  
  # Build cell-level edge list (ref indices)
  n_cells <- length(id_order)
  from_ref <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  to_ref   <- unlist(rook_neighbors_unique, use.names = FALSE)
  valid    <- to_ref > 0L & to_ref <= n_cells
  cell_edges <- data.table(
    from_id = id_order[from_ref[valid]],
    to_id   = id_order[to_ref[valid]]
  )
  
  # Initialize result columns
  for (var_name in neighbor_source_vars) {
    cell_data[, paste0(var_name, "_nb_max")  := NA_real_]
    cell_data[, paste0(var_name, "_nb_min")  := NA_real_]
    cell_data[, paste0(var_name, "_nb_mean") := NA_real_]
  }
  
  years <- sort(unique(cell_data$year))
  
  for (yr in years) {
    cat("Year:", yr, "\n")
    
    # Subset to this year
    yr_dt <- cell_data[year == yr, c("id", "row_idx", neighbor_source_vars), with = FALSE]
    setkey(yr_dt, id)
    
    # Build year-specific edge list with row indices
    edges_yr <- merge(cell_edges, yr_dt[, .(id, row_idx)],
                      by.x = "from_id", by.y = "id", all.x = FALSE, sort = FALSE)
    setnames(edges_yr, "row_idx", "from_row")
    edges_yr <- merge(edges_yr, yr_dt[, .(id, row_idx)],
                      by.x = "to_id", by.y = "id", all.x = FALSE, sort = FALSE)
    setnames(edges_yr, "row_idx", "to_row")
    
    n_yr <- nrow(yr_dt)
    # Local row indices within the year subset for sparse matrix
    yr_dt[, local_idx := .I]
    local_from <- yr_dt[.(edges_yr$from_row), on = "row_idx", local_idx]
    local_to   <- yr_dt[.(edges_yr$to_row),   on = "row_idx", local_idx]
    
    # Actually, simpler: work directly with global row indices and edge_dt
    for (var_name in neighbor_source_vars) {
      vals <- cell_data[[var_name]]
      edges_yr[, val := vals[to_row]]
      ev <- edges_yr[!is.na(val)]
      
      if (nrow(ev) > 0) {
        stats <- ev[, .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)),
                    by = from_row]
        
        cell_data[stats$from_row, paste0(var_name, "_nb_max")  := stats$nb_max]
        cell_data[stats$from_row, paste0(var_name, "_nb_min")  := stats$nb_min]
        cell_data[stats$from_row, paste0(var_name, "_nb_mean") := stats$nb_mean]
      }
    }
  }
  
  cell_data[, row_idx := NULL]
  return(cell_data)
}
```

---

## 5. Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string pastes + named vector lookups | Integer merge via `data.table` |
| **Stats computation** | `lapply` over 6.46M rows Ã— 5 vars | Sparse matrix multiply (mean) + `data.table` grouped ops (max/min) |
| **Result assembly** | `do.call(rbind, 6.46M-element list)` | Direct vector assignment |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes |
| **RAM usage** | Moderate (but slow) | ~2â€“4 GB for sparse matrix + edge list (fits in 16 GB) |
| **Numerical equivalence** | Baseline | Identical â€” same neighbor sets, same `max/min/mean` logic, same NA handling |
| **RF model** | Unchanged | Unchanged â€” no retraining |
END_RESPONSE>>>
