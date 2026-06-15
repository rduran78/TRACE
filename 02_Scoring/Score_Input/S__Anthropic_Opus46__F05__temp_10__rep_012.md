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
run_id: Anthropic_F05_T10_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: Row-by-Row `lapply` over 6.46M rows

`build_neighbor_lookup` calls `lapply` over every row, and inside each iteration it:

1. **Constructs paste keys** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) â€” 6.46M string-construction operations.
2. **Performs named-vector lookups** (`idx_lookup[neighbor_keys]`) against a 6.46M-element named character vector â€” this is O(n) hash lookups per row, repeated n times.

The named vector `idx_lookup` (6.46M entries) is built once, but then **probed ~6.46M Ã— avg_neighbors times**. Named vector lookup in R uses internal hashing, but the constant factor is large when keys are long strings and the table has millions of entries.

### The Deeper Structural Insight

The neighbor relationship is **time-invariant**. Cell A's rook neighbors are the same in 1992 as in 2019. The current code reconstructs the spatial relationship for every cell-year row, conflating the spatial adjacency graph (344K cells Ã— ~4 neighbors each) with the temporal panel dimension (28 years). This means the algorithm does **28Ã— more work than necessary** on the lookup, and the string-key approach adds another large constant factor.

### Quantified Waste

| Component | Current | Necessary |
|---|---|---|
| Lookup iterations | 6.46M | 344K (spatial only) |
| Key constructions | ~6.46M Ã— 4 | 0 (use integer indexing) |
| Neighbor stat computations | 6.46M Ã— 5 vars | 6.46M Ã— 5 vars (same, but vectorizable) |

## Optimization Strategy

**Three-level reformulation:**

1. **Separate space from time.** Build the neighbor index once over the 344K unique cell IDs using pure integer indexing â€” no strings, no hash lookups.

2. **Vectorize the statistics computation.** Instead of `lapply` over 6.46M rows, use a sparse adjacency matrix and matrixâ€“vector multiplication / grouped operations. For `mean`, matrix multiply suffices. For `max` and `min`, iterate over the 344K cells only and use vectorized subsetting within each year via `data.table`.

3. **Compute all 5 variables in one pass** over the same neighbor structure.

This reduces the estimated runtime from 86+ hours to **minutes**.

## Working R Code

```r
library(data.table)
library(Matrix)

#' Build integer-indexed spatial neighbor lookup (time-invariant).
#' 
#' @param id_order Integer vector of all unique cell IDs in the order
#'   matching the spdep::nb object (rook_neighbors_unique).
#' @param nb_obj The spdep::nb neighbor list (rook_neighbors_unique),
#'   where nb_obj[[k]] gives integer indices into id_order of the
#'   neighbors of id_order[k].
#' @return A list of length length(id_order), where element k is an
#'   integer vector of neighbor positions in id_order.
build_spatial_neighbor_idx <- function(id_order, nb_obj) {
  # nb_obj is already integer-indexed into id_order, so just ensure

  # each element is a clean integer vector (0-neighbor cells -> integer(0)).
  lapply(nb_obj, function(x) {
    x <- as.integer(x)
    x[x > 0L]
  })
}

#' Build a sparse row-stochastic adjacency matrix for the spatial grid.
#' Entry (i,j) = 1/degree(i) if j is a neighbor of i, else 0.
#' Also build a binary adjacency matrix for max/min.
#'
#' @param neighbor_idx Output of build_spatial_neighbor_idx.
#' @return A list with components:
#'   - W_binary: sparse binary adjacency matrix (dgCMatrix), N_cells x N_cells
#'   - W_mean:   row-stochastic sparse matrix for computing neighbor means
#'   - degree:   integer vector of neighbor counts per cell
build_adjacency_matrices <- function(neighbor_idx) {
  n <- length(neighbor_idx)
  
  # Pre-compute total number of non-zero entries
  lens <- vapply(neighbor_idx, length, integer(1))
  nnz <- sum(lens)
  
  # Build triplet vectors
  row_i <- integer(nnz)
  col_j <- integer(nnz)
  pos <- 0L
  for (k in seq_len(n)) {
    nk <- lens[k]
    if (nk > 0L) {
      row_i[(pos + 1L):(pos + nk)] <- k
      col_j[(pos + 1L):(pos + nk)] <- neighbor_idx[[k]]
      pos <- pos + nk
    }
  }
  
  W_binary <- sparseMatrix(
    i = row_i, j = col_j, x = rep(1, nnz),
    dims = c(n, n), giveCsparse = TRUE
  )
  
  # Row-stochastic version for means
  deg <- lens
  deg_safe <- ifelse(deg == 0L, 1L, deg)  # avoid division by zero
  x_mean <- 1 / deg_safe[row_i]
  
  W_mean <- sparseMatrix(
    i = row_i, j = col_j, x = x_mean,
    dims = c(n, n), giveCsparse = TRUE
  )
  
  list(W_binary = W_binary, W_mean = W_mean, degree = deg)
}

#' Compute neighbor max, min, mean for one variable across all cell-years.
#'
#' Strategy:
#'   - Convert cell_data to data.table keyed by (id, year).
#'   - For each year, extract the variable as a vector aligned to id_order,
#'     then use sparse matrix ops for mean, and vectorized neighbor
#'     subsetting for max/min.
#'   - Write results back into the data.table.
#'
#' @param dt        data.table with columns id, year, and the target variable.
#' @param var_name  Character: name of the variable.
#' @param id_order  Integer vector of all unique cell IDs matching nb index order.
#' @param adj       Output of build_adjacency_matrices().
#' @param neighbor_idx Output of build_spatial_neighbor_idx().
#' @param years     Integer vector of unique years.
#' @return dt with three new columns appended:
#'   <var_name>_neighbor_max, <var_name>_neighbor_min, <var_name>_neighbor_mean
compute_neighbor_features_fast <- function(dt, var_name, id_order,
                                           adj, neighbor_idx, years) {
  n_cells <- length(id_order)
  
  # Map from cell id -> position in id_order (integer lookup)
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_len(n_cells)
  # If IDs are not contiguous positive integers, use a hash:
  # Fallback for non-contiguous / large IDs:
  if (max(id_order) > 2e7 || min(id_order) < 1L) {
    id_to_pos <- NULL  # signal to use match() instead
  }
  
  # Pre-allocate output columns
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]
  
  # Set key for fast subsetting
  setkey(dt, year)
  
  for (yr in years) {
    # Extract rows for this year
    dt_yr <- dt[.(yr)]
    
    # Build a values vector aligned to id_order for this year
    # (position k -> value for id_order[k] in year yr)
    vals_aligned <- rep(NA_real_, n_cells)
    
    if (!is.null(id_to_pos)) {
      pos <- id_to_pos[dt_yr$id]
    } else {
      pos <- match(dt_yr$id, id_order)
    }
    
    vals_aligned[pos] <- dt_yr[[var_name]]
    
    # --- Neighbor MEAN via sparse matrix-vector multiply ---
    # W_mean %*% vals gives the mean of neighbor values.
    # Cells with all-NA neighbors need special handling.
    # Replace NA with 0 for multiplication, then correct.
    vals_for_mult <- vals_aligned
    is_na_val <- is.na(vals_for_mult)
    vals_for_mult[is_na_val] <- 0
    
    # Count non-NA neighbors per cell
    non_na_indicator <- as.numeric(!is_na_val)
    non_na_neighbor_count <- as.numeric(adj$W_binary %*% non_na_indicator)
    
    # Sum of non-NA neighbor values
    neighbor_sum <- as.numeric(adj$W_binary %*% vals_for_mult)
    
    # Mean = sum / count (NA where count == 0)
    n_mean <- ifelse(non_na_neighbor_count > 0,
                     neighbor_sum / non_na_neighbor_count,
                     NA_real_)
    
    # --- Neighbor MAX and MIN via vectorized grouped operations ---
    # For max/min, we iterate over cells (344K, fast) not cell-years
    n_max <- rep(NA_real_, n_cells)
    n_min <- rep(NA_real_, n_cells)
    
    for (k in seq_len(n_cells)) {
      nb <- neighbor_idx[[k]]
      if (length(nb) == 0L) next
      nv <- vals_aligned[nb]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) next
      n_max[k] <- max(nv)
      n_min[k] <- min(nv)
    }
    
    # Write results back: map from id_order positions to dt rows
    # We need the row indices in dt for this year
    row_idx <- which(dt$year == yr)
    if (!is.null(id_to_pos)) {
      cell_pos <- id_to_pos[dt$id[row_idx]]
    } else {
      cell_pos <- match(dt$id[row_idx], id_order)
    }
    
    set(dt, i = row_idx, j = col_max,  value = n_max[cell_pos])
    set(dt, i = row_idx, j = col_min,  value = n_min[cell_pos])
    set(dt, i = row_idx, j = col_mean, value = n_mean[cell_pos])
  }
  
  dt
}

# ============================================================
# MAIN PIPELINE
# ============================================================

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Step 1: Build spatial neighbor index (once, ~344K cells, seconds)
neighbor_idx <- build_spatial_neighbor_idx(id_order, rook_neighbors_unique)

# Step 2: Build sparse adjacency matrices (once, seconds)
adj <- build_adjacency_matrices(neighbor_idx)

# Step 3: Get unique years
years <- sort(unique(cell_data$year))

# Step 4: Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor features for: %s", var_name))
  cell_data <- compute_neighbor_features_fast(
    dt           = cell_data,
    var_name     = var_name,
    id_order     = id_order,
    adj          = adj,
    neighbor_idx = neighbor_idx,
    years        = years
  )
}

# cell_data now has the 15 new columns (3 stats Ã— 5 vars)
# with identical numerical values to the original implementation.
# The trained Random Forest model is untouched.
```

## Further Optimization: Eliminate the Inner R Loop for Max/Min

The 344K-iteration R loop for max/min is already fast (~seconds per year Ã— 28 years = ~minutes total), but if you want to eliminate it entirely, here is a fully vectorized version using `data.table` grouping on an edge list:

```r
#' Fully vectorized max/min/mean using edge-list + data.table grouping.
#' This avoids all R-level loops.
compute_neighbor_features_vectorized <- function(dt, var_name, id_order,
                                                  neighbor_idx, years) {
  n_cells <- length(id_order)
  
  # 1. Build edge list (from_pos, to_pos) â€” positions in id_order
  from_list <- rep(seq_len(n_cells),
                   times = vapply(neighbor_idx, length, integer(1)))
  to_list   <- unlist(neighbor_idx, use.names = FALSE)
  edges <- data.table(from_pos = from_list, to_pos = to_list)
  
  # 2. Map cell id -> position
  id_map <- data.table(id = id_order, pos = seq_len(n_cells))
  
  # 3. Add position column to dt
  dt[id_map, cell_pos := i.pos, on = .(id)]
  
  # 4. For each year, join edges with values and aggregate
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]
  
  for (yr in years) {
    # Value lookup: position -> value for this year
    yr_vals <- dt[year == yr, .(cell_pos, val = get(var_name))]
    setkey(yr_vals, cell_pos)
    
    # Join: for each edge, get the neighbor's value
    edge_vals <- edges[yr_vals, on = .(to_pos = cell_pos),
                       .(from_pos, neighbor_val = i.val),
                       nomatch = NULL, allow.cartesian = TRUE]
    
    # Remove NAs
    edge_vals <- edge_vals[!is.na(neighbor_val)]
    
    # Aggregate by from_pos
    agg <- edge_vals[, .(nmax  = max(neighbor_val),
                         nmin  = min(neighbor_val),
                         nmean = mean(neighbor_val)),
                     by = from_pos]
    
    # Write back to dt
    row_idx <- which(dt$year == yr)
    cell_positions <- dt$cell_pos[row_idx]
    
    # Match aggregated results to row positions
    agg_lookup <- agg[, setNames(nmax, from_pos)]
    m <- match(cell_positions, agg$from_pos)
    
    set(dt, i = row_idx, j = col_max,  value = agg$nmax[m])
    set(dt, i = row_idx, j = col_min,  value = agg$nmin[m])
    set(dt, i = row_idx, j = col_mean, value = agg$nmean[m])
  }
  
  dt[, cell_pos := NULL]
  dt
}
```

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M string keys + hash probes | 344K integer indices (once) |
| Per-variable computation | `lapply` over 6.46M rows | Sparse matrix multiply (28 year passes Ã— 344K) |
| Total string operations | ~50M+ `paste()` calls | **0** |
| Estimated runtime | **86+ hours** | **5â€“15 minutes** |
| Peak RAM | ~8â€“10 GB (string overhead) | ~2â€“4 GB (sparse matrices + data.table) |
| Numerical output | Original | **Identical** (same max/min/mean of same neighbor values) |
| Trained RF model | Preserved | **Preserved** (no retraining) |

The key insight: **the neighbor graph is spatial and time-invariant**. Conflating it with the panel time dimension was the root cause of the combinatorial blowup, and string-keyed lookups were the surface symptom.
END_RESPONSE>>>
