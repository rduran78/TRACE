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
run_id: Anthropic_F08_T00_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup of length ~6.46 million, where each entry maps a cell-year row to the row indices of its neighbors *in that same year*. This is enormously wasteful because:

1. **The neighbor topology is static.** Cell A's neighbors are the same in 1992 as in 2019. Yet the current code rebuilds string keys (`paste(id, year)`) and performs hash lookups for every single cell-year combination â€” ~6.46M Ã— ~4 neighbors â‰ˆ 25M+ string operations and hash lookups just to build the lookup.

2. **`compute_neighbor_stats` iterates over 6.46M list elements** with an R-level `lapply`, extracting and summarizing neighbor values one row at a time. This is inherently slow in interpreted R.

3. **The lookup is monolithic.** Because it fuses topology (which cell is whose neighbor) with panel indexing (which row corresponds to which cell-year), it cannot exploit the fact that within any single year, the neighbor structure is identical. The same topology is redundantly encoded 28 times.

**Estimated complexity of current approach:**
- `build_neighbor_lookup`: O(N_cells Ã— N_years Ã— avg_neighbors) string operations â‰ˆ 25M+
- `compute_neighbor_stats`: O(N_cells Ã— N_years) R-level list iterations, called 5 times â‰ˆ 32M iterations
- Total wall time: 86+ hours (as reported)

## Optimization Strategy

**Key insight:** Separate the *static topology* from the *year-varying data*. 

1. **Build the neighbor lookup once at the cell level (344K entries), not at the cell-year level (6.46M entries).** Store it as a simple list: `cell_neighbors[[i]]` = integer vector of neighbor cell indices (positional indices into `id_order`).

2. **Reshape the year-varying data into a matrix** of dimension `(N_cells Ã— N_years)` for each variable, where rows are cells (in `id_order` order) and columns are years. This allows vectorized column-wise (i.e., year-wise) operations.

3. **Vectorize the neighbor aggregation.** Convert the cell-level neighbor list into a sparse adjacency matrix (using the `Matrix` package). Then for each variable, the neighbor max/min/mean across all cells and all years can be computed via sparse matrix operations â€” replacing ~32M R-level iterations with a handful of sparse matrix multiplications.

   - **Neighbor mean:** `A %*% X / degree` where `A` is the binary adjacency matrix and `degree` is the number of neighbors per cell.
   - **Neighbor max and min:** Iterate over years (28 iterations) using the sparse structure to gather neighbor values, then apply vectorized `pmax`/`pmin` reductions. Alternatively, use a CSR-style loop in C++ via `Rcpp`, or use a grouped operation.

4. **Merge results back** into the original `cell_data` data frame in the correct row order.

**Expected speedup:** From 86+ hours to **minutes**. The sparse matrix approach reduces the problem to ~28 sparse matrix-vector products per variable (for mean), and similarly efficient operations for max/min.

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build the static cell-level neighbor adjacency ONCE
# ==============================================================================

build_cell_adjacency <- function(id_order, neighbors_nb) {

  # neighbors_nb: spdep nb object, list of integer vectors (indices into id_order)
  # Returns: a sparse binary adjacency matrix of dimension (n_cells x n_cells)
  
  n <- length(id_order)
  stopifnot(length(neighbors_nb) == n)
  
  # Build COO (coordinate) representation
  from <- rep(seq_len(n), times = lengths(neighbors_nb))
  to   <- unlist(neighbors_nb)
  
  # Remove any 0-neighbor sentinel values that spdep uses
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  adj <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  return(adj)
}

# ==============================================================================
# STEP 2: Build cell-to-row index mapping and variable matrices
# ==============================================================================

compute_all_neighbor_features <- function(cell_data, id_order, neighbors_nb,
                                          neighbor_source_vars) {
  
  # --- Convert to data.table for speed ---
  dt <- as.data.table(cell_data)
  
  # --- Establish cell index: position of each cell's id within id_order ---
  id_to_cellidx <- setNames(seq_along(id_order), as.character(id_order))
  dt[, cell_idx := id_to_cellidx[as.character(id)]]
  
  # --- Get sorted unique years ---
  years <- sort(unique(dt$year))
  n_years <- length(years)
  n_cells <- length(id_order)
  year_to_colidx <- setNames(seq_along(years), as.character(years))
  dt[, year_idx := year_to_colidx[as.character(year)]]
  
  # --- Build sparse adjacency matrix (static, built once) ---
  cat("Building sparse adjacency matrix...\n")
  adj <- build_cell_adjacency(id_order, neighbors_nb)
  
  # Degree vector (number of neighbors per cell), used for mean
  degree <- as.numeric(rowSums(adj))  # length n_cells
  degree[degree == 0] <- NA  # avoid division by zero; will produce NA
  
  # --- CSC structure for neighbor gathering (for max/min) ---
  # For each cell i, adj[i, ] gives its neighbors.
  # We extract the neighbor list once from the sparse matrix.
  cat("Extracting neighbor list from sparse matrix...\n")
  adj_csr <- as(adj, "RsparseMatrix")  # Row-compressed for row-wise access
  # Actually, let's just use the original nb object directly for max/min
  # since we need to iterate per cell anyway for those.
  # But we'll do it year-by-year (28 iterations) instead of cell-year (6.46M).
  
  # Pre-extract neighbor indices as a simple list (from the nb object)
  # Clean up 0-entries (spdep convention for no neighbors)
  cell_neighbors <- lapply(neighbors_nb, function(nb) {
    nb <- as.integer(nb)
    nb[nb > 0L]
  })
  
  # --- For each variable, build matrix and compute neighbor stats ---
  cat("Computing neighbor features for", length(neighbor_source_vars), "variables...\n")
  
  # Pre-allocate result columns in dt
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
  }
  
  # Key dt by (cell_idx, year_idx) for fast assignment
  setkey(dt, cell_idx, year_idx)
  
  for (var_name in neighbor_source_vars) {
    cat("  Processing variable:", var_name, "\n")
    
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    # --- Build (n_cells x n_years) matrix for this variable ---
    # Initialize with NA
    var_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # Fill from data.table (vectorized)
    vals <- dt[[var_name]]
    cidx <- dt$cell_idx
    yidx <- dt$year_idx
    var_mat[cbind(cidx, yidx)] <- vals
    
    # --- Neighbor MEAN via sparse matrix multiplication ---
    # adj %*% var_mat gives, for each cell, the sum of neighbor values per year
    # Divide by degree to get mean
    neighbor_sum <- as.matrix(adj %*% var_mat)  # n_cells x n_years
    neighbor_mean_mat <- neighbor_sum / degree   # recycling degree along columns
    
    # --- Neighbor MAX and MIN: year-by-year vectorized approach ---
    # For max/min we cannot use matrix multiplication directly.
    # Strategy: loop over 28 years (not 6.46M rows), and within each year
    # use vapply over cells. But 344K cells Ã— 28 years = 9.6M iterations
    # of vapply is still slow. Instead, use an unrolled sparse approach:
    
    # Flatten the neighbor list into a "from-to" edge list
    from_cell <- rep(seq_len(n_cells), times = lengths(cell_neighbors))
    to_cell   <- unlist(cell_neighbors)
    n_edges   <- length(from_cell)
    
    # Pre-allocate result matrices
    neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # Process year by year (only 28 iterations)
    for (yi in seq_len(n_years)) {
      col_vals <- var_mat[, yi]  # values for all cells in this year
      
      # Get neighbor values for every edge
      neighbor_vals <- col_vals[to_cell]  # length = n_edges
      
      # Use data.table for fast grouped max/min
      edge_dt <- data.table(
        from = from_cell,
        nval = neighbor_vals
      )
      
      # Remove NA neighbor values before aggregation
      edge_dt <- edge_dt[!is.na(nval)]
      
      if (nrow(edge_dt) > 0) {
        agg <- edge_dt[, .(nmax = max(nval), nmin = min(nval)), by = from]
        neighbor_max_mat[agg$from, yi] <- agg$nmax
        neighbor_min_mat[agg$from, yi] <- agg$nmin
      }
    }
    
    # --- Write results back into dt ---
    # Extract values at the (cell_idx, year_idx) positions that exist in dt
    idx_mat <- cbind(cidx, yidx)
    
    set(dt, j = max_col,  value = neighbor_max_mat[idx_mat])
    set(dt, j = min_col,  value = neighbor_min_mat[idx_mat])
    set(dt, j = mean_col, value = neighbor_mean_mat[idx_mat])
  }
  
  # --- Clean up helper columns ---
  dt[, c("cell_idx", "year_idx") := NULL]
  
  # --- Return as data.frame (or data.table, depending on downstream needs) ---
  return(as.data.frame(dt))
}

# ==============================================================================
# STEP 3: Drop-in replacement for the outer loop
# ==============================================================================

# Original outer loop:
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }

# Replacement (single call):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data        = cell_data,
  id_order         = id_order,
  neighbors_nb     = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now contains the same 15 new columns:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... etc.
# 
# Proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup size** | 6.46M entries (cell-year) | 344K entries (cell only) â€” built once |
| **String key construction** | ~25M `paste()` + hash lookups | Zero â€” integer indexing only |
| **Neighbor stat computation** | 6.46M R-level `lapply` iterations Ã— 5 vars | 28 year-iterations Ã— 5 vars = 140 vectorized passes |
| **Mean calculation** | Per-row R loop | Single sparse matrix multiply `adj %*% var_mat` |
| **Max/Min calculation** | Per-row R loop | Vectorized edge-list + `data.table` grouped aggregation per year |
| **Estimated runtime** | 86+ hours | **~5â€“15 minutes** |
| **RAM usage** | Dominated by 6.46M-element list | Sparse matrix (~5.5M non-zeros) + dense matrices (344K Ã— 28) â‰ˆ manageable within 16 GB |
| **Numerical results** | â€” | **Identical** (same max, min, mean values) |
| **RF model** | â€” | **Untouched** (pre-trained, used only for `predict()`) |
END_RESPONSE>>>
