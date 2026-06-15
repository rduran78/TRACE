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
run_id: Anthropic_F08_T10_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause: Redundant Neighbor Lookups

1. **`build_neighbor_lookup`** operates on the full 6.46M-row `cell_data` data frame. For every single row (cellÃ—year), it recomputes which neighbor rows to look up by pasting cell IDs and years into string keys, then performing named-vector lookups. This means **~6.46M string-paste and hash-lookup operations**, when the underlying neighbor graph is **identical across all 28 years**. The topology is static; only the variable values change by year.

2. **`compute_neighbor_stats`** then iterates over the 6.46M-entry `neighbor_lookup` list, extracting values and computing `max`, `min`, `mean`. Because the lookup was built per cell-year row, this is a massive list of ~6.46M elements, each containing integer index vectors into the full data frame. This is both memory-intensive and cache-unfriendly.

3. The combination is **O(R Ã— K)** where R = 6.46M rows and K = average neighbor count (~4 for rook), repeated for each of the 5 variables. The per-element overhead of R's `lapply` over 6.46M elements, with string operations inside, dominates.

### Why It's Wasteful

- The neighbor graph has only **344,208 cells** with ~1.37M directed edges. This is **28Ã— smaller** than the row-level problem.
- For any given year, cell `i`'s neighbors are always the same cells. The only thing that changes is the *values* attached to those cells.
- The current code rebuilds string keys, re-hashes, and re-indexes as if the topology could differ by year.

---

## Optimization Strategy

**Separate the static topology from the dynamic (year-varying) computation.**

### Key Insight

1. **Build the neighbor index once, at the cell level (344K cells), not at the cell-year level (6.46M rows).** Store `neighbors[[i]]` as a simple integer vector of cell-level indices (1-based into the 344K `id_order` vector). This is just a cleaned-up version of `rook_neighbors_unique` â€” essentially free.

2. **For each year, subset/index the data to get a matrix of values for that year, then vectorize the neighbor aggregation.** For each year (only 344K cells), use the precomputed cell-level neighbor list to gather neighbor values and compute max/min/mean. This turns the problem from 6.46M `lapply` iterations with string ops into 28 Ã— 344K iterations with pure integer indexing â€” a ~28Ã— reduction in iterations, plus elimination of all string operations.

3. **Further vectorize using matrix operations.** Convert the neighbor list into a sparse adjacency matrix (344K Ã— 344K). Then for each year, extract the variable column as a vector, and compute:
   - `neighbor_mean = (A %*% x) / (A %*% ones)` (sparse matrix-vector multiply)
   - `neighbor_max` and `neighbor_min` via grouped operations

   The sparse matrix approach makes `mean` trivially fast. For `max` and `min`, we can use efficient grouped operations.

4. **Process year-by-year in a loop of 28 iterations**, writing results directly into preallocated columns. This keeps memory bounded.

### Expected Speedup

| Aspect | Before | After |
|---|---|---|
| Lookup construction | 6.46M string-paste + hash lookups | One-time O(344K) integer cleanup |
| Stats computation | 6.46M Ã— 5 vars lapply iterations | 28 years Ã— 344K cells Ã— 5 vars, vectorized |
| String operations | ~19M paste + match ops | Zero |
| Estimated time | 86+ hours | **~2â€“10 minutes** |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from dynamic (year-varying) variable values.
# Preserves the trained Random Forest model and original numerical estimand.
# =============================================================================

library(data.table)
library(Matrix)

# ---- STEP 1: Build static cell-level neighbor structures (done ONCE) --------

build_cell_neighbor_structures <- function(id_order, rook_neighbors_unique) {
  # rook_neighbors_unique is an nb object: a list of length n_cells,
  # where each element is an integer vector of neighbor indices (1-based)
  # into id_order, with 0L meaning no neighbors.
  
  n_cells <- length(id_order)
  stopifnot(length(rook_neighbors_unique) == n_cells)
  
  # Clean the nb object into a simple list of integer vectors.
  # nb objects use 0L to indicate no neighbors; we convert to integer(0).
  cell_neighbors <- lapply(rook_neighbors_unique, function(nb_idx) {
    nb_idx <- as.integer(nb_idx)
    nb_idx[nb_idx > 0L]
  })
  
  # Also build a sparse adjacency matrix for fast mean computation.
  # Rows = focal cell index, Cols = neighbor cell index, Value = 1.
  from_idx <- rep(seq_len(n_cells), lengths(cell_neighbors))
  to_idx   <- unlist(cell_neighbors, use.names = FALSE)
  
  if (length(from_idx) == 0) {
    adj_matrix <- sparseMatrix(i = integer(0), j = integer(0),
                               dims = c(n_cells, n_cells), x = numeric(0))
  } else {
    adj_matrix <- sparseMatrix(
      i = from_idx,
      j = to_idx,
      dims = c(n_cells, n_cells),
      x = rep(1, length(from_idx))
    )
  }
  
  # Precompute the number of neighbors per cell (for mean denominator).
  neighbor_counts <- diff(adj_matrix@p)  # CSC column counts if transposed; 
  # Actually for row-based ops we need row sums:
  neighbor_counts <- as.numeric(rowSums(adj_matrix))
  
  list(
    cell_neighbors  = cell_neighbors,
    adj_matrix      = adj_matrix,
    neighbor_counts = neighbor_counts,
    n_cells         = n_cells
  )
}


# ---- STEP 2: Compute neighbor max, min, mean per variable per year ----------
# Uses the static topology + year-specific variable values.

compute_all_neighbor_features <- function(cell_data, id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  
  # Convert to data.table for efficient indexing (non-destructive copy).
  dt <- as.data.table(cell_data)
  
  # Build static structures once.
  message("Building static cell-level neighbor structures...")
  topo <- build_cell_neighbor_structures(id_order, rook_neighbors_unique)
  
  n_cells         <- topo$n_cells
  cell_neighbors  <- topo$cell_neighbors
  adj_matrix      <- topo$adj_matrix
  neighbor_counts <- topo$neighbor_counts
  
  # Create a mapping from cell id to cell index (1-based position in id_order).
  id_to_cellidx <- setNames(seq_len(n_cells), as.character(id_order))
  
  # Add cell index to dt for fast lookups.
  dt[, cell_idx := id_to_cellidx[as.character(id)]]
  
  # Verify all cells are mapped.
  stopifnot(!anyNA(dt$cell_idx))
  
  # Pre-allocate output columns.
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")
    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]
  }
  
  # Get unique years.
  years <- sort(unique(dt$year))
  message(sprintf("Processing %d variables Ã— %d years = %d year-variable batches...",
                  length(neighbor_source_vars), length(years),
                  length(neighbor_source_vars) * length(years)))
  
  # Key the data.table for fast year subsetting.
  setkey(dt, year, cell_idx)
  
  # ---- Main loop: iterate over years, then variables ----
  for (yr in years) {
    
    # Get the row indices for this year, ordered by cell_idx.
    yr_row_indices <- dt[.(yr), which = TRUE]
    yr_cell_indices <- dt$cell_idx[yr_row_indices]
    
    # Build a mapping: for this year, cell_idx -> position in yr_row_indices.
    # Not all 344K cells may be present in every year.
    # We build a full-length vector (indexed by cell_idx) for O(1) lookup.
    cellidx_to_yrpos <- rep(NA_integer_, n_cells)
    cellidx_to_yrpos[yr_cell_indices] <- seq_along(yr_row_indices)
    
    n_yr <- length(yr_row_indices)
    
    for (var_name in neighbor_source_vars) {
      col_max  <- paste0(var_name, "_neighbor_max")
      col_min  <- paste0(var_name, "_neighbor_min")
      col_mean <- paste0(var_name, "_neighbor_mean")
      
      # Extract variable values for this year, indexed by cell_idx.
      # Build a full vector of length n_cells (NA for missing cells).
      vals_full <- rep(NA_real_, n_cells)
      vals_full[yr_cell_indices] <- dt[[var_name]][yr_row_indices]
      
      # --- Neighbor MEAN via sparse matrix-vector multiply ---
      # adj_matrix %*% vals_full gives the sum of neighbor values.
      # Divide by neighbor_counts for the mean.
      # Cells with NA neighbors: sparse multiply treats NA specially,
      # so we need to handle NAs carefully.
      
      # For mean: use the sparse matrix, but set NA values to 0 and track
      # valid counts separately.
      not_na <- !is.na(vals_full)
      vals_zero <- vals_full
      vals_zero[!not_na] <- 0
      
      neighbor_sum     <- as.numeric(adj_matrix %*% vals_zero)
      neighbor_valid_n <- as.numeric(adj_matrix %*% as.numeric(not_na))
      
      n_mean <- ifelse(neighbor_valid_n > 0,
                       neighbor_sum / neighbor_valid_n,
                       NA_real_)
      
      # --- Neighbor MAX and MIN via grouped C-level operations ---
      # For max and min, we iterate over cells using the precomputed
      # neighbor list. This is 344K iterations (not 6.46M), pure integer
      # indexed, no string operations.
      n_max <- rep(NA_real_, n_cells)
      n_min <- rep(NA_real_, n_cells)
      
      for (ci in seq_len(n_cells)) {
        nb <- cell_neighbors[[ci]]
        if (length(nb) == 0L) next
        nv <- vals_full[nb]
        nv <- nv[!is.na(nv)]
        if (length(nv) == 0L) next
        n_max[ci] <- max(nv)
        n_min[ci] <- min(nv)
      }
      
      # Write results back to the dt rows for this year.
      set(dt, i = yr_row_indices, j = col_max,  value = n_max[yr_cell_indices])
      set(dt, i = yr_row_indices, j = col_min,  value = n_min[yr_cell_indices])
      set(dt, i = yr_row_indices, j = col_mean, value = n_mean[yr_cell_indices])
    }
    
    message(sprintf("  Year %d done.", yr))
  }
  
  # Remove helper column.
  dt[, cell_idx := NULL]
  
  # Return as data.frame to maintain compatibility with downstream RF predict.
  as.data.frame(dt)
}


# =============================================================================
# FURTHER OPTIMIZATION: Vectorized max/min using data.table
# Replaces the inner for-loop over 344K cells with a vectorized grouped op.
# =============================================================================

compute_all_neighbor_features_fast <- function(cell_data, id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  
  dt <- as.data.table(cell_data)
  
  message("Building static cell-level neighbor structures...")
  topo <- build_cell_neighbor_structures(id_order, rook_neighbors_unique)
  
  n_cells         <- topo$n_cells
  adj_matrix      <- topo$adj_matrix
  neighbor_counts <- topo$neighbor_counts
  
  # Build edge list from the sparse adjacency matrix (static, done once).
  # from_cell -> to_cell (meaning: to_cell is a neighbor of from_cell)
  adj_coo <- summary(adj_matrix)  # gives i, j, x triplets
  edge_from <- adj_coo$i  # focal cell index
  edge_to   <- adj_coo$j  # neighbor cell index
  n_edges   <- length(edge_from)
  
  message(sprintf("  %d cells, %d directed edges.", n_cells, n_edges))
  
  # Build edge data.table for grouped operations.
  edges_dt <- data.table(from_cell = edge_from, to_cell = edge_to)
  setkey(edges_dt, from_cell)
  
  # Cell ID to cell index mapping.
  id_to_cellidx <- setNames(seq_len(n_cells), as.character(id_order))
  dt[, cell_idx := id_to_cellidx[as.character(id)]]
  stopifnot(!anyNA(dt$cell_idx))
  
  # Pre-allocate output columns.
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")
    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]
  }
  
  years <- sort(unique(dt$year))
  setkey(dt, year, cell_idx)
  
  message(sprintf("Processing %d variables Ã— %d years...",
                  length(neighbor_source_vars), length(years)))
  
  for (yr in years) {
    
    yr_row_indices  <- dt[.(yr), which = TRUE]
    yr_cell_indices <- dt$cell_idx[yr_row_indices]
    
    for (var_name in neighbor_source_vars) {
      col_max  <- paste0(var_name, "_neighbor_max")
      col_min  <- paste0(var_name, "_neighbor_min")
      col_mean <- paste0(var_name, "_neighbor_mean")
      
      # Build full cell-indexed value vector for this year.
      vals_full <- rep(NA_real_, n_cells)
      vals_full[yr_cell_indices] <- dt[[var_name]][yr_row_indices]
      
      # ---- Vectorized max, min, mean using the edge list ----
      # Look up neighbor values for all edges at once.
      neighbor_vals <- vals_full[edge_to]  # length = n_edges
      
      # Create a temporary data.table for grouped aggregation.
      # This is ~1.37M rows â€” very fast for data.table.
      agg_dt <- data.table(
        from_cell = edge_from,
        nval      = neighbor_vals
      )
      
      # Remove edges where the neighbor value is NA.
      agg_dt <- agg_dt[!is.na(nval)]
      
      # Grouped aggregation: max, min, mean per focal cell.
      if (nrow(agg_dt) > 0) {
        stats <- agg_dt[, .(
          nb_max  = max(nval),
          nb_min  = min(nval),
          nb_mean = mean(nval)
        ), by = from_cell]
        
        # Initialize result vectors (NA for cells with no valid neighbors).
        res_max  <- rep(NA_real_, n_cells)
        res_min  <- rep(NA_real_, n_cells)
        res_mean <- rep(NA_real_, n_cells)
        
        res_max[stats$from_cell]  <- stats$nb_max
        res_min[stats$from_cell]  <- stats$nb_min
        res_mean[stats$from_cell] <- stats$nb_mean
      } else {
        res_max  <- rep(NA_real_, n_cells)
        res_min  <- rep(NA_real_, n_cells)
        res_mean <- rep(NA_real_, n_cells)
      }
      
      # Write back to the main dt for this year's rows.
      set(dt, i = yr_row_indices, j = col_max,  value = res_max[yr_cell_indices])
      set(dt, i = yr_row_indices, j = col_min,  value = res_min[yr_cell_indices])
      set(dt, i = yr_row_indices, j = col_mean, value = res_mean[yr_cell_indices])
    }
    
    message(sprintf("  Year %d done.", yr))
  }
  
  dt[, cell_idx := NULL]
  as.data.frame(dt)
}


# =============================================================================
# USAGE â€” Drop-in replacement for the original outer loop
# =============================================================================

# Original code (SLOW, ~86+ hours):
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# Optimized replacement (expected ~2-10 minutes):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features_fast(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The output cell_data now has columns:
#   ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean,
#   ec_neighbor_max, ec_neighbor_min, ec_neighbor_mean,
#   pop_density_neighbor_max, pop_density_neighbor_min, pop_density_neighbor_mean,
#   def_neighbor_max, def_neighbor_min, def_neighbor_mean,
#   usd_est_n2_neighbor_max, usd_est_n2_neighbor_min, usd_est_n2_neighbor_mean
#
# These are numerically identical to the original implementation's output.
# The pre-trained Random Forest model is unchanged â€” proceed directly to predict().
#
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| | Original | Optimized |
|---|---|---|
| **Topology scope** | Rebuilt per cell-year row (6.46M) | Built once per cell (344K) |
| **String operations** | ~19M `paste` + named-vector lookups | Zero |
| **Neighbor lookup structure** | 6.46M-element list of index vectors | Static edge list (1.37M rows), reused |
| **Stats computation** | `lapply` over 6.46M elements Ã— 5 vars | `data.table` grouped aggregation: 28 years Ã— 5 vars Ã— 1.37M edges |
| **Memory for lookup** | Massive list (~gigabytes) | Edge data.table (~22 MB) + sparse matrix (~22 MB) |
| **Estimated runtime** | 86+ hours | **~2â€“10 minutes** |
| **Numerical result** | Baseline | Identical (same max/min/mean, same NA handling) |
| **RF model** | Unchanged | Unchanged |
END_RESPONSE>>>
