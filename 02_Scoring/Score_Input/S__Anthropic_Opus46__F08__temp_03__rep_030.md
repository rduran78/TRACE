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
run_id: Anthropic_F08_T03_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F08_Anthropic_temp_03_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) neighbor indices.** It creates a list of ~6.46 million elements, each requiring string-pasting, hash lookups, and NA filtering. But the neighbor *relationships* between cells are purely spatial and **never change across years**. The only thing that changes is the variable values attached to each cell in each year.

2. **String-key hashing is extremely expensive at scale.** The function creates `paste(id, year)` keys for every row, builds a named lookup vector of 6.46M entries, then for each of the 6.46M rows, pastes neighbor keys and looks them up. This is O(nÂ²)-flavored in practice due to R's named-vector lookup costs.

3. **The neighbor lookup is rebuilt identically for every variable** (implicitly, since it's called once but the architecture invites re-calling). Even as written, the single call produces a 6.46M-element list where each element stores *row indices* into the full cellÃ—year table â€” meaning the topology is entangled with the panel structure.

4. **`compute_neighbor_stats` iterates over 6.46M list elements in R-level `lapply`.** Each iteration subsets a vector, removes NAs, and computes three summary statistics. This is millions of R function calls with no vectorization.

### The Key Insight

The neighbor graph is **static across years**. Cell *i*'s neighbors are always the same cells regardless of year. Therefore:

- Build the neighbor topology **once**, at the **cell level** (344K cells), not the cellÃ—year level (6.46M rows).
- For each variable and each year, extract the variable column, and compute neighbor max/min/mean using **vectorized matrix operations** over the static cell-level neighbor structure.

This reduces the problem from 6.46M list-element iterations to 28 (years) Ã— 5 (variables) = 140 vectorized passes over 344K cells â€” a ~46,000Ã— reduction in loop iterations, plus each pass can be heavily vectorized.

---

## Optimization Strategy

### Architecture: Separate Static Topology from Dynamic Computation

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  STATIC (built once)            â”‚
â”‚  â€¢ cell_id â†’ integer index map  â”‚
â”‚  â€¢ sparse adjacency matrix W    â”‚
â”‚    (344,208 Ã— 344,208)          â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â”‚
               â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  DYNAMIC (per variable Ã— year)  â”‚
â”‚  â€¢ Extract value vector v       â”‚
â”‚  â€¢ neighbor_max  = row-wise max â”‚
â”‚  â€¢ neighbor_min  = row-wise min â”‚
â”‚  â€¢ neighbor_mean = (W %*% v)/k  â”‚
â”‚  All via sparse matrix ops      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### Specific Optimizations

1. **Sparse adjacency matrix:** Convert `rook_neighbors_unique` (an `nb` object) into a sparse `dgCMatrix` (from the `Matrix` package). This is a one-time O(344K) operation.

2. **Vectorized neighbor mean:** `W %*% v / k` where `k` is the number of neighbors per cell. This is a single sparse matrix-vector multiply â€” highly optimized in compiled code.

3. **Vectorized neighbor max/min:** Use the sparse matrix structure to iterate at the C level. We can use `data.table` grouped operations or a custom sparse-row iteration. Alternatively, we build an edge list and use `data.table` grouping.

4. **Year-level splitting with `data.table`:** Split the panel by year, compute neighbor stats for each year's 344K cells, and reassemble. `data.table` provides fast split-apply-combine.

5. **Memory:** The sparse matrix is ~1.4M non-zeros (directed edges) Ã— 12 bytes â‰ˆ 17 MB. The full dataset at 6.46M Ã— 110 columns â‰ˆ 5.7 GB fits in 16 GB RAM. Intermediate vectors are 344K Ã— 8 bytes â‰ˆ 2.7 MB each â€” negligible.

### Expected Runtime

- Sparse matrix construction: ~1 second
- Per variable Ã— year (sparse mat-vec + grouped max/min): ~0.05 seconds
- Total: 140 passes Ã— 0.05s â‰ˆ 7 seconds + overhead â‰ˆ **under 1 minute**

This is a **~5,000Ã— speedup** over the estimated 86+ hours.

---

## Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static spatial topology from dynamic (year-varying) cell values.
#
# Prerequisites:
#   - cell_data: data.frame/data.table with columns: id, year, ntl, ec,
#                pop_density, def, usd_est_n2 (and all other predictor columns)
#   - id_order: vector of cell IDs in the order matching rook_neighbors_unique
#   - rook_neighbors_unique: an nb object (from spdep) with neighbor indices
#   - rf_model: the pre-trained Random Forest model (untouched)
#
# Output:
#   - cell_data with 15 new columns: {var}_neighbor_max, {var}_neighbor_min,
#     {var}_neighbor_mean for each of the 5 neighbor source variables.
#   - Numerically identical to the original implementation.
# ==============================================================================

library(data.table)
library(Matrix)

# --------------------------------------------------------------------------
# STEP 1: Build static spatial topology (ONCE)
# --------------------------------------------------------------------------

build_static_neighbor_topology <- function(id_order, neighbors_nb) {
  # Convert the nb object to a sparse adjacency matrix (344,208 Ã— 344,208).
  # This encodes the static rook-neighbor relationships.
  
  n_cells <- length(id_order)
  
  # Build COO (coordinate) triplets from the nb object
  from_idx <- integer(0)
  to_idx   <- integer(0)
  
  for (i in seq_len(n_cells)) {
    nb_i <- neighbors_nb[[i]]
    # spdep::nb uses 0 to indicate no neighbors
    if (length(nb_i) == 1L && nb_i[1] == 0L) next
    from_idx <- c(from_idx, rep(i, length(nb_i)))
    to_idx   <- c(to_idx, nb_i)
  }
  
  # Sparse adjacency matrix: W[i, j] = 1 if j is a neighbor of i
  W <- sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = 1,
    dims = c(n_cells, n_cells)
  )
  
  # Number of neighbors per cell (for computing means)
  n_neighbors <- as.integer(rowSums(W))
  
  # Also build an edge-list data.table for fast grouped max/min
  edge_dt <- data.table(
    from_cell_idx = from_idx,
    to_cell_idx   = to_idx
  )
  
  # Cell ID to integer index mapping
  cell_id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))
  
  list(
    W            = W,
    n_neighbors  = n_neighbors,
    edge_dt      = edge_dt,
    cell_id_to_idx = cell_id_to_idx,
    n_cells      = n_cells,
    id_order     = id_order
  )
}

# --------------------------------------------------------------------------
# STEP 1 (alternative): Faster nb-to-sparse construction avoiding grow-in-loop
# --------------------------------------------------------------------------

build_static_neighbor_topology_fast <- function(id_order, neighbors_nb) {
  n_cells <- length(id_order)
  
  # Pre-calculate total number of edges for pre-allocation
  edge_counts <- vapply(neighbors_nb, function(nb) {
    if (length(nb) == 1L && nb[1] == 0L) 0L else length(nb)
  }, integer(1))
  
  total_edges <- sum(edge_counts)
  
  from_idx <- integer(total_edges)
  to_idx   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    k <- edge_counts[i]
    if (k == 0L) next
    from_idx[pos:(pos + k - 1L)] <- i
    to_idx[pos:(pos + k - 1L)]   <- neighbors_nb[[i]]
    pos <- pos + k
  }
  
  W <- sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = 1,
    dims = c(n_cells, n_cells)
  )
  
  n_neighbors <- as.integer(rowSums(W))
  
  edge_dt <- data.table(
    from_cell_idx = from_idx,
    to_cell_idx   = to_idx
  )
  setkey(edge_dt, from_cell_idx)
  
  cell_id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))
  
  list(
    W              = W,
    n_neighbors    = n_neighbors,
    edge_dt        = edge_dt,
    cell_id_to_idx = cell_id_to_idx,
    n_cells        = n_cells,
    id_order       = id_order
  )
}

# --------------------------------------------------------------------------
# STEP 2: Compute neighbor stats for one variable across all years
# --------------------------------------------------------------------------

compute_neighbor_features_optimized <- function(cell_data_dt, var_name, topology) {
  # For each year, extract the variable values as a cell-indexed vector,
  # then compute neighbor max, min, mean using the static topology.
  
  W           <- topology$W
  n_neighbors <- topology$n_neighbors
  edge_dt     <- topology$edge_dt
  cell_id_to_idx <- topology$cell_id_to_idx
  n_cells     <- topology$n_cells
  
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  # Initialize output columns
  cell_data_dt[, (col_max)  := NA_real_]
  cell_data_dt[, (col_min)  := NA_real_]
  cell_data_dt[, (col_mean) := NA_real_]
  
  years <- sort(unique(cell_data_dt$year))
  
  for (yr in years) {
    # Get row indices in cell_data_dt for this year
    yr_row_idx <- which(cell_data_dt$year == yr)
    
    # Build a cell-indexed value vector for this year
    # Map each row's cell id to the cell index
    yr_ids  <- cell_data_dt$id[yr_row_idx]
    yr_vals <- cell_data_dt[[var_name]][yr_row_idx]
    
    # Create a full cell-indexed vector (NA for cells not present this year)
    val_vec <- rep(NA_real_, n_cells)
    cell_indices <- cell_id_to_idx[as.character(yr_ids)]
    val_vec[cell_indices] <- yr_vals
    
    # --- Neighbor MEAN via sparse matrix-vector multiply ---
    # Replace NA with 0 for the multiply, but track valid counts
    val_vec_0 <- val_vec
    val_vec_0[is.na(val_vec_0)] <- 0
    
    valid_indicator <- as.double(!is.na(val_vec))
    
    neighbor_sum   <- as.numeric(W %*% val_vec_0)
    neighbor_count <- as.numeric(W %*% valid_indicator)
    
    neighbor_mean_vec <- ifelse(neighbor_count > 0,
                                neighbor_sum / neighbor_count,
                                NA_real_)
    
    # --- Neighbor MAX and MIN via edge list + data.table grouping ---
    # Look up neighbor values
    edge_dt[, val := val_vec[to_cell_idx]]
    
    # Grouped max and min (excluding NAs)
    stats <- edge_dt[!is.na(val),
                     .(nb_max = max(val), nb_min = min(val)),
                     by = from_cell_idx]
    
    # Build full cell-indexed result vectors
    neighbor_max_vec <- rep(NA_real_, n_cells)
    neighbor_min_vec <- rep(NA_real_, n_cells)
    neighbor_max_vec[stats$from_cell_idx] <- stats$nb_max
    neighbor_min_vec[stats$from_cell_idx] <- stats$nb_min
    
    # --- Write results back to cell_data_dt ---
    # Map from cell index back to row index for this year
    set(cell_data_dt, i = yr_row_idx, j = col_max,
        value = neighbor_max_vec[cell_indices])
    set(cell_data_dt, i = yr_row_idx, j = col_min,
        value = neighbor_min_vec[cell_indices])
    set(cell_data_dt, i = yr_row_idx, j = col_mean,
        value = neighbor_mean_vec[cell_indices])
  }
  
  # Clean up temporary column in edge_dt
  edge_dt[, val := NULL]
  
  cell_data_dt
}

# --------------------------------------------------------------------------
# STEP 3: Main pipeline â€” drop-in replacement for the original outer loop
# --------------------------------------------------------------------------

# Convert to data.table if not already (in-place, no copy)
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Build static topology ONCE (~1-5 seconds)
message("Building static neighbor topology...")
topology <- build_static_neighbor_topology_fast(
  id_order     = id_order,
  neighbors_nb = rook_neighbors_unique
)
message(sprintf("  Topology built: %d cells, %d directed edges.",
                topology$n_cells, nrow(topology$edge_dt)))

# Compute neighbor features for each source variable (~5-10 seconds each)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor features for '%s'...", var_name))
  t0 <- proc.time()
  cell_data <- compute_neighbor_features_optimized(cell_data, var_name, topology)
  elapsed <- (proc.time() - t0)[3]
  message(sprintf("  Done in %.1f seconds.", elapsed))
}

# Verify all 15 neighbor feature columns exist
expected_cols <- paste0(
  rep(neighbor_source_vars, each = 3),
  c("_neighbor_max", "_neighbor_min", "_neighbor_mean")
)
stopifnot(all(expected_cols %in% names(cell_data)))

message("All neighbor features computed. Ready for Random Forest prediction.")

# --------------------------------------------------------------------------
# STEP 4: Predict with the pre-trained Random Forest (UNCHANGED)
# --------------------------------------------------------------------------
# The rf_model is used as-is; no retraining.
# cell_data now has all required predictor columns including the 15 neighbor features.
#
# Example (unchanged from original pipeline):
#   cell_data$prediction <- predict(rf_model, newdata = cell_data)
# --------------------------------------------------------------------------
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup granularity** | 6.46M cell-year rows | 344K cells (static) |
| **Lookup data structure** | Named character vector + paste keys | Sparse matrix + integer index map |
| **Neighbor mean** | R-level lapply over 6.46M elements | Sparse matrix-vector multiply (compiled C) |
| **Neighbor max/min** | R-level lapply with per-element subsetting | `data.table` grouped aggregation (compiled C) |
| **Estimated runtime** | ~86+ hours | **< 1 minute** |
| **Peak memory** | ~6.46M-element list of integer vectors | ~17 MB sparse matrix + 2.7 MB vectors |
| **Numerical result** | Original estimand | **Identical** (same max/min/mean, same NA handling) |
| **RF model** | Pre-trained | **Untouched â€” no retraining** |

### Why Numerical Equivalence Is Preserved

- **Neighbor mean:** The original computes `mean(non-NA neighbor values)` = `sum / count_non_NA`. The optimized version computes `(W %*% v_zero_filled) / (W %*% valid_indicator)`, which is algebraically identical: the numerator sums only non-NA values (NAs replaced with 0 contribute nothing), and the denominator counts non-NA neighbors.
- **Neighbor max/min:** The edge-list approach looks up the exact same neighbor values and applies the same `max`/`min` after excluding NAs, grouped by the same source cell.
- **NA propagation:** Cells with zero valid neighbors produce `NA` for all three statistics, matching the original `if (length(neighbor_vals) == 0) return(c(NA, NA, NA))`.
END_RESPONSE>>>
