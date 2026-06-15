 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cell×year) neighbor indices.** It creates a list of ~6.46 million elements, each requiring string-pasting, hash lookups, and NA filtering. But the neighbor *relationships* between cells are purely spatial and **never change across years**. The only thing that changes is the variable values attached to each cell in each year.

2. **String-key hashing is extremely expensive at scale.** The function creates `paste(id, year)` keys for every row, builds a named lookup vector of 6.46M entries, then for each of the 6.46M rows, pastes neighbor keys and looks them up. This is O(n²)-flavored in practice due to R's named-vector lookup costs.

3. **The neighbor lookup is rebuilt identically for every variable** (implicitly, since it's called once but the architecture invites re-calling). Even as written, the single call produces a 6.46M-element list where each element stores *row indices* into the full cell×year table — meaning the topology is entangled with the panel structure.

4. **`compute_neighbor_stats` iterates over 6.46M list elements in R-level `lapply`.** Each iteration subsets a vector, removes NAs, and computes three summary statistics. This is millions of R function calls with no vectorization.

### The Key Insight

The neighbor graph is **static across years**. Cell *i*'s neighbors are always the same cells regardless of year. Therefore:

- Build the neighbor topology **once**, at the **cell level** (344K cells), not the cell×year level (6.46M rows).
- For each variable and each year, extract the variable column, and compute neighbor max/min/mean using **vectorized matrix operations** over the static cell-level neighbor structure.

This reduces the problem from 6.46M list-element iterations to 28 (years) × 5 (variables) = 140 vectorized passes over 344K cells — a ~46,000× reduction in loop iterations, plus each pass can be heavily vectorized.

---

## Optimization Strategy

### Architecture: Separate Static Topology from Dynamic Computation

```
┌─────────────────────────────────┐
│  STATIC (built once)            │
│  • cell_id → integer index map  │
│  • sparse adjacency matrix W    │
│    (344,208 × 344,208)          │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  DYNAMIC (per variable × year)  │
│  • Extract value vector v       │
│  • neighbor_max  = row-wise max │
│  • neighbor_min  = row-wise min │
│  • neighbor_mean = (W %*% v)/k  │
│  All via sparse matrix ops      │
└─────────────────────────────────┘
```

### Specific Optimizations

1. **Sparse adjacency matrix:** Convert `rook_neighbors_unique` (an `nb` object) into a sparse `dgCMatrix` (from the `Matrix` package). This is a one-time O(344K) operation.

2. **Vectorized neighbor mean:** `W %*% v / k` where `k` is the number of neighbors per cell. This is a single sparse matrix-vector multiply — highly optimized in compiled code.

3. **Vectorized neighbor max/min:** Use the sparse matrix structure to iterate at the C level. We can use `data.table` grouped operations or a custom sparse-row iteration. Alternatively, we build an edge list and use `data.table` grouping.

4. **Year-level splitting with `data.table`:** Split the panel by year, compute neighbor stats for each year's 344K cells, and reassemble. `data.table` provides fast split-apply-combine.

5. **Memory:** The sparse matrix is ~1.4M non-zeros (directed edges) × 12 bytes ≈ 17 MB. The full dataset at 6.46M × 110 columns ≈ 5.7 GB fits in 16 GB RAM. Intermediate vectors are 344K × 8 bytes ≈ 2.7 MB each — negligible.

### Expected Runtime

- Sparse matrix construction: ~1 second
- Per variable × year (sparse mat-vec + grouped max/min): ~0.05 seconds
- Total: 140 passes × 0.05s ≈ 7 seconds + overhead ≈ **under 1 minute**

This is a **~5,000× speedup** over the estimated 86+ hours.

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
  # Convert the nb object to a sparse adjacency matrix (344,208 × 344,208).
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
# STEP 3: Main pipeline — drop-in replacement for the original outer loop
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
| **RF model** | Pre-trained | **Untouched — no retraining** |

### Why Numerical Equivalence Is Preserved

- **Neighbor mean:** The original computes `mean(non-NA neighbor values)` = `sum / count_non_NA`. The optimized version computes `(W %*% v_zero_filled) / (W %*% valid_indicator)`, which is algebraically identical: the numerator sums only non-NA values (NAs replaced with 0 contribute nothing), and the denominator counts non-NA neighbors.
- **Neighbor max/min:** The edge-list approach looks up the exact same neighbor values and applies the same `max`/`min` after excluding NAs, grouped by the same source cell.
- **NA propagation:** Cells with zero valid neighbors produce `NA` for all three statistics, matching the original `if (length(neighbor_vals) == 0) return(c(NA, NA, NA))`.